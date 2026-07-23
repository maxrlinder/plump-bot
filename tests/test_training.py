import csv
import copy
import dataclasses
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from plump.env import PlumpEnv
from plump.modeling import ModelConfig, SCHEMA_VERSION, encode_observation
from plump.modeling.torch_model import (
    PlumpTransformerModel,
    encoded_observations_to_batch,
)
from plump.policies import ModelPolicy, RandomPolicy
from plump.rounds import RoundSpec, round_game_config
from plump.training import (
    PPOTrainer,
    PositionBaseline,
    TrainingConfig,
    TrainingRunLogger,
    compute_relative_rewards,
    training_config_snapshot,
)
from plump.training.ppo import (
    _argmax_capped_probabilities,
    _explained_variance,
    _non_argmax_tempered_probabilities,
    _non_argmax_uniform_probabilities,
    _trick_implied_relative_values,
    solve_meta_mixture,
)


class TrainingTest(unittest.TestCase):
    def _trainer(self, *, player_counts=(3,), hand_sizes=(3,), **overrides) -> PPOTrainer:
        model_config = ModelConfig(
            max_seq_len=48,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            context_hidden_dim=64,
            dropout=0.0,
        )
        config = TrainingConfig(
            player_counts=player_counts,
            hand_sizes=hand_sizes,
            rounds_per_configuration=1,
            num_envs=2,
            ppo_epochs=1,
            minibatch_size=64,
            learning_rate=1e-3,
            self_play_fraction=1.0,
            historical_fraction=0.0,
            heuristic_fraction=0.0,
            mixed_fraction=0.0,
            seed=3,
            device="cpu",
            model_config=model_config,
            **overrides,
        )
        return PPOTrainer(PlumpTransformerModel(model_config), config)

    def test_compute_relative_rewards(self):
        rewards = compute_relative_rewards({0: 10, 1: 4, 2: 1})
        self.assertEqual(rewards, {0: 7.5, 1: -1.5, 2: -6.0})

    def test_balanced_schedule_and_macro_round_weights(self):
        trainer = self._trainer(player_counts=(3, 4), hand_sizes=(3, 4))
        schedule = trainer.balanced_round_specs()
        self.assertEqual(len(schedule), 4)
        self.assertEqual(set(schedule), set(trainer.config.specs))

        buffer = trainer.collect_rollouts()
        counts = {spec: 0 for spec in trainer.config.specs}
        for outcome in buffer.round_outcomes:
            counts[outcome.spec] += 1
        self.assertEqual(set(counts.values()), {1})
        # Self-play rounds train every seat, so each round's weight is split
        # evenly across its seat trajectories.
        for sample in buffer.samples:
            self.assertEqual(
                sample.round_weight,
                1.0 / (4.0 * sample.spec.num_players),
            )

    def test_historical_arm_falls_back_to_self_play_without_history(self):
        trainer = self._trainer()
        trainer.config = dataclasses.replace(
            trainer.config,
            self_play_fraction=0.3,
            historical_fraction=0.2,
            heuristic_fraction=0.2,
            mixed_fraction=0.3,
        )

        fractions = trainer._effective_arm_fractions()

        self.assertEqual(fractions["historical"], 0.0)
        self.assertEqual(fractions["self"], 0.5)

    def test_opponent_mix_is_gradient_weighted_30_30_30_10(self):
        trainer = self._trainer()
        trainer.config = dataclasses.replace(
            trainer.config,
            rounds_per_configuration=16,
            self_play_fraction=0.3,
            heuristic_fraction=0.3,
            mixed_fraction=0.3,
            historical_fraction=0.1,
        )
        trainer.add_historical_policy(RandomPolicy(19))

        buffer = trainer.collect_rollouts()
        arm_counts = {
            arm: sum(outcome.opponent_arm == arm for outcome in buffer.round_outcomes)
            for arm in ("self", "heuristic", "mixed", "historical")
        }
        weight_totals = {
            arm: sum(
                sample.round_weight
                for sample in buffer.samples
                if sample.opponent_arm == arm
            )
            for arm in ("self", "heuristic", "mixed", "historical")
        }
        total_weight = sum(weight_totals.values())

        self.assertEqual(arm_counts, {
            "self": 5,
            "heuristic": 5,
            "mixed": 5,
            "historical": 1,
        })
        self.assertAlmostEqual(weight_totals["self"] / total_weight, 0.3)
        self.assertAlmostEqual(weight_totals["heuristic"] / total_weight, 0.3)
        self.assertAlmostEqual(weight_totals["mixed"] / total_weight, 0.3)
        self.assertAlmostEqual(weight_totals["historical"] / total_weight, 0.1)

    def test_current_policy_seats_are_trained_and_frozen_arms_store_focal_only(self):
        trainer = self._trainer()
        trainer.config = dataclasses.replace(
            trainer.config,
            rounds_per_configuration=16,
            self_play_fraction=0.3,
            heuristic_fraction=0.3,
            mixed_fraction=0.3,
            historical_fraction=0.1,
        )
        trainer.add_historical_policy(RandomPolicy(23))

        buffer = trainer.collect_rollouts()
        arms_by_episode = {
            outcome.episode_id: outcome.opponent_arm
            for outcome in buffer.round_outcomes
        }
        acting_players_by_episode: dict[int, set[int]] = {}
        for sample in buffer.samples:
            acting_players_by_episode.setdefault(sample.episode_id, set()).add(
                sample.acting_player
            )

        self.assertEqual(len(acting_players_by_episode), 16)
        num_players = trainer.config.specs[0].num_players
        decisions_per_seat = 1 + trainer.config.specs[0].hand_size
        for episode_id, players in acting_players_by_episode.items():
            arm = arms_by_episode[episode_id]
            if arm == "self":
                self.assertEqual(len(players), num_players)
            elif arm in ("heuristic", "historical"):
                self.assertEqual(len(players), 1)
            episode_samples = [
                sample
                for sample in buffer.samples
                if sample.episode_id == episode_id
            ]
            self.assertEqual(
                len(episode_samples),
                len(players) * decisions_per_seat,
            )

    def test_self_play_seat_returns_are_relative_and_sum_to_zero(self):
        trainer = self._trainer()
        buffer = trainer.collect_rollouts()

        last_return_by_player: dict[int, float] = {}
        for sample in buffer.samples:
            last_return_by_player[sample.acting_player] = float(
                sample.return_target
            )
        self.assertEqual(len(last_return_by_player), 3)
        self.assertAlmostEqual(
            sum(last_return_by_player.values()),
            0.0,
            places=6,
        )

    def test_mixed_arm_seats_current_and_historical_fifty_fifty(self):
        trainer = self._trainer()
        historical = RandomPolicy(29)
        trainer.add_historical_policy(historical)
        counts = {"self": 0, "historical": 0}

        for episode_id in range(200):
            episode = trainer._new_active_episode(
                trainer.config.specs[0],
                "mixed",
                episode_id,
                iteration=1,
            )
            for policy in episode.opponent_policies.values():
                if policy is None:
                    counts["self"] += 1
                elif policy is historical:
                    counts["historical"] += 1
                else:
                    self.fail("mixed arm seated a heuristic opponent")

        total = counts["self"] + counts["historical"]
        self.assertGreater(counts["self"], total * 0.35)
        self.assertGreater(counts["historical"], total * 0.35)

    def test_per_arm_explore_eps_overrides_fall_back_to_global(self):
        from types import SimpleNamespace

        trainer = self._trainer(
            explore_eps_bid=0.08,
            explore_eps_play=0.02,
            explore_eps_by_arm={"historical": (0.25, 0.10)},
        )
        request = lambda arm, phase, collect=True: SimpleNamespace(
            opponent_arm=arm,
            phase=phase,
            collect=collect,
            explore_uniform=False,
        )
        self.assertEqual(trainer._request_explore_eps(request("historical", "bid")), 0.25)
        self.assertEqual(trainer._request_explore_eps(request("historical", "play")), 0.10)
        self.assertEqual(trainer._request_explore_eps(request("self", "bid")), 0.08)
        self.assertEqual(trainer._request_explore_eps(request("mixed", "play")), 0.02)
        # Frozen-copy seats (collect=False) always sample raw.
        self.assertEqual(
            trainer._request_explore_eps(request("historical", "bid", collect=False)),
            0.0,
        )

    def test_explore_eps_by_arm_validates_arm_names_and_ranges(self):
        with self.assertRaises(ValueError):
            self._trainer(explore_eps_by_arm={"bogus": (0.1, 0.1)})
        with self.assertRaises(ValueError):
            self._trainer(explore_eps_by_arm={"self": (1.5, 0.1)})

    def test_uniform_league_draws_every_member(self):
        trainer = self._trainer(league_meta_solver="uniform")
        for index in range(3):
            trainer.add_historical_policy(RandomPolicy(index), snapshot_id=f"snap_{index}")
        drawn = {
            trainer._draw_historical_snapshot_unbatched().snapshot_id
            for _ in range(200)
        }
        self.assertEqual(drawn, {"snap_0", "snap_1", "snap_2"})

    def test_replace_historical_snapshots_keeps_loaded_members(self):
        from pathlib import Path

        trainer = self._trainer(league_meta_solver="uniform")
        for index in range(1, 4):
            trainer.add_historical_policy(
                RandomPolicy(index),
                snapshot_id=f"plump_v4_iter_{index:05d}",
            )
        kept = trainer.historical_snapshots[1]
        trainer.replace_historical_snapshots(
            [Path("nowhere/plump_v4_iter_00002.pt")]
        )
        self.assertEqual(
            [snapshot.snapshot_id for snapshot in trainer.historical_snapshots],
            ["plump_v4_iter_00002"],
        )
        self.assertIs(trainer.historical_snapshots[0], kept)

    def test_mixed_arm_without_history_uses_current_policy_only(self):
        trainer = self._trainer()
        episode = trainer._new_active_episode(
            trainer.config.specs[0],
            "mixed",
            0,
            iteration=1,
        )
        self.assertTrue(
            all(policy is None for policy in episode.opponent_policies.values())
        )

    def test_rollouts_assign_residual_value_and_auxiliary_targets(self):
        trainer = self._trainer()
        buffer = trainer.collect_rollouts()

        self.assertEqual(len(buffer.round_outcomes), 1)
        self.assertTrue(buffer.samples)
        self.assertTrue(trainer.position_baseline.values)
        for sample in buffer.samples:
            self.assertIsNotNone(sample.return_target)
            self.assertIsNotNone(sample.value_target)
            self.assertEqual(sample.position_intercept, 0.0)
            self.assertEqual(len(sample.final_trick_targets), 5)
            self.assertEqual(len(sample.final_bid_targets), 5)
            self.assertEqual(len(sample.owner_targets), 52)
            active_owner_targets = [target for target in sample.owner_targets if target >= 0]
            self.assertTrue(active_owner_targets)
            self.assertTrue(all(target < trainer.config.model_config.owner_class_count for target in active_owner_targets))
            self.assertFalse(sample.encoded.game_context_enabled)

    def test_heuristic_reward_uses_only_focal_round_rewards(self):
        trainer = self._trainer()
        trainer.config = dataclasses.replace(
            trainer.config,
            self_play_fraction=0.0,
            historical_fraction=0.0,
            heuristic_fraction=1.0,
            mixed_fraction=0.0,
        )
        buffer = trainer.collect_rollouts()
        stats = trainer.summarize_rollout(buffer)

        self.assertEqual(stats.heuristic_rounds, 1)
        self.assertEqual(buffer.round_outcomes[0].opponent_arm, "heuristic")
        self.assertEqual(
            stats.heuristic_relative_reward,
            buffer.round_outcomes[0].focal_reward,
        )

    def test_trick_baseline_folds_implied_value_into_intercepts(self):
        trainer = self._trainer()
        trainer.config = dataclasses.replace(trainer.config, trick_baseline=True)

        buffer = trainer.collect_rollouts()
        stats = trainer.update(buffer)

        self.assertTrue(math.isfinite(stats.total_loss))
        play_intercepts = [
            sample.position_intercept
            for sample in buffer.samples
            if sample.phase == "play"
        ]
        self.assertTrue(any(abs(value) > 0.0 for value in play_intercepts))
        for sample in buffer.samples:
            self.assertAlmostEqual(
                sample.old_value,
                sample.position_intercept + sample.old_residual_value,
                places=4,
            )
            self.assertAlmostEqual(
                sample.value_target,
                sample.return_target - sample.position_intercept,
                places=4,
            )

    def test_position_baseline_is_lagged(self):
        baseline = PositionBaseline(decay=0.5)
        key = (3, 3, 0)
        self.assertEqual(baseline.get(key), 0.0)
        baseline.update_many([(key, 10.0), (key, 6.0)])
        self.assertEqual(baseline.get(key), 8.0)
        baseline.update_many([(key, 4.0)])
        self.assertEqual(baseline.get(key), 6.0)

    def test_update_and_diagnostics_are_finite(self):
        trainer = self._trainer()
        buffer = trainer.collect_rollouts()
        stats = trainer.update(buffer)
        prediction = trainer.compute_prediction_stats(buffer, max_samples=8, minibatch_size=4)

        for value in (
            stats.total_loss,
            stats.policy_loss,
            stats.value_loss,
            stats.entropy,
            stats.auxiliary_loss,
            stats.trick_loss,
            stats.owner_loss,
            stats.owner_ce_loss,
            stats.owner_capacity_loss,
            prediction.value_mse,
            prediction.owner_brier,
            prediction.owner_opponent_accuracy,
            prediction.owner_opponent_true_prob,
            prediction.owner_capacity_mae,
            prediction.owner_raw_capacity_mae,
            prediction.hit_prob_brier,
            prediction.trick_implied_value_explained_variance,
            prediction.bid_trick_implied_value_explained_variance,
            prediction.play_trick_implied_value_explained_variance,
        ):
            self.assertTrue(math.isfinite(value))
        self.assertEqual(stats.configurations, 1)
        self.assertGreaterEqual(prediction.owner_accuracy, 0.0)
        self.assertLessEqual(prediction.owner_accuracy, 1.0)
        self.assertGreaterEqual(prediction.owner_opponent_accuracy, 0.0)
        self.assertLessEqual(prediction.owner_opponent_accuracy, 1.0)
        self.assertLess(prediction.owner_capacity_max_error, 1e-4)
        self.assertGreater(prediction.owner_raw_capacity_mae, 0.0)

    def test_trick_implied_value_uses_expected_relative_score(self):
        probabilities = torch.zeros((1, 3, 4))
        probabilities[0, 0, 1] = 0.8
        probabilities[0, 0, 0] = 0.2
        probabilities[0, 1, 0] = 0.5
        probabilities[0, 1, 1] = 0.5
        probabilities[0, 2, 2] = 0.25
        probabilities[0, 2, 0] = 0.75
        bids = torch.tensor([[1, 0, 2]])
        active = torch.tensor([[True, True, True]])

        implied = _trick_implied_relative_values(
            probabilities,
            bids,
            active,
        )

        self.assertAlmostEqual(float(implied[0]), 6.05, places=5)

    def test_explained_variance_ignores_constant_prediction_offset(self):
        self.assertAlmostEqual(
            _explained_variance(
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
            ),
            1.0,
        )

    def test_sinkhorn_iteration_count_must_be_positive(self):
        model_config = dataclasses.replace(
            self._trainer().config.model_config,
            owner_sinkhorn_iterations=0,
        )
        config = dataclasses.replace(
            self._trainer().config,
            model_config=model_config,
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            PPOTrainer(PlumpTransformerModel(model_config), config)

    def test_focal_decision_gae_bootstraps_across_opponent_actions(self):
        trainer = self._trainer()
        buffer = trainer.collect_rollouts()
        samples = [copy.deepcopy(buffer.samples[0]) for _ in range(3)]
        for sample, value in zip(samples, (1.0, 2.0, 3.0)):
            sample.old_value = value
            sample.position_intercept = 0.0

        trainer._assign_gae(
            samples,
            terminal_reward=10.0,
            gae_lambda=0.95,
        )

        self.assertAlmostEqual(samples[2].return_target, 10.0)
        self.assertAlmostEqual(samples[1].return_target, 9.65)
        self.assertAlmostEqual(samples[0].return_target, 9.2675)

    def test_routed_states_disable_only_ppo_policy_and_entropy(self):
        trainer = self._trainer()
        trainer.config = dataclasses.replace(
            trainer.config,
            value_coef=1.0,
            trick_coef=0.0,
            owner_coef=0.0,
        )
        buffer = trainer.collect_rollouts()
        before = {
            key: value.detach().clone()
            for key, value in trainer.model.value_head.state_dict().items()
        }
        for sample in buffer.samples:
            sample.ppo_policy_enabled = False

        stats = trainer.update(buffer)

        self.assertEqual(stats.policy_loss, 0.0)
        self.assertEqual(stats.entropy, 0.0)
        self.assertTrue(
            any(
                not torch.equal(before[key], value)
                for key, value in trainer.model.value_head.state_dict().items()
            )
        )

    def test_ppo_entropy_uses_only_post_mask_legal_actions(self):
        trainer = self._trainer()
        sample = next(
            row
            for row in trainer.collect_rollouts().samples
            if row.phase == "bid"
        )
        legal = [
            index
            for index, valid in enumerate(sample.encoded.legal_bid_mask)
            if valid
        ]
        illegal = next(
            index
            for index, valid in enumerate(sample.encoded.legal_bid_mask)
            if not valid
        )
        with torch.no_grad():
            trainer.model.bid_head.weight.zero_()
            trainer.model.bid_head.bias.zero_()
            trainer.model.bid_head.bias[illegal] = 1_000.0
            output = trainer.model(
                encoded_observations_to_batch(
                    [sample.encoded],
                    device="cpu",
                )
            )
            _, entropy = trainer._logprobs_and_entropy(
                output,
                [sample],
            )

        self.assertAlmostEqual(
            float(entropy[0]),
            math.log(len(legal)),
            places=5,
        )

    def test_complete_game_rollout_uses_masked_context_and_terminal_targets(self):
        trainer = self._trainer()
        trainer.config = dataclasses.replace(
            trainer.config,
            training_mode="game",
            game_schedule=(3, 2, 3),
            games_per_player_seat=1,
        )
        buffer = trainer.collect_rollouts()

        self.assertEqual(len(buffer.round_outcomes), 9)
        self.assertTrue(buffer.samples)
        self.assertTrue(
            all(sample.encoded.game_context_enabled for sample in buffer.samples)
        )
        self.assertTrue(
            all(sample.return_target is not None for sample in buffer.samples)
        )
        episode_ids = {sample.episode_id for sample in buffer.samples}
        self.assertEqual(len(episode_ids), 3)

    def test_microbatch_accumulation_matches_logical_minibatch(self):
        full = self._trainer()
        buffer = full.collect_rollouts()
        micro_config = dataclasses.replace(full.config, microbatch_size=3)
        micro_model = copy.deepcopy(full.model)
        full.optimizer = torch.optim.SGD(full.model.parameters(), lr=1e-3)
        micro = PPOTrainer(
            micro_model,
            micro_config,
            optimizer=torch.optim.SGD(micro_model.parameters(), lr=1e-3),
        )
        micro.rng.setstate(full.rng.getstate())

        full_stats = full.update(buffer)
        micro_stats = micro.update(buffer)

        for full_parameter, micro_parameter in zip(
            full.model.parameters(),
            micro.model.parameters(),
        ):
            self.assertTrue(
                torch.allclose(full_parameter, micro_parameter, atol=2e-6, rtol=2e-5)
            )
        self.assertAlmostEqual(full_stats.total_loss, micro_stats.total_loss, places=5)
        self.assertAlmostEqual(full_stats.owner_loss, micro_stats.owner_loss, places=5)

    def test_checkpoint_is_v4_and_older_resume_is_rejected(self):
        trainer = self._trainer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v4.pt"
            trainer.save_checkpoint(path, iteration=7)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
            self.assertIn("rules_fingerprint", payload)
            self.assertIn("position_baseline", payload)
            self.assertEqual(trainer.load_checkpoint(path)["iteration"], 7)

            legacy = Path(tmp) / "v1.pt"
            torch.save({"model_state_dict": {}}, legacy)
            with self.assertRaisesRegex(ValueError, "evaluation-only"):
                trainer.load_checkpoint(legacy)

    def test_schema_v4_optimizer_state_resumes_with_runtime_optimizations(self):
        trainer = self._trainer()
        trainer.update(trainer.collect_rollouts())
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "resume.pt"
            trainer.save_checkpoint(checkpoint, iteration=9)
            expected = trainer.optimizer.state_dict()

            restored = self._trainer()
            restored.config = dataclasses.replace(
                restored.config,
                event_length_buckets=(8, 16, 32, 64),
                batch_packing="numpy",
                lean_rollout_forward=True,
                batched_league_sampling=True,
            )
            info = restored.load_checkpoint(checkpoint, load_optimizer=True)
            actual = restored.optimizer.state_dict()

            self.assertEqual(info["iteration"], 9)
            self.assertTrue(info["optimizer_loaded"])
            self.assertEqual(expected["param_groups"], actual["param_groups"])
            self.assertEqual(set(expected["state"]), set(actual["state"]))
            for parameter_id, expected_state in expected["state"].items():
                actual_state = actual["state"][parameter_id]
                self.assertEqual(set(expected_state), set(actual_state))
                for name, expected_value in expected_state.items():
                    actual_value = actual_state[name]
                    if isinstance(expected_value, torch.Tensor):
                        self.assertTrue(
                            torch.equal(expected_value, actual_value),
                            (parameter_id, name),
                        )
                    else:
                        self.assertEqual(expected_value, actual_value)

    def test_single_optimizer_group_migrates_to_belief_split(self):
        trainer = self._trainer()
        trainer.update(trainer.collect_rollouts())
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "resume.pt"
            trainer.save_checkpoint(checkpoint, iteration=9)
            saved = trainer.optimizer.state_dict()

            restored = self._trainer(belief_head_only=True)
            info = restored.load_checkpoint(checkpoint, load_optimizer=True)
            actual = restored.optimizer.state_dict()

            self.assertTrue(info["optimizer_group_migrated"])
            self.assertEqual(
                [group["name"] for group in actual["param_groups"]],
                ["policy_value", "belief_heads"],
            )
            self.assertEqual(len(saved["state"]), len(actual["state"]))

            saved_ids = saved["param_groups"][0]["params"]
            saved_id_by_parameter = {
                id(parameter): saved_id
                for parameter, saved_id in zip(
                    restored.model.parameters(),
                    saved_ids,
                    strict=True,
                )
            }
            actual_id_by_parameter = {}
            for live_group, serialized_group in zip(
                restored.optimizer.param_groups,
                actual["param_groups"],
                strict=True,
            ):
                actual_id_by_parameter.update(
                    {
                        id(parameter): serialized_id
                        for parameter, serialized_id in zip(
                            live_group["params"],
                            serialized_group["params"],
                            strict=True,
                        )
                    }
                )
            for parameter in (
                restored.model.bid_head.weight,
                restored.model.trick_count_head.weight,
            ):
                saved_state = saved["state"][saved_id_by_parameter[id(parameter)]]
                actual_state = actual["state"][actual_id_by_parameter[id(parameter)]]
                for name in saved_state:
                    if isinstance(saved_state[name], torch.Tensor):
                        self.assertTrue(torch.equal(saved_state[name], actual_state[name]))
                    else:
                        self.assertEqual(saved_state[name], actual_state[name])

    def test_logger_uses_only_v4_metrics(self):
        trainer = self._trainer()
        buffer = trainer.collect_rollouts()
        rollout = trainer.summarize_rollout(buffer)
        update = trainer.update(buffer)
        prediction = trainer.compute_prediction_stats(buffer, max_samples=4, minibatch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            logger = TrainingRunLogger(tmp)
            logger.write_config({"training_config": training_config_snapshot(trainer.config)})
            logger.log_iteration(
                iteration=1,
                elapsed_sec=1.0,
                timings={"iteration_sec": 1.0},
                update=update,
                rollout=rollout,
                collection=trainer.last_collection_stats,
                prediction=prediction,
                evaluation=None,
                checkpoint_path=Path(tmp) / "checkpoint.pt",
            )
            fields = next(csv.reader((Path(tmp) / "metrics.csv").open()))
            self.assertIn("loss_owner", fields)
            self.assertIn("loss_owner_ce", fields)
            self.assertIn("loss_owner_capacity", fields)
            self.assertIn("pred_owner_accuracy", fields)
            self.assertIn("pred_owner_opponent_accuracy", fields)
            self.assertIn("pred_owner_capacity_mae", fields)
            self.assertIn(
                "pred_trick_implied_value_explained_variance",
                fields,
            )
            self.assertIn(
                "search_bid_sampler_infeasible_rejection_rate",
                fields,
            )
            self.assertIn("search_bid_entropy_floor_loss", fields)
            self.assertNotIn("loss_point", fields)
            self.assertNotIn("loss_hand", fields)


class OracleLeagueMmdTest(unittest.TestCase):
    def _trainer(self, **overrides) -> PPOTrainer:
        model_config = ModelConfig(
            max_seq_len=48,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            context_hidden_dim=64,
            dropout=0.0,
            oracle_critic=overrides.pop("oracle_critic_model", False),
            suit_presence_head=overrides.pop("suit_presence_head_model", False),
        )
        config = TrainingConfig(
            player_counts=(3,),
            hand_sizes=(3,),
            rounds_per_configuration=overrides.pop(
                "rounds_per_configuration",
                8,
            ),
            num_envs=2,
            ppo_epochs=overrides.pop("ppo_epochs", 1),
            minibatch_size=overrides.pop("minibatch_size", 64),
            learning_rate=1e-3,
            self_play_fraction=overrides.pop("self_play_fraction", 0.3),
            heuristic_fraction=overrides.pop("heuristic_fraction", 0.3),
            mixed_fraction=overrides.pop("mixed_fraction", 0.3),
            historical_fraction=overrides.pop("historical_fraction", 0.1),
            historical_max_snapshots=overrides.pop(
                "historical_max_snapshots",
                4,
            ),
            league_meta_solver=overrides.pop(
                "league_meta_solver",
                "softmax_ema",
            ),
            seed=3,
            device="cpu",
            model_config=model_config,
            **overrides,
        )
        return PPOTrainer(PlumpTransformerModel(model_config), config)

    def test_oracle_critic_drives_gae_baseline_and_losses(self):
        trainer = self._trainer(oracle_critic_model=True, oracle_critic=True)
        trainer.add_historical_policy(RandomPolicy(11))

        buffer = trainer.collect_rollouts()
        stats = trainer.update(buffer)

        self.assertTrue(math.isfinite(stats.oracle_value_loss))
        self.assertGreater(stats.oracle_value_loss, 0.0)
        # Oracle and plain heads are independently initialized, so the
        # GAE baseline should not simply equal the plain residual.
        differences = [
            abs((sample.old_value - sample.position_intercept) - sample.old_residual_value)
            for sample in buffer.samples
        ]
        self.assertGreater(max(differences), 1e-6)

    def test_oracle_critic_disabled_keeps_plain_baseline_and_zero_loss(self):
        trainer = self._trainer()
        trainer.add_historical_policy(RandomPolicy(11))

        buffer = trainer.collect_rollouts()
        stats = trainer.update(buffer)

        self.assertEqual(stats.oracle_value_loss, 0.0)
        self.assertEqual(stats.magnet_kl, 0.0)
        for sample in buffer.samples:
            self.assertAlmostEqual(
                sample.old_value - sample.position_intercept,
                sample.old_residual_value,
                places=5,
            )

    def test_oracle_critic_config_requires_model_head(self):
        with self.assertRaises(ValueError):
            self._trainer(oracle_critic=True)

    def test_league_draw_prefers_harder_snapshots(self):
        trainer = self._trainer(
            league_temperature=1.0,
            league_meta_solver="softmax_ema",
        )
        trainer.add_historical_policy(RandomPolicy(5), snapshot_id="easy")
        trainer.add_historical_policy(RandomPolicy(7), snapshot_id="hard")
        trainer.league_reward_ema["easy"] = 4.0
        trainer.league_reward_ema["hard"] = -4.0

        draws = [
            trainer._draw_historical_snapshot().snapshot_id
            for _ in range(200)
        ]
        self.assertGreater(draws.count("hard") / len(draws), 0.9)

    def test_league_unseen_snapshot_scored_as_hardest(self):
        trainer = self._trainer(
            league_temperature=1.0,
            league_meta_solver="softmax_ema",
        )
        trainer.add_historical_policy(RandomPolicy(5), snapshot_id="easy")
        trainer.add_historical_policy(RandomPolicy(7), snapshot_id="new")
        trainer.league_reward_ema["easy"] = 6.0

        draws = [
            trainer._draw_historical_snapshot().snapshot_id
            for _ in range(200)
        ]
        self.assertGreater(draws.count("new") / len(draws), 0.9)

    def test_batched_league_sampling_is_bounded_and_preserves_marginal(self):
        trainer = self._trainer(
            batched_league_sampling=True,
            league_probe_fraction=0.10,
            league_meta_solver="regret_matching",
        )
        for index, snapshot_id in enumerate(("a", "b", "c")):
            trainer.add_historical_policy(
                RandomPolicy(index),
                snapshot_id=snapshot_id,
            )
        trainer.league_meta_mixture = {"a": 0.6, "b": 0.3, "c": 0.1}

        counts = {snapshot_id: 0 for snapshot_id in ("a", "b", "c")}
        iterations = 20_000
        for _ in range(iterations):
            trainer._prepare_historical_iteration_sampling()
            available = {
                snapshot.snapshot_id
                for snapshot in (
                    trainer._iteration_exploit_snapshot,
                    trainer._iteration_probe_snapshot,
                )
                if snapshot is not None
            }
            self.assertLessEqual(len(available), 2)
            selected = trainer._draw_historical_snapshot().snapshot_id
            self.assertIn(selected, available)
            counts[selected] += 1

        for snapshot_id, meta_probability in trainer.league_meta_mixture.items():
            expected = 0.9 * meta_probability + 0.1 / 3.0
            self.assertAlmostEqual(
                counts[snapshot_id] / iterations,
                expected,
                delta=0.015,
            )

    def test_batched_league_sampling_edges_and_fallback(self):
        empty = self._trainer(
            batched_league_sampling=True,
            league_meta_solver="regret_matching",
        )
        empty._prepare_historical_iteration_sampling()
        self.assertIsNone(empty._iteration_exploit_snapshot)
        self.assertIsNone(empty._iteration_probe_snapshot)
        with self.assertRaisesRegex(RuntimeError, "No historical snapshots"):
            empty._draw_historical_snapshot()

        single = self._trainer(
            batched_league_sampling=True,
            league_meta_solver="regret_matching",
        )
        single.add_historical_policy(RandomPolicy(1), snapshot_id="only")
        single.league_meta_mixture = {"only": 1.0}
        single._prepare_historical_iteration_sampling()
        self.assertIs(
            single._iteration_exploit_snapshot,
            single._iteration_probe_snapshot,
        )
        self.assertEqual(
            {single._draw_historical_snapshot().snapshot_id for _ in range(50)},
            {"only"},
        )

        fallback = self._trainer(
            batched_league_sampling=True,
            league_meta_solver="softmax_ema",
        )
        fallback.add_historical_policy(RandomPolicy(1), snapshot_id="a")
        fallback.add_historical_policy(RandomPolicy(2), snapshot_id="b")
        fallback._prepare_historical_iteration_sampling()
        self.assertIsNotNone(fallback._iteration_exploit_snapshot)
        self.assertIsNone(fallback._iteration_probe_snapshot)

    def test_collection_reports_at_most_two_historical_model_policies(self):
        trainer = self._trainer(
            self_play_fraction=0.0,
            heuristic_fraction=0.0,
            mixed_fraction=0.0,
            historical_fraction=1.0,
            batched_league_sampling=True,
            league_meta_solver="regret_matching",
        )
        report = SimpleNamespace(macro_relative_reward=0.0)
        with patch("plump.training.ppo.evaluate_policy", return_value=report):
            for index, snapshot_id in enumerate(("a", "b", "c")):
                torch.manual_seed(index + 100)
                trainer.add_historical_policy(
                    ModelPolicy(
                        PlumpTransformerModel(trainer.config.model_config),
                        device="cpu",
                        greedy=False,
                    ),
                    snapshot_id=snapshot_id,
                )
        trainer.league_meta_mixture = {"a": 0.6, "b": 0.3, "c": 0.1}

        buffer = trainer.collect_rollouts()
        snapshot_ids = {
            outcome.opponent_snapshot_id
            for outcome in buffer.round_outcomes
        }

        self.assertGreater(trainer.last_collection_stats.historical_policy_count, 0)
        self.assertLessEqual(
            trainer.last_collection_stats.historical_policy_count,
            2,
        )
        self.assertLessEqual(len(snapshot_ids), 2)

    def test_collection_metrics_report_bucketing_and_lean_calls(self):
        trainer = self._trainer(
            self_play_fraction=1.0,
            heuristic_fraction=0.0,
            mixed_fraction=0.0,
            historical_fraction=0.0,
            event_length_buckets=(8, 16, 32, 64),
            lean_rollout_forward=True,
            batch_packing="numpy",
        )

        buffer = trainer.collect_rollouts()
        stats = trainer.last_collection_stats

        self.assertTrue(buffer.samples)
        self.assertGreater(stats.total_sec, 0.0)
        self.assertGreater(stats.current_forward_calls, 0)
        self.assertGreater(stats.current_forward_rows, 0)
        self.assertEqual(stats.historical_forward_calls, 0)
        self.assertEqual(stats.historical_policy_count, 0)
        self.assertGreaterEqual(
            stats.processed_event_tokens,
            stats.valid_event_tokens,
        )
        self.assertLess(
            stats.processed_event_tokens,
            stats.current_forward_rows * trainer.config.model_config.max_seq_len,
        )

    def test_historical_episodes_share_and_attribute_one_snapshot(self):
        trainer = self._trainer()
        trainer.add_historical_policy(RandomPolicy(5), snapshot_id="a")
        trainer.add_historical_policy(RandomPolicy(7), snapshot_id="b")

        buffer = trainer.collect_rollouts()
        historical = [
            outcome
            for outcome in buffer.round_outcomes
            if outcome.opponent_arm == "historical"
        ]
        self.assertTrue(historical)
        for outcome in historical:
            self.assertIn(outcome.opponent_snapshot_id, {"a", "b"})
            self.assertIn(outcome.opponent_snapshot_id, trainer.league_reward_ema)
        non_historical = [
            outcome
            for outcome in buffer.round_outcomes
            if outcome.opponent_arm != "historical"
        ]
        for outcome in non_historical:
            self.assertIsNone(outcome.opponent_snapshot_id)

    def test_league_checkpoint_roundtrip(self):
        trainer = self._trainer(league_meta_solver="regret_matching")
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = Path(tmp) / "plump_v4_iter_00001.pt"
            trainer.save_checkpoint(snapshot_path, iteration=1)
            report = SimpleNamespace(macro_relative_reward=0.0)
            with patch("plump.training.ppo.evaluate_policy", return_value=report):
                trainer.add_historical_checkpoint(snapshot_path)
            trainer.league_reward_ema["plump_v4_iter_00001"] = -1.25
            members = ["heuristic", "plump_v4_iter_00001", "current"]
            trainer.league_payoffs = {
                (row, column): float(row_index - column_index)
                for row_index, row in enumerate(members)
                for column_index, column in enumerate(members)
            }
            trainer._refresh_league_mixture()
            expected_payoffs = dict(trainer.league_payoffs)
            expected_mixture = dict(trainer.league_meta_mixture)
            checkpoint_path = Path(tmp) / "resume.pt"
            trainer.save_checkpoint(checkpoint_path, iteration=2)

            restored = self._trainer(league_meta_solver="regret_matching")
            with patch("plump.training.ppo.evaluate_policy", return_value=report):
                info = restored.load_checkpoint(checkpoint_path)

            self.assertEqual(info["league_snapshots_loaded"], 1)
            self.assertEqual(info["league_snapshots_missing"], [])
            self.assertEqual(
                [snapshot.snapshot_id for snapshot in restored.historical_snapshots],
                ["plump_v4_iter_00001"],
            )
            self.assertEqual(
                restored.league_reward_ema,
                {"plump_v4_iter_00001": -1.25},
            )
            self.assertEqual(restored.league_payoffs, expected_payoffs)
            self.assertEqual(restored.league_meta_mixture, expected_mixture)

    def test_league_snapshots_restore_beside_relocated_checkpoint(self):
        trainer = self._trainer()
        report = SimpleNamespace(macro_relative_reward=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            source_dir.mkdir()
            snapshot_path = source_dir / "plump_v4_iter_00001.pt"
            trainer.save_checkpoint(snapshot_path, iteration=1)
            with patch("plump.training.ppo.evaluate_policy", return_value=report):
                trainer.add_historical_checkpoint(snapshot_path)
            checkpoint_path = source_dir / "resume.pt"
            trainer.save_checkpoint(checkpoint_path, iteration=2)

            # Simulate a host migration: the run directory moves, so the
            # absolute snapshot paths stored in the checkpoint no longer exist.
            moved_dir = Path(tmp) / "moved"
            moved_dir.mkdir()
            for file in (snapshot_path, checkpoint_path):
                shutil.move(file, moved_dir / file.name)

            restored = self._trainer()
            with patch("plump.training.ppo.evaluate_policy", return_value=report):
                info = restored.load_checkpoint(moved_dir / "resume.pt")

            self.assertEqual(info["league_snapshots_loaded"], 1)
            self.assertEqual(info["league_snapshots_missing"], [])
            self.assertEqual(
                [snapshot.snapshot_id for snapshot in restored.historical_snapshots],
                ["plump_v4_iter_00001"],
            )

    def test_rollout_model_sync_tracks_updates(self):
        trainer = self._trainer(pipeline_rollouts=True)
        trainer.add_historical_policy(RandomPolicy(11))
        self.assertIsNotNone(trainer.rollout_model)

        buffer = trainer.collect_rollouts()
        trainer.update(buffer)

        live = torch.cat([p.flatten() for p in trainer.model.parameters()])
        snapshot = torch.cat(
            [p.flatten() for p in trainer.rollout_model.parameters()]
        )
        self.assertFalse(torch.equal(live, snapshot))
        trainer.sync_rollout_model()
        snapshot = torch.cat(
            [p.flatten() for p in trainer.rollout_model.parameters()]
        )
        self.assertTrue(torch.equal(live, snapshot))

    def test_pipelined_collection_trains_on_stale_buffer(self):
        from concurrent.futures import ThreadPoolExecutor

        trainer = self._trainer(pipeline_rollouts=True)
        trainer.add_historical_policy(RandomPolicy(11))
        with ThreadPoolExecutor(max_workers=1) as executor:
            buffer = trainer.collect_rollouts(iteration=1)
            future = executor.submit(trainer.collect_rollouts, iteration=2)
            first_stats = trainer.update(buffer)
            stale_buffer = future.result()
            trainer.sync_rollout_model()
            # Collected under the pre-update snapshot, trained on the updated
            # model: the one-step staleness the pipelined loop produces.
            stale_stats = trainer.update(stale_buffer)

        self.assertEqual(
            len(stale_buffer.round_outcomes),
            trainer.config.rounds_per_batch,
        )
        for stats in (first_stats, stale_stats):
            self.assertTrue(math.isfinite(stats.total_loss))
            self.assertTrue(math.isfinite(stats.approx_kl))
            self.assertGreaterEqual(stats.epochs_run, 1)

    def test_act_encoded_matches_act_many(self):
        trainer = self._trainer()
        policy = ModelPolicy(trainer.model, device="cpu", greedy=True)
        envs = []
        for seed in range(4):
            env = PlumpEnv(
                round_game_config(RoundSpec(3, 3), bidding_start_player=0),
                seed=seed,
            )
            env.reset()
            envs.append(env)

        via_envs = policy.act_many(envs)
        players = [env.current_player() for env in envs]
        phases = [env.phase() for env in envs]
        encoded = [
            encode_observation(
                env.get_observation(player),
                policy.model_config,
                include_game_context=False,
            )
            for env, player in zip(envs, players)
        ]
        via_encoded = policy.act_encoded(encoded, phases=phases, players=players)

        self.assertEqual(via_envs, via_encoded)

    def test_env_worker_collection_produces_trainable_buffer(self):
        from plump.training.env_workers import EnvWorkerPool

        trainer = self._trainer()
        trainer.add_historical_policy(
            ModelPolicy(
                PlumpTransformerModel(trainer.config.model_config),
                device="cpu",
                greedy=False,
            ),
            snapshot_id="worker_snap",
        )
        pool = EnvWorkerPool(
            num_workers=2,
            model_config=trainer.config.model_config,
            include_game_context=False,
            num_envs=trainer.config.num_envs,
        )
        try:
            trainer.env_pool = pool
            buffer = trainer.collect_rollouts()
        finally:
            trainer.env_pool = None
            pool.close()

        self.assertEqual(
            len(buffer.round_outcomes),
            trainer.config.rounds_per_batch,
        )
        self.assertEqual(
            {outcome.opponent_arm for outcome in buffer.round_outcomes},
            {"self", "heuristic", "mixed", "historical"},
        )
        stats = trainer.update(buffer)
        self.assertTrue(math.isfinite(stats.total_loss))
        self.assertTrue(math.isfinite(stats.approx_kl))
        self.assertGreater(stats.samples, 0)

    def test_explore_eps_records_behavior_mixture_logprobs(self):
        # eps=1 makes the behavior policy exactly uniform-over-legal, so every
        # recorded logprob must invert to an integer legal-action count.
        trainer = self._trainer(explore_eps_bid=1.0, explore_eps_play=1.0)
        trainer.add_historical_policy(RandomPolicy(11))

        buffer = trainer.collect_rollouts()

        sampled_bids = {
            sample.action_index
            for sample in buffer.samples
            if sample.phase == "bid"
        }
        self.assertGreaterEqual(len(sampled_bids), 3)
        for sample in buffer.samples:
            n_legal = 1.0 / math.exp(sample.old_logprob)
            self.assertAlmostEqual(n_legal, round(n_legal), places=3)
            self.assertGreaterEqual(round(n_legal), 1)
            self.assertIsNotNone(sample.old_policy_logprob)
        stats = trainer.update(buffer)
        self.assertTrue(math.isfinite(stats.approx_kl))
        # approx_kl gates on policy movement, not on the (large by
        # construction) policy-vs-exploration-mixture divergence.
        self.assertLess(stats.approx_kl, 0.05)

    def test_importance_weighted_kl_is_stable_for_exploratory_rare_actions(self):
        trainer = self._trainer(explore_eps_bid=0.5, explore_eps_play=0.5)
        trainer.add_historical_policy(RandomPolicy(11))
        buffer = trainer.collect_rollouts()
        rare = buffer.samples[0]
        # This action was plausible under the exploratory behavior but had
        # effectively zero probability under the collecting policy. The
        # importance-weighted form must stay finite even though its naive
        # exp(log-ratio) intermediate would overflow.
        rare.old_policy_logprob = rare.old_logprob - 1_000.0

        stats = trainer.update(buffer)

        self.assertTrue(math.isfinite(stats.approx_kl))
        self.assertTrue(math.isfinite(stats.total_loss))
        self.assertEqual(stats.skipped_steps, 0)

    def test_explore_eps_zero_keeps_policy_sampling(self):
        trainer = self._trainer()
        trainer.add_historical_policy(RandomPolicy(11))
        buffer = trainer.collect_rollouts()
        # Sharp-policy logprobs are the policy's own; nothing forces them to
        # invert to integer counts (regression guard for the default path).
        stats = trainer.update(buffer)
        self.assertTrue(math.isfinite(stats.total_loss))

    def test_snapshot_pool_eviction_prunes_reward_ema(self):
        trainer = self._trainer(historical_max_snapshots=2)
        for index in range(3):
            trainer.add_historical_policy(
                RandomPolicy(index),
                snapshot_id=f"snap_{index}",
            )
            trainer.league_reward_ema[f"snap_{index}"] = float(index)

        self.assertEqual(
            [snapshot.snapshot_id for snapshot in trainer.historical_snapshots],
            ["snap_1", "snap_2"],
        )
        self.assertEqual(set(trainer.league_reward_ema), {"snap_1", "snap_2"})

    def test_mmd_magnet_kl_reported_and_magnet_tracks_model(self):
        trainer = self._trainer(mmd_enabled=True, mmd_magnet_decay=0.5)
        self.assertIsNotNone(trainer.magnet_model)
        initial_magnet = copy.deepcopy(trainer.magnet_model)

        buffer = trainer.collect_rollouts()
        stats = trainer.update(buffer)

        self.assertTrue(math.isfinite(stats.magnet_kl))
        self.assertGreaterEqual(stats.magnet_kl, 0.0)
        moved = sum(
            float((updated - initial).abs().sum())
            for updated, initial in zip(
                trainer.magnet_model.parameters(),
                initial_magnet.parameters(),
            )
        )
        self.assertGreater(moved, 0.0)
        distance_to_model = sum(
            float((magnet - current).abs().sum())
            for magnet, current in zip(
                trainer.magnet_model.parameters(),
                trainer.model.parameters(),
            )
        )
        initial_distance_to_model = sum(
            float((magnet - current).abs().sum())
            for magnet, current in zip(
                initial_magnet.parameters(),
                trainer.model.parameters(),
            )
        )
        self.assertLess(distance_to_model, initial_distance_to_model)

    def test_mmd_magnet_logits_are_cached_across_ppo_epochs(self):
        trainer = self._trainer(
            mmd_enabled=True,
            lean_rollout_forward=True,
            ppo_epochs=3,
            minibatch_size=4,
            microbatch_size=2,
        )
        buffer = trainer.collect_rollouts()
        expected_cache_forwards = math.ceil(len(buffer.samples) / 2)

        with patch.object(
            trainer.magnet_model,
            "forward_policy",
            wraps=trainer.magnet_model.forward_policy,
        ) as forward:
            trainer.update(buffer)

        self.assertEqual(forward.call_count, expected_cache_forwards)

    def test_suit_presence_targets_label_only_opponents(self):
        trainer = self._trainer(suit_presence_head_model=True)
        buffer = trainer.collect_rollouts()

        for sample in buffer.samples:
            targets = sample.suit_presence_targets
            self.assertEqual(
                len(targets),
                trainer.config.model_config.max_players,
            )
            # Observer slot and padding players carry no labels.
            self.assertEqual(targets[0], [-100, -100, -100, -100])
            for relative in range(1, sample.spec.num_players):
                self.assertTrue(
                    all(value in (0, 1) for value in targets[relative])
                )
                # Three-card hands cover at most three suits, so at least
                # one suit must be absent.
                self.assertLess(sum(targets[relative]), 4)
            for relative in range(
                sample.spec.num_players,
                trainer.config.model_config.max_players,
            ):
                self.assertEqual(targets[relative], [-100, -100, -100, -100])

    def test_suit_presence_loss_trains_and_owner_loss_can_be_disabled(self):
        trainer = self._trainer(
            suit_presence_head_model=True,
            suit_coef=0.1,
            owner_coef=0.0,
        )
        buffer = trainer.collect_rollouts()
        stats = trainer.update(buffer)

        self.assertTrue(math.isfinite(stats.suit_presence_loss))
        self.assertGreater(stats.suit_presence_loss, 0.0)
        self.assertEqual(stats.owner_loss, 0.0)
        self.assertEqual(stats.owner_ce_loss, 0.0)
        self.assertEqual(stats.owner_capacity_loss, 0.0)

        prediction = trainer.compute_prediction_stats(buffer, max_samples=64)
        self.assertGreaterEqual(prediction.suit_presence_accuracy, 0.0)
        self.assertLessEqual(prediction.suit_presence_accuracy, 1.0)
        self.assertGreater(prediction.suit_presence_brier, 0.0)

    def test_suit_presence_loss_zero_without_head(self):
        trainer = self._trainer()
        buffer = trainer.collect_rollouts()
        stats = trainer.update(buffer)

        self.assertEqual(stats.suit_presence_loss, 0.0)
        self.assertGreater(stats.owner_loss, 0.0)

    def test_nonfinite_gradients_skip_the_optimizer_step(self):
        trainer = self._trainer()
        buffer = trainer.collect_rollouts()
        trainer.model.bid_head.weight.register_hook(
            lambda gradient: gradient * float("nan")
        )
        before = copy.deepcopy(trainer.model.state_dict())

        stats = trainer.update(buffer)

        self.assertGreater(stats.skipped_steps, 0)
        after = trainer.model.state_dict()
        for key, tensor in after.items():
            self.assertTrue(torch.isfinite(tensor).all(), key)
            self.assertTrue(torch.equal(tensor, before[key]), key)

    def test_mmd_magnet_state_persists_in_checkpoints(self):
        trainer = self._trainer(mmd_enabled=True, mmd_magnet_decay=0.5)
        buffer = trainer.collect_rollouts()
        trainer.update(buffer)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "resume.pt"
            trainer.save_checkpoint(checkpoint_path, iteration=1)

            restored = self._trainer(mmd_enabled=True, mmd_magnet_decay=0.5)
            restored.load_checkpoint(checkpoint_path)

            distance = sum(
                float((restored_param - trained_param).abs().sum())
                for restored_param, trained_param in zip(
                    restored.magnet_model.parameters(),
                    trainer.magnet_model.parameters(),
                )
            )
            self.assertEqual(distance, 0.0)

    def test_regret_matching_spreads_mass_across_rps_cycle(self):
        members = ["rock", "paper", "scissors"]
        payoffs = {
            ("rock", "scissors"): 1.0,
            ("scissors", "paper"): 1.0,
            ("paper", "rock"): 1.0,
            ("scissors", "rock"): -1.0,
            ("paper", "scissors"): -1.0,
            ("rock", "paper"): -1.0,
        }

        mixture = solve_meta_mixture(members, payoffs, iterations=2_000)

        for member in members:
            self.assertAlmostEqual(mixture[member], 1.0 / 3.0, delta=0.03)

    def test_regret_matching_removes_dominated_member(self):
        members = ["a", "b", "dominated"]
        payoffs = {
            (row, column): (-2.0 if row == "dominated" else 1.0)
            for row in members
            for column in members
        }

        mixture = solve_meta_mixture(members, payoffs, iterations=2_000)

        self.assertLess(mixture["dominated"], 0.01)

    def test_league_caches_frozen_cells_and_refreshes_current_on_schedule(self):
        trainer = self._trainer(
            league_meta_solver="regret_matching",
            league_eval_every=5,
        )
        report = SimpleNamespace(macro_relative_reward=0.25)
        with patch("plump.training.ppo.evaluate_policy", return_value=report) as evaluate:
            trainer.add_historical_policy(RandomPolicy(5), snapshot_id="a")
            admission_calls = evaluate.call_count
            frozen_value = trainer.league_payoffs[("a", "heuristic")]

            self.assertFalse(trainer.refresh_league_payoffs(iteration=4))
            self.assertEqual(evaluate.call_count, admission_calls)
            self.assertTrue(trainer.refresh_league_payoffs(iteration=5))
            self.assertGreater(evaluate.call_count, admission_calls)
            refresh_calls = evaluate.call_count

            trainer.add_historical_policy(RandomPolicy(7), snapshot_id="b")
            self.assertGreater(evaluate.call_count, refresh_calls)
            self.assertEqual(
                trainer.league_payoffs[("a", "heuristic")],
                frozen_value,
            )

    def test_league_eviction_uses_lowest_meta_mass_and_protects_newest(self):
        trainer = self._trainer(
            historical_max_snapshots=2,
            league_meta_solver="regret_matching",
        )
        report = SimpleNamespace(macro_relative_reward=0.0)
        with patch("plump.training.ppo.evaluate_policy", return_value=report):
            trainer.add_historical_policy(RandomPolicy(1), snapshot_id="a")
            trainer.add_historical_policy(RandomPolicy(2), snapshot_id="b")
            trainer.league_meta_mixture = {
                "heuristic": 0.1,
                "a": 0.8,
                "b": 0.01,
                "current": 0.09,
            }
            with patch.object(trainer, "_refresh_league_mixture"):
                trainer.add_historical_policy(RandomPolicy(3), snapshot_id="new")

        self.assertEqual(
            [row.snapshot_id for row in trainer.historical_snapshots],
            ["a", "new"],
        )
        self.assertFalse(
            any("b" in key for key in trainer.league_payoffs)
        )

    def test_target_kl_stops_after_first_high_kl_epoch(self):
        trainer = self._trainer(ppo_epochs=4, target_kl=0.001)
        buffer = trainer.collect_rollouts()
        for sample in buffer.samples:
            sample.old_logprob += 5.0
            sample.old_policy_logprob = sample.old_logprob

        stats = trainer.update(buffer)

        self.assertEqual(stats.epochs_run, 1)
        self.assertGreater(stats.approx_kl, trainer.config.target_kl)

    def test_target_kl_none_runs_every_epoch(self):
        trainer = self._trainer(ppo_epochs=3, target_kl=None)
        buffer = trainer.collect_rollouts()

        stats = trainer.update(buffer)

        self.assertEqual(stats.epochs_run, 3)

    def test_best_response_opponent_mix_is_pinned_and_pool_does_not_grow(self):
        trainer = self._trainer(
            rounds_per_configuration=4,
            self_play_fraction=0.0,
            heuristic_fraction=0.0,
            mixed_fraction=0.0,
            historical_fraction=1.0,
            historical_max_snapshots=1,
            league_meta_solver="softmax_ema",
        )
        candidate = RandomPolicy(91)
        trainer.add_historical_policy(candidate, snapshot_id="candidate")

        buffer = trainer.collect_rollouts()

        self.assertEqual(len(trainer.historical_snapshots), 1)
        self.assertTrue(
            all(row.opponent_arm == "historical" for row in buffer.round_outcomes)
        )
        self.assertTrue(
            all(
                row.opponent_snapshot_id == "candidate"
                for row in buffer.round_outcomes
            )
        )


class WeightedSamplingTest(unittest.TestCase):
    def _trainer(self, *, player_counts=(3,), hand_sizes=(3,), **overrides) -> PPOTrainer:
        model_config = ModelConfig(
            max_seq_len=48,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            context_hidden_dim=64,
            dropout=0.0,
        )
        config = TrainingConfig(
            player_counts=player_counts,
            hand_sizes=hand_sizes,
            rounds_per_configuration=overrides.pop("rounds_per_configuration", 1),
            num_envs=overrides.pop("num_envs", 2),
            ppo_epochs=1,
            minibatch_size=64,
            learning_rate=1e-3,
            self_play_fraction=overrides.pop("self_play_fraction", 1.0),
            heuristic_fraction=overrides.pop("heuristic_fraction", 0.0),
            mixed_fraction=overrides.pop("mixed_fraction", 0.0),
            historical_fraction=overrides.pop("historical_fraction", 0.0),
            seed=3,
            device="cpu",
            model_config=model_config,
            **overrides,
        )
        return PPOTrainer(PlumpTransformerModel(model_config), config)

    def test_weighted_spec_quotas_follow_joint_ramp(self):
        trainer = self._trainer(
            player_counts=(3, 4, 5),
            hand_sizes=(3, 4, 5),
            rounds_per_configuration=36,
            player_count_weights=(2.0, 3.0, 4.0),
            hand_size_weights=(1.0, 2.0, 3.0),
        )
        quotas = trainer.config.spec_round_quotas()
        self.assertEqual(sum(quotas.values()), trainer.config.rounds_per_batch)
        # joint weight is the product; 36 rounds/config * 9 cells = 324 total,
        # raw shares are weight/54 so every cell lands exactly on an integer
        for spec, quota in quotas.items():
            weight = {3: 2, 4: 3, 5: 4}[spec.num_players] * (spec.hand_size - 2)
            self.assertEqual(quota, 324 * weight // 54)
        # monotone: more players and more cards always get more rounds
        self.assertLess(
            quotas[RoundSpec(3, 3)],
            quotas[RoundSpec(5, 5)],
        )

    def test_weighted_schedule_matches_quotas_with_exact_arm_mix(self):
        trainer = self._trainer(
            player_counts=(3, 4, 5),
            hand_sizes=(3, 4, 5),
            rounds_per_configuration=36,
            player_count_weights=(2.0, 3.0, 4.0),
            hand_size_weights=(1.0, 2.0, 3.0),
            self_play_fraction=0.35,
            mixed_fraction=0.35,
            historical_fraction=0.30,
            heuristic_fraction=0.0,
        )
        schedule = trainer.balanced_round_schedule()
        quotas = trainer.config.spec_round_quotas()
        counts = {}
        arm_counts = {}
        for spec, arm in schedule:
            counts[spec] = counts.get(spec, 0) + 1
            arm_counts[arm] = arm_counts.get(arm, 0) + 1
        self.assertEqual(counts, quotas)
        total = len(schedule)
        # no historical snapshots loaded -> historical folds into self
        self.assertEqual(arm_counts.get("heuristic", 0), 0)
        self.assertEqual(arm_counts["self"], round(total * 0.65))
        self.assertEqual(arm_counts["mixed"], round(total * 0.35))

    def test_sampling_weight_validation(self):
        with self.assertRaises(ValueError):
            self._trainer(
                player_counts=(3, 4, 5),
                player_count_weights=(1.0, 2.0),
            )
        with self.assertRaises(ValueError):
            self._trainer(hand_sizes=(3, 4), hand_size_weights=(0.0, 0.0))
        with self.assertRaises(ValueError):
            self._trainer(explore_temperature_fraction=1.5)
        with self.assertRaises(ValueError):
            self._trainer(
                explore_temperature_fraction=0.5,
                explore_temperature_bid=0.5,
            )

    def test_weighted_cells_carry_matching_gradient_mass(self):
        trainer = self._trainer(
            player_counts=(3,),
            hand_sizes=(3, 4),
            rounds_per_configuration=4,
            num_envs=8,
            player_count_weights=(1.0,),
            hand_size_weights=(1.0, 3.0),
        )
        buffer = trainer.collect_rollouts()
        trainer._assign_round_weights(buffer)
        mass = {}
        for sample in buffer.samples:
            key = sample.spec.hand_size
            mass[key] = mass.get(key, 0.0) + sample.round_weight
        # per-decision mass follows cell share x steps per trajectory: every
        # decision carries its round's weight, so the 1:3 cell shares appear
        # as 3 * (1+4)/(1+3) once the bid+plays trajectory lengths differ
        self.assertAlmostEqual(mass[4] / mass[3], 3.0 * 5.0 / 4.0, places=5)

    def test_tempered_rounds_flatten_the_behavior_policy(self):
        trainer = self._trainer(
            explore_temperature_fraction=1.0,
            explore_temperature_bid=2.0,
            explore_temperature_play=1.5,
        )
        buffer = trainer.collect_rollouts()
        self.assertTrue(buffer.samples)
        diffs = [
            abs(sample.old_logprob - sample.old_policy_logprob)
            for sample in buffer.samples
            if sample.old_policy_logprob is not None
        ]
        # behavior (tempered) and raw-policy logprobs must diverge somewhere
        self.assertTrue(diffs)
        self.assertGreater(max(diffs), 1e-4)
        # the off-policy surrogate (clip on the policy ratio, behavior weight
        # outside the min) must update cleanly on tempered data: before the
        # first optimizer step the policy ratio is 1, so nothing is clipped
        # even though behavior != policy
        metrics = trainer.update(buffer)
        self.assertTrue(math.isfinite(metrics.policy_loss))
        self.assertTrue(math.isfinite(metrics.approx_kl))
        self.assertTrue(math.isfinite(metrics.clip_fraction))

    def test_temperature_fraction_zero_keeps_classic_sampling(self):
        torch.manual_seed(11)
        buffer_plain = self._trainer().collect_rollouts()
        torch.manual_seed(11)
        buffer_flagged = self._trainer(
            explore_temperature_fraction=0.0,
            explore_temperature_bid=2.0,
            explore_temperature_play=1.5,
        ).collect_rollouts()
        self.assertEqual(
            [sample.action_index for sample in buffer_plain.samples],
            [sample.action_index for sample in buffer_flagged.samples],
        )


class CleanExploreArmsTest(unittest.TestCase):
    def _trainer(self, *, player_counts=(3,), hand_sizes=(3,), **overrides) -> PPOTrainer:
        model_config = ModelConfig(
            max_seq_len=48,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            context_hidden_dim=64,
            dropout=0.0,
        )
        config = TrainingConfig(
            player_counts=player_counts,
            hand_sizes=hand_sizes,
            rounds_per_configuration=overrides.pop("rounds_per_configuration", 4),
            num_envs=overrides.pop("num_envs", 4),
            ppo_epochs=1,
            minibatch_size=64,
            learning_rate=1e-3,
            self_play_fraction=overrides.pop("self_play_fraction", 0.0),
            heuristic_fraction=overrides.pop("heuristic_fraction", 0.0),
            mixed_fraction=overrides.pop("mixed_fraction", 0.0),
            historical_fraction=overrides.pop("historical_fraction", 0.0),
            seed=3,
            device="cpu",
            model_config=model_config,
            **overrides,
        )
        return PPOTrainer(PlumpTransformerModel(model_config), config)

    def _explore_noise(self) -> dict:
        return dict(
            explore_temperature_fraction=1.0,
            explore_temperature_bid=3.0,
            explore_temperature_play=2.0,
            explore_temperature_arms=("explore_self", "explore_historical"),
            explore_uniform_round_probability=0.3,
            explore_noise_hand_normalized=True,
        )

    def test_argmax_cap_triggers_only_above_eighty_percent(self):
        minimum = torch.finfo(torch.float32).min
        logits = torch.tensor(
            [
                [6.0, 2.0, 0.0, minimum],
                [1.0, 1.0, 1.0, minimum],
                [minimum, 1.0, minimum, minimum],
            ]
        )

        probabilities = _argmax_capped_probabilities(
            logits,
            0.8,
            alternative_temperature=4.0,
        )

        alternative = torch.softmax(torch.tensor([2.0, 0.0]) / 4.0, dim=0) * 0.2
        self.assertAlmostEqual(probabilities[0, 0].item(), 0.8, places=6)
        self.assertAlmostEqual(
            probabilities[0, 1].item(),
            alternative[0].item(),
            places=6,
        )
        self.assertAlmostEqual(
            probabilities[0, 2].item(),
            alternative[1].item(),
            places=6,
        )
        self.assertEqual(probabilities[0, 3].item(), 0.0)
        self.assertTrue(
            torch.allclose(
                probabilities[1],
                torch.tensor([1.0 / 3.0] * 3 + [0.0]),
            )
        )
        self.assertEqual(probabilities[2].tolist(), [0.0, 1.0, 0.0, 0.0])
        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=-1),
                torch.ones(3),
            )
        )

    def test_non_argmax_uniform_excludes_top_unless_forced(self):
        minimum = torch.finfo(torch.float32).min
        logits = torch.tensor(
            [
                [6.0, 2.0, 0.0, minimum],
                [1.0, 1.0, 1.0, minimum],
                [minimum, 1.0, minimum, minimum],
            ]
        )

        probabilities = _non_argmax_uniform_probabilities(logits)

        self.assertEqual(probabilities[0].tolist(), [0.0, 0.5, 0.5, 0.0])
        self.assertEqual(probabilities[1].tolist(), [0.0, 0.5, 0.5, 0.0])
        self.assertEqual(probabilities[2].tolist(), [0.0, 1.0, 0.0, 0.0])
        self.assertTrue(
            torch.allclose(probabilities.sum(dim=-1), torch.ones(3))
        )

    def test_single_deviation_preserves_alternative_logit_ranking(self):
        minimum = torch.finfo(torch.float32).min
        logits = torch.tensor(
            [
                [8.0, 4.0, 1.0, minimum],
                [minimum, 2.0, minimum, minimum],
            ]
        )

        probabilities = _non_argmax_tempered_probabilities(
            logits,
            temperature=3.0,
        )

        expected_alternatives = torch.softmax(
            torch.tensor([4.0, 1.0]) / 3.0,
            dim=0,
        )
        self.assertEqual(probabilities[0, 0].item(), 0.0)
        self.assertAlmostEqual(
            probabilities[0, 1].item(),
            expected_alternatives[0].item(),
            places=6,
        )
        self.assertAlmostEqual(
            probabilities[0, 2].item(),
            expected_alternatives[1].item(),
            places=6,
        )
        self.assertEqual(probabilities[0, 3].item(), 0.0)
        self.assertEqual(probabilities[1].tolist(), [0.0, 1.0, 0.0, 0.0])
        self.assertTrue(
            torch.allclose(probabilities.sum(dim=-1), torch.ones(2))
        )

    def test_six_equal_arms_and_focal_only_exploration(self):
        sixth = 1.0 / 6.0
        fractions = dict(
            self_play_fraction=sixth,
            historical_fraction=sixth,
            tempered_self_fraction=0.0,
            tempered_historical_fraction=0.0,
            capped_self_fraction=sixth,
            capped_historical_fraction=sixth,
            epsilon_self_fraction=sixth,
            epsilon_historical_fraction=sixth,
        )
        trainer = self._trainer(
            rounds_per_configuration=6,
            num_envs=6,
            explore_argmax_cap=0.8,
            explore_argmax_rest_temperature=4.0,
            explore_eps_by_arm={
                "epsilon_self": (0.15, 0.30),
                "epsilon_historical": (0.15, 0.30),
            },
            **fractions,
        )
        trainer.add_historical_policy(RandomPolicy(11))

        buffer = trainer.collect_rollouts()
        counts: dict[str, int] = {}
        arm_by_episode = {}
        for outcome in buffer.round_outcomes:
            counts[outcome.opponent_arm] = (
                counts.get(outcome.opponent_arm, 0) + 1
            )
            arm_by_episode[outcome.episode_id] = outcome.opponent_arm

        expected_arms = {
            "self",
            "historical",
            "capped_self",
            "capped_historical",
            "epsilon_self",
            "epsilon_historical",
        }
        self.assertEqual(counts, {arm: 1 for arm in expected_arms})

        players_by_episode: dict[int, set[int]] = {}
        for sample in buffer.samples:
            players_by_episode.setdefault(sample.episode_id, set()).add(
                sample.acting_player
            )
        for episode_id, arm in arm_by_episode.items():
            if arm in {
                "historical",
                "capped_self",
                "capped_historical",
                "epsilon_self",
                "epsilon_historical",
            }:
                self.assertEqual(len(players_by_episode[episode_id]), 1)
        self_episode = next(
            episode_id
            for episode_id, arm in arm_by_episode.items()
            if arm == "self"
        )
        self.assertEqual(
            len(players_by_episode[self_episode]),
            3,
        )

        for arm in {
            "epsilon_self",
            "epsilon_historical",
        }:
            differences = [
                abs(sample.old_logprob - sample.old_policy_logprob)
                for sample in buffer.samples
                if sample.opponent_arm == arm
                and sample.old_policy_logprob is not None
            ]
            self.assertTrue(differences)
            self.assertGreater(max(differences), 1e-6)

        rollout_stats = trainer.summarize_rollout(buffer)
        update_stats = trainer.update(buffer)
        self.assertTrue(math.isfinite(update_stats.total_loss))
        self.assertEqual(
            sum(
                (
                    rollout_stats.self_play_rounds,
                    rollout_stats.historical_rounds,
                    rollout_stats.tempered_self_rounds,
                    rollout_stats.tempered_historical_rounds,
                    rollout_stats.capped_self_rounds,
                    rollout_stats.capped_historical_rounds,
                    rollout_stats.epsilon_self_rounds,
                    rollout_stats.epsilon_historical_rounds,
                )
            ),
            6,
        )
        self.assertEqual(rollout_stats.tempered_self_rounds, 0)
        self.assertEqual(rollout_stats.tempered_historical_rounds, 0)

    def test_explore_fractions_fold_without_history(self):
        trainer = self._trainer(
            self_play_fraction=0.30,
            historical_fraction=0.30,
            explore_self_fraction=0.20,
            explore_historical_fraction=0.20,
        )
        fractions = trainer._effective_arm_fractions()
        self.assertAlmostEqual(fractions["self"], 0.60)
        self.assertAlmostEqual(fractions["explore_self"], 0.40)
        self.assertEqual(fractions["historical"], 0.0)
        self.assertEqual(fractions["explore_historical"], 0.0)
        # With a pool present, the configured split stands.
        trainer.historical_snapshots.append(
            SimpleNamespace(policy=object(), snapshot_id="fake")
        )
        fractions = trainer._effective_arm_fractions()
        self.assertAlmostEqual(fractions["historical"], 0.30)
        self.assertAlmostEqual(fractions["explore_historical"], 0.20)

    def test_explore_self_trains_focal_only_with_noised_behavior(self):
        trainer = self._trainer(
            explore_self_fraction=1.0,
            rounds_per_configuration=6,
            **self._explore_noise(),
        )
        buffer = trainer.collect_rollouts()
        # Only the focal seat produces samples: exactly bid + hand_size
        # decisions per round, not seats x trajectory.
        self.assertEqual(len(buffer.samples), 6 * (1 + 3))
        players_by_episode: dict[int, set[int]] = {}
        for sample in buffer.samples:
            players_by_episode.setdefault(sample.episode_id, set()).add(
                sample.acting_player
            )
        self.assertTrue(
            all(len(players) == 1 for players in players_by_episode.values())
        )
        # The focal behavior policy is tempered+eps, never raw.
        diffs = [
            abs(sample.old_logprob - sample.old_policy_logprob)
            for sample in buffer.samples
            if sample.old_policy_logprob is not None
        ]
        self.assertEqual(len(diffs), len(buffer.samples))
        self.assertGreater(max(diffs), 1e-4)
        metrics = trainer.update(buffer)
        self.assertTrue(math.isfinite(metrics.policy_loss))
        self.assertTrue(math.isfinite(metrics.approx_kl))

    def test_league_ema_ignores_explore_outcomes(self):
        trainer = self._trainer(self_play_fraction=1.0)
        outcome = dict(
            episode_id=0,
            spec=RoundSpec(3, 3),
            focal_reward=5.0,
            bid_hit_count=1,
            bid_player_count=3,
            bid_abs_error_mean=0.0,
            focal_bid_hit=1,
            focal_bid_abs_error=0.0,
            position_rewards={},
            opponent_snapshot_id="snap",
        )
        from plump.training.ppo import RoundOutcome

        trainer._update_league_ema(
            [RoundOutcome(opponent_arm="explore_historical", **outcome)]
        )
        self.assertNotIn("snap", trainer.league_reward_ema)
        trainer._update_league_ema(
            [RoundOutcome(opponent_arm="historical", **outcome)]
        )
        self.assertIn("snap", trainer.league_reward_ema)

    def test_clean_metrics_exclude_explore_rounds(self):
        trainer = self._trainer(
            self_play_fraction=0.5,
            explore_self_fraction=0.5,
            rounds_per_configuration=8,
            num_envs=8,
            **self._explore_noise(),
        )
        buffer = trainer.collect_rollouts()
        stats = trainer.summarize_rollout(buffer)
        self.assertEqual(stats.self_play_rounds + stats.explore_self_rounds, 8)
        self.assertGreater(stats.explore_self_rounds, 0)
        # Grouped bid-hit covers only the clean cells that actually played.
        self.assertEqual(set(stats.bid_hit_rate_by_players), {3})
        self.assertEqual(set(stats.bid_hit_rate_by_hand_bucket), {"3_5"})
        clean_hits = [
            float(outcome.focal_bid_hit)
            for outcome in buffer.round_outcomes
            if outcome.opponent_arm == "self"
        ]
        self.assertAlmostEqual(
            stats.bid_hit_rate,
            sum(clean_hits) / len(clean_hits),
        )
        prediction = trainer.compute_prediction_stats(buffer, max_samples=128)
        for value in (
            prediction.trick_count_accuracy_bidtime,
            prediction.trick_count_accuracy_early,
            prediction.trick_count_accuracy_mid,
            prediction.trick_count_accuracy_late,
        ):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertEqual(set(prediction.trick_count_accuracy_by_players), {3})
        self.assertEqual(
            set(prediction.trick_count_accuracy_by_hand_bucket),
            {"3_5"},
        )

    def test_explore_uniform_injects_at_most_one_action_per_round(self):
        trainer = self._trainer(
            explore_self_fraction=1.0,
            rounds_per_configuration=8,
            num_envs=8,
            explore_uniform_round_probability=1.0,
        )
        buffer = trainer.collect_rollouts()
        diverged_by_episode: dict[int, int] = {}
        for sample in buffer.samples:
            diverged_by_episode.setdefault(sample.episode_id, 0)
            if (
                sample.old_policy_logprob is not None
                and abs(sample.old_logprob - sample.old_policy_logprob) > 1e-6
            ):
                diverged_by_episode[sample.episode_id] += 1
        # Probability 1.0 and no round tempering: at most one policy-shaped
        # non-argmax (behavior != policy) decision per round — zero only when
        # the selected decision has one legal action.
        self.assertTrue(set(diverged_by_episode.values()) <= {0, 1})
        self.assertIn(1, diverged_by_episode.values())

    def test_hand_normalization_scales_noise_down(self):
        trainer = self._trainer(
            hand_sizes=(3, 10),
            self_play_fraction=1.0,
            explore_temperature_bid=3.0,
            explore_temperature_play=2.0,
            explore_noise_hand_normalized=True,
        )
        self.assertAlmostEqual(trainer._hand_noise_scale(3), 1.0)
        self.assertAlmostEqual(trainer._hand_noise_scale(10), 4.0 / 11.0)
        request = lambda phase, hand: SimpleNamespace(
            collect=True,
            explore_tempered=True,
            explore_uniform=False,
            phase=phase,
            hand_size=hand,
        )
        self.assertAlmostEqual(
            trainer._request_explore_temperature(request("bid", 3)),
            3.0,
        )
        self.assertAlmostEqual(
            trainer._request_explore_temperature(request("bid", 10)),
            1.0 + 2.0 * 4.0 / 11.0,
        )
        self.assertAlmostEqual(
            trainer._request_explore_temperature(request("play", 10)),
            1.0 + 1.0 * 4.0 / 11.0,
        )
        with self.assertRaises(ValueError):
            self._trainer(
                self_play_fraction=1.0,
                explore_uniform_round_probability=1.5,
            )
        with self.assertRaises(ValueError):
            self._trainer(
                self_play_fraction=1.0,
                explore_single_deviation_temperature=0.0,
            )

    def test_owner_warmup_detaches_trunk_gradients(self):
        trainer = self._trainer(self_play_fraction=1.0, owner_coef=0.05)
        model = trainer.model
        env = PlumpEnv(
            round_game_config(RoundSpec(3, 3), bidding_start_player=0),
            seed=5,
        )
        env.reset()
        observation = env.get_observation(env.current_player())
        encoded = encode_observation(observation, model.config)
        batch = encoded_observations_to_batch(
            [encoded],
            device=torch.device("cpu"),
        )
        for detach, expect_trunk_grad in ((True, False), (False, True)):
            model.zero_grad(set_to_none=True)
            output = model(batch, need_owner=True, detach_owner_trunk=detach)
            loss = output.owner_probs.pow(2).sum()
            loss.backward()
            owner_grad = model.owner_card_mlp[0].weight.grad
            self.assertIsNotNone(owner_grad)
            self.assertGreater(float(owner_grad.abs().sum()), 0.0)
            trunk_grad = model.final_norm.weight.grad
            if expect_trunk_grad:
                self.assertIsNotNone(trunk_grad)
                self.assertGreater(float(trunk_grad.abs().sum()), 0.0)
            else:
                self.assertTrue(
                    trunk_grad is None or float(trunk_grad.abs().sum()) == 0.0
                )

    def test_belief_head_only_detaches_trunk_gradients(self):
        trainer = self._trainer(self_play_fraction=1.0, belief_head_only=True)
        model = trainer.model
        env = PlumpEnv(
            round_game_config(RoundSpec(3, 3), bidding_start_player=0),
            seed=5,
        )
        env.reset()
        observation = env.get_observation(env.current_player())
        encoded = encode_observation(observation, model.config)
        batch = encoded_observations_to_batch(
            [encoded],
            device=torch.device("cpu"),
        )
        for detach, expect_trunk_grad in ((True, False), (False, True)):
            model.zero_grad(set_to_none=True)
            output = model(batch, detach_belief_trunk=detach)
            loss = output.trick_count_logits.pow(2).sum()
            loss.backward()
            belief_grad = model.trick_count_head.weight.grad
            self.assertIsNotNone(belief_grad)
            self.assertGreater(float(belief_grad.abs().sum()), 0.0)
            trunk_grad = model.final_norm.weight.grad
            if expect_trunk_grad:
                self.assertIsNotNone(trunk_grad)
                self.assertGreater(float(trunk_grad.abs().sum()), 0.0)
            else:
                self.assertTrue(
                    trunk_grad is None or float(trunk_grad.abs().sum()) == 0.0
                )

        group_names = [group["name"] for group in trainer.optimizer.param_groups]
        self.assertEqual(group_names, ["policy_value", "belief_heads"])
        belief_ids = {id(parameter) for parameter in trainer._belief_parameters()}
        self.assertEqual(
            {id(parameter) for parameter in trainer.optimizer.param_groups[1]["params"]},
            belief_ids,
        )

    def test_owner_active_since_persists_across_checkpoints(self):
        trainer = self._trainer(
            self_play_fraction=1.0,
            owner_coef=0.05,
            owner_warmup_iterations=50,
        )
        buffer = trainer.collect_rollouts(iteration=7)
        trainer.update(buffer)
        self.assertEqual(trainer.owner_active_since, 7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            trainer.save_checkpoint(path, iteration=7)
            resumed = self._trainer(
                self_play_fraction=1.0,
                owner_coef=0.05,
                owner_warmup_iterations=50,
            )
            resumed.load_checkpoint(path)
            self.assertEqual(resumed.owner_active_since, 7)

    def test_metrics_header_migration_adds_grouped_columns(self):
        from plump.training.run_logger import METRIC_FIELDS

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            old_fields = [
                field
                for field in METRIC_FIELDS
                if not field.startswith("rollout_bid_hit_p")
                and not field.startswith("pred_trick_accuracy")
                and "explore" not in field
            ]
            metrics_path = log_dir / "metrics.csv"
            with metrics_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=old_fields)
                writer.writeheader()
                writer.writerow({field: "1" for field in old_fields})
            TrainingRunLogger(log_dir)
            with metrics_path.open(newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, METRIC_FIELDS)
                row = next(reader)
            self.assertEqual(row["iteration"], "1")
            self.assertEqual(row["rollout_bid_hit_p5"], "")
            self.assertEqual(row["pred_trick_accuracy_bidtime"], "")
            self.assertEqual(row["rollout_explore_self_rounds"], "")


if __name__ == "__main__":
    unittest.main()
