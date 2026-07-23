import random
import unittest
import copy

import torch
from plump.cards import Card, Rank, Suit
from plump.env import PlumpEnv
from plump.modeling import ModelConfig, encode_observation
from plump.modeling.torch_model import PlumpTransformerModel
from plump.policies import ModelPolicy, RandomPolicy
from plump.rounds import RoundSpec, round_game_config
from plump.search import (
    ConstrainedDeterminizationSampler,
    RootSearchPolicy,
    SearchConfig,
    reconstruct_public_state,
)
from plump.state import BidAction, GameConfig, PlayCardAction


def _manual_env(swapped=False):
    hand0 = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.CLUBS, Rank.ACE),
    ]
    hand1 = [
        Card(Suit.SPADES, Rank.KING),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.CLUBS, Rank.KING),
    ]
    hand2 = [
        Card(Suit.SPADES, Rank.QUEEN),
        Card(Suit.HEARTS, Rank.QUEEN),
        Card(Suit.CLUBS, Rank.QUEEN),
    ]
    if swapped:
        hand1, hand2 = hand2, hand1
    env = PlumpEnv(
        GameConfig(
            num_players=3,
            hand_sizes=[3],
            manual_hands={0: hand0, 1: hand1, 2: hand2},
            bidding_start_players=[0],
        )
    )
    env.reset()
    return env


class SearchTest(unittest.TestCase):
    def test_sampler_respects_capacities_kitty_and_voids(self):
        env = PlumpEnv(round_game_config(RoundSpec(3, 3)), seed=2)
        env.reset()
        observation = env.get_observation(0)
        observation.voids[1][Suit.SPADES] = True
        sampler = ConstrainedDeterminizationSampler()
        sampled = sampler.sample(
            observation,
            owner_probs=None,
            model_config=ModelConfig(),
            rng=random.Random(3),
        )

        self.assertEqual(len(sampled.current_hands[1]), 3)
        self.assertEqual(len(sampled.current_hands[2]), 3)
        self.assertEqual(len(sampled.undealt_cards), 43)
        self.assertFalse(any(card.suit == Suit.SPADES for card in sampled.current_hands[1]))
        all_hidden = [
            *sampled.current_hands[1],
            *sampled.current_hands[2],
            *sampled.undealt_cards,
        ]
        self.assertEqual(len(all_hidden), len(set(all_hidden)))
        self.assertFalse(set(all_hidden) & set(observation.my_hand))
        self.assertEqual(sampler.draws_attempted, 1)
        self.assertEqual(sampler.draws_succeeded, 1)
        self.assertEqual(sampler.draws_failed, 0)
        self.assertGreaterEqual(sampler.candidate_attempts, 6)
        self.assertLessEqual(
            sampler.infeasible_candidate_rejections,
            sampler.candidate_attempts,
        )

    def test_public_history_reconstruction_matches_decision_point(self):
        env = _manual_env()
        env.step(BidAction(0, 1))
        env.step(BidAction(1, 0))
        env.step(BidAction(2, 0))
        env.step(PlayCardAction(0, Card(Suit.SPADES, Rank.ACE)))
        observation = env.get_observation(1)
        determinization = ConstrainedDeterminizationSampler().sample(
            observation,
            owner_probs=None,
            model_config=ModelConfig(),
            rng=random.Random(4),
        )
        rebuilt = reconstruct_public_state(observation, determinization)

        self.assertEqual(rebuilt.phase(), env.phase())
        self.assertEqual(rebuilt.current_player(), env.current_player())
        self.assertEqual(rebuilt.legal_actions(), env.legal_actions())

    def test_search_does_not_depend_on_actual_hidden_hands(self):
        first = _manual_env(False)
        second = _manual_env(True)
        config = SearchConfig(
            min_determinizations=2,
            max_determinizations=2,
            batch_determinizations=2,
            forward_pass_budget=100,
            seed=5,
        )
        first_search = RootSearchPolicy(RandomPolicy(), RandomPolicy(), config=config)
        second_search = RootSearchPolicy(RandomPolicy(), RandomPolicy(), config=config)
        first_decision = first_search.search(
            first.get_observation(0),
            legal_actions=first.legal_actions(),
            rng=random.Random(7),
        )
        second_decision = second_search.search(
            second.get_observation(0),
            legal_actions=second.legal_actions(),
            rng=random.Random(7),
        )

        self.assertEqual(first_decision.action_values, second_decision.action_values)
        self.assertEqual(first_decision.action, second_decision.action)
        self.assertEqual(
            encode_observation(first.get_observation(0)),
            encode_observation(second.get_observation(0)),
        )

    def test_batched_root_search_matches_serial_search(self):
        torch.manual_seed(11)
        model_config = ModelConfig(
            max_seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            context_hidden_dim=64,
            game_hidden_dim=32,
            schedule_heads=4,
        )
        model = PlumpTransformerModel(model_config)
        serial = RootSearchPolicy(
            ModelPolicy(copy.deepcopy(model), device="cpu", greedy=False),
            RandomPolicy(),
            config=SearchConfig(
                min_determinizations=4,
                max_determinizations=4,
                exact_tricks_remaining=0,
            ),
        )
        batched = RootSearchPolicy(
            ModelPolicy(copy.deepcopy(model), device="cpu", greedy=False),
            RandomPolicy(),
            config=serial.config,
        )
        envs = [_manual_env(False), _manual_env(True)]
        serial_decisions = [
            serial.search(
                env.get_observation(0),
                legal_actions=env.legal_actions(),
                rng=random.Random(seed),
            )
            for env, seed in zip(envs, (17, 23))
        ]
        batched_decisions = batched.search_many(
            [
                (
                    env.get_observation(0),
                    env.legal_actions(),
                    random.Random(seed),
                )
                for env, seed in zip(envs, (17, 23))
            ]
        )

        for expected, actual in zip(serial_decisions, batched_decisions):
            self.assertEqual(expected.action, actual.action)
            self.assertEqual(expected.action_values, actual.action_values)
            for key in expected.action_probabilities:
                self.assertAlmostEqual(
                    expected.action_probabilities[key],
                    actual.action_probabilities[key],
                    places=8,
                )

    def test_same_policy_batched_continuations_match_serial_roles(self):
        torch.manual_seed(19)
        model_config = ModelConfig(
            max_seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            context_hidden_dim=64,
            game_hidden_dim=32,
            schedule_heads=4,
        )
        model = PlumpTransformerModel(model_config)
        config = SearchConfig(
            min_determinizations=2,
            max_determinizations=2,
            exact_tricks_remaining=0,
            use_owner_beliefs=False,
        )
        shared_policy = ModelPolicy(
            copy.deepcopy(model),
            device="cpu",
            greedy=False,
        )
        batched = RootSearchPolicy(
            shared_policy,
            shared_policy,
            config=config,
        )
        serial = RootSearchPolicy(
            ModelPolicy(copy.deepcopy(model), device="cpu", greedy=False),
            ModelPolicy(copy.deepcopy(model), device="cpu", greedy=False),
            config=config,
        )
        env = _manual_env(False)

        batched_decision = batched.search(
            env.get_observation(0),
            legal_actions=env.legal_actions(),
            rng=random.Random(29),
        )
        serial_decision = serial.search(
            env.get_observation(0),
            legal_actions=env.legal_actions(),
            rng=random.Random(29),
        )

        self.assertEqual(batched_decision.action, serial_decision.action)
        self.assertEqual(
            batched_decision.action_values,
            serial_decision.action_values,
        )
        self.assertEqual(
            batched_decision.prior_probabilities,
            serial_decision.prior_probabilities,
        )

    def test_forced_actions_skip_search(self):
        env = _manual_env()
        env.step(BidAction(0, 1))
        env.step(BidAction(1, 0))
        env.step(BidAction(2, 0))
        env.step(PlayCardAction(0, Card(Suit.SPADES, Rank.ACE)))
        policy = RootSearchPolicy(RandomPolicy(), RandomPolicy())
        action = policy.act(env, rng=random.Random(1))

        self.assertEqual(action, PlayCardAction(1, Card(Suit.SPADES, Rank.KING)))
        self.assertEqual(policy.last_decision.forward_passes, 0)


if __name__ == "__main__":
    unittest.main()
