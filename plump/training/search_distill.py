"""Gated counterfactual search routing and replay for schema v4."""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

import torch
from torch import Tensor

from plump.modeling import EncodedObservation, SCHEMA_VERSION
from plump.modeling.torch_model import (
    PlumpTransformerModel,
    best_torch_device,
    encoded_observations_to_batch,
)
from plump.policies import ModelPolicy
from plump.rounds import RoundSpec, rules_fingerprint
from plump.search import RootSearchPolicy, SearchConfig, SearchDecision
from plump.state import BidAction, Observation, PlayCardAction


DistillPhase = Literal["bid", "play"]
ReplayKey = tuple[int, int, int, DistillPhase, int, str]

if TYPE_CHECKING:
    from .ppo import RolloutBuffer, RolloutSample


@dataclass
class SearchReplaySample:
    encoded: EncodedObservation
    phase: DistillPhase
    target_probabilities: list[float]
    spec: RoundSpec
    bidding_position: int
    iteration: int = 0
    trick_position: int = -1
    opponent_arm: str = "unknown"
    regrets: list[float] | None = None

    @property
    def key(self) -> ReplayKey:
        return (
            self.spec.num_players,
            self.spec.hand_size,
            self.bidding_position,
            self.phase,
            self.trick_position,
            self.opponent_arm,
        )


@dataclass
class SearchRoutingStats:
    phase: DistillPhase
    eligible: bool
    probed: int
    accepted: int
    accepted_rate: float
    argmax_agreement: float
    median_target_js: float
    paired_improvement: float
    paired_ci_low: float
    paired_ci_high: float
    worst_cell_improvement: float
    sampler_infeasible_rejection_rate: float
    sampler_failed_draw_rate: float
    gate_passed: bool
    routed: int


@dataclass
class SearchUpdateStats:
    phase: DistillPhase
    samples: int
    loss: float
    cross_entropy_loss: float
    entropy_floor_loss: float
    policy_entropy: float
    target_entropy_floor: float
    kl: float
    maximum_stratum_kl: float
    kl_cap: float
    regret_matching_fraction: float
    backtracks: int
    applied: bool


class CounterfactualSearchRouter:
    """Probe current focal states and route accepted policy supervision."""

    def __init__(
        self,
        model: PlumpTransformerModel,
        *,
        device: str | torch.device | None = None,
        precision: str = "fp32",
        minimum_iteration: int = 250,
        explained_variance_threshold: float = 0.30,
        states_per_phase: int = 24,
        bid_states_per_iteration: int | None = None,
        play_states_per_iteration: int | None = None,
        replay_capacity: int = 50_000,
        replay_max_age: int = 250,
        search_config: SearchConfig | None = None,
        opponent_arms: tuple[str, ...] = (),
        assume_value_ready: bool = False,
        event_length_buckets: tuple[int, ...] = (),
        batch_packing: Literal["torch", "numpy"] = "torch",
        lean_action_forward: bool = False,
        seed: int = 1,
    ) -> None:
        self.model = model
        self.device = (
            torch.device(device)
            if device is not None
            else best_torch_device()
        )
        self.precision = precision
        self.minimum_iteration = minimum_iteration
        self.explained_variance_threshold = explained_variance_threshold
        self.states_per_phase: dict[DistillPhase, int] = {
            "bid": (
                states_per_phase
                if bid_states_per_iteration is None
                else bid_states_per_iteration
            ),
            "play": (
                states_per_phase
                if play_states_per_iteration is None
                else play_states_per_iteration
            ),
        }
        if any(value < 0 for value in self.states_per_phase.values()):
            raise ValueError("search states per iteration must be nonnegative.")
        self.search_config = search_config or SearchConfig()
        self.opponent_arms = frozenset(opponent_arms)
        self.event_length_buckets = event_length_buckets
        self.batch_packing = batch_packing
        self.lean_action_forward = lean_action_forward
        self.rng = random.Random(seed)
        self.replay = StratifiedReplayWindow(
            replay_capacity,
            replay_max_age,
        )
        self.ev_history: dict[DistillPhase, deque[float]] = {
            "bid": deque(maxlen=3),
            "play": deque(maxlen=3),
        }
        self.phase_gates: dict[DistillPhase, bool] = {
            "bid": False,
            "play": False,
        }
        self.eligible_updates: dict[DistillPhase, int] = {
            "bid": 0,
            "play": 0,
        }
        self.ramp_readiness: dict[DistillPhase, deque[bool]] = {
            "bid": deque(maxlen=3),
            "play": deque(maxlen=3),
        }
        if assume_value_ready:
            seeded_ev = max(self.explained_variance_threshold, 0.0)
            for phase in ("bid", "play"):
                self.ev_history[phase].extend([seeded_ev] * 3)

    def update_diagnostics(
        self,
        *,
        bid_explained_variance: float,
        play_explained_variance: float,
    ) -> None:
        self.ev_history["bid"].append(bid_explained_variance)
        self.ev_history["play"].append(play_explained_variance)

    def route(
        self,
        buffer: "RolloutBuffer",
        *,
        iteration: int,
    ) -> list[SearchRoutingStats]:
        reports = []
        base = ModelPolicy(
            self.model,
            device=self.device,
            greedy=False,
            include_game_context=False,
            precision=self.precision,
            event_length_buckets=self.event_length_buckets,
            batch_packing=self.batch_packing,
            lean_action_forward=self.lean_action_forward,
            name="search-current",
        )
        # Search evaluates every root action against the same raw current
        # policy that generated the self-play data.  Using a hand-written
        # heuristic here teaches heuristic-specific exploits and can move a
        # mature policy away from its actual league objective.  Sharing the
        # wrapper also enables batched continuation waves in RootSearchPolicy.
        search = RootSearchPolicy(
            base,
            base,
            config=self.search_config,
        )
        for phase in ("bid", "play"):
            state_count = self.states_per_phase[phase]
            eligible = state_count > 0 and self._eligible(phase, iteration)
            candidates = [
                sample
                for sample in buffer.samples
                if sample.phase == phase
                and sample.observation is not None
                and len(_legal_actions(sample)) > 1
                and (
                    not self.opponent_arms
                    or sample.opponent_arm in self.opponent_arms
                )
            ]
            selected = (
                _targeted_rollout_samples(
                    candidates,
                    state_count,
                    self.rng,
                )
                if eligible
                else []
            )
            sampler_before = search.sampler.counters()
            searched = search.search_many(
                [
                    (
                        sample.observation,
                        _legal_actions(sample),
                        random.Random(self.rng.randrange(2**31)),
                    )
                    for sample in selected
                ]
            )
            sampler_after = search.sampler.counters()
            sampler_delta = tuple(
                after - before
                for before, after in zip(
                    sampler_before,
                    sampler_after,
                )
            )
            decisions = list(zip(selected, searched))

            report = self._phase_report(
                phase,
                eligible,
                decisions,
                iteration,
                sampler_delta,
            )
            reports.append(report)
        return reports

    def regret_matching_fraction(self, phase: DistillPhase) -> float:
        readiness = self.ramp_readiness[phase]
        if (
            not self.phase_gates[phase]
            or len(readiness) < 3
            or not all(readiness)
        ):
            return 0.0
        return min(0.5, 0.5 * self.eligible_updates[phase] / 500.0)

    def _eligible(self, phase: DistillPhase, iteration: int) -> bool:
        history = self.ev_history[phase]
        return (
            iteration >= self.minimum_iteration
            and len(history) == 3
            and all(
                value >= self.explained_variance_threshold
                for value in history
            )
        )

    def _phase_report(
        self,
        phase: DistillPhase,
        eligible: bool,
        decisions: list[tuple["RolloutSample", object]],
        iteration: int,
        sampler_delta: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
    ) -> SearchRoutingStats:
        accepted = [
            (sample, decision)
            for sample, decision in decisions
            if decision.accepted
        ]
        improvements = [
            _decision_improvement(decision)
            for _, decision in decisions
        ]
        low, high = _bootstrap_mean_ci(
            improvements,
            samples=1_000,
            rng=self.rng,
        )
        by_cell: dict[tuple[int, int, int], list[float]] = defaultdict(list)
        for (sample, _), improvement in zip(decisions, improvements):
            by_cell[
                (
                    sample.spec.num_players,
                    sample.spec.hand_size,
                    sample.bidding_position,
                )
            ].append(improvement)
        worst_cell = min(
            (_mean(values) for values in by_cell.values()),
            default=0.0,
        )
        accepted_rate = len(accepted) / len(decisions) if decisions else 0.0
        agreement = (
            _mean(
                [
                    float(decision.split_half_argmax_agreement)
                    for _, decision in decisions
                ]
            )
            if decisions
            else 0.0
        )
        divergences = sorted(
            decision.target_js_divergence
            for _, decision in decisions
            if math.isfinite(decision.target_js_divergence)
        )
        median_js = (
            divergences[len(divergences) // 2]
            if divergences
            else float("inf")
        )
        passed = (
            bool(decisions)
            and low > 0.0
            and accepted_rate >= 0.80
            and agreement >= 0.80
            and worst_cell >= -0.5
        )
        self.phase_gates[phase] = self.phase_gates[phase] or passed
        stable_for_ramp = (
            eligible
            and accepted_rate >= 0.80
            and agreement >= 0.80
            and median_js <= 0.05
        )
        if eligible:
            self.ramp_readiness[phase].append(stable_for_ramp)
            if (
                len(self.ramp_readiness[phase]) == 3
                and all(self.ramp_readiness[phase])
            ):
                self.eligible_updates[phase] += 1

        routed = 0
        if self.phase_gates[phase]:
            for sample, decision in accepted:
                replay_sample = _search_replay_sample(
                    sample,
                    decision,
                    iteration,
                )
                self.replay.add(replay_sample)
                sample.search_target_probabilities = list(
                    replay_sample.target_probabilities
                )
                sample.ppo_policy_enabled = False
                routed += 1
        draw_attempts, _, failed_draws, candidate_attempts, rejected_candidates = (
            sampler_delta
        )
        return SearchRoutingStats(
            phase=phase,
            eligible=eligible,
            probed=len(decisions),
            accepted=len(accepted),
            accepted_rate=accepted_rate,
            argmax_agreement=agreement,
            median_target_js=median_js,
            paired_improvement=_mean(improvements),
            paired_ci_low=low,
            paired_ci_high=high,
            worst_cell_improvement=worst_cell,
            sampler_infeasible_rejection_rate=(
                rejected_candidates / candidate_attempts
                if candidate_attempts
                else 0.0
            ),
            sampler_failed_draw_rate=(
                failed_draws / draw_attempts
                if draw_attempts
                else 0.0
            ),
            gate_passed=self.phase_gates[phase],
            routed=routed,
        )


class SearchTrustRegionUpdater:
    """Stateless SGD search update with a measured hard reverse-KL cap."""

    def __init__(
        self,
        model: PlumpTransformerModel,
        *,
        device: str | torch.device | None = None,
        learning_rate: float = 1e-4,
        minibatch_size: int = 256,
        max_grad_norm: float = 1.0,
        entropy_floor_coef: float = 0.002,
        kl_cap_start: float = 0.005,
        kl_cap_end: float = 0.003,
        seed: int = 1,
    ) -> None:
        self.model = model
        self.device = (
            torch.device(device)
            if device is not None
            else best_torch_device()
        )
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.entropy_floor_coef = entropy_floor_coef
        if kl_cap_start <= 0.0 or kl_cap_end <= 0.0:
            raise ValueError("search KL caps must be positive.")
        self.kl_cap_start = kl_cap_start
        self.kl_cap_end = kl_cap_end
        self.rng = random.Random(seed)

    def update(
        self,
        replay: StratifiedReplayWindow,
        *,
        phase: DistillPhase,
        iteration: int,
        regret_matching_fraction: float,
    ) -> SearchUpdateStats:
        rows = replay.balanced_samples(
            self.rng,
            current_iteration=iteration,
            phase=phase,
            max_samples=self.minibatch_size,
        )
        cap = self.kl_cap_start + (
            self.kl_cap_end - self.kl_cap_start
        ) * (
            regret_matching_fraction / 0.5
            if regret_matching_fraction > 0.0
            else 0.0
        )
        if not rows:
            return SearchUpdateStats(
                phase=phase,
                samples=0,
                loss=0.0,
                cross_entropy_loss=0.0,
                entropy_floor_loss=0.0,
                policy_entropy=0.0,
                target_entropy_floor=0.0,
                kl=0.0,
                maximum_stratum_kl=0.0,
                kl_cap=cap,
                regret_matching_fraction=regret_matching_fraction,
                backtracks=0,
                applied=False,
            )

        batch = encoded_observations_to_batch(
            [row.encoded for row in rows],
            device=self.device,
        )
        previous_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                old_logits = self._phase_logits(self.model(batch), phase).float()
                old_probs = torch.softmax(old_logits, dim=-1)
            targets = _mixed_search_targets(
                rows,
                regret_matching_fraction,
                self.device,
            )
            logits = self._phase_logits(self.model(batch), phase).float()
            log_probs = torch.log_softmax(logits, dim=-1)
            cross_entropy_loss = -(targets * log_probs).sum(dim=-1).mean()
            policy_probs = torch.softmax(logits, dim=-1)
            policy_entropy = _categorical_entropy(policy_probs)
            anchored_targets = torch.tensor(
                [row.target_probabilities for row in rows],
                dtype=torch.float32,
                device=self.device,
            )
            target_entropy_floor = _categorical_entropy(anchored_targets)
            entropy_floor_loss = (
                torch.relu(target_entropy_floor - policy_entropy).mean()
                if regret_matching_fraction > 0.0
                else torch.zeros((), device=self.device)
            )
            loss = (
                cross_entropy_loss
                + self.entropy_floor_coef * entropy_floor_loss
            )
            parameters = [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            gradients = torch.autograd.grad(
                loss,
                parameters,
                allow_unused=True,
            )
            parameter_gradients = [
                (parameter, gradient)
                for parameter, gradient in zip(parameters, gradients)
                if gradient is not None
            ]
            if not parameter_gradients:
                raise RuntimeError("Search loss produced no model gradients.")
            norm = torch.sqrt(
                sum(
                    gradient.detach().float().pow(2).sum()
                    for _, gradient in parameter_gradients
                )
            )
            gradient_scale = min(
                1.0,
                self.max_grad_norm / max(float(norm.cpu()), 1e-12),
            )
            backups = [
                parameter.detach().clone()
                for parameter, _ in parameter_gradients
            ]

            for backtracks in range(9):
                step_size = self.learning_rate * (0.5**backtracks)
                with torch.no_grad():
                    for (parameter, gradient), backup in zip(
                        parameter_gradients,
                        backups,
                    ):
                        parameter.copy_(
                            backup - step_size * gradient_scale * gradient
                        )
                    new_logits = self._phase_logits(
                        self.model(batch),
                        phase,
                    ).float()
                    new_probs = torch.softmax(new_logits, dim=-1)
                    per_row_kl = (
                        old_probs
                        * (
                            torch.log(old_probs.clamp_min(1e-12))
                            - torch.log(new_probs.clamp_min(1e-12))
                        )
                    ).sum(dim=-1)
                    overall_kl = float(per_row_kl.mean().cpu())
                    maximum_stratum_kl = _maximum_stratum_mean(
                        per_row_kl,
                        rows,
                    )
                if overall_kl <= cap and maximum_stratum_kl <= cap:
                    return SearchUpdateStats(
                        phase=phase,
                        samples=len(rows),
                        loss=float(loss.detach().cpu()),
                        cross_entropy_loss=float(
                            cross_entropy_loss.detach().cpu()
                        ),
                        entropy_floor_loss=float(
                            entropy_floor_loss.detach().cpu()
                        ),
                        policy_entropy=float(
                            policy_entropy.mean().detach().cpu()
                        ),
                        target_entropy_floor=float(
                            target_entropy_floor.mean().detach().cpu()
                        ),
                        kl=overall_kl,
                        maximum_stratum_kl=maximum_stratum_kl,
                        kl_cap=cap,
                        regret_matching_fraction=regret_matching_fraction,
                        backtracks=backtracks,
                        applied=True,
                    )

            with torch.no_grad():
                for (parameter, _), backup in zip(
                    parameter_gradients,
                    backups,
                ):
                    parameter.copy_(backup)
            return SearchUpdateStats(
                phase=phase,
                samples=len(rows),
                loss=float(loss.detach().cpu()),
                cross_entropy_loss=float(
                    cross_entropy_loss.detach().cpu()
                ),
                entropy_floor_loss=float(
                    entropy_floor_loss.detach().cpu()
                ),
                policy_entropy=float(
                    policy_entropy.mean().detach().cpu()
                ),
                target_entropy_floor=float(
                    target_entropy_floor.mean().detach().cpu()
                ),
                kl=overall_kl,
                maximum_stratum_kl=maximum_stratum_kl,
                kl_cap=cap,
                regret_matching_fraction=regret_matching_fraction,
                backtracks=8,
                applied=False,
            )
        finally:
            self.model.train(previous_training)

    @staticmethod
    def _phase_logits(output, phase: DistillPhase) -> Tensor:
        return (
            output.masked_bid_logits
            if phase == "bid"
            else output.masked_card_logits
        )


class StratifiedReplayWindow:
    """Bounded replay with equal sampling pressure across configuration strata."""

    def __init__(self, capacity: int = 200_000, max_age: int = 250):
        if capacity < 1:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self.max_age = max_age
        self._rows: dict[ReplayKey, deque[SearchReplaySample]] = {}

    def add(self, sample: SearchReplaySample) -> None:
        if sample.key not in self._rows:
            self._rebalance_limits(len(self._rows) + 1)
            self._rows[sample.key] = deque(maxlen=self._per_stratum_capacity())
        self._rows[sample.key].append(sample)

    def extend(self, samples: Iterable[SearchReplaySample]) -> None:
        for sample in samples:
            self.add(sample)

    def balanced_samples(
        self,
        rng: random.Random,
        *,
        current_iteration: int | None = None,
        phase: DistillPhase | None = None,
        max_samples: int | None = None,
    ) -> list[SearchReplaySample]:
        if current_iteration is not None:
            minimum_iteration = current_iteration - self.max_age
            for key, rows in list(self._rows.items()):
                kept = [
                    row
                    for row in rows
                    if row.iteration >= minimum_iteration
                ]
                if kept:
                    self._rows[key] = deque(kept, maxlen=rows.maxlen)
                else:
                    del self._rows[key]
        selected_rows = {
            key: rows
            for key, rows in self._rows.items()
            if phase is None or key[3] == phase
        }
        if not selected_rows:
            return []
        count = min(len(rows) for rows in selected_rows.values())
        selected: list[SearchReplaySample] = []
        for key in sorted(selected_rows):
            rows = list(selected_rows[key])
            selected.extend(rng.sample(rows, count) if len(rows) > count else rows)
        rng.shuffle(selected)
        return selected[:max_samples] if max_samples is not None else selected

    @property
    def strata(self) -> int:
        return len(self._rows)

    def __len__(self) -> int:
        return sum(len(rows) for rows in self._rows.values())

    def save(self, path: str | Path, *, gate_report_path: str | Path) -> None:
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "rules_fingerprint": rules_fingerprint(),
                "gate_report_path": str(gate_report_path),
                "capacity": self.capacity,
                "max_age": self.max_age,
                "samples": [sample for rows in self._rows.values() for sample in rows],
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "StratifiedReplayWindow":
        payload = _torch_load(path, "cpu")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Search replay does not use schema v4.")
        if payload.get("rules_fingerprint") != rules_fingerprint():
            raise ValueError("Search replay rules fingerprint does not match.")
        replay = cls(
            int(payload["capacity"]),
            int(payload.get("max_age", 250)),
        )
        replay.extend(payload["samples"])
        return replay

    def _per_stratum_capacity(self) -> int:
        return max(self.capacity // max(len(self._rows), 1), 1)

    def _rebalance_limits(self, stratum_count: int) -> None:
        limit = max(self.capacity // stratum_count, 1)
        for key, rows in list(self._rows.items()):
            self._rows[key] = deque(list(rows)[-limit:], maxlen=limit)


def _legal_actions(
    sample: "RolloutSample",
) -> list[BidAction | PlayCardAction]:
    observation = sample.observation
    if not isinstance(observation, Observation):
        return []
    if sample.phase == "bid":
        return [
            BidAction(observation.player_id, bid)
            for bid in observation.legal_bids
        ]
    return [
        PlayCardAction(observation.player_id, card)
        for card in observation.legal_cards
    ]


def _stratified_rollout_samples(
    samples: list["RolloutSample"],
    count: int,
    rng: random.Random,
) -> list["RolloutSample"]:
    if count <= 0 or not samples:
        return []
    strata: dict[tuple[int, int, int, int, str], list["RolloutSample"]] = (
        defaultdict(list)
    )
    for sample in samples:
        strata[
            (
                sample.spec.num_players,
                sample.spec.hand_size,
                sample.bidding_position,
                sample.trick_position,
                sample.opponent_arm,
            )
        ].append(sample)
    queues: list[deque["RolloutSample"]] = []
    for key in sorted(strata):
        rows = strata[key]
        rng.shuffle(rows)
        queues.append(deque(rows))
    rng.shuffle(queues)
    selected: list["RolloutSample"] = []
    while queues and len(selected) < count:
        remaining = []
        for queue in queues:
            if queue and len(selected) < count:
                selected.append(queue.popleft())
            if queue:
                remaining.append(queue)
        queues = remaining
    return selected


def _targeted_rollout_samples(
    samples: list["RolloutSample"],
    count: int,
    rng: random.Random,
) -> list["RolloutSample"]:
    """Split search budget between confident blind spots and broad coverage.

    Half of each probe is spent on the lowest normalized raw-policy entropy
    states (the failure mode we are trying to challenge).  The other half is
    stratified across player count, hand size, position, and opponent arm so
    search does not overfit one narrow corner of the game.  Forced decisions
    have already been removed by the caller.
    """

    if count <= 0 or not samples:
        return []
    challenge_count = min((count + 1) // 2, len(samples))

    def normalized_entropy(sample: "RolloutSample") -> float:
        legal_mask = (
            sample.encoded.legal_bid_mask
            if sample.phase == "bid"
            else sample.encoded.legal_card_mask
        )
        legal_count = sum(bool(value) for value in legal_mask)
        denominator = math.log(max(legal_count, 2))
        return float(sample.old_entropy) / denominator

    # Random tie-breaks keep equal-entropy masked states from always selecting
    # the same deterministic ordering.
    ranked = sorted(
        ((normalized_entropy(sample), rng.random(), sample) for sample in samples),
        key=lambda row: (row[0], row[1]),
    )
    challenged = [row[2] for row in ranked[:challenge_count]]
    challenged_ids = {id(sample) for sample in challenged}
    remainder = [sample for sample in samples if id(sample) not in challenged_ids]
    covered = _stratified_rollout_samples(
        remainder,
        min(count - len(challenged), len(remainder)),
        rng,
    )
    selected = [*challenged, *covered]
    rng.shuffle(selected)
    return selected


def _decision_improvement(decision: SearchDecision) -> float:
    return sum(
        (
            decision.action_probabilities.get(key, 0.0)
            - decision.prior_probabilities.get(key, 0.0)
        )
        * value
        for key, value in decision.action_values.items()
    )


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    estimates = sorted(
        _mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(samples)
    )
    return (
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    )


def _search_replay_sample(
    sample: "RolloutSample",
    decision: SearchDecision,
    iteration: int,
) -> SearchReplaySample:
    target_size = (
        len(sample.encoded.legal_bid_mask)
        if sample.phase == "bid"
        else len(sample.encoded.legal_card_mask)
    )
    probabilities = [0.0] * target_size
    regrets = [0.0] * target_size
    for key, probability in decision.action_probabilities.items():
        index = int(key.split(":", maxsplit=1)[1])
        probabilities[index] = probability
    for key, regret in decision.action_regrets.items():
        index = int(key.split(":", maxsplit=1)[1])
        regrets[index] = regret
    return SearchReplaySample(
        encoded=sample.encoded,
        phase=sample.phase,
        target_probabilities=probabilities,
        spec=sample.spec,
        bidding_position=sample.bidding_position,
        iteration=iteration,
        trick_position=sample.trick_position,
        opponent_arm=sample.opponent_arm,
        regrets=regrets,
    )


def _mixed_search_targets(
    rows: list[SearchReplaySample],
    regret_matching_fraction: float,
    device: torch.device,
) -> Tensor:
    anchored = torch.tensor(
        [row.target_probabilities for row in rows],
        dtype=torch.float32,
        device=device,
    )
    if regret_matching_fraction <= 0.0:
        return anchored
    positive_targets = torch.zeros_like(anchored)
    for index, row in enumerate(rows):
        if row.regrets is None:
            positive_targets[index] = anchored[index]
            continue
        regrets = torch.tensor(
            row.regrets,
            dtype=torch.float32,
            device=device,
        )
        legal = anchored[index] > 0.0
        positive = regrets.clamp_min(0.0) * legal
        if float(positive.sum().cpu()) <= 0.0:
            legal_indices = torch.nonzero(legal, as_tuple=False).flatten()
            if len(legal_indices) == 0:
                positive_targets[index] = anchored[index]
                continue
            best = legal_indices[torch.argmax(regrets[legal_indices])]
            positive[best] = 1.0
        positive_targets[index] = positive / positive.sum().clamp_min(1e-12)
    mixed = (
        (1.0 - regret_matching_fraction) * anchored
        + regret_matching_fraction * positive_targets
    )
    return mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _categorical_entropy(probabilities: Tensor) -> Tensor:
    """Return row entropy; zero-probability masked actions contribute nothing."""

    probabilities = probabilities.float()
    return -(
        probabilities
        * torch.log(probabilities.clamp_min(1e-12))
    ).sum(dim=-1)


def _maximum_stratum_mean(
    values: Tensor,
    rows: list[SearchReplaySample],
) -> float:
    by_stratum: dict[ReplayKey, list[float]] = defaultdict(list)
    for row, value in zip(rows, values.detach().cpu().tolist()):
        by_stratum[row.key].append(float(value))
    return max(
        (_mean(stratum_values) for stratum_values in by_stratum.values()),
        default=0.0,
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _torch_load(path: str | Path, device: str | torch.device) -> dict:
    try:
        return torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover
        return torch.load(Path(path), map_location=device)
