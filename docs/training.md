# Training

## Objective

### NeuRD

The current preset uses paper-backed NeuRD with a multi-action control-variate
estimator. For frozen pre-sampling value prediction `b`, action-value sample
`Q(a)`, and exact candidate-inclusion probability `q(a)`:

```text
Q_hat(a) = b + 1[a observed] * (Q(a) - b) / q(a)
A_hat(a) = Q_hat(a) - sum_x pi_old(x) Q_hat(x)
L_policy = -sum_a stop_gradient(A_hat(a)) * logit(a)
```

With exponent one and no clip/cap, every component has expectation
`Q_pi(a) - V_pi`; unobserved legal actions receive the centered control
variate rather than a fabricated zero. This is the sampled NeuRD update, not a
policy-gradient or self-normalized mirror approximation.

The collector uses fixed-width stratified old-policy sampling. If the legal
set fits the branch budget it is fully enumerated. Otherwise legal actions are
partitioned into disjoint, policy-mass-balanced strata and one action is drawn
from each stratum under `pi_old(. | stratum)`. For stratum mass `M_g`:

```text
backup/reach weight = M_g
q(a) = pi_old(a) / M_g
V_hat = sum_g M_g Q(a_g)
```

The representatives are exactly distinct, their weights sum to one, and the
backup and represented descendant distribution are unbiased under the old
policy. Full enumeration has `q(a)=1` and weight `pi_old(a)`.

Every branch child carries its represented old-policy reach. Value, belief,
and descendant policy losses are weighted by that reach. Cache blocking
changes estimator variance, not the expected objective.

After Adam proposes an update, KL is measured on every policy row. Both
weighted mean and weighted p99 caps must pass; the current preset uses `0.01`
and `0.05`, respectively. Failed proposals restore the exact model and
optimizer state and retry at a geometrically smaller learning rate.

The first, nominal Adam proposal is always evaluated over the complete policy
row set before any reduction. Metrics retain its weighted mean, p95, p99, and
max KL plus booleans for which caps it exceeded. The existing `policy_kl*`
fields describe the final accepted proposal, not the rejected one. Intermediate
retries may still use the proof-only p99 early exit because they are immediately
discarded and are not used as the durable pre-backtrack diagnostic.

KL percentiles use the same objective weights as the loss. With equal
per-tree weighting and negative branch-depth weighting, a shallow decision can
carry at least one percent of total weight; weighted p99 can therefore equal
the maximum. This is intentional in the current implementation and should be
remembered when interpreting or changing the tail cap.

Backtracking scales the shared trunk and bid/card action heads because those
parameters can move the policy. It does not scale the value, suit-presence,
trick-count, or bid-hit readout heads: those heads only consume the shared
representation and cannot change policy KL. The current preset starts both the
core and auxiliary readouts at `2.5e-5`; the older
`learning_rate` field remains the fallback when a group-specific value is
omitted. Core and readout gradients are clipped independently, so an exact
NeuRD correction with a large norm cannot rescale the readout-head sample
before Adam. Older one-group checkpoints are split losslessly at load time.

### Sampled mirror target

`training.policy_objective="sampled_mirror"` remains an explicit
bias/variance alternative. It exponentiates the same stochastic full-action
advantage vector and fits the resulting target. The target is exact for that
realized vector, but expectation does not commute with exponentiation, so it
is intentionally described as sampled mirror improvement rather than an
unbiased full-information mirror-descent update.

For both objectives, per-tree weighting prevents large long-game trees from
silently dominating. A negative branch-depth exponent keeps bidding and early
play meaningful despite the multiplicity of deep branch nodes. The preset uses
`(1 + branch_depth)^-0.5`, a moderate early-decision preference, plus five
distinct bid strata and four distinct play strata. Smaller legal sets are fully
enumerated. The depth factors are normalized within each tree, so this changes
where its policy weight lands without changing that tree's total weight.

The optimizer also trains relative value plus suit-presence and final
trick-count auxiliaries. Both belief auxiliaries use coefficient `0.05`; value
remains `0.5`. Bid hit and entropy remain disabled.

Value is a control variate for the focal policy update, so the current
objective is normalized MSE at exactly the focal bid/play readout positions:

```text
L_value = 0.5 * ((V(s) - backed_return) / value_reward_scale)^2
```

MSE is intentional: it learns the conditional mean `E[R | s]` required by a
variance-reducing state value. Smooth-L1 becomes mostly absolute error for the
game's discrete ±5–20 point returns and therefore approaches a conditional
median, which can stay close to zero even when the expected return is
state-dependent. Decision rows use the same per-tree, reach, and branch-depth
weights as the policy objective. This both preserves the old-policy state
distribution and avoids spending most value supervision on event-token
positions where the baseline is never consumed. `value_positions="all"` and
`value_objective="smooth_l1"` remain explicit diagnostic alternatives.

Counterfactual branches do not turn these into off-policy labels. Descendant
positions carry their represented old-policy reach; the reaches sum to the
ordinary live-policy state distribution in expectation. Decision-state values
use the recursive unbiased branch backup, while outcome beliefs use the
originally sampled on-policy continuation. Consequently, residual value error
can reflect partial observability and Monte Carlo variance rather than an
unweighted exploratory policy.

The beliefs are chosen so each asks something the prefix does not already
answer. Own suit presence is excluded because the observer's hand and its plays
are both in its own token stream — supervising it would fit an identity. Own
trick count is included because the observer's final total depends on how the
rest of the round resolves. Bid hit is off by default: measured against a
per-position base rate it is the weakest of the three, and unlike the others it
needs a hidden layer, since `won == bid` is a bump rather than a threshold and
a linear readout cannot express two decision boundaries.

### Branch-free PPO

`training.policy_objective="ppo"` selects an independent, branch-free path;
the NeuRD and sampled-mirror implementations remain unchanged and selectable.
Each deal has exactly one sampled leaf. In self-play, every seat is a learned
trajectory by default; against the heuristic or a historical policy, only the
focal current-policy seat is learned.

For game `g`, let `C_g` be its learned seats. Every learned decision has weight
`1 / (number_of_games * C_g)`. The loss is not divided by the number of
decisions, so longer hands contribute more decisions naturally:

```text
L = (1 / games) sum_g (1 / C_g) sum_i sum_t L[g, i, t]
```

The clipped actor objective uses the exact legally masked behavior policy:

```text
ratio = exp(log pi_new(action|observation) - log pi_old(action|observation))
L_clip = -min(ratio * advantage,
              clip(ratio, 1-epsilon, 1+epsilon) * advantage)
advantage = terminal_relative_reward - V_old(global_state, seat)
```

There is no epsilon-random behavior policy. Sampling from an external mixture
while storing `pi_old` probabilities would make ordinary PPO off-policy.

PPO uses an independent critic trunk. In the default
`ppo_critic_mode="oracle"`, update batching builds one canonical sequence per
environment game rather than one sequence per learned observer. Its prefix has
one distinct HAND token for every dealt card, ordered by absolute owner seat;
each token contains owner, exact-card, rank, and suit fields. The public suffix
also uses absolute seat ids. The value head emits `max_players` columns, and
column `s` is always the value for the same seat `s` named by card-owner and
event-actor fields. An assertion checks this tie for every PPO decision while
building the update batch.

At each learned pre-action state the critic is trained against all active
players' undiscounted terminal relative returns. Its loss averages the player
axis, so adding output columns does not multiply the weight of larger tables.
For the actor advantage only the acting player's column is selected. The
critic runs after rollout collection, uses full causal forwards grouped by
player/hand shape, and never changes actor tokens or exposes hidden cards at
deployment. The older `privileged` pooled-deal observer critic and the
non-privileged `independent` critic remain available as ablations.

For oracle PPO runs, the dashboard's value panel becomes **Oracle critic
dynamics**. It compares acting-seat RMSE, all-seat RMSE, and the zero-baseline
RMSE; plots acting/all-seat return correlation and fractional critic-loss
reduction from the first to last epoch within each update; and includes the
separately clipped critic gradient norm in the gradient panel. The raw first
and last epoch losses are retained in `metrics.csv` as
`critic_loss_first_epoch` and `critic_loss_last_epoch`.

Entropy is calculated over legal actions and normalized by `log(legal_count)`.
Forced moves are excluded. Bid and play temperatures are separate. Adaptive
mode increases a positive temperature when measured normalized entropy falls
below its configured target and decreases it above target. Full masked
`KL(pi_old || pi_new)` mean and p99 guards still backtrack or reject an Adam
proposal independently of PPO ratio clipping. `ppo_behavior_replay_kl` checks
the pre-update numerical difference between FP16-KV cached collection and the
full-sequence update replay; it should remain near zero.

`ppo_trainable_policies=1` shares actor weights across learned seats. Larger
values create genuinely independent actor parameters. Self-play seats are
assigned those actors round-robin with a rotating deal offset; focal actors in
anchor games rotate the same way. All actor weights, optimizer state, critic,
entropy temperatures, and assignment cursor are checkpointed. Actor zero is
the deployment/evaluation model and the source of historical league snapshots.

The MPS preset is `configs/ppo-mps.toml`. It uses BF16 autocast with FP32 master
parameters and Adam state, FP16 KV storage, and FP32 attention softmax, masked
log-softmax, ratios, KL, entropy, returns, and advantages. Rollout games and
update microbatch positions are independent memory knobs.

## Rollout packing and opponent curriculum

The preset keeps the existing 48-deal update budget and apportions it exactly:
24 current-policy self-play games and 24 anchor-opponent games. Each of the 24
`(players, cards)` cells contributes one of each. Through five cards the pair
shares one wave loop; from six cards upward each deal runs in its own wave loop
so a wide tree gets the full cache budget. `rollout.opponent_fraction`,
`rollout.opponent_packing`, `rollout.deals_per_batch`, and
`rollout.parallel_deals_max_hand_size` configure these behaviors.

The anchor initially consists of deterministic heuristic opponents. Heuristic
seats run through the batched wave scheduler but consume no neural forward or
KV-cache rows; only the focal current policy is encoded. Inline evaluation uses
reproducible policy sampling against the same heuristic. After mean relative
reward is above `evaluation.opponent_switch_reward` for
`evaluation.opponent_switch_consecutive` consecutive evaluations, the anchor
switches permanently to historical league opponents. The phase and streak are
checkpointed. Only historical checkpoints with iteration in
`[ceil(current / 2), current]` are eligible for sampling.

An existing run can adopt an explicitly changed configuration with
`plump train RUN --config ... --reconfigure --reconfigure-reason ...`. This
writes a config-compatible resume checkpoint and archives the previous config
before continuing; ordinary resume still rejects accidental config drift.

NeuRD follows [Neural Replicator
Dynamics](https://arxiv.org/abs/1906.00190). The candidate correction is a
Horvitz–Thompson/control-variate estimate with closed-form inclusion
probabilities. This is not labelled CFR, Deep CFR, or R-NaD: those methods
require cumulative/average-strategy state or reference-policy transformations
beyond this local optimizer.

## Configuration

`configs/train.toml` is the source of truth for a new run. Its defaults match
the optimized schema-v6 training entrypoint:

- balanced player/card/bidding-position schedule;
- schema-v6 model dimensions and token flags;
- branch rule and per-shape rate schedule;
- cache and microbatch limits;
- loss coefficients, KL cap, and warmup;
- policy objective, sampled-mirror target, and KL backtracking;
- evaluation, dashboard, checkpoint, and league cadence.

Use typed dotted overrides:

```bash
uv run plump train experiment \
  --set model.d_model=320 \
  --set training.core_learning_rate=0.00002 \
  --set training.auxiliary_learning_rate=0.0002 \
  --set rollout.kv_dtype=\"fp16\"
```

Unknown keys are rejected. The fully resolved configuration is stored in the
run and must match on resume.

## Benchmarks

Measurement-only tools live in `tools/benchmarks/`. They cover collect/update
throughput, KV-cache scaling, schedule calibration, per-shape cost, branch
rate grids, rollout sweeps, and isolated solo-versus-paired shape grids.
Generated results are explicitly scoped to a selected run or output path.
The production PPO actor/critic path has a dedicated benchmark:

```bash
.venv/bin/python tools/benchmarks/benchmark_ppo.py \
  --games-per-shape 128 \
  --microbatch-positions 16384 \
  --warmup 1 --repeats 3
```
