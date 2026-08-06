# Training

## Objective

### NeuRD

The counterfactual preset in `configs/train.toml` uses paper-backed NeuRD with
a multi-action control-variate estimator. For frozen pre-sampling value
prediction `b`, action-value sample
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

After Adam proposes an update, KL is measured on every policy row. The
weighted mean cap must pass. A positive weighted-p99 cap adds a second tail
guard; setting `policy_kl_p99_cap = 0` disables only that acceptance guard and
retains p95, p99, and max diagnostics. Failed proposals restore the exact
model and optimizer state and retry at a geometrically smaller learning rate.

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
representation and cannot change policy KL. The NeuRD preset starts both the
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

Value is a control variate for the focal policy update, so the NeuRD
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
trajectory only when `ppo_self_play_seats="all"`. The active profile uses
`ppo_self_play_seats="focal"`: one current-policy seat samples and contributes
PPO rows in every game, while all other neural seats take legal-action argmax.
This includes both current-policy self-play rivals and historical rivals.

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
The oracle also predicts every absolute seat's final trick count at every
causal prefix, using the same feasibility-masked classes as the actor. This is
an outcome auxiliary, not a hidden-information belief: the oracle already sees
the full deal. Oracle suit presence is intentionally not trained because it is
visible verbatim in the owned HAND tokens and would be a trivial identity.
For the actor advantage only the acting player's column is selected. The
critic runs after rollout collection, uses full causal forwards coalesced by
padded sequence length, and never changes actor tokens or exposes hidden cards
at deployment. The older `privileged` pooled-deal observer critic and the
non-privileged `independent` critic remain available as ablations.

For oracle PPO runs, the dashboard's value panel becomes **Oracle critic
dynamics**. It compares acting-seat RMSE, all-seat RMSE, and the zero-baseline
RMSE; plots acting/all-seat return correlation and fractional critic-loss
reduction from the first to last epoch within each update; and includes the
separately clipped critic gradient norm in the gradient panel. The raw first
and last epoch losses are retained in `metrics.csv` as
`critic_loss_first_epoch` and `critic_loss_last_epoch`.

When `suit_coef` or `trick_coef` is nonzero, PPO actor replay also trains the
ordinary observer-limited heads from the same full-sequence hidden states used
by the policy loss. Each game has unit auxiliary mass, split across learned
seats and then evenly across that observer's genuine token positions; padding
has zero weight. This keeps auxiliary loss scale independent of sequence
length without changing the deliberately length-weighted policy objective.
For ten-card games, metrics report opponent-suit bit accuracy and exact
per-seat final-trick accuracy immediately before the observer's first, fifth,
and ninth play—after that observer has played 0, 4, and 8 cards.

Entropy is calculated over legal actions and normalized by `log(legal_count)`.
Forced moves are excluded. Bid and play temperatures are separate. Adaptive
mode increases a positive temperature when measured normalized entropy falls
below its configured target and decreases it above target. Full masked
`KL(pi_old || pi_new)` guards still backtrack or reject an Adam proposal
independently of PPO ratio clipping. The mean guard is mandatory and the p99
guard is optional. `ppo_behavior_replay_kl` checks
the pre-update numerical difference between reduced-precision cached
collection and full-sequence update replay; it should remain near zero.

`ppo_trainable_policies=1` shares actor weights across learned seats. Larger
values create genuinely independent actor parameters. Self-play seats are
assigned those actors round-robin with a rotating deal offset; focal actors in
anchor games rotate the same way. All actor weights, optimizer state, critic,
entropy temperatures, and assignment cursor are checkpointed. Actor zero is
the deployment/evaluation model and the source of historical league snapshots.

The MPS preset is `configs/ppo-mps.toml`. It uses BF16 autocast and FP16 KV
storage with FP32 master parameters and Adam state. Attention softmax, masked
log-softmax, ratios, KL, entropy, returns, and advantages remain FP32. On the
M5 Pro, this mixed pair measured 99.7 rollout games/s, versus 82.8 for full
BF16, 73.3 for full FP16, and 56.9 for full FP32. Rollout games and update
microbatch positions are independent memory knobs.

Independent spawned rollout producers do not improve MPS throughput. For the
same 768-game workload, four BF16-compute/FP16-cache producers reached 88.5
games/s versus 99.7 for one and used 4.55 GB rather than 1.15 GB after warmup.
They contend for the same Metal execution path instead of creating independent
GPU compute lanes. The production collector therefore remains single-process;
future rollout work should coalesce ready environments across shapes into
larger forwards inside that process rather than replicate the model.

PPO update groups can be coalesced with `ppo_sequence_bucket_width`. Each
actor/oracle sequence is tail-padded to the next bucket boundary (capped at its
model's positional capacity), then shapes with the same padded length are
processed together. Causality makes selected pre-padding readouts exactly
unchanged. The PPO MPS preset uses width 32. In the 768-game production
profile this reduced the 24 small per-shape groups to 3 actor and 4 critic
length buckets. PPO replay
also evaluates bid/card heads only on actual decision rows rather than every
token. Measured together, these changes cut steady update time while keeping
`microbatch_positions=16384`; larger microbatches increased memory without a
repeatable speed gain.

## Rollout packing and opponent curriculum

The counterfactual `configs/train.toml` preset keeps the 48-deal update budget
and apportions it exactly:
24 current-policy self-play games and 24 anchor-opponent games. Each of the 24
`(players, cards)` cells contributes one of each. Through five cards the pair
shares one wave loop; from six cards upward each deal runs in its own wave loop
so a wide tree gets the full cache budget. `rollout.opponent_fraction`,
`rollout.opponent_packing`, `rollout.deals_per_batch`, and
`rollout.parallel_deals_max_hand_size` configure these behaviors.

The current local PPO production profile uses 32 independent games for
each of the 24 `(players, cards)` shapes: 768 complete games/update, split into
384 self-play and 384 historical games. Equivalently, each player-count bucket
gets 256 games across its eight hand sizes. PPO never branches and exactly one
focal seat per game contributes policy rows. The focal samples on-policy;
every non-focal seat uses argmax. `deals_per_batch=128` is only a rollout wave
capacity and does not change the 768-game objective batch.

At the start of each update, five distinct checkpoints are sampled uniformly
from every retained checkpoint at iteration 3500 or later. They are preloaded
and assigned round-robin across historical batches, so a single checkpoint
cannot dominate an update. The current-policy arm and historical arm each
receive exactly half of the fixed game budget.

Training saves every 100 updates. Evaluation is due every 200 updates and
always runs both reproducible policy sampling and deterministic argmax against
the fixed heuristic deal bank. Both best-checkpoint selection and any optional
heuristic-to-history gate use argmax reward by default.

Evaluation and dashboard rendering are not part of the trainer process. The
pipeline launches `tools/watch_evaluation_dashboard.sh`, which starts a fresh
`plump monitor` process on each polling pass. Consequently evaluator,
selection, and dashboard code/config can change without restarting training.
Compact `*.summary.json` sidecars keep each fresh pass below a second in the
no-new-checkpoint case; the full per-round evaluation reports remain intact.
Launch the two detached processes together with:

```bash
tools/run_training_pipeline.sh ppo-oracle-mps-768-v2 configs/ppo-mps.toml \
  --reconfigure --reconfigure-reason "focal PPO versus argmax league"
```

Omitting `--from-checkpoint` creates and records `iter_000000.pt`; the monitor
then supplies its sample and argmax baselines independently. Matching cached
results are reused.

An existing run can adopt an explicitly changed configuration with
`plump train RUN --config ... --reconfigure --reconfigure-reason ...`. This
writes a config-compatible resume checkpoint and archives the previous config
before continuing; ordinary resume still rejects accidental config drift.
`--resume-checkpoint ITERATION` can intentionally rewind within a run. Metric
rows and evaluation directories newer than that checkpoint, plus derived
monitor/gate state, are automatically removed before continuation; interval
checkpoint files themselves are retained.

NeuRD follows [Neural Replicator
Dynamics](https://arxiv.org/abs/1906.00190). The candidate correction is a
Horvitz–Thompson/control-variate estimate with closed-form inclusion
probabilities. This is not labelled CFR, Deep CFR, or R-NaD: those methods
require cumulative/average-strategy state or reference-policy transformations
beyond this local optimizer.

## Configuration

`configs/train.toml` is the counterfactual NeuRD source of truth;
`configs/ppo-mps.toml` is the branch-free PPO/MPS preset. Both use the same
typed schema-v6 training entrypoint and run/checkpoint machinery:

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
  --games-per-shape 32 \
  --bucket-width 32 \
  --microbatch-positions 16384 \
  --warmup 1 --repeats 3
```
