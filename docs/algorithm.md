# How Plump training works

This document explains the original counterfactual NeuRD training algorithm
from first principles. The repository also has a branch-free PPO path; after
the original derivation, [Section 18](#18-the-branch-free-ppo-option) derives
that method and explains what changes and what stays invariant. The main
question is:

> If training samples actions and counterfactual branches, why should
> minimizing its loss produce a policy that gets high game reward?

The short answer is that the policy loss is built from estimates of each
action's **advantage**: how much better or worse that action is than the
policy's average action at the same decision. The estimates are corrected for
the probability that an action was inspected. In expectation, a good action's
logit is pushed up and a bad action's logit is pushed down. A KL guard prevents
one neural update from moving the policy too far at once.

This is a statistically coherent reward-improvement method. It is not a proof
that every finite neural update increases measured reward, nor a proof of
convergence to a globally optimal or Nash policy. The distinction is explained
in [What is guaranteed, and what is not](#14-what-is-guaranteed-and-what-is-not).

## The whole loop in one picture

```text
observable game history h
        |
        v
old policy pi_old(a | h) and old value prediction b = V_model(h)
        |
        v
sample the live action; sometimes clone the game and try other legal actions
        |
        v
roll every branch to the end -> terminal relative rewards
        |
        v
construct corrected estimates Q_hat(h,a) and A_hat(h,a)
        |
        v
push logits up for positive A_hat and down for negative A_hat
        |
        v
Adam proposes new weights -> measure KL(old policy || new policy)
        |
        +-- too large: restore and retry with a smaller step
        |
        +-- acceptable: keep the weights
```

The value, suit-presence, and trick-count losses train useful predictions from
the same transformer representation. The policy loss is the part that directly
turns game reward into action preference.

## 1. What is the policy?

At a bidding or play decision, the model sees only the information available
to the focal player: its hand and the public event history. Call this observable
history $h$. It does not receive the opponents' hidden cards.

The model emits one real number, or **logit**, $z_\theta(h,a)$, for every
possible action. Illegal actions are masked. A softmax converts the remaining
logits into a probability distribution:

$$
\pi_\theta(a\mid h)
=
\frac{\exp z_\theta(h,a)}
     {\sum_{x\in\mathcal A(h)} \exp z_\theta(h,x)}.
$$

- $\theta$ means all trainable model weights.
- $\mathcal A(h)$ is the legal action set.
- $\pi_\theta(a\mid h)$ is the probability of taking action $a$.

Adding the same constant to every legal logit changes no probabilities. What
matters is the difference between logits.

During training, actions are sampled from this distribution. At inference one
can either sample from it or choose the action with the largest logit
(`argmax`). The stochastic distribution is the policy the training math
directly optimizes.

## 2. What is the reward?

After a completed round, the focal player $i$ receives

$$
R_i
=
\operatorname{score}_i-
\frac{1}{P-1}\sum_{j\ne i}\operatorname{score}_j,
$$

where $P$ is the number of players.

So positive reward means the focal player outscored its opponents on average;
negative reward means it did worse. These relative rewards sum to zero across
the table. Increasing the focal player's expected reward therefore means
improving its score relative to the other players, not merely predicting an
unrelated label.

Everything below would still be mathematically valid for a different terminal
reward. The learned behavior is only as desirable as the selected reward.

## 3. V, Q, and advantage

These three quantities answer different questions at the same observable
history $h$.

| Symbol | Question | Definition |
| --- | --- | --- |
| $V^\pi(h)$ | How well will I do before choosing an action? | Expected terminal reward when the policy acts normally from $h$. |
| $Q^\pi(h,a)$ | How well will I do if I choose this particular action now? | Expected terminal reward after taking $a$, then following the rollout policies. |
| $A^\pi(h,a)$ | Is this action better or worse than my usual choice here? | $Q^\pi(h,a)-V^\pi(h)$. |

More formally,

$$
Q^\pi(h,a)
=
\mathbb E[R\mid h,\ a\text{ now, rollout policies thereafter}],
$$

and

$$
V^\pi(h)
=
\sum_{a\in\mathcal A(h)}\pi(a\mid h)Q^\pi(h,a).
$$

Therefore

$$
A^\pi(h,a)=Q^\pi(h,a)-V^\pi(h),
\qquad
\sum_a \pi(a\mid h)A^\pi(h,a)=0.
$$

An advantage of $+2$ means that action is expected to earn two more reward
points than the policy's normal mixture at that history. An advantage of
$-2$ means two fewer.

### There is no learned Q-head

This implementation does **not** ask a neural network to guess every
$Q(h,a)$. It tries an action in a cloned environment and rolls the resulting
game forward. The resulting return, or a recursive branch backup, is a sample
of $Q(h,a)$.

The model does have a learned scalar value head. Its output is an estimate of
$V(h)$, but the policy algorithm uses it as a variance-reducing baseline, not
as the source of truth for which action was good. A poor value prediction can
make updates noisier without, by itself, biasing the exact estimator described
below.

Because Plump has hidden information, $h$ is an observation history rather
than the omniscient game state. Thus $V(h)$ and $Q(h,a)$ average over hidden
hands consistent with that information, as encountered across many randomly
dealt games.

## 4. How one rollout produces Q samples

Each training deal designates one **focal player**. At non-focal decisions the
appropriate opponent simply acts: the current policy in self-play, the
heuristic in heuristic games, or a saved policy in historical games.

At a focal decision:

1. The current, frozen rollout policy $\pi_{\text{old}}$ produces legal-action
   probabilities.
2. One action is sampled for the ordinary on-policy path, called the spine.
3. At selected decisions, the environment and transformer cache are cloned.
   Several legal actions are taken in separate children.
4. Each child continues according to the same rollout policies until the game
   ends, possibly creating deeper counterfactual branches.
5. Terminal relative rewards are backed up through the tree.

Taking a counterfactual action does not mean the training objective has changed
to a uniformly random policy. It is a way of observing what would have happened
after a one-action deviation. The inclusion and reach corrections below keep
the estimated objective tied to $\pi_{\text{old}}$.

The focal player's bid is always branched. Play decisions are branched according
to a per-shape branch rate and the cache budget. If a decision is not branched,
the single sampled action still yields a valid, but higher-variance, policy
estimate.

## 5. Sampling only some actions without bias

Suppose a legal action $a$ is inspected with known inclusion probability

$$
q(a)=\Pr(a\text{ is included among the candidates}\mid h).
$$

Let $I(a)$ be 1 when it was included and 0 otherwise. Let $Y(a)$ be the
sampled return obtained after taking it. Conditional on taking that action,

$$
\mathbb E[Y(a)\mid h,a]=Q^\pi(h,a).
$$

Before candidate sampling, the model also produced a frozen scalar baseline
$b$. The implementation constructs

$$
\widehat Q(h,a)
=
b
+
\frac{I(a)}{q(a)}\bigl(Y(a)-b\bigr).
$$

This is a control-variate Horvitz--Thompson estimator. Its key property is

$$
\begin{aligned}
\mathbb E[\widehat Q(h,a)]
&=b+\mathbb E\left[\frac{I(a)}{q(a)}(Y(a)-b)\right]\\
&=b+(Q^\pi(h,a)-b)\\
&=Q^\pi(h,a).
\end{aligned}
$$

This equality does not require $b$ to be accurate. It requires the baseline
to be fixed before action selection, the reported $q(a)$ to be correct, and
the continuation return to be an unbiased $Q$ sample.

An unobserved legal action receives $\widehat Q=b$ on that particular update.
That is not a claim that its true value equals $b$. Across repeated candidate
samples, the occasional $1/q$-weighted residual supplies exactly the missing
expectation.

The estimated advantage is the complete legal-action vector centered under the
old policy:

$$
\widehat V(h)=\sum_x\pi_{\text{old}}(x\mid h)\widehat Q(h,x),
$$

$$
\widehat A(h,a)=\widehat Q(h,a)-\widehat V(h).
$$

For every realized sample—not merely in expectation—

$$
\sum_a\pi_{\text{old}}(a\mid h)\widehat A(h,a)=0.
$$

With the current exact settings (inclusion exponent 1, no inclusion cap, and
no advantage clipping),

$$
\mathbb E[\widehat A(h,a)]
=
Q^\pi(h,a)-V^\pi(h)
=
A^\pi(h,a).
$$

Clipping the residual, capping $1/q$, or changing its exponent can reduce
variance, but would deliberately give up this exact unbiasedness. Those knobs
exist for experiments and are disabled for the active NeuRD preset.

### The one-action case

If no counterfactual branch is built, only the ordinary sampled action is
observed. Its inclusion probability is simply

$$
q(a)=\pi_{\text{old}}(a\mid h).
$$

The same equation remains unbiased. It can be noisy when a low-probability
action is sampled because $1/q(a)$ is large, which is why evaluating several
stratified candidates is useful.

### The stratified candidate rule

The active preset uses five candidate strata for bidding and four for play.
If the legal set is no larger than that budget, every legal action is evaluated
and $q(a)=1$.

Otherwise, legal actions are partitioned into disjoint groups with roughly
balanced policy mass. For stratum $G_g$, define

$$
M_g=\sum_{a\in G_g}\pi_{\text{old}}(a\mid h).
$$

Each group contributes one representative, drawn according to the old policy
conditioned on that group. Hence

$$
q(a)=\frac{\pi_{\text{old}}(a\mid h)}{M_g}
\quad\text{for }a\in G_g.
$$

The implementation does not draw all of these independently. The live sampled
action is reused as the representative of whichever stratum contains it, and
only the remaining strata draw fresh representatives. This does not change
$q$. Conditional on the live action landing in $G_g$ — which happens with
probability $M_g$ — it is already distributed as $\pi_{\text{old}}$ restricted
to that group, so for $a\in G_g$

$$
\Pr(a\text{ represents }G_g)
=
M_g\frac{\pi_{\text{old}}(a\mid h)}{M_g}
+(1-M_g)\frac{\pi_{\text{old}}(a\mid h)}{M_g}
=\frac{\pi_{\text{old}}(a\mid h)}{M_g},
$$

which is the same $q(a)$ either way. What the reuse does change is that
representatives are no longer independent across strata: the spine is shared
with the tree rather than sampled twice. Every unbiasedness claim below
survives this, because each one is an expectation of a linear function of the
representatives and linearity does not require independence. Variance
statements would not carry over unchanged.

Representatives are distinct, every stratum is represented, and the masses
sum to one. A state-value backup is

$$
\widehat V(h)=\sum_g M_gY(a_g),
\qquad a_g\sim\pi_{\text{old}}(\cdot\mid G_g).
$$

Its expectation is exactly

$$
\mathbb E[\widehat V(h)]
=
\sum_g\sum_{a\in G_g}\pi_{\text{old}}(a\mid h)Q(h,a)
=
V^\pi(h).
$$

So the branching tree gets broad action coverage without replacing the
old-policy value with a uniform-action value.

### A numerical example

Suppose the old policy has four actions:

| Action | Old probability | True/sample $Q$ when inspected |
| --- | ---: | ---: |
| $a_1$ | 0.50 | 2 |
| $a_2$ | 0.25 | not inspected |
| $a_3$ | 0.15 | 4 |
| $a_4$ | 0.10 | not inspected |

Use two strata: $G_1=\{a_1\}$ with mass 0.5 and
$G_2=\{a_2,a_3,a_4\}$ with mass 0.5. Imagine $a_1$ and $a_3$ are the
representatives and the baseline is $b=1$.

For $a_1$, $q(a_1)=1$, so $\widehat Q(a_1)=2$. For $a_3$,
$q(a_3)=0.15/0.5=0.3$, so

$$
\widehat Q(a_3)=1+\frac{4-1}{0.3}=11.
$$

The unobserved actions receive 1 on this draw. Thus

$$
\widehat Q=[2,1,11,1]
$$

and

$$
\widehat V
=0.50(2)+0.25(1)+0.15(11)+0.10(1)
=3.
$$

This is also the stratified backup $0.5(2)+0.5(4)=3$. The realized estimate
for $a_3$ looks extreme because it represents the occasions on which other
members of its stratum would have been selected. Across repeated draws, each
action's mean estimate is its true $Q$.

The corresponding advantage sample is

$$
\widehat A=[-1,-2,8,-2],
$$

whose old-policy-weighted mean is exactly zero.

## 6. Why descendant branches do not become off-policy training data

A sampled candidate does more than provide a $Q$ value at its parent. Its
descendant decisions also produce policy, value, and belief rows. Those rows
must not all receive equal weight, because the branching mechanism deliberately
created states more often than the live policy would visit them.

Each child therefore carries the old-policy mass it represents:

- exhaustive child for action $a$: reach factor $\pi(a\mid h)$;
- stratified representative from $G_g$: reach factor $M_g$;
- deeper child: product of all reach factors along its path.

Loss rows are multiplied by this reach. In expectation, descendant state
weight matches the state distribution generated by the old policy. Recursive
branch backups use the same masses, so an upstream $Q$ sample remains
unbiased even when its continuation contains more branches.

This is the answer to the apparent paradox: the collector can deliberately
try actions that the live policy did not take, while the **weighted estimator**
still describes the live policy's values and advantages.

The training preset also gives each dealt tree equal total importance rather
than allowing a large ten-card tree to outweigh a small tree merely because it
created more rows. A depth factor of $(1+d)^{-0.5}$ moves some weight toward
bids and early plays without changing a tree's total weight. These are
intentional positive reweightings of which decision states matter; they do not
reverse the good-action/up, bad-action/down direction at a state.

## 7. Turning advantage into a policy update

The default policy objective is sampled NeuRD. For a policy row, its main term
is

$$
L_{\text{regret}}(h)
=
-\sum_{a\in\mathcal A(h)}
\operatorname{stopgrad}\!\left[\widehat A(h,a)\right]
z_\theta(h,a).
$$

`stopgrad` means the optimizer treats the sampled advantage as a fixed target.
It does not change the rollout return or the value baseline to make the loss
look smaller.

The derivative with respect to an independently controllable logit is

$$
\frac{\partial L_{\text{regret}}}{\partial z(h,a)}=-\widehat A(h,a).
$$

Gradient descent therefore does exactly the intuitive thing:

- positive advantage $\Rightarrow$ increase the action's logit;
- negative advantage $\Rightarrow$ decrease the action's logit;
- zero advantage $\Rightarrow$ no direct preference change.

The full row loss also contains a forward KL anchor,

$$
L_{\text{policy}}
=L_{\text{regret}}
+\lambda_{\mathrm{KL}}
D_{\mathrm{KL}}\!\left(
\pi_{\text{old}}\,\Vert\,\pi_\theta
\right).
$$

With the current one-epoch update, the model equals the old policy when this
loss is differentiated, so this KL term starts with zero gradient. It becomes
an optimization anchor if multiple epochs are used. The active protection in
the current configuration is the post-update KL acceptance test described
below.

### Why this is a reward hill-climbing direction in the clean case

Consider the idealized tabular case: every history has independent logits,
the exact advantages are known, opponents and environment are fixed, and the
step is infinitesimal. Ignoring a positive per-row weighting constant, NeuRD
moves logits according to

$$
\frac{d z(h,a)}{dt}=A^\pi(h,a).
$$

Differentiating the softmax gives

$$
\frac{d\pi(a\mid h)}{dt}
=
\pi(a\mid h)
\left(A(h,a)-\sum_x\pi(x\mid h)A(h,x)\right).
$$

The centered-advantage identity makes the second term zero:

$$
\frac{d\pi(a\mid h)}{dt}=\pi(a\mid h)A(h,a).
$$

The first-order change in the value of that decision is therefore

$$
\begin{aligned}
\frac{d}{dt}\sum_a\pi(a\mid h)Q(h,a)
&=\sum_a Q(h,a)\pi(a\mid h)A(h,a)\\
&=\sum_a\pi(a\mid h)A(h,a)^2\\
&\ge 0.
\end{aligned}
$$

For a finite-horizon game, let $d^\pi(h)$ be the old policy's visitation weight
and let $c(h)>0$ be any positive training weight applied to that history. The
policy-gradient theorem adds the same terms over histories:

$$
\frac{dJ}{dt}
=
\sum_h d^\pi(h)c(h)
\mathbb E_{a\sim\pi(\cdot\mid h)}[A(h,a)^2]
\ge 0.
$$

This is the precise hill-climbing argument: with exact tabular values and a
fixed opponent, the NeuRD direction has non-negative first-order expected
reward improvement. It is strictly positive wherever a reached and positively
weighted history mixes actions with different values. The implementation's
tree and depth weights change $c(h)$, but remain non-negative.

Equivalently, an independent finite logit step has the form

$$
\pi_{\text{new}}(a\mid h)
\propto
\pi_{\text{old}}(a\mid h)\exp\!\left(\eta A(h,a)\right),
$$

so probability moves toward higher-advantage actions.

### Why use NeuRD instead of an ordinary sampled policy gradient?

An ordinary softmax policy-gradient signal on logit $a$ is proportional to
$\pi(a)A(a)$. An action whose probability has become tiny then recovers very
slowly, even if it later becomes strongly advantageous as self-play opponents
change.

NeuRD applies the advantage directly to the logit. A suppressed action can
therefore recover. This corresponds to replicator/Hedge-style no-regret
dynamics in the ideal tabular case.

## 8. The KL acceptance guard

The actual model is a shared transformer, and Adam updates millions of coupled
parameters. A modest parameter step can unexpectedly move some action
distributions a great deal. The trainer therefore treats Adam's result as a
proposal.

After the proposal it recomputes every policy row and measures

$$
D_{\mathrm{KL}}
\left(\pi_{\text{old}}(\cdot\mid h)
\,\Vert\,
\pi_{\text{new}}(\cdot\mid h)\right).
$$

The standard local preset requires both the objective-weighted mean KL to be
at most 0.01 and the weighted p99 to be at most 0.05. These numbers are
configuration values, not mathematical constants.

If either guard fails, the exact pre-update model and Adam state are restored,
and the same proposal is retried with half the policy-sensitive learning rate.
There are up to eight backtracking attempts. If none pass, the complete update
is rolled back.

The guard does not prove that reward increased. It prevents an estimator or
function-approximation error from causing an uncontrolled policy jump, making
the local hill-climbing approximation more credible.

## 9. Learning the value V

The value head is trained at focal decision positions toward the backed return
$Y(h)$:

$$
L_V
=
\frac{1}{2}
\left(\frac{V_\theta(h)-Y(h)}{5}\right)^2.
$$

Rows carry the same tree, reach, and depth weights used by the policy data.
Mean-squared error is intentional: its population optimum is

$$
V_\theta(h)=\mathbb E[Y\mid h],
$$

the conditional mean required for a good control variate. The division by 5
only changes numerical scale.

The value target before a branch is the recursively backed expected return.
After a branch, each child uses the return of the continuation it represents.
There is no temporal-difference bootstrap from a later value prediction; the
targets ultimately resolve to completed-game rewards.

An accurate baseline makes $Y-b$ smaller and reduces variance in
$\widehat Q$. It is not required for the expectation to be correct because
the same $b$ is added back outside the inclusion correction.

## 10. Auxiliary predictions

The current loss also trains:

- whether each opponent still holds each suit;
- every player's final trick count;
- optionally, whether each player exactly hits its bid (disabled in the active
  preset).

These labels never become policy inputs. They encourage the shared transformer
to represent strategically relevant information. The active coefficients are

$$
L_{\text{total}}
=
1.0L_{\text{policy}}
+0.5L_V
+0.05L_{\text{suit}}
+0.05L_{\text{trick}},
$$

with bid-hit and entropy coefficients zero.

Although the auxiliary readout parameters have their own optimizer group,
their gradients also pass through the shared transformer. They can help the
policy representation, but they are another reason the neural update is not a
literal tabular policy-improvement proof. The KL guard observes their net
effect on the policy.

## 11. What one current local update contains

The standard local configuration covers all 24 combinations of

- 3, 4, or 5 players; and
- 3 through 10 cards.

It deals two games per combination, for 48 games total:

- 24 current-policy self-play games;
- 24 games with the focal policy against the deterministic heuristic.

Only focal decisions create policy-loss rows and counterfactual branches.
Across games, focal seats and bidding positions are randomized so the same
policy learns to act from every role.

The opponent arm initially uses the heuristic. Every 100 updates, the sampled
policy is evaluated against it on a fixed deal bank. After four consecutive
evaluations with positive mean relative reward, the anchor permanently changes
to recent historical checkpoints. This avoids spending the rest of training
specializing against one deterministic opponent after it has been beaten.

For one update, opponents and $\pi_{\text{old}}$ are frozen. In self-play the
gradient does not differentiate through the opponents' sampled actions; it
improves the focal policy against the behavior encountered in that rollout.
The next update recollects games with the new policy.

## 12. Why sampling can eventually produce a good policy

The chain of reasoning is:

1. A completed game supplies the exact selected terminal reward.
2. Counterfactual continuations supply Monte Carlo samples of $Q(h,a)$.
3. Exact inclusion probabilities turn partial candidate coverage into an
   unbiased full-action $Q$ and advantage estimate.
4. Reach weights prevent the artificial branch count from changing the
   represented continuation policy.
5. In expectation, the NeuRD loss raises logits for actions with positive true
   advantage and lowers logits for actions with negative true advantage.
6. In the exact tabular, fixed-opponent limit, this direction has first-order
   reward derivative
   $\sum_h d^\pi(h)c(h)\mathbb E_{a\sim\pi}[A(h,a)^2]\ge0$.
7. Repeated recollection updates the $Q$ estimates after the policy changes.
8. KL backtracking keeps each accepted neural policy change local.

Thus action sampling is not noise with no relationship to the objective. It is
a Monte Carlo method for estimating the direction in which expected terminal
reward improves.

## 13. Sampling versus argmax at inference

The policy's expected sampled reward is the quantity most directly aligned
with training. If repeated improvement makes one action consistently better,
its logit and probability rise; eventually both sampling and argmax usually
select it.

However, there is an important game-theoretic caveat. Some competitive states
require a mixed strategy. A stochastic policy can have high expected reward
and be difficult to exploit, while always taking its single most probable
action can become predictable. Therefore:

- sampled evaluation is the primary test of the policy that was trained;
- argmax evaluation is still useful and may be excellent against the
  deterministic heuristic;
- high sampled reward does not mathematically guarantee equally high argmax
  reward against every opponent.

Both modes should be measured if both will be used.

## 14. What is guaranteed, and what is not

### What the math supports

Under the current exact estimator settings:

- each candidate action's $\widehat Q$ is unbiased for its old-policy action
  value;
- each $\widehat A$ is unbiased for $Q^\pi-V^\pi$;
- recursive stratified backups are unbiased for old-policy values;
- descendant reach weights represent the old-policy state distribution in
  expectation;
- the expected independent-logit NeuRD direction is a local reward-improvement
  direction against fixed opponents;
- every accepted neural step is bounded by configured mean and p99 policy KL.

The repository has statistical and deterministic tests for inclusion
probabilities, stratified backups, represented reach, full-action advantage
expectation, unbranched gradients, and KL backtracking.

### What it does not guarantee

It does not guarantee that:

- every individual update raises evaluation reward;
- a finite sample has low variance, especially for a very small inclusion
  probability;
- a shared neural parameter update moves every history in its requested
  direction—gradients from different histories and auxiliary tasks can
  interfere;
- Adam follows the exact independent-logit NeuRD step;
- self-play converges rather than cycles;
- performance generalizes beyond the mixture of self, heuristic, historical,
  and evaluation opponents used during training;
- the logit direction the policy is invariant to stays bounded. Shifting every
  legal logit at a history by one constant leaves $\pi$ unchanged, but the
  NeuRD loss $-\sum_a\widehat A(a)z(a)$ has gradient $-\sum_a\widehat A(a)$
  along it, and that sum is not zero in general — only the
  $\pi_{\text{old}}$-weighted one is. The KL guard measures policy KL, so it
  cannot see this direction. The `policy_logit_shift` metric reports the
  weighted mean legal logit so a drift is visible;
- argmax is as robust as sampling when a mixed strategy is needed;
- the globally best policy is representable by this model and found by this
  optimizer.

These are reasons to rely on held-out sampled and argmax game evaluation, not
reasons the loss is disconnected from reward. The estimator supplies a
mathematically correct expected direction; evaluation checks whether finite
neural optimization realizes the intended improvement.

## 15. Compact pseudocode

```text
repeat for each training update:
    freeze current policy as pi_old for data collection

    for each scheduled deal:
        choose a focal player and opponent arm
        play from pi_old / heuristic / historical policies

        at each focal decision h:
            save pi_old(. | h) and baseline b
            sample the live action
            sometimes select stratified counterfactual candidates
            roll each candidate continuation to terminal reward
            recursively back up child values
            record each candidate's exact inclusion probability q
            record each descendant's represented old-policy reach

    for each focal policy row:
        Q_hat(a) = b + I(a) / q(a) * (Y(a) - b)
        A_hat(a) = Q_hat(a) - sum_x pi_old(x) * Q_hat(x)
        L_policy += row_weight * [-sum_a stopgrad(A_hat(a)) * logit(a)]

    L_total = L_policy + value and enabled auxiliary losses
    backpropagate through the shared model
    let Adam propose an update

    if weighted mean and p99 KL(pi_old || pi_new) pass:
        accept
    else:
        restore model and optimizer; retry with a smaller core step
```

## 16. A useful mental model

- **Reward** says what winning means.
- **$Q(h,a)$** asks what reward follows one particular choice.
- **$V(h)$** is the policy's average $Q$ before choosing.
- **Advantage** is the difference $Q-V$.
- **Counterfactual branches** obtain more $Q$ samples from one deal.
- **Inclusion correction** prevents selective action inspection from biasing
  those samples.
- **Reach correction** prevents cloned branches from pretending to be more
  common than they are.
- **NeuRD** converts advantage into direct up/down pressure on action logits.
- **KL backtracking** prevents a noisy neural proposal from moving too far.
- **Evaluation** determines whether those locally coherent updates actually
  produce the desired game behavior at finite scale.

## 17. Where the implementation lives

- `plump/rewards.py` defines terminal relative reward.
- `plump/seq/rollout.py` samples actions, builds counterfactual trees, records
  inclusion probabilities, propagates reach, and backs up terminal values.
- `plump/seq/trainer.py` constructs $\widehat Q$ and $\widehat A$, evaluates
  the NeuRD/value/auxiliary losses, and performs KL backtracking.
- `configs/train.toml` records the active local algorithm settings and loss
  coefficients.
- `tests/test_seq_rollout.py` and `tests/test_seq_trainer.py` pin the estimator
  and optimizer invariants described above.

## 18. The branch-free PPO option

Set `training.policy_objective="ppo"` to collect one ordinary sampled game per
deal. There are no cloned states and no candidate inclusion correction. At a
learned decision, collection stores the sampled action, the complete legally
masked old distribution, and ultimately that seat's terminal relative reward.

The critic has separate weights. The default oracle critic receives one
canonical sequence per environmental game:

$$
[\mathrm{GAME}],
[\mathrm{HAND}(0,c)]_{c\in H_0},\ldots,
[\mathrm{HAND}(P-1,c)]_{c\in H_{P-1}},
[\text{public events}].
$$

Every card remains a separate token with its absolute owner, exact-card, rank,
and suit fields. The deal is not reduced to one pooled vector. At every causal
state the critic emits

$$
V_\phi(s_t)=(V_\phi(s_t,0),\ldots,V_\phi(s_t,P-1)).
$$

Input player id $i$ and output column $i$ always refer to the same absolute
seat. For an action sampled by seat $i$, let the frozen pre-update prediction
be $V_{\phi_{old}}(s_t,i)$. With no discounting or bootstrapping, the advantage
sample is

$$
A_t = R_i - V_{\phi_{old}}(s_t,i).
$$

The critic changes variance, not the expected policy gradient. Although it sees
hidden cards unavailable to the actor, it is a valid baseline because those
cards are part of the pre-action environment state and the critic does not see
which current action will be sampled. Conditional on that state,
$\mathbb E_{a\sim\pi}[\nabla\log\pi(a) V_\phi(s,i)]=0$. At every learned
pre-action state, all active output columns are trained by Monte Carlo MSE
against $(R_0,\ldots,R_{P-1})$; the player axis is averaged so it does not
reweight games by player count.

For the actor, define

$$
r_t(\theta)=
\exp\left[
\log\pi_\theta(a_t\mid h_t)-
\log\pi_{old}(a_t\mid h_t)
\right].
$$

The minimized PPO loss is

$$
-\min\left(
r_t A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right).
$$

At the start of an exact update, $r_t=1$. A positive advantage therefore
increases the sampled action's log probability and a negative advantage lowers
it. Unlike the counterfactual estimator, there is no explicit $1/q(a)$ factor:
an action contributes when the live policy samples it, which is the ordinary
on-policy score-function estimator.

Sampling must be from the same masked $\pi_{old}$ whose log probability is
stored. An external epsilon-random behavior policy would make this off-policy
unless the mixture itself were defined and optimized as the policy. The PPO
implementation therefore has no epsilon explorer.

### Game and seat weighting

If game $g$ has learned-seat set $C_g$, the implemented objective is

$$
L=\frac{1}{G}\sum_g\frac{1}{|C_g|}
\sum_{i\in C_g}\sum_t L_{g,i,t}.
$$

Self-play can learn from every seat without counting one deal as $P$
independent games. The sum is deliberately not divided by decision count, so a
longer hand contributes more policy decisions.

### Preventing entropy collapse

For a non-forced decision with legal set $\mathcal A_t$, use normalized entropy

$$
\bar H_t = \frac{H(\pi_t)}{\log|\mathcal A_t|}.
$$

Separate positive temperatures for bids and plays add $\alpha\bar H$ to the
maximized objective. In adaptive mode, each temperature increases below its
configured target and decreases above it. Forced actions are omitted from this
constraint. PPO clipping is supplemented by exact full-distribution masked KL
mean and p99 acceptance guards.

### Multiple actor weights and precision

One actor can be shared by every learned seat, or several independent actors
can be assigned round-robin. Their parameters and optimizer moments are
distinct. Actor zero remains the evaluated/deployed policy and league snapshot
source.

On MPS, BF16 autocast lowers eligible transformer operations while parameters
and Adam state stay FP32. Cached K/V is independently FP16. Attention softmax,
masked policy log-softmax, likelihood ratios, KL, entropy, returns, advantages,
and loss accumulation are explicitly FP32.

The PPO implementation is in `plump/seq/ppo.py`; the default critic is
`SeqPPOOracleCritic` in `plump/seq/model.py`; dispatch, optimization, entropy
duals, and checkpoint migration are in `plump/seq/trainer.py`. Its focused
invariants are in `tests/test_seq_ppo.py`.
