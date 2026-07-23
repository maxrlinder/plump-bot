#!/bin/zsh
set -euo pipefail

repo_dir="${0:A:h:h}"
run_dir="${PLUMP_LAPTOP_RUN_DIR:-$repo_dir/checkpoints/local/v9_8m_laptop_seed1}"
league_archive="${PLUMP_LAPTOP_LEAGUE_ARCHIVE:-$repo_dir/checkpoints/ladder/v9_8m_wideppo_seed1/ckpts}"
iterations="${PLUMP_LAPTOP_ITERATIONS:-18000}"

cd "$repo_dir"
mkdir -p "$run_dir"

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.95
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.88
export PYTORCH_MPS_FAST_MATH=1
unset PYTORCH_MPS_PREFER_METAL

args=(
  --iterations "$iterations"
  --seed 1
  --training-mode round
  --rounds-per-configuration 16
  --num-envs 768
  --event-length-buckets 8,16,32,64
  --batch-packing numpy
  --lean-rollout-forward
  --ppo-epochs 4
  --target-kl 0.015
  --minibatch-size 2880
  --microbatch-size 1152
  --lr 2e-4
  --entropy-coef 0.0
  --branch-rollouts
  --branch-decision-budget-per-arm 30000
  --branch-update-decisions-per-arm 2400
  --branch-max-active 1536
  --branch-bid-max-actions 4
  --branch-policy-coef 1.0
  --branch-policy-objective neurd
  --branch-neurd-regret-coef 0.25
  --branch-neurd-kl-coef 1.0
  --branch-kl-cap 0.005
  --branch-tree-decision-budget-per-arm 10000
  --branch-tree-update-decisions-per-arm 800
  --trick-baseline
  --belief-head-only
  --oracle-critic
  --oracle-value-coef 0.5
  --suit-presence-head
  --suit-coef 0.1
  --owner-coef 0.0
  --owner-warmup-iters 200
  --owner-capacity-coef 0.1
  --owner-sinkhorn-iterations 16
  --player-count-weights 2,3,4
  --hand-size-weights 1,2,3,4,5,6,7,8
  --self-play-fraction 0.5
  --heuristic-fraction 0.0
  --mixed-fraction 0.0
  --historical-fraction 0.5
  --historical-max-snapshots 8
  --league-meta-solver uniform
  --league-uniform-min-iteration 4000
  --league-archive-dir "$league_archive"
  --league-resample-every 30
  --no-historical-current-snapshots
  --league-temperature 2.0
  --league-reward-decay 0.95
  --batched-league-sampling
  --league-probe-fraction 0.10
  --league-eval-every 50
  --league-eval-deals-per-configuration 4
  --no-counterfactual-search
  --diag-every 10
  --diag-samples 2048
  --diag-batch-size 512
  --eval-every 0
  --save-every-minutes 20
  --plot-every 0
  --checkpoint-dir "$run_dir"
  --log-dir "$run_dir"
  --precision bf16
  --max-seq-len 100
  --d-model 320
  --n-layers 6
  --n-heads 10
  --d-ff 896
  --context-hidden-dim 256
)
if [[ "${PLUMP_LAPTOP_PIPELINE_ROLLOUTS:-0}" == "1" ]]; then
  args+=(--pipeline-rollouts)
fi

checkpoints=("$run_dir"/plump_v4_iter_*.pt(N))
if (( ${#checkpoints} == 0 )); then
  print -u2 -r -- "No resume checkpoint found in $run_dir"
  exit 1
fi
newest="${checkpoints[-1]}"
args+=(--resume-from "$newest" --resume-optimizer)

print -r -- "$$" > "$run_dir/train.pid"
print -r -- "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] laptop launch resume=$newest iterations=$iterations" \
  >> "$run_dir/supervisor.log"

# Keep the Mac awake while preserving this shell PID across exec so train.pid
# remains the actual Python training process.
caffeinate -ims -w $$ >/dev/null 2>&1 &

exec "$repo_dir/.venv/bin/python" -u examples/train_ppo.py "${args[@]}" \
  >> "$run_dir/train.log" 2>> "$run_dir/train.err.log"
