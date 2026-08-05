# Modal L40S Training

The Modal path is a sidecar around the ordinary run/checkpoint system. The
trainer, objective, metrics, checkpoint validation, and curriculum are shared
with local execution; `infra/modal_training.py` supplies only the image, CUDA
resources, persistent Volume, retries, and detached entrypoint.

This guide documents the checked-in `configs/modal-l40s.toml` NeuRD profile.
It is separate from the local `configs/ppo-mps.toml` oracle-PPO profile; the
PPO length bucketing and microbatch measurements in the local training guide
do not imply tuned CUDA values. The Modal profile uses:

- CUDA is explicit and float32 matrix multiplies use PyTorch's `high` mode;
- each update has 192 games: 96 self-play plus 96 anchor games;
- each four-game wave has two self and two anchor games through ten cards;
- the fp16 KV ceiling is 262,144 rows, or 32.61 GB for this model;
- update microbatches contain up to 131,072 positions;
- the two optimizer-group rates are 5e-5 (2x the laptop run) and the accepted
  policy-KL p99 ceiling is 0.20 (4x its previous value);
- evaluation runs every 25 updates and checkpoints every 12, approximately
  preserving their prior cadence in games after the 4x update-size increase;
- the container requests one L40S, eight physical CPU cores, and 64 GiB RAM.

Rollout KV pools are released before the update, so the 32.61 GB cache ceiling
and the larger backward activation batch are separate peaks. Neural outputs
still cross to the CPU once per game wave because rules and branching are CPU
state; host inputs cross to CUDA once per batched forward. No tensor relies on
Apple unified memory.

## Prepare and upload

Create a self-contained run without starting it locally. Resetting the league
avoids copying historical checkpoint files from the parent; by the earliest
possible four-evaluation curriculum switch, the remote run will already have
eight recent interval checkpoints of its own.

```bash
uv run plump train stratified-modal-8m \
  --config configs/modal-l40s.toml \
  --from-checkpoint runs/stratified-8m/checkpoints/iter_001350.pt \
  --reset-league --prepare-only
```

Create the persistent Volume and upload the prepared directory:

```bash
MODAL_PROFILE=max-r-linder uvx modal volume create plump-training-runs
MODAL_PROFILE=max-r-linder uvx modal volume put -f \
  plump-training-runs runs/stratified-modal-8m /
```

## Smoke test and detached launch

The smoke function copies the prepared run to container-local storage and
performs real updates there. It cannot advance or alter the persistent run.

```bash
MODAL_PROFILE=max-r-linder uvx modal run \
  infra/modal_training.py::smoke --run-name stratified-modal-8m --updates 2
```

After the smoke test succeeds, launch the persistent job:

```bash
MODAL_PROFILE=max-r-linder uvx modal run --detach \
  infra/modal_training.py --run-name stratified-modal-8m
```

`--detach` lets the Function continue if the terminal disconnects or the
laptop closes. Modal background-commits the mounted Volume every few seconds;
the wrapper also commits when an attempt exits. The Function has a 24-hour
attempt timeout and ten retries, so each retry discards post-checkpoint metric
rows and resumes the checksum-verified `latest.json` checkpoint.

To inspect or retrieve artifacts later:

```bash
MODAL_PROFILE=max-r-linder uvx modal volume ls \
  plump-training-runs stratified-modal-8m/checkpoints
MODAL_PROFILE=max-r-linder uvx modal volume get \
  plump-training-runs stratified-modal-8m runs/stratified-modal-8m-download
```
