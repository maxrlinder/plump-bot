"""All-resident randomized checkpoint tournament: pooled cross-matchup batching.

Same statistical philosophy as checkpoint_tournament.py — replay everything,
field frozen at launch, fresh time-based seed, every edge played from both
seats on the same deals, weighted margin fit with a global focal-bias
intercept — but executed by the arena engine (plump/evaluation_workers.py).
The bootstrap tree, adjacent chain, and each random-matching layer use fresh
deal banks and action seeds, so a pair drawn again in a later stage contributes
new evidence instead of replaying an identical result. In the arena engine
(plump/evaluation_workers.py), every selected checkpoint stays resident in
memory, CPU workers step and encode thousands of concurrent rounds, and the
parent batches forwards per model across all concurrent matchups so the GPU
stays saturated instead of idling between one-matchup-at-a-time waves.

The schedule is sparse, sized from measured noise rather than all pairs (see
build_arena_chunks): bisection tree for early full-range identification, the
adjacent chain for local shape, then random perfect-matching layers for graph
conductance. Defaults (30 checkpoints, 8 matchings, 800-round matchups) give
per-node rating SE ~0.04 pts/round — about 2% of the field's spread — in
roughly 45-60 minutes on an M-series MPS machine.

The field is every checkpoint from --min-iteration onward, uniformly
subsampled to --max-checkpoints (endpoints always included).

    .venv/bin/python examples/checkpoint_arena.py            # full fresh run

Outputs (rewritten continuously, same paths as the tournament): tournament.csv,
the plot elo_ladder.png, and the raw stream arena_results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_elo_ladder import (
    _spread_order,
    checkpoint_iterations,
    pull_checkpoint,
    volume_iterations,
)
from checkpoint_tournament import _bisection_tree, write_outputs

from plump.evaluation import DealBank
from plump.evaluation_workers import (
    ArenaMatchup,
    run_matchups_pooled,
    subsample_uniform,
)
from plump.policies import ModelPolicy

REPO_DIR = Path(__file__).resolve().parents[1]


def build_arena_chunks(
    keys: list[str],
    *,
    matchings: int,
    tree_chunk: int,
    seed: int,
    rng: random.Random,
) -> list[list[ArenaMatchup]]:
    """Bootstrap-tree chunks (probes interleaved), then chain + matching layers.

    Sparse design sized from measured noise instead of all pairs. Every edge
    is played from both seats on the same deals, so the focal-seat bias and
    deal luck cancel in the pair difference; the margin fit then only needs
    enough graph conductance per node. Measured on this run's bank, a
    combined both-direction pair estimate has sigma ~0.107 pts/round at 1200
    rounds/direction, and noise is nearly independent of the strength gap —
    so random long-range pairings are as informative as local ones, and
    random perfect matchings (uniform degree, expander-like) buy the most
    rating precision per matchup. Rating SE per node ~ sigma/sqrt(degree).

    Each stage/layer uses a fresh deal bank and action-sampling seed, so
    repeated opponent pairs remain statistically useful. Layers, in schedule
    order:
    - bisection tree: connects the whole field coarse-to-fine so ratings are
      identified for every played checkpoint from the first publishes;
    - adjacent chain: pins the local shape of the ladder between neighbours;
    - ``matchings`` random perfect matchings: each adds +1 to every node's
      degree with fresh uniformly random opponents.
    """

    counter = 0

    def matchup(
        a: str,
        b: str,
        *,
        bank_ref: str,
        rng_seed: int,
    ) -> ArenaMatchup:
        nonlocal counter
        counter += 1
        return ArenaMatchup(counter, "matchup", a, b, bank_ref, rng_seed=rng_seed)

    def probe(a: str) -> ArenaMatchup:
        nonlocal counter
        counter += 1
        return ArenaMatchup(counter, "probe", a, a, "probe", rng_seed=seed + 1)

    tree = _bisection_tree(len(keys))
    probe_order = [keys[index] for index in _spread_order(len(keys))]
    tree_items: list[ArenaMatchup] = []
    for position, (a, b) in enumerate(tree):
        tree_items.append(
            matchup(
                keys[a],
                keys[b],
                bank_ref="match_tree",
                rng_seed=seed + 10,
            )
        )
        tree_items.append(
            matchup(
                keys[b],
                keys[a],
                bank_ref="match_tree",
                rng_seed=seed + 10,
            )
        )
        if position < len(probe_order):
            tree_items.append(probe(probe_order[position]))
    tree_items.extend(probe(key) for key in probe_order[len(tree):])
    chunks = [
        tree_items[start : start + tree_chunk]
        for start in range(0, len(tree_items), tree_chunk)
    ]

    chain_items: list[ArenaMatchup] = []
    for a, b in zip(keys, keys[1:]):
        chain_items.append(
            matchup(
                a,
                b,
                bank_ref="match_chain",
                rng_seed=seed + 11,
            )
        )
        chain_items.append(
            matchup(
                b,
                a,
                bank_ref="match_chain",
                rng_seed=seed + 11,
            )
        )
    chunks.extend(
        chain_items[start : start + 2 * tree_chunk]
        for start in range(0, len(chain_items), 2 * tree_chunk)
    )

    for layer in range(matchings):
        shuffled = rng.sample(keys, len(keys))
        pairs = list(zip(shuffled[::2], shuffled[1::2]))
        if len(shuffled) % 2:
            leftover = shuffled[-1]
            pairs.append((leftover, rng.choice([k for k in keys if k != leftover])))
        layer_items: list[ArenaMatchup] = []
        bank_ref = f"match_layer_{layer:02d}"
        for a, b in pairs:
            layer_items.append(
                matchup(
                    a,
                    b,
                    bank_ref=bank_ref,
                    rng_seed=seed + 100 + layer,
                )
            )
            layer_items.append(
                matchup(
                    b,
                    a,
                    bank_ref=bank_ref,
                    rng_seed=seed + 100 + layer,
                )
            )
        chunks.append(layer_items)
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh randomized tournament with all checkpoints resident in memory.",
    )
    parser.add_argument("--run-name", default="v9_8m_wideppo_seed1")
    parser.add_argument(
        "--min-iteration",
        type=int,
        default=1000,
        help="Ignore checkpoints trained for fewer iterations than this.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=30,
        help=(
            "Field size. If more checkpoints exist, a uniformly spaced subset "
            "(endpoints included) is used. 30 models is ~0.5 GB of fp16 "
            "weights; the sparse matching schedule keeps per-node rating SE "
            "near 0.04 pts/round in under an hour."
        ),
    )
    parser.add_argument(
        "--matchings",
        type=int,
        default=8,
        help=(
            "Random perfect-matching layers after the tree+chain bootstrap. "
            "Each layer gives every checkpoint one fresh random opponent "
            "(both directions); rating SE per node ~ sigma/sqrt(degree)."
        ),
    )
    parser.add_argument("--deals-per-configuration", type=int, default=2)
    parser.add_argument("--probe-deals", type=int, default=2)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Tournament seed. Defaults to the current time: every rerun is fresh.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--tree-chunk",
        type=int,
        default=16,
        help="Directed matchups per chunk in the bootstrap-tree phase.",
    )
    parser.add_argument(
        "--max-active-per-worker",
        type=int,
        default=1024,
        help=(
            "Concurrent rounds per worker. Bounds wave size: bigger waves batch "
            "better only until MPS activation memory pressure sets in."
        ),
    )
    parser.add_argument(
        "--forward-batch",
        type=int,
        default=2048,
        help="Rows per forward (per-row cost is flat; this bounds activations).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default="fp16",
        help=(
            "Weight precision: fp16/bf16 CONVERT the resident weights "
            "(measured ~2.1x forward throughput on MPS; legal-action logits "
            "agree with fp32 to ~0.03). Autocast is never used."
        ),
    )
    parser.add_argument(
        "--publish-every",
        type=int,
        default=16,
        help="Rewrite tournament.csv and the plot after this many completed items.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip the Modal volume listing/download; use local checkpoints only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is None:
        args.seed = int(time.time())
    if args.matchings < 0:
        raise SystemExit("--matchings must be at least 0")
    archive_dir = REPO_DIR / "checkpoints" / args.run_name
    out_dir = REPO_DIR / "checkpoints" / "ladder" / args.run_name
    ckpt_dir = out_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    local_paths = checkpoint_iterations(archive_dir)
    pulled_paths = checkpoint_iterations(ckpt_dir)
    volume_iters = set() if args.no_sync else set(volume_iterations(args.run_name))

    available = sorted(
        iteration
        for iteration in set(local_paths) | set(pulled_paths) | volume_iters
        if iteration >= args.min_iteration
    )
    if len(available) < 2:
        raise SystemExit("Need at least two checkpoints above --min-iteration.")
    ladder = subsample_uniform(available, args.max_checkpoints)
    keys = [f"iter_{iteration:05d}" for iteration in ladder]
    print(
        f"arena: {len(ladder)}/{len(available)} checkpoints, iters "
        f"{ladder[0]}-{ladder[-1]}, seed {args.seed}, "
        f"tree+chain+{args.matchings} matchings",
        flush=True,
    )

    policies: dict[str, ModelPolicy] = {}
    load_started = time.monotonic()
    for position, iteration in enumerate(ladder, start=1):
        key = f"iter_{iteration:05d}"
        if iteration in local_paths:
            path = local_paths[iteration]
        elif iteration in pulled_paths:
            path = pulled_paths[iteration]
        else:
            path = ckpt_dir / f"plump_v4_iter_{iteration:05d}.pt"
            if iteration not in volume_iters:
                raise SystemExit(f"iter {iteration} absent from the volume")
            pull_checkpoint(args.run_name, iteration, path)
        policy = ModelPolicy.from_checkpoint(
            path,
            device=args.device,
            greedy=False,
            event_length_buckets=(8, 16, 32, 64),
            batch_packing="numpy",
            lean_action_forward=True,
            name=key,
        )
        if args.precision == "fp16":
            policy.model.half()
        elif args.precision == "bf16":
            policy.model.bfloat16()
        policies[key] = policy
        if position % 25 == 0 or position == len(ladder):
            print(f"  loaded {position}/{len(ladder)} models", flush=True)
    resident_mb = sum(
        parameter.numel() * parameter.element_size()
        for policy in policies.values()
        for parameter in policy.model.parameters()
    ) / 1e6
    device = next(iter(policies.values())).device
    print(
        f"resident: {len(policies)} models, {resident_mb:,.0f} MB weights on "
        f"{device} ({time.monotonic() - load_started:.0f}s to load)",
        flush=True,
    )

    rng = random.Random(args.seed)
    chunks = build_arena_chunks(
        keys,
        matchings=args.matchings,
        tree_chunk=args.tree_chunk,
        seed=args.seed,
        rng=rng,
    )
    match_bank_refs = sorted(
        {
            item.bank_ref
            for chunk in chunks
            for item in chunk
            if item.kind == "matchup"
        }
    )

    def match_bank_seed(bank_ref: str) -> int:
        if bank_ref == "match_tree":
            return args.seed + 10
        if bank_ref == "match_chain":
            return args.seed + 11
        return args.seed + 100 + int(bank_ref.rsplit("_", 1)[1])

    banks = {
        bank_ref: DealBank.generate(
            player_counts=(3, 4, 5),
            hand_sizes=tuple(range(3, 11)),
            deals_per_configuration=args.deals_per_configuration,
            seed=match_bank_seed(bank_ref),
        )
        for bank_ref in match_bank_refs
    }
    banks["probe"] = DealBank.generate(
        player_counts=(3, 4, 5),
        hand_sizes=tuple(range(3, 11)),
        deals_per_configuration=args.probe_deals,
        seed=args.seed + 1,
    )
    total_items = sum(len(chunk) for chunk in chunks)
    total_matchups = sum(
        1 for chunk in chunks for item in chunk if item.kind == "matchup"
    )
    total_probes = total_items - total_matchups
    rounds_per_matchup = sum(
        deal.spec.num_players**2 for deal in banks[match_bank_refs[0]].deals
    )
    print(
        f"schedule: {total_matchups} directed matchups x {rounds_per_matchup} "
        f"rounds = {total_matchups * rounds_per_matchup:,} rounds, plus "
        f"{total_probes} behavior probes, {len(chunks)} chunks "
        f"(bootstrap tree, adjacent chain, then {args.matchings} random "
        f"matchings)",
        flush=True,
    )

    header = (
        f"Arena tournament, seed {args.seed}: {len(keys)} checkpoints, "
        f"tree+chain+{args.matchings} random matchings"
    )
    results_path = out_dir / "arena_results.jsonl"
    matchup_rows: list[dict] = []
    behavior_rows: list[dict] = []
    probes_done = 0
    matchups_done = 0
    stats: dict = {}
    run_started = time.monotonic()
    with results_path.open("w") as stream:
        for position, row in enumerate(
            run_matchups_pooled(
                chunks,
                banks,
                policies,
                num_workers=args.num_workers,
                forward_batch=args.forward_batch,
                max_active_per_worker=args.max_active_per_worker,
                stats=stats,
            ),
            start=1,
        ):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if row["kind"] == "behavior":
                behavior_rows.append(row)
                probes_done += 1
                progress = (
                    f"[probe {probes_done}/{total_probes}] {row['key']}: "
                    f"avg_bid {row['avg_bid']:.2f} "
                    f"hit {100 * row['bid_hit_rate']:.1f}% "
                    f"zero {100 * row['zero_bid_rate']:.1f}% "
                    f"all {100 * row['all_bid_rate']:.1f}%"
                )
            else:
                matchup_rows.append(row)
                matchups_done += 1
                progress = (
                    f"[{matchups_done}/{total_matchups}] "
                    f"{row['focal']} vs {row['table']}: "
                    f"reward {row['macro_relative_reward']:+.3f}"
                )
            if position % args.publish_every == 0 or position == total_items:
                write_outputs(matchup_rows, behavior_rows, out_dir, header)
                elapsed = time.monotonic() - run_started
                print(
                    f"  published: {stats['rounds']:,} rounds, "
                    f"{stats['decisions']:,} decisions, "
                    f"{stats['decisions'] / max(elapsed, 1e-9):,.0f} decisions/s "
                    f"(waves {stats['waves']}, gather {stats['gather_sec']:.0f}s, "
                    f"forward {stats['forward_sec']:.0f}s, "
                    f"send {stats['send_sec']:.0f}s)",
                    flush=True,
                )
            print(progress, flush=True)
    print(f"arena complete: {out_dir / 'elo_ladder.png'}")


if __name__ == "__main__":
    main()
