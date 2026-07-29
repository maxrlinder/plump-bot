"""Checkpoint-scoped evaluation artifacts for a live or completed run."""

from __future__ import annotations

import dataclasses
import gc
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from plump.evaluation import DealBank, EvaluationReport, evaluate_policy
from plump.policies import HeuristicPolicy
from plump.runs import RunDirectory, atomic_write_json, file_sha256
from plump.seq.policy import SeqModelPolicy


EVALUATION_FORMAT_VERSION = 1
_INTERVAL_CHECKPOINT = re.compile(r"iter_(\d+)\.pt")


@dataclass(frozen=True)
class EvaluationProtocol:
    """Inputs that make checkpoint scores directly comparable."""

    opponent: str
    player_counts: tuple[int, ...]
    hand_sizes: tuple[int, ...]
    deals_per_configuration: int
    deal_seed: int
    action_seed: int
    bootstrap_samples: int
    batch_size: int
    greedy: bool = True

    def as_json(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["player_counts"] = list(self.player_counts)
        value["hand_sizes"] = list(self.hand_sizes)
        return value


def discover_interval_checkpoints(run: RunDirectory) -> list[Path]:
    """Return complete interval checkpoints in iteration order."""

    checkpoints: list[tuple[int, Path]] = []
    for path in run.checkpoints.glob("iter_*.pt"):
        match = _INTERVAL_CHECKPOINT.fullmatch(path.name)
        if match is not None and path.is_file():
            checkpoints.append((int(match.group(1)), path.resolve()))
    return [path for _, path in sorted(checkpoints)]


def checkpoint_iteration(checkpoint: str | Path) -> int:
    path = Path(checkpoint)
    match = _INTERVAL_CHECKPOINT.fullmatch(path.name)
    if match is not None:
        return int(match.group(1))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("iteration", 0))


def evaluation_output(
    run: RunDirectory,
    iteration: int,
    opponent: str,
) -> Path:
    return run.evaluations / f"iter_{iteration:06d}" / f"{opponent}.json"


def load_evaluation(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def result_matches_protocol(
    path: str | Path,
    protocol: EvaluationProtocol,
) -> bool:
    try:
        value = load_evaluation(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return (
        value.get("format_version") == EVALUATION_FORMAT_VERSION
        and value.get("protocol") == protocol.as_json()
    )


def evaluate_checkpoint(
    run: RunDirectory,
    checkpoint: str | Path,
    *,
    protocol: EvaluationProtocol,
    deal_bank: DealBank,
    device: str | torch.device,
    force: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Evaluate one checkpoint, atomically storing a reproducible report."""

    checkpoint = Path(checkpoint).resolve()
    iteration = checkpoint_iteration(checkpoint)
    output = evaluation_output(run, iteration, protocol.opponent)
    if not force and result_matches_protocol(output, protocol):
        return load_evaluation(output), False
    if protocol.opponent != "heuristic":
        raise ValueError(f"Unsupported evaluation opponent: {protocol.opponent}")

    started = time.perf_counter()
    policy: SeqModelPolicy | None = None
    try:
        policy = SeqModelPolicy.from_checkpoint(
            checkpoint,
            device=device,
            greedy=protocol.greedy,
            name=checkpoint.stem,
        )
        report = evaluate_policy(
            policy,
            HeuristicPolicy(),
            deal_bank,
            bootstrap_samples=protocol.bootstrap_samples,
            seed=protocol.action_seed,
            batch_size=protocol.batch_size,
        )
        payload = _evaluation_payload(
            run,
            checkpoint,
            iteration,
            protocol,
            report,
            elapsed_sec=time.perf_counter() - started,
        )
        atomic_write_json(output, payload)
        return payload, True
    finally:
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _evaluation_payload(
    run: RunDirectory,
    checkpoint: Path,
    iteration: int,
    protocol: EvaluationProtocol,
    report: EvaluationReport,
    *,
    elapsed_sec: float,
) -> dict[str, Any]:
    return {
        "format_version": EVALUATION_FORMAT_VERSION,
        "run": run.name,
        "iteration": iteration,
        "checkpoint": {
            "path": checkpoint.name,
            "size": checkpoint.stat().st_size,
            "sha256": file_sha256(checkpoint),
        },
        "opponent": protocol.opponent,
        "protocol": protocol.as_json(),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": elapsed_sec,
        "report": dataclasses.asdict(report),
    }
