"""Run-directory ownership, metadata, and checkpoint resolution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import tomllib
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from plump.run_config import PROJECT_ROOT, dump_toml


RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RunDirectory:
    def __init__(self, name: str, *, root: Path | None = None) -> None:
        if not RUN_NAME.fullmatch(name):
            raise ValueError(
                "Run names may contain letters, digits, '.', '_', and '-'."
            )
        self.name = name
        configured_root = os.environ.get("PLUMP_RUNS_DIR")
        default_root = (
            Path(configured_root).expanduser()
            if configured_root
            else PROJECT_ROOT / "runs"
        )
        self.root = (root or default_root).resolve()
        self.path = self.root / name
        self.checkpoints = self.path / "checkpoints"
        self.evaluations = self.path / "evaluations"
        self.analysis = self.path / "analysis"
        self.metrics = self.path / "metrics.csv"
        self.dashboard = self.path / "dashboard.png"
        self.evaluation_dashboard = self.path / "dashboard-eval.png"
        self.config = self.path / "config.toml"
        self.metadata = self.path / "metadata.json"
        self.train_log = self.path / "train.log"
        self.lock = self.path / "run.lock"

    @property
    def exists(self) -> bool:
        return self.path.is_dir()

    def create(self, raw_config: dict[str, Any], command: list[str]) -> None:
        self.checkpoints.mkdir(parents=True, exist_ok=False)
        self.evaluations.mkdir()
        self.analysis.mkdir()
        atomic_write_text(self.config, dump_toml(raw_config))
        atomic_write_json(
            self.metadata,
            {
                "run": self.name,
                "created_at": _timestamp(),
                "command": command,
                "git": _git_metadata(),
                "python": sys.version,
                "torch": torch.__version__,
                "status": "created",
            },
        )

    def recorded_config(self) -> dict[str, Any]:
        with self.config.open("rb") as handle:
            return tomllib.load(handle)

    def update_metadata(self, **updates: Any) -> None:
        data: dict[str, Any] = {}
        if self.metadata.exists():
            data = json.loads(self.metadata.read_text())
        data.update(updates)
        data["updated_at"] = _timestamp()
        atomic_write_json(self.metadata, data)

    def interval_checkpoint(self, iteration: int) -> Path:
        return self.checkpoints / f"iter_{iteration:06d}.pt"

    def record_latest(self, checkpoint: Path, iteration: int) -> None:
        atomic_write_json(
            self.checkpoints / "latest.json",
            checkpoint_manifest(checkpoint, iteration),
        )

    def record_best(
        self,
        checkpoint: Path,
        iteration: int,
        metric: float,
    ) -> None:
        manifest = checkpoint_manifest(checkpoint, iteration)
        manifest["metric"] = metric
        atomic_write_json(self.checkpoints / "best.json", manifest)

    def best_metric(self) -> float | None:
        path = self.checkpoints / "best.json"
        if not path.exists():
            return None
        return float(json.loads(path.read_text())["metric"])

    def resolve_checkpoint(self, selector: str | int) -> Path:
        if isinstance(selector, int) or str(selector).isdigit():
            path = self.interval_checkpoint(int(selector))
        elif selector == "latest":
            path = self._resolve_manifest(self.checkpoints / "latest.json")
        elif selector == "best":
            path = self._resolve_manifest(self.checkpoints / "best.json")
        else:
            candidate = Path(str(selector)).expanduser()
            path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()

    def _resolve_manifest(self, manifest_path: Path) -> Path:
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        data = json.loads(manifest_path.read_text())
        path = self.checkpoints / data["path"]
        if path.stat().st_size != int(data["size"]):
            raise RuntimeError(f"Checkpoint size mismatch: {path}")
        if file_sha256(path) != data["sha256"]:
            raise RuntimeError(f"Checkpoint checksum mismatch: {path}")
        return path

    @contextmanager
    def acquire_lock(self) -> Iterator[None]:
        self.path.mkdir(parents=True, exist_ok=True)
        if self.lock.exists():
            try:
                pid = int(self.lock.read_text().strip())
            except ValueError:
                pid = -1
            if pid > 0 and _process_exists(pid):
                raise RuntimeError(
                    f"Run {self.name!r} is already owned by process {pid}."
                )
            self.lock.unlink()
        descriptor = os.open(
            self.lock,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode())
            os.close(descriptor)
            yield
        finally:
            try:
                self.lock.unlink()
            except FileNotFoundError:
                pass


def checkpoint_manifest(checkpoint: Path, iteration: int) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "path": checkpoint.name,
        "size": checkpoint.stat().st_size,
        "sha256": file_sha256(checkpoint),
        "written_at": _timestamp(),
    }


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}
