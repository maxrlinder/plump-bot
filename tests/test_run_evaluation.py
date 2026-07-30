"""Low-level sidecar evaluation discovery and cache matching."""

from __future__ import annotations

from plump.run_evaluation import (
    EVALUATION_FORMAT_VERSION,
    EvaluationProtocol,
    checkpoint_iteration,
    discover_interval_checkpoints,
    evaluation_output,
    result_matches_protocol,
)
from plump.runs import RunDirectory, atomic_write_json


def _protocol() -> EvaluationProtocol:
    return EvaluationProtocol(
        opponent="heuristic",
        player_counts=(3, 4, 5),
        hand_sizes=(3, 4),
        deals_per_configuration=6,
        deal_seed=1234,
        action_seed=17,
        bootstrap_samples=2000,
        batch_size=64,
    )


def test_discovers_only_complete_interval_checkpoints(tmp_path):
    run = RunDirectory("sample", root=tmp_path)
    run.checkpoints.mkdir(parents=True)
    for name in (
        "iter_000100.pt",
        "iter_000000.pt",
        "best.pt",
        ".iter_000050.pt.partial",
    ):
        (run.checkpoints / name).touch()

    checkpoints = discover_interval_checkpoints(run)

    assert [path.name for path in checkpoints] == [
        "iter_000000.pt",
        "iter_000100.pt",
    ]
    assert checkpoint_iteration(checkpoints[0]) == 0
    assert checkpoint_iteration(checkpoints[1]) == 100


def test_cached_result_requires_identical_protocol(tmp_path):
    output = tmp_path / "heuristic.json"
    protocol = _protocol()
    atomic_write_json(
        output,
        {
            "format_version": EVALUATION_FORMAT_VERSION,
            "protocol": protocol.as_json(),
        },
    )

    assert result_matches_protocol(output, protocol)
    assert not result_matches_protocol(
        output,
        EvaluationProtocol(
            **{
                **protocol.__dict__,
                "deals_per_configuration": 12,
            }
        ),
    )


def test_sampled_evaluation_has_a_separate_sidecar(tmp_path):
    run = RunDirectory("sample", root=tmp_path)

    assert evaluation_output(run, 50, "heuristic").name == "heuristic.json"
    assert (
        evaluation_output(run, 50, "heuristic", greedy=False).name
        == "heuristic_sample.json"
    )
