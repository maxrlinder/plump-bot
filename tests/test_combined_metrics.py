from scripts.render_combined_training_metrics import (
    active_resume_branch,
    merge_metric_rows,
)


def _row(iteration: int, timestamp: str, source: str) -> dict[str, str]:
    return {
        "iteration": str(iteration),
        "timestamp_utc": timestamp,
        "source": source,
    }


def test_combined_metrics_switches_to_latest_local_branch() -> None:
    modal = [
        _row(10, "2026-01-01T00:00:00Z", "modal"),
        _row(11, "2026-01-01T00:01:00Z", "modal"),
        _row(12, "2026-01-01T00:02:00Z", "modal"),
    ]
    local = [
        _row(11, "2026-01-01T01:00:00Z", "old-local"),
        _row(12, "2026-01-01T01:01:00Z", "old-local"),
        _row(12, "2026-01-01T02:01:00Z", "current-local"),
        _row(13, "2026-01-01T02:02:00Z", "current-local"),
    ]

    combined = merge_metric_rows(modal, local)

    assert [
        (int(row["iteration"]), row["source"])
        for row in combined
    ] == [
        (10, "modal"),
        (11, "old-local"),
        (12, "current-local"),
        (13, "current-local"),
    ]


def test_combined_metrics_discards_abandoned_future_after_rollback() -> None:
    local = [
        _row(14_484, "2026-01-01T00:00:00Z", "shared"),
        _row(14_485, "2026-01-01T00:01:00Z", "shared"),
        _row(16_309, "2026-01-01T01:00:00Z", "abandoned"),
        _row(14_487, "2026-01-01T02:00:00Z", "resumed"),
        _row(14_488, "2026-01-01T02:01:00Z", "resumed"),
    ]

    active, resume_iteration = active_resume_branch(local)

    assert resume_iteration == 14_487
    assert [
        (int(row["iteration"]), row["source"])
        for row in active
    ] == [
        (14_484, "shared"),
        (14_485, "shared"),
        (14_487, "resumed"),
        (14_488, "resumed"),
    ]
