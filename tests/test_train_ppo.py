from pathlib import Path

from examples.train_ppo import _uniform_league_draw


def test_uniform_league_draw_combines_archive_and_run_checkpoints(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    run = tmp_path / "run"
    archive.mkdir()
    run.mkdir()
    for iteration in (3000, 4000, 5000, 6000):
        (archive / f"plump_v4_iter_{iteration:05d}.pt").touch()
    replacement = run / "plump_v4_iter_05000.pt"
    replacement.touch()
    newest = run / "plump_v4_iter_07000.pt"
    newest.touch()

    selected = _uniform_league_draw(
        run,
        min_iteration=4000,
        pool_size=10,
        seed=1,
        archive_dir=archive,
    )

    assert [int(path.stem.rsplit("_", 1)[1]) for path in selected] == [
        4000,
        5000,
        6000,
        7000,
    ]
    assert selected[1] == replacement


def test_uniform_league_draw_is_seeded_across_archive(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    run = tmp_path / "run"
    archive.mkdir()
    run.mkdir()
    for iteration in range(4000, 4010):
        (archive / f"plump_v4_iter_{iteration:05d}.pt").touch()

    first = _uniform_league_draw(
        run,
        min_iteration=4000,
        pool_size=4,
        seed=17,
        archive_dir=archive,
    )
    second = _uniform_league_draw(
        run,
        min_iteration=4000,
        pool_size=4,
        seed=17,
        archive_dir=archive,
    )

    assert first == second
    assert len(first) == 4
