"""Schema-v6 analysis writes only checkpoint-scoped run artifacts."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import torch

from plump.analysis import card_geometry
from plump.seq.config import SEQ_SCHEMA_VERSION, SeqModelConfig
from plump.seq.model import SeqPlumpModel


def test_analysis_loads_seq_checkpoint_and_scopes_every_output(
    tmp_path,
    monkeypatch,
):
    config = SeqModelConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
    )
    model = SeqPlumpModel(config)
    checkpoint = tmp_path / "run" / "checkpoints" / "iter_000003.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "schema_version": SEQ_SCHEMA_VERSION,
            "iteration": 3,
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )

    def touch(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")

    def fake_pca(model, source, output, **kwargs):
        touch(output)
        return SimpleNamespace(
            explained_variance=(0.5, 0.25, 0.125, 0.0625),
            nearest_suit_fraction=0.5,
            nearest_rank_gap=1.0,
        )

    def fake_mds(distances, title, label, output, seed, dpi):
        touch(output)
        return 0.1

    def fake_umap(vectors, title, label, output, seed, dpi):
        touch(output)

    def fake_heatmap(vectors, title, label, output, dpi):
        touch(output)

    def fake_probes(
        vectors,
        title,
        label,
        output,
        permutations,
        seed,
        dpi,
    ):
        touch(output)
        return 0.5, 0.25, 1.0, 0.25

    monkeypatch.setattr(card_geometry, "save_pca", fake_pca)
    monkeypatch.setattr(card_geometry, "plot_mds", fake_mds)
    monkeypatch.setattr(card_geometry, "plot_umap", fake_umap)
    monkeypatch.setattr(card_geometry, "plot_similarity_heatmap", fake_heatmap)
    monkeypatch.setattr(card_geometry, "plot_probe_baselines", fake_probes)

    output = tmp_path / "run" / "analysis" / checkpoint.stem
    report = card_geometry.analyze_checkpoint(
        checkpoint,
        output,
        permutations=1,
        dpi=20,
    )

    assert report["iteration"] == 3
    assert len(report["outputs"]) == 10
    report_paths = [type(output)(path) for path in report["outputs"]]
    assert all(path.is_relative_to(output) for path in report_paths)
    assert all(path.is_file() for path in report_paths)
    assert (output / "report.json").is_file()
