"""Static coverage for the optional Modal L40S preset."""

from plump.run_config import PROJECT_ROOT, load_training_config


def test_modal_l40s_preset_preserves_schedule_and_uses_cuda_capacity():
    resolved = load_training_config(PROJECT_ROOT / "configs" / "modal-l40s.toml")
    options = resolved.training.rollout

    assert resolved.run["device"] == "cuda"
    assert sum(cell.games for cell in resolved.training.schedule_cells) == 192
    assert options.opponent_mode == "heuristic_then_historical"
    assert options.opponent_fraction == 0.5
    assert options.opponent_packing == "concurrent"
    assert options.deals_per_batch == 4
    assert options.parallel_deals_max_hand_size == 10
    assert options.max_cache_rows == 262_144
    assert resolved.training.microbatch_positions == 131_072
    assert resolved.training.learning_rate == 4e-4
    assert resolved.training.core_lr == 5e-5
    assert resolved.training.auxiliary_lr == 5e-5
    assert resolved.training.policy_kl_p99_cap == 0.20
    assert resolved.run["checkpoint_every"] == 12
    assert resolved.evaluation["every"] == 25
