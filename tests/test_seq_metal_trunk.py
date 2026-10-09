"""The fused Metal trunk against the PyTorch trunk it stands in for.

Both run bf16 autocast, so they cannot agree bit-for-bit: the checks compare
each against an fp32 run of the PyTorch trunk and require the fused error to
be of the same size as the PyTorch bf16 error.

Batches are at least 32 sequences on purpose. Below a few hundred rows MPS
autograd itself is not reproducible on this model -- the plain PyTorch trunk
returned gradients differing by up to 100% between identical runs at 6
sequences, and agreed to 3e-7 at 32 -- so a smaller test would be measuring
that, not the kernels.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from plump.env import PlumpEnv
from plump.rounds import RoundSpec, round_game_config
from plump.seq import tokens as seq_tokens
from plump.seq.config import SeqModelConfig
from plump.seq.metal_trunk import DualTrunkPass, trunk_parameters, unsupported_reason
from plump.seq.model import SeqPlumpModel

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="the Metal trunk needs MPS"
)

DEVICE = torch.device("mps")


def metal_config(**overrides) -> SeqModelConfig:
    # The rl-v6 trunk at two layers: GQA 10/2, PLE 64, alternating value
    # embeddings, factored cards.
    fields = dict(
        d_model=240,
        n_layers=2,
        n_heads=10,
        n_kv_heads=2,
        d_ff=720,
        embedding_rms_norm=True,
        trick_win_token=True,
        turn_token="bid",
        per_layer_embedding_dim=64,
        value_embedding_mode="alternating",
        card_embedding_mode="factored",
        absolute_position_embedding=False,
    )
    fields.update(overrides)
    return SeqModelConfig(**fields)


def token_batch(config, num_players, hand_size, seeds) -> torch.Tensor:
    rows = []
    for seed in seeds:
        start = random.Random(seed).randrange(num_players)
        env = PlumpEnv(
            round_game_config(
                RoundSpec(num_players, hand_size), bidding_start_player=start
            ),
            seed=seed,
        )
        env.reset(seed=seed)
        rng = random.Random(seed + 1)
        while not env.is_done():
            env.step(rng.choice(env.legal_actions()))
        observer = seed % num_players
        rows.append(
            seq_tokens.build_seat_tokens(
                config,
                env.state.event_log,
                observer,
                num_players,
                hand_size,
                env.state.current_round.initial_hands[observer],
                start,
            )
        )
    return torch.from_numpy(np.stack(rows)).to(DEVICE)


def make_model(config, seed=0) -> SeqPlumpModel:
    torch.manual_seed(seed)
    model = SeqPlumpModel(config)
    # Fresh additive tables start at zero; give every path real signal.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
    return model.to(DEVICE).train()


def run(model, tokens, probe, *, precision, metal):
    model.metal_trunk = None
    if metal:
        model.enable_metal_trunk()
    model.zero_grad(set_to_none=True)
    if precision == "bf16":
        with torch.autocast("mps", dtype=torch.bfloat16):
            hidden = model.forward_hidden(tokens)
    else:
        hidden = model.forward_hidden(tokens)
    (hidden.float() * probe).sum().backward()
    grads = {
        name: p.grad.detach().float().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    model.metal_trunk = None
    return hidden.detach().float(), grads


def relative(a, b) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        # Group size 2, and a width the 64-wide PLE does not divide.
        dict(d_model=96, n_heads=4, n_kv_heads=2, d_ff=192),
        # PLE width divides d_model; every layer has a value embedding.
        dict(d_model=192, n_heads=8, n_kv_heads=1, d_ff=384, value_embedding_mode="all"),
    ],
)
def test_forward_and_backward_match_the_pytorch_trunk(overrides):
    config = metal_config(**overrides)
    assert unsupported_reason(config) is None
    model = make_model(config)
    tokens = token_batch(config, 4, 7, seeds=range(32))
    probe = torch.randn(*tokens.shape[:2], config.d_model, device=DEVICE)

    exact, exact_grads = run(model, tokens, probe, precision="fp32", metal=False)
    torch_bf16, torch_grads = run(model, tokens, probe, precision="bf16", metal=False)
    fused, fused_grads = run(model, tokens, probe, precision="bf16", metal=True)

    torch_error = relative(torch_bf16, exact)
    assert relative(fused, exact) < 2 * torch_error + 1e-3
    assert set(fused_grads) == set(torch_grads)
    ratios = []
    for name, grad in exact_grads.items():
        if grad.norm() == 0:
            continue
        fused_error = relative(fused_grads[name], grad)
        torch_error = relative(torch_grads[name], grad)
        assert fused_error < 0.05, name
        ratios.append(fused_error / max(torch_error, 1e-4))
    assert np.median(ratios) < 1.5
    assert max(ratios) < 4


def test_dual_pass_returns_each_losss_own_gradients():
    config = metal_config()
    model = make_model(config, seed=1)
    tokens = token_batch(config, 5, 6, seeds=range(10, 42))
    params = [p for p in model.parameters() if p.requires_grad]

    def losses(hidden):
        # Two objectives that share the trunk but reach different heads.
        auxiliary = model.suit_presence_head(hidden).float().square().mean()
        policy = model.bid_head(hidden).float().logsumexp(-1).mean()
        return auxiliary, policy

    with torch.autocast("mps", dtype=torch.bfloat16):
        hidden = model.forward_hidden(tokens)
        auxiliary, policy = losses(hidden)
    expected = [
        torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        for loss in (auxiliary, policy)
    ]

    model.enable_metal_trunk()
    with torch.autocast("mps", dtype=torch.bfloat16):
        dual = DualTrunkPass(model, model.metal_trunk, tokens)
        auxiliary, policy = losses(dual.hidden)
    got = dual.gradients([auxiliary, policy], params)

    trunk = {id(p) for p in trunk_parameters(model)}
    for stream in range(2):
        assert len(got[stream]) == len(params)
        for p, want, have in zip(params, expected[stream], got[stream]):
            if want is None:
                assert have is None
                continue
            assert have is not None
            assert have.shape == p.shape and have.dtype == p.dtype
            tolerance = 0.06 if id(p) in trunk else 0.03
            assert relative(have.float(), want.float()) < tolerance


def test_trunk_only_replaces_bf16_autocast_on_mps():
    config = metal_config()
    model = make_model(config)
    model.enable_metal_trunk()
    tokens = token_batch(config, 3, 4, seeds=[0])
    assert not model.uses_metal_trunk(tokens)
    with torch.autocast("mps", dtype=torch.float16):
        assert not model.uses_metal_trunk(tokens)
    with torch.autocast("mps", dtype=torch.bfloat16):
        assert model.uses_metal_trunk(tokens)
        assert not model.uses_metal_trunk(tokens.cpu())


def test_unsupported_configurations_are_named():
    assert unsupported_reason(metal_config()) is None
    assert unsupported_reason(metal_config(dropout=0.1)) is not None
    assert unsupported_reason(metal_config(per_layer_embedding_dim=32)) is not None
    # 64-wide heads would not leave K and V room in threadgroup memory.
    assert unsupported_reason(metal_config(n_heads=4, n_kv_heads=2)) is not None


def test_merged_decode_matches_the_pytorch_cached_decode():
    """Ragged merged waves: rows at different positions, padded runs."""

    config = metal_config()
    model = make_model(config, seed=2).eval()
    tokens = token_batch(config, 4, 7, seeds=range(40, 72))
    batch, length, _ = tokens.shape
    rng = random.Random(0)
    caches = {
        name: model.new_cache(batch, torch.float16, max_len=config.max_seq_len + 1)
        for name in ("torch", "metal")
    }
    # Every row starts with a shared 9-token prefill, then advances by its
    # own 1-3 tokens per wave, so positions drift apart immediately.
    cursor = [0] * batch
    waves = 0
    with torch.inference_mode(), torch.autocast("mps", dtype=torch.bfloat16):
        while min(cursor) < length:
            counts = [
                0 if cursor[row] >= length
                else min(9 if waves == 0 else rng.randint(1, 3), length - cursor[row])
                for row in range(batch)
            ]
            run_length = max(counts)
            wave = torch.zeros(batch, run_length, tokens.shape[-1], dtype=tokens.dtype, device=DEVICE)
            for row in range(batch):
                start = min(cursor[row], length - 1)
                stop = max(cursor[row] + counts[row], start + 1)
                block = tokens[row, start:stop]
                # Pad with the row's last real token, as the merged loop does.
                wave[row, : block.shape[0]] = block
                wave[row, block.shape[0]:] = block[-1]
            positions = torch.tensor(
                [min(c, length - 1) for c in cursor], device=DEVICE
            )
            offsets = torch.tensor(
                [max(c, 1) - 1 for c in counts], device=DEVICE
            )
            hidden = {}
            for name, cache in caches.items():
                model.metal_trunk = None
                if name == "metal":
                    model.enable_metal_trunk()
                hidden[name] = model.forward_step_ragged(
                    wave, positions, cache, readout_offsets=offsets
                ).float()
            live = torch.tensor([c > 0 for c in counts], device=DEVICE)
            assert relative(hidden["metal"][live], hidden["torch"][live]) < 1e-2
            cursor = [c + n for c, n in zip(cursor, counts)]
            waves += 1
    assert waves > 5
    for layer in range(config.n_layers):
        for store in ("k", "v"):
            got = getattr(caches["metal"], store)[layer][:, :, :length].float()
            want = getattr(caches["torch"], store)[layer][:, :, :length].float()
            assert relative(got, want) < 1e-2


def test_cached_bf16_weights_follow_a_fused_optimizer_step():
    """The fused MPS Adam step does not bump Tensor._version, so a cache keyed
    on versions alone would keep serving the pre-step weights."""

    config = metal_config()
    model = make_model(config)
    model.enable_metal_trunk()
    params = trunk_parameters(model)
    before = model.metal_trunk._bf16_weights(params)[0][0].clone()
    optimizer = torch.optim.Adam(params, lr=1e-2, fused=True)
    for p in params:
        p.grad = torch.ones_like(p)
    optimizer.step()
    after = model.metal_trunk._bf16_weights(params)[0][0]
    assert torch.equal(after, params[2].detach().to(torch.bfloat16))
    assert not torch.equal(after, before)
