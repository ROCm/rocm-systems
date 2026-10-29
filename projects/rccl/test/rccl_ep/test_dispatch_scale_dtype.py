# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""ElasticBuffer.dispatch's Python-side scale handling, with the C library stubbed."""

import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "algorithms", "rccl_ep", "python")
)
rccl_ep = pytest.importorskip("rccl_ep")

TOKENS, HIDDEN, TOPK = 4, 256, 2


def _make_buffer():
    buf = object.__new__(rccl_ep.ElasticBuffer)
    buf.num_ranks = 2
    buf.num_max_tokens_per_rank = 8
    buf.hidden = HIDDEN
    buf.num_experts = 4
    buf.num_topk = TOPK
    buf._h = 0
    buf._configured = (4, TOPK)
    return buf


def _install_stubs(monkeypatch):
    lib = types.SimpleNamespace(
        ep_configure=lambda *a: 0,
        ep_plan=lambda *a: 0,
        ep_recv_counts=lambda *a: 0,
        ep_expert_counts=lambda *a: 0,
        ep_dispatch=lambda *a: 0,
    )
    monkeypatch.setattr(rccl_ep, "_LIB", lib)
    seen = []
    orig = rccl_ep._ptr

    def rec(t):
        seen.append(t)
        return orig(t)

    monkeypatch.setattr(rccl_ep, "_ptr", rec)
    return seen


def _topk():
    return torch.zeros((TOKENS, TOPK), dtype=torch.int64, device="cuda")


@pytest.mark.parametrize("sf_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_dispatch_fp8_scales_are_float32_contiguous(monkeypatch, sf_dtype):
    seen = _install_stubs(monkeypatch)
    x = torch.zeros((TOKENS, HIDDEN), device="cuda").to(torch.float8_e4m3fn)
    base = torch.arange(TOKENS * 2, device="cuda", dtype=torch.float32) * 0.5 + 1.0
    sf = base.reshape(2, TOKENS).to(sf_dtype).t()
    assert not sf.is_contiguous()

    _make_buffer().dispatch((x, sf), topk_idx=_topk())

    passed = seen[0]
    assert passed is not None
    assert passed.dtype == torch.float32
    assert passed.is_contiguous()
    assert passed.shape == sf.shape
    assert torch.equal(passed, sf.float())


def test_dispatch_bf16_passes_no_scales(monkeypatch):
    seen = _install_stubs(monkeypatch)
    x = torch.zeros((TOKENS, HIDDEN), dtype=torch.bfloat16, device="cuda")

    _make_buffer().dispatch(x, topk_idx=_topk())

    assert seen[0] is None


def test_dispatch_expanded_col_major_scales(monkeypatch):
    _install_stubs(monkeypatch)
    rows = 3
    rccl_ep._LIB.ep_expand_build = lambda *a: rows
    rccl_ep._LIB.ep_expand_scatter = lambda *a: 0
    x = torch.zeros((TOKENS, HIDDEN), device="cuda").to(torch.float8_e4m3fn)
    sf = torch.ones((TOKENS, (HIDDEN + 127) // 128), dtype=torch.float32, device="cuda")

    (ex, esf), _, _, _, _ = _make_buffer().dispatch(
        (x, sf), topk_idx=_topk(), do_expand=True, use_tma_aligned_col_major_sf=True
    )

    assert ex.shape == (rows, HIDDEN)
    assert esf.shape == (rows, (HIDDEN + 127) // 128)
    assert esf.stride() == (1, rows)
