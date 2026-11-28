# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# See LICENSE.txt for license information
"""ElasticBuffer's Python-side input checks, internode opt-in and plan bookkeeping, with the C library
stubbed."""

import ctypes
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
import rccl_ep  # noqa: E402

TOKENS, HIDDEN, TOPK, EXPERTS = 4, 256, 2, 4
RECV = 3  # rows the stubbed ep_dispatch reports received


def _make_buffer(internode=False):
    buf = object.__new__(rccl_ep.ElasticBuffer)
    # Internode: two nodes of two ranks, as far as the Python layer can tell.
    buf.num_ranks = 4 if internode else 2
    buf.internode = internode
    buf.num_max_tokens_per_rank = 8
    buf.hidden = HIDDEN
    buf.num_experts = EXPERTS
    buf.num_topk = TOPK
    buf.allow_multiple_reduction = False
    buf._h = 0
    buf._configured = (EXPERTS, TOPK)
    return buf


class _Calls(dict):
    """The stubbed calls' arguments by name, and `order`: their names in call order."""

    def __init__(self):
        super().__init__()
        self.order = []


def _install_stubs(monkeypatch, expanded_rows=None, inter_counts=()):
    """Stubs every entry point a dispatch and combine reach; returns the calls.

    For an internode buffer, `inter_counts` holds the (sendc, recvc) lists that successive
    ep_inter_counts calls return. ep_inter_bind records the counts it reads at the call, with
    send_list and num_tokens, and ep_dispatch then reports the bound receive total, or fails
    with nothing bound, as the library does."""
    calls = _Calls()
    pending = list(inter_counts)
    ranks = len(inter_counts[0][0]) if inter_counts else 0
    bound = []

    def record(name, *a):
        calls.setdefault(name, []).append(a)
        calls.order.append(name)

    def stub(name, rc=0):
        def f(*a):
            record(name, *a)
            return rc
        return f

    def ep_inter_counts(h, sendc, out, stream):
        record("ep_inter_counts", h, sendc, out, stream)
        s, r = pending.pop(0)
        (ctypes.c_int32 * (2 * ranks)).from_address(out)[:] = s + r
        return 0

    def ep_inter_bind(h, sendc, recvc, send_list, num_tokens):
        s = list((ctypes.c_int32 * ranks).from_address(sendc))
        r = list((ctypes.c_int32 * ranks).from_address(recvc))
        record("ep_inter_bind", s, r, send_list, num_tokens)
        bound[:] = [sum(r)]
        return 0

    def ep_dispatch(*a):
        record("ep_dispatch", *a)
        if not inter_counts:
            return RECV
        return bound[0] if bound else -1

    lib = types.SimpleNamespace(
        ep_configure=stub("ep_configure"),
        ep_plan=stub("ep_plan"),
        ep_inter_counts=ep_inter_counts,
        ep_inter_bind=ep_inter_bind,
        ep_dispatch=ep_dispatch,
        ep_recv_counts=stub("ep_recv_counts"),
        ep_expert_counts=stub("ep_expert_counts"),
        ep_expand_build=stub("ep_expand_build", expanded_rows),
        ep_expand_scatter=stub("ep_expand_scatter"),
        ep_combine=stub("ep_combine"),
    )
    monkeypatch.setattr(rccl_ep, "_LIB", lib)
    return calls


def _x():
    return torch.zeros((TOKENS, HIDDEN), dtype=torch.bfloat16, device="cuda")


def _topk():
    return torch.zeros((TOKENS, TOPK), dtype=torch.int64, device="cuda")


def test_dispatch_tuple_without_scales_is_refused(monkeypatch):
    calls = _install_stubs(monkeypatch)
    with pytest.raises(ValueError, match="needs x_sf"):
        _make_buffer().dispatch((_x(), None), topk_idx=_topk())
    assert not calls


def test_dispatch_clamps_int64_ids_before_narrowing(monkeypatch):
    _install_stubs(monkeypatch)
    ids = torch.tensor([[2**32 + 1, 1], [-(2**32) + 2, 3], [EXPERTS, -5], [0, 2**31]],
                       dtype=torch.int64, device="cuda")

    _, _, _, handle, _ = _make_buffer().dispatch(_x(), topk_idx=ids)

    # Truncated, 2**32 + 1 would be expert 1 and -(2**32) + 2 expert 2; clamped, every id outside
    # [0, num_experts) stays outside it.
    want = torch.tensor([[EXPERTS, 1], [-1, 3], [EXPERTS, -1], [0, EXPERTS]], dtype=torch.int32)
    assert torch.equal(handle.topk_idx_i32.cpu(), want)


def test_cached_dispatch_configures_the_handles_expert_count(monkeypatch):
    calls = _install_stubs(monkeypatch)
    buf = _make_buffer()
    _, _, _, handle, _ = buf.dispatch(_x(), topk_idx=_topk())
    buf._configure(2 * EXPERTS, TOPK)  # another layer's dispatch re-sizes the window

    buf.dispatch(_x(), handle=handle)

    assert [a[1:] for a in calls["ep_configure"]] == [(2 * EXPERTS, TOPK), (EXPERTS, TOPK)]


def test_combine_refuses_a_window_reconfigured_since_dispatch(monkeypatch):
    calls = _install_stubs(monkeypatch)
    buf = _make_buffer()
    _, _, _, handle, _ = buf.dispatch(_x(), topk_idx=_topk())
    buf._configure(2 * EXPERTS, TOPK)
    rows = torch.zeros((RECV, HIDDEN), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError, match="configured for"):
        buf.combine(rows, handle)
    assert "ep_combine" not in calls


def _bad_weights(expanded, rows):
    shape = (rows,) if expanded else (rows, TOPK)
    wrong = ([("per token, not per row", (rows, TOPK)), ("too few rows", (rows - 1,))] if expanded
             else [("too few columns", (rows, TOPK - 1)), ("flattened", (rows * TOPK,))])
    return [(what, torch.zeros(s, device="cuda")) for what, s in wrong] + [
        ("integer", torch.zeros(shape, dtype=torch.int32, device="cuda")),
        ("on the host", torch.zeros(shape, device="cpu")),
    ]


@pytest.mark.parametrize("expanded", [False, True])
def test_combine_checks_topk_weights_against_the_handle(monkeypatch, expanded):
    erows = 5
    calls = _install_stubs(monkeypatch, expanded_rows=erows)
    buf = _make_buffer()
    _, _, _, handle, _ = buf.dispatch(_x(), topk_idx=_topk(), do_expand=expanded)
    rows = erows if expanded else RECV
    shape = (rows,) if expanded else (rows, TOPK)
    x = torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device="cuda")

    for what, w in _bad_weights(expanded, rows):
        with pytest.raises(ValueError, match="topk_weights"):
            buf.combine(x, handle, topk_weights=w)
        assert "ep_combine" not in calls, what

    _, out_w, _ = buf.combine(x, handle, topk_weights=torch.ones(shape, device="cuda"))
    assert len(calls["ep_combine"]) == 1
    assert out_w.shape == (TOKENS, TOPK)


def test_combine_bias_must_be_on_the_current_device(monkeypatch):
    calls = _install_stubs(monkeypatch)
    buf = _make_buffer()
    _, _, _, handle, _ = buf.dispatch(_x(), topk_idx=_topk())
    x = torch.zeros((RECV, HIDDEN), dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros((TOKENS, HIDDEN), dtype=torch.bfloat16, device="cpu")

    with pytest.raises(ValueError, match="bias0 is on"):
        buf.combine(x, handle, bias=bias)
    assert "ep_combine" not in calls


def _fp8(dtype=None):
    return torch.zeros((TOKENS, HIDDEN), device="cuda").to(dtype or rccl_ep._FP8_DTYPES[0])


def _sf(rows=TOKENS, cols=(HIDDEN + 127) // 128, dtype=torch.float32, device="cuda"):
    return torch.ones((rows, cols), dtype=dtype, device=device)


BAD_DISPATCH = [
    ("x is 3-D", lambda: dict(x=_x().view(1, TOKENS, HIDDEN)), "x must be 2-D"),
    ("x has another hidden", lambda: dict(x=torch.zeros((TOKENS, HIDDEN + 8), dtype=torch.bfloat16,
                                                          device="cuda")), "x must be 2-D"),
    ("x is float32", lambda: dict(x=torch.zeros((TOKENS, HIDDEN), device="cuda")), "x must be bfloat16"),
    ("x is on the host", lambda: dict(x=_x().to("cpu")), "x is on"),
    ("an fp8 tuple holds bf16 x", lambda: dict(x=(_x(), _sf())), "an fp8 dispatch's x"),
    ("x_sf has too few blocks", lambda: dict(x=(_fp8(), _sf(cols=1))), "x_sf must be"),
    ("x_sf is integer", lambda: dict(x=(_fp8(), _sf(dtype=torch.int32))), "x_sf must be"),
    ("x_sf is on the host", lambda: dict(x=(_fp8(), _sf(device="cpu"))), "x_sf is on"),
    ("topk_idx is float", lambda: dict(topk_idx=_topk().float()), "topk_idx must be"),
    ("topk_idx has another row count", lambda: dict(topk_idx=_topk()[1:]), "topk_idx must be"),
    ("topk_idx is on the host", lambda: dict(topk_idx=_topk().to("cpu")), "topk_idx is on"),
    ("topk_weights has another shape", lambda: dict(topk_weights=torch.ones((TOKENS, TOPK + 1), device="cuda")),
     "topk_weights must have"),
    ("topk_weights is on the host", lambda: dict(topk_weights=torch.ones((TOKENS, TOPK), device="cpu")),
     "topk_weights is on"),
    ("another capacity", lambda: dict(num_max_tokens_per_rank=16), "num_max_tokens_per_rank"),
]


@pytest.mark.parametrize("what,args,match", BAD_DISPATCH, ids=[c[0] for c in BAD_DISPATCH])
def test_dispatch_checks_its_inputs_before_anything_runs(monkeypatch, what, args, match):
    calls = _install_stubs(monkeypatch)
    kw = dict(x=_x(), topk_idx=_topk())
    kw.update(args())
    with pytest.raises(ValueError, match=match):
        _make_buffer().dispatch(kw.pop("x"), **kw)
    assert not calls, what


def test_cached_dispatch_refuses_another_expert_count(monkeypatch):
    _install_stubs(monkeypatch)
    buf = _make_buffer()
    _, _, _, handle, _ = buf.dispatch(_x(), topk_idx=_topk())

    with pytest.raises(ValueError, match="cached dispatch: num_experts"):
        buf.dispatch(_x(), handle=handle, num_experts=2 * EXPERTS)


@pytest.mark.parametrize("dtype", rccl_ep._FP8_DTYPES, ids=str)
def test_fp8_outputs_keep_the_inputs_e4m3_variant(monkeypatch, dtype):
    _install_stubs(monkeypatch, expanded_rows=5)
    buf = _make_buffer()

    (rx, rsf), _, _, _, _ = buf.dispatch((_fp8(dtype), _sf()), topk_idx=_topk())
    (ex, esf), _, _, _, _ = buf.dispatch((_fp8(dtype), _sf()), topk_idx=_topk(), do_expand=True)

    assert rx.dtype == dtype and ex.dtype == dtype
    assert rsf.dtype == torch.float32 and esf.dtype == torch.float32


def test_combine_checks_x_against_the_handle(monkeypatch):
    calls = _install_stubs(monkeypatch)
    buf = _make_buffer()
    _, _, _, handle, _ = buf.dispatch(_x(), topk_idx=_topk())
    bad = [
        (torch.zeros((RECV + 1, HIDDEN), dtype=torch.bfloat16, device="cuda"), "x has"),
        (torch.zeros((RECV, HIDDEN), device="cuda"), "x must be bfloat16"),
        (torch.zeros((RECV, HIDDEN + 8), dtype=torch.bfloat16, device="cuda"), "x must be bfloat16"),
        (torch.zeros((RECV, HIDDEN), dtype=torch.bfloat16, device="cpu"), "x is on"),
    ]
    for x, match in bad:
        with pytest.raises(ValueError, match=match):
            buf.combine(x, handle)
    assert "ep_combine" not in calls


def test_combine_checks_bias_dtype_and_shape(monkeypatch):
    calls = _install_stubs(monkeypatch)
    buf = _make_buffer()
    _, _, _, handle, _ = buf.dispatch(_x(), topk_idx=_topk())
    x = torch.zeros((RECV, HIDDEN), dtype=torch.bfloat16, device="cuda")
    good = torch.zeros((TOKENS, HIDDEN), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(TypeError, match="bias0 must be bfloat16"):
        buf.combine(x, handle, bias=good.float())
    with pytest.raises(ValueError, match="bias0 must be"):
        buf.combine(x, handle, bias=good[1:])
    with pytest.raises(ValueError, match="bias1 must be"):
        buf.combine(x, handle, bias=(good, good[:, 8:]))
    assert "ep_combine" not in calls


def _stub_group(monkeypatch, hosts, opted_in, rank):
    """A stubbed group of len(hosts) ranks, this one `rank`, on hosts[rank]: rank r's entry in
    the constructor's all-gather is (hosts[r], opted_in[r]), except this rank's, which is what
    the constructor contributes. Returns the ep_create_nodes calls."""
    monkeypatch.setenv("NCCL_HOSTID", hosts[rank])

    def all_gather_object(out, obj, group=None):
        out[:] = [obj if r == rank else p for r, p in enumerate(zip(hosts, opted_in))]

    monkeypatch.setattr(rccl_ep, "dist", types.SimpleNamespace(
        get_world_size=lambda group=None: len(hosts), get_rank=lambda group=None: rank,
        get_backend=lambda group=None: "gloo", broadcast=lambda *a, **k: None,
        all_gather_object=all_gather_object))
    created = []

    def ep_create_nodes(*a):
        created.append(a)
        return 1

    monkeypatch.setattr(rccl_ep, "_LIB", types.SimpleNamespace(
        ep_unique_id_size=lambda: 8, ep_get_unique_id=lambda buf: 0, ep_create_nodes=ep_create_nodes))
    return created


TWO_NODES = ["a", "a", "b", "b"]
REFUSED = (NotImplementedError, "opt-in: pass allow_internode=True")
MIXED = (ValueError, "must be given on every rank")
LAYOUT = (ValueError, "needs equal nodes")
# (what, hosts, the other ranks' opt-in, how this rank opts in, ranks per node or (error, match))
CONSTRUCTOR_CASES = [
    ("one node", ["a"] * 4, [False] * 3, None, 4),
    ("two nodes, no opt-in", TWO_NODES, [False] * 3, None, REFUSED),
    ("two nodes, keyword", TWO_NODES, [True] * 3, "keyword", 2),
    ("two nodes, one other rank not opted in", TWO_NODES, [True, True, False], "keyword", MIXED),
    ("two nodes, this rank not opted in", TWO_NODES, [True] * 3, None, MIXED),
    ("interleaved nodes", ["a", "b", "a", "b"], [True] * 3, "keyword", LAYOUT),
    ("unequal nodes", ["a", "a", "a", "b"], [True] * 3, "keyword", LAYOUT),
]


@pytest.mark.parametrize("hybrid", [False, True], ids=["plain", "hybrid"])
@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("what,hosts,others,mine,want", CONSTRUCTOR_CASES, ids=[c[0] for c in CONSTRUCTOR_CASES])
def test_constructor_takes_the_internode_path_only_when_every_rank_opts_in(
        monkeypatch, capsys, what, hosts, others, mine, want, rank, hybrid):
    created = _stub_group(monkeypatch, hosts, others[:rank] + [None] + others[rank:], rank)
    kw = {"allow_hybrid_mode": hybrid}
    if mine == "keyword":
        kw["allow_internode"] = True

    if isinstance(want, tuple):
        # Every rank sees the same gathered list, so each raises the same error, before the
        # unique-id broadcast and ep_create_nodes that would wait on its peers.
        with pytest.raises(want[0], match=want[1]):
            rccl_ep.ElasticBuffer(None, 8, HIDDEN, **kw)
        assert not created, what
        return
    buf = rccl_ep.ElasticBuffer(None, 8, HIDDEN, **kw)
    assert [c[-1] for c in created] == [want]
    assert buf.internode == (want < len(hosts))
    # One line from group rank 0 when the group takes the internode path, none otherwise.
    noted = buf.internode and rank == 0
    assert capsys.readouterr().err.count("internode send/recv path") == (1 if noted else 0)


# (sendc, recvc) per rank, as ep_inter_counts returns them.
COUNTS_A = ([1, 2, 0, 3], [2, 0, 1, 4])
COUNTS_B = ([0, 1, 1, 1], [1, 1, 0, 0])


def test_internode_dispatch_binds_its_plan_and_sizes_outputs_to_the_counts(monkeypatch):
    calls = _install_stubs(monkeypatch, inter_counts=[COUNTS_A])
    buf = _make_buffer(internode=True)

    rx, _, rtw, h, _ = buf.dispatch(_x(), topk_idx=_topk())

    steps = ("ep_plan", "ep_inter_counts", "ep_inter_bind", "ep_dispatch")
    assert [c for c in calls.order if c in steps] == list(steps)
    assert calls["ep_inter_counts"][0][1] == h.sendc.data_ptr()
    assert calls["ep_inter_bind"] == [(*COUNTS_A, h.send_list.data_ptr(), TOKENS)]
    # Sized to the receive total the exchange reported, not sliced from num_ranks * capacity rows.
    assert rx.shape[0] == h.num_recv == sum(COUNTS_A[1])
    for t in (rx, h.recv_topk_idx, rtw):
        assert t.untyped_storage().nbytes() == t.numel() * t.element_size()


def test_internode_combine_and_replay_bind_their_own_handle(monkeypatch):
    calls = _install_stubs(monkeypatch, inter_counts=[COUNTS_A, COUNTS_B])
    buf = _make_buffer(internode=True)
    _, _, _, a, _ = buf.dispatch(_x(), topk_idx=_topk())
    _, _, _, b, _ = buf.dispatch(_x()[:3], topk_idx=_topk()[:3])
    bind_a = (*COUNTS_A, a.send_list.data_ptr(), TOKENS)
    assert calls["ep_inter_bind"][-1] == (*COUNTS_B, b.send_list.data_ptr(), 3)

    # The library still holds b's binding; combining a must bind a's first.
    buf.combine(torch.zeros((a.num_recv, HIDDEN), dtype=torch.bfloat16, device="cuda"), a)
    assert calls.order[-2:] == ["ep_inter_bind", "ep_combine"]
    assert calls["ep_inter_bind"][-1] == bind_a

    # A replay reuses the handle's counts, with no new exchange, and binds them again.
    buf.dispatch(_x()[:3], handle=b)
    buf.dispatch(_x(), handle=a)
    assert len(calls["ep_inter_counts"]) == 2
    last = len(calls.order) - 1 - calls.order[::-1].index("ep_dispatch")
    assert calls.order[last - 1] == "ep_inter_bind"
    assert calls["ep_inter_bind"][-1] == bind_a
