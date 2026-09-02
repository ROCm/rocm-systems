# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# The Python surface: a ctypes binding to the rccl_ep C ABI.
#
# See LICENSE.txt for license information

import ctypes
import os
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist

__version__ = "0.1.0"

# The public API uses int64 topk indices.
__all__ = [
    "ElasticBuffer",
    "EventOverlap",
    "EPHandle",
    "get_bf16_lanes_per_token",
    "get_threads_per_cta",
    "get_waves_per_cta",
    "topk_idx_t",
]

topk_idx_t = torch.int64

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        # An EP-capable librccl, loaded RTLD_GLOBAL before the extension so
        # that it and torch resolve to the same one. Without it the loader
        # binds our DT_NEEDED to whichever librccl arrived first, typically
        # torch's, which predates the device API this path needs.
        rccl = os.environ.get("RCCL_EP_LIBRCCL")
        if rccl and os.path.exists(rccl):
            ctypes.CDLL(rccl, mode=ctypes.RTLD_GLOBAL)

        path = os.environ.get(
            "RCCL_EP_LIB",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "librccl_ep.so"),
        )
        # RTLD_LOCAL for the extension itself: it exports the same symbol
        # names as librccl, and there is no reason to put those in the global
        # namespace. RTLD_DEEPBIND is not an option -- it would also redirect
        # the C++ runtime and HIP symbols shared with torch, and segfaults.
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
        P, I = ctypes.c_size_t, ctypes.c_int
        lib.ep_create.restype = ctypes.c_void_p
        # The unique id is an opaque binary blob, not a string: raw pointers
        # rather than c_char_p, which is for NUL-terminated data.
        lib.ep_create.argtypes = [I, I, ctypes.c_void_p, I, I, I]
        lib.ep_configure.argtypes = [ctypes.c_void_p, I, I]
        lib.ep_window_bytes.restype = ctypes.c_size_t
        lib.ep_window_bytes.argtypes = [ctypes.c_void_p]
        lib.ep_destroy.argtypes = [ctypes.c_void_p]
        # Every data-path entry takes the caller's stream as a trailing argument.
        lib.ep_barrier.argtypes = [ctypes.c_void_p, P]
        lib.ep_plan.argtypes = [I, I, I, P, I, P, P, P, P, I, I, P]
        lib.ep_dispatch.restype = I
        lib.ep_dispatch.argtypes = [
            ctypes.c_void_p,
            P,
            P,
            P,
            P,
            I,
            P,
            P,
            I,
            P,
            P,
            I,
            I,
            P,
            P,
            P,
            P,
            P,
            P,
            P,
        ]
        lib.ep_recv_counts.argtypes = [ctypes.c_void_p, P, P]
        lib.ep_expert_counts.argtypes = [ctypes.c_void_p, P, I, P, P]
        lib.ep_expand_build.restype = I
        lib.ep_expand_build.argtypes = [ctypes.c_void_p, P, I, I, P, P, P, P, P, P]
        lib.ep_expand_scatter.argtypes = [ctypes.c_void_p, I, P, P, P, P, I, P, P,
                                          I, I, P, I, I, P, P, I, P]
        lib.ep_combine.restype = I
        lib.ep_combine.argtypes = [ctypes.c_void_p, P, P, P, P, P, I, P, I, P, P, I, P, P, I, P]
        lib.ep_get_unique_id.argtypes = [ctypes.c_void_p]
        lib.ep_unique_id_size.restype = I
        lib.ep_threads_per_cta.restype = I
        lib.ep_waves_per_cta.restype = I
        lib.ep_bf16_lanes_per_token.restype = I
        _LIB = lib
    return _LIB


def _ptr(t: Optional[torch.Tensor]) -> int:
    return 0 if t is None else t.data_ptr()


def _align(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def get_threads_per_cta() -> int:
    return int(_lib().ep_threads_per_cta())


def get_waves_per_cta() -> int:
    return int(_lib().ep_waves_per_cta())


def get_bf16_lanes_per_token() -> int:
    return int(_lib().ep_bf16_lanes_per_token())


class EventOverlap:
    """Completion handle returned by the data path.

    rccl_ep queues work on the caller's current stream rather than a private
    communication stream. Same-stream consumers are ordered without an event;
    this wrapper preserves the DeepEP-compatible call surface.
    """

    def __init__(self, event=None):
        self.event = event

    def current_stream_wait(self):
        if self.event is not None:
            torch.cuda.current_stream().wait_event(self.event)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class EPHandle:
    """Routing metadata produced by dispatch and consumed by combine.

    Attribute names are part of the public API, because callers read them
    directly.
    """

    def __init__(self):
        self.topk_idx = None                            # as passed, or a copy
        self.topk_idx_i32 = None
        self.slot = None                                # routing plan
        self.send_list = None                           # dense slot -> token
        self.sendc = None
        self.num_tokens = 0
        self.num_recv = 0
        self.recv_counts_host = None
        self.recv_rank_offsets = None
        self.recv_rank_prefix = None
        self.recv_src_metadata = None                   # [n, 2 + num_topk]
        self.recv_topk_idx = None                       # local ids, [n, num_topk]
        self.dst_buffer_slot_idx = None
        self.psum_num_recv_tokens_per_scaleup_rank = None
        self.psum_num_recv_tokens_per_expert = None
        self.num_recv_tokens_per_expert_list = None
        self.expert_counts = None
        self.expert_offsets = None
        self.row_map = None                             # expanded layout only
        self.expert_alignment = 1
        self.expanded = False
        self.num_expanded_rows = 0


class ElasticBuffer:
    """Intranode expert-parallel dispatch/combine buffer.

    Implemented: BF16 and FP8 dispatch, expanded (grouped-by-expert) dispatch
    with alignment and zero padding, cached replay, both combine reduction
    recipes, one or two bias tensors, and deterministic ordering. The normal
    DeepEP V2-style route planner assigns slots atomically; deterministic mode
    selects the ordered planner so repeated runs remain byte-identical.

    Scale-up only: every peer must be reachable by load and store, so a
    communicator spanning more than one node is rejected at construction.
    """

    def __init__(self, group, num_max_tokens_per_rank, hidden,
                 num_experts=None, num_topk=None,
                 deterministic=False, allow_hybrid_mode=False,
                 allow_multiple_reduction=False,
                 prefer_overlap_with_compute=False, sl_idx=0,
                 num_allocated_qps=0, explicitly_destroy=False,
                 num_gpu_timeout_secs=100, num_cpu_timeout_secs=100, **kwargs):
        # Standard callers pass True for both, so refusing them outright would
        # leave the standard invocation unable to run at all.
        #
        # hybrid mode: on a single node there is no scale-out leg, so
        # num_scaleout_ranks is 1 and every hybrid branch degenerates to the
        # pure scale-up path we implement. Accepting it is therefore honest
        # HERE and only here -- across nodes it would silently do the wrong
        # thing, so the single-node property is verified rather than assumed.
        if allow_hybrid_mode:
            import socket
            hosts = [None] * dist.get_world_size(group)
            dist.all_gather_object(hosts, socket.gethostname(), group)
            if len(set(hosts)) != 1:
                raise NotImplementedError(
                    f"allow_hybrid_mode needs a scale-out leg, which this path does "
                    f"not provide, and this communicator spans {len(set(hosts))} "
                    f"nodes. It is accepted only on a single node, where it "
                    f"degenerates to the scale-up path.")
        # multiple reduction: the expanded combine has to apply the per-rank
        # grouped reduction before sending, instead of shipping one row per
        # (token, expert). Implemented; see combine().
        self.allow_multiple_reduction = bool(allow_multiple_reduction)
        self.allow_hybrid_mode = bool(allow_hybrid_mode)

        self.group = group
        self.rank_idx = dist.get_rank(group)
        self.num_ranks = dist.get_world_size(group)
        self.scaleup_rank_idx = self.rank_idx
        self.scaleout_rank_idx = 0
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.num_experts = num_experts
        self.num_topk = num_topk
        self.deterministic = deterministic
        self.num_allocated_qps = 0        # LSA path uses no QPs
        self.explicitly_destroy = explicitly_destroy
        self._configured = None

        lib = _lib()
        n = lib.ep_unique_id_size()
        buf = ctypes.create_string_buffer(n)
        # Unchecked, a failure here broadcasts the zero-filled buffer and every
        # rank hangs inside ncclCommInitRank on an id that is not one.
        if self.rank_idx == 0 and lib.ep_get_unique_id(buf) != 0:
            raise RuntimeError("ep_get_unique_id failed")
        # The bootstrap group may be gloo (CPU-only) or nccl (GPU-only), and
        # neither accepts the other's tensors, so follow the backend.
        t = torch.frombuffer(bytearray(buf.raw), dtype=torch.uint8).clone()
        t = t.cuda() if dist.get_backend(group) != "gloo" else t.cpu()
        # `src` is a global rank even when a group is given, while rank_idx above is the
        # rank within the group. They coincide only for the group that happens to contain
        # global rank 0, so passing 0 here works on a single node and raises "Global rank 0
        # is not part of group" on every other node's expert-parallel group.
        dist.broadcast(t, src=dist.get_global_rank(group, 0), group=group)
        t = t.cpu()
        raw = bytes(bytearray(t.tolist()))

        self._h = lib.ep_create(self.rank_idx, self.num_ranks, raw,
                                num_max_tokens_per_rank, hidden,
                                torch.cuda.current_device())
        if not self._h:
            raise RuntimeError("ep_create failed")

    # ---- informational -------------------------------------------------
    def get_logical_domain_size(self, *a, **k):
        return (1, self.num_ranks)          # (scaleout, scaleup)

    def get_theoretical_num_sms(self, *a, **k):
        # Measured on gfx950: peer-copy bandwidth is flat at ~6.0-6.7 GB/s per
        # CU up to 64 CUs, and dispatch reaches 81% of the achievable ceiling
        # at 128 total CTAs, which is where it stops improving. Constants
        # fitted on Hopper would under-provision here by roughly 2x.
        #
        # This is a TOTAL CTA budget across the grid, which is how num_sms is
        # defined here and how ep_dispatch divides it up.
        return 128

    def get_theoretical_num_qps(self, *a, **k):
        return 0

    def capture(self):
        ev = torch.cuda.Event()
        ev.record()
        return EventOverlap(ev)

    def barrier(self):
        _lib().ep_barrier(self._h, torch.cuda.current_stream().cuda_stream)

    def destroy(self):
        if getattr(self, "_h", None):
            _lib().ep_destroy(self._h)
            self._h = None

    # ---- internals -------------------------------------------------------
    def _configure(self, num_experts, num_topk):
        key = (num_experts, num_topk)
        if self._configured == key:
            return
        if _lib().ep_configure(self._h, num_experts, num_topk) != 0:
            raise RuntimeError(f"ep_configure({num_experts}, {num_topk}) failed")
        self._configured = key
        self.num_experts, self.num_topk = num_experts, num_topk

    @staticmethod
    def plan_layout(
        topk_idx, num_ranks, num_experts, deterministic=False, num_sms=0
    ):
        """Build the routing plan and counts consumed by dispatch."""
        ti32 = topk_idx.to(torch.int32).contiguous()
        num_tokens, num_topk = ti32.shape
        slot = torch.empty(
            (num_ranks, num_tokens), dtype=torch.int32, device=ti32.device
        )
        send_list = torch.empty_like(slot)
        sendc = torch.empty((num_ranks,), dtype=torch.int32, device=ti32.device)
        expertc = torch.empty((num_experts,), dtype=torch.int32, device=ti32.device)
        stream = torch.cuda.current_stream().cuda_stream
        if _lib().ep_plan(
            num_ranks,
            num_experts,
            num_topk,
            ti32.data_ptr(),
            num_tokens,
            slot.data_ptr(),
            send_list.data_ptr(),
            sendc.data_ptr(),
            expertc.data_ptr(),
            1 if deterministic else 0,
            num_sms,
            stream,
        ) != 0:
            raise RuntimeError("ep_plan failed")
        return ti32, slot, send_list, sendc, expertc

    def plan(self, topk_idx, num_experts, num_sms=0):
        return self.plan_layout(
            topk_idx,
            self.num_ranks,
            num_experts,
            deterministic=self.deterministic,
            num_sms=num_sms,
        )

    @property
    def num_local_experts(self):
        return self.num_experts // self.num_ranks

    def _expert_metadata(self, h, counts, expert_alignment, counts_list=None):
        """Fill in the two different per-expert prefix sums the handle reports.

        They are genuinely different quantities: the plain handle reports a
        prefix over ALIGNED counts (there is no expanded tensor, so the numbers
        only describe how a consumer would lay one out), while the expanded
        handle reports the end of each expert's REAL rows inside the tensor that
        was actually produced.
        """
        cl = counts.tolist() if counts_list is None else counts_list
        aligned = [_align(c, expert_alignment) for c in cl]
        h.num_recv_tokens_per_expert_list = aligned
        h.expert_counts = counts
        if h.expanded:
            psum, acc = [], 0
            for c, a in zip(cl, aligned):
                psum.append(acc + c)
                acc += a
        else:
            psum, acc = [], 0
            for a in aligned:
                acc += a
                psum.append(acc)
        h.psum_num_recv_tokens_per_expert = torch.tensor(
            psum, dtype=torch.int32, device="cuda")

    # ---- data path -------------------------------------------------------
    def dispatch(self, x, topk_idx=None, topk_weights=None,
                 num_max_tokens_per_rank=None, num_experts=None,
                 num_sms=0, num_qps=0, expert_alignment=1,
                 async_with_compute_stream=0, allocate_on_comm_stream=0,
                 do_handle_copy=1, do_cpu_sync=1,
                 do_expand=False, use_tma_aligned_col_major_sf=False,
                 do_zero_padding=False, handle=None,
                 routing_plan=None,
                 recv_rank_counts=None,
                 recv_rank_offsets=None,
                 recv_rank_prefix=None,
                 recv_expert_counts=None,
                 cumulative_local_expert_recv_stats=None,
                 previous_event=None, **kwargs):
        lib = _lib()
        use_fp8 = isinstance(x, tuple)
        x, x_sf = x if use_fp8 else (x, None)
        cached = handle is not None

        # `async_with_compute_stream` and `allocate_on_comm_stream` are accepted
        # for API compatibility and ignored: work stays ordered on the caller's
        # stream rather than moving to a private communication stream.
        if previous_event is not None:
            previous_event.current_stream_wait()
        stream = torch.cuda.current_stream().cuda_stream

        if cached:
            topk_idx = handle.topk_idx if topk_idx is None else topk_idx
            num_experts = self.num_experts
            recv_rank_offsets = handle.recv_rank_offsets
            recv_rank_prefix = handle.recv_rank_prefix
            # A cached call replays the handle's plan verbatim, so the alignment
            # it was built with is the only one that keeps the replay exact.
            expert_alignment = handle.expert_alignment
        if num_experts is None:
            num_experts = self.num_experts
            if num_experts is None:
                raise ValueError(
                    "num_experts must be passed to dispatch() or to the "
                    "ElasticBuffer constructor")

        num_tokens = x.shape[0]
        if cached and num_tokens != handle.num_tokens:
            raise ValueError(
                f"cached dispatch has {num_tokens} tokens, but its handle was "
                f"built for {handle.num_tokens}"
            )
        if num_tokens > self.num_max_tokens_per_rank:
            raise ValueError(
                f"num_tokens {num_tokens} exceeds num_max_tokens_per_rank "
                f"{self.num_max_tokens_per_rank}")

        num_topk = topk_idx.shape[1]
        self._configure(num_experts, num_topk)

        x = x.contiguous()
        x_sf = x_sf.contiguous().float() if x_sf is not None else None
        ti32 = (
            routing_plan[0]
            if routing_plan is not None
            else topk_idx.to(torch.int32).contiguous()
        )
        tw = (topk_weights if topk_weights is not None
              else torch.zeros(topk_idx.shape, dtype=torch.float32,
                               device=x.device)).to(torch.float32).contiguous()

        # Routing plan. Reused verbatim on a cached call, which is what makes the
        # replay bit-identical rather than merely equivalent.
        if cached:
            h, slot, sendc = handle, handle.slot, handle.sendc
            send_list = handle.send_list
        else:
            h = EPHandle()
            h.num_tokens = num_tokens
            h.expert_alignment = expert_alignment
            # do_handle_copy is observable: callers compare data_ptr identity.
            h.topk_idx = topk_idx.clone() if do_handle_copy else topk_idx
            h.topk_idx_i32 = ti32
            if routing_plan is None:
                ti32, slot, send_list, sendc, _ = self.plan(
                    topk_idx, num_experts, num_sms
                )
                h.topk_idx_i32 = ti32
            else:
                _, slot, send_list, sendc, _ = routing_plan
            h.slot, h.sendc, h.send_list = slot, sendc, send_list
            if recv_rank_counts is not None:
                h.recv_counts_host = (ctypes.c_int32 * self.num_ranks)(
                    *(int(value) for value in recv_rank_counts)
                )
            h.recv_rank_offsets = recv_rank_offsets
            h.recv_rank_prefix = recv_rank_prefix

        cap = self.num_ranks * self.num_max_tokens_per_rank
        hidden_sf = (self.hidden + 127) // 128
        rx = (torch.empty((cap, self.hidden), dtype=torch.float8_e4m3fn, device=x.device)
              if use_fp8 else
              torch.empty((cap, self.hidden), dtype=torch.bfloat16, device=x.device))
        rsf = torch.empty((cap, hidden_sf), dtype=torch.float32, device=x.device) if use_fp8 else None
        rtk = torch.empty((cap, num_topk), dtype=torch.int32, device=x.device)
        rtw = torch.empty((cap, num_topk), dtype=torch.float32, device=x.device)
        rsrc = torch.empty((cap,), dtype=torch.int32, device=x.device)
        epr = self.num_local_experts
        reused_counts = cached and h.expert_counts is not None
        supplied_counts = recv_expert_counts is not None and not do_expand
        if reused_counts:
            counts = h.expert_counts
            counts_list = h.num_recv_tokens_per_expert_list
        elif supplied_counts:
            counts, counts_list = recv_expert_counts
        else:
            counts = torch.zeros((epr,), dtype=torch.int32, device=x.device)
            counts_list = None

        expected_num_recv = (
            h.num_recv
            if cached
            else sum(recv_rank_counts) if recv_rank_counts is not None else -1
        )
        n = lib.ep_dispatch(self._h, x.data_ptr(), _ptr(x_sf), ti32.data_ptr(), tw.data_ptr(),
                            num_tokens, send_list.data_ptr(), sendc.data_ptr(),
                            expected_num_recv,
                            ctypes.addressof(h.recv_counts_host)
                            if h.recv_counts_host is not None else 0,
                            _ptr(h.recv_rank_offsets),
                            1 if use_fp8 else 0, num_sms,
                            rx.data_ptr(), _ptr(rsf), rtk.data_ptr(), rtw.data_ptr(),
                            rsrc.data_ptr(),
                            0 if do_expand or supplied_counts or reused_counts else counts.data_ptr(), stream)
        if n < 0:
            raise RuntimeError("ep_dispatch failed")

        rx, rsf = rx[:n], (rsf[:n] if use_fp8 else None)
        rtk, rtw, rsrc = rtk[:n], rtw[:n], rsrc[:n]
        h.num_recv, h.recv_topk_idx = n, rtk

        # Per-source-rank prefix sum.
        if h.recv_rank_prefix is not None:
            h.psum_num_recv_tokens_per_scaleup_rank = h.recv_rank_prefix
        else:
            rc = torch.empty((self.num_ranks,), dtype=torch.int32, device=x.device)
            lib.ep_recv_counts(self._h, rc.data_ptr(), stream)
            if h.recv_counts_host is None:
                h.recv_counts_host = (ctypes.c_int32 * self.num_ranks)(
                    *(int(value) for value in rc.tolist())
                )
            h.psum_num_recv_tokens_per_scaleup_rank = torch.cumsum(
                rc, 0, dtype=torch.int32
            )
            h.recv_rank_offsets = (
                h.psum_num_recv_tokens_per_scaleup_rank - rc
            )
            h.recv_rank_prefix = h.psum_num_recv_tokens_per_scaleup_rank
        h.dst_buffer_slot_idx = slot

        if not do_expand:
            h.expanded = False
            if not reused_counts:
                self._expert_metadata(h, counts, expert_alignment, counts_list)
            # recv_src_metadata: column 0 is src_token_global_idx (the only
            # column callers rely on); column 1 carries the source rank,
            # and the remaining num_topk columns are the expand map, absent here.
            meta = torch.empty((n, 2 + num_topk), dtype=torch.int32, device=x.device)
            meta[:, 0] = rsrc
            meta[:, 1] = rsrc // self.num_max_tokens_per_rank
            meta[:, 2:] = -1
            h.recv_src_metadata = meta
            if cumulative_local_expert_recv_stats is not None:
                cumulative_local_expert_recv_stats += counts
            out_x = (rx, rsf) if use_fp8 else rx
            return out_x, rtk.to(topk_idx_t), rtw, h, EventOverlap()

        # ---- expanded layout ------------------------------------------------
        offsets = torch.empty((epr,), dtype=torch.int32, device=x.device)
        psum = torch.empty((epr,), dtype=torch.int32, device=x.device)
        hist = torch.empty((max(n, 1), epr), dtype=torch.int32, device=x.device)
        row_map = torch.empty((n, num_topk), dtype=torch.int32, device=x.device)
        rows = lib.ep_expand_build(self._h, rtk.data_ptr(), n, expert_alignment,
                                   counts.data_ptr(), offsets.data_ptr(), psum.data_ptr(),
                                   hist.data_ptr(), row_map.data_ptr(), stream)
        if rows < 0:
            raise RuntimeError("ep_expand_build failed")

        ex = (torch.empty((rows, self.hidden), dtype=torch.float8_e4m3fn, device=x.device)
              if use_fp8 else
              torch.empty((rows, self.hidden), dtype=torch.bfloat16, device=x.device))
        ew = torch.empty((rows,), dtype=torch.float32, device=x.device)
        esf, sf_rs, sf_cs = None, 0, 0
        if use_fp8:
            if use_tma_aligned_col_major_sf:
                # No TMA on AMD, but the column-major layout is still what a
                # downstream GEMM wants, so it is produced rather than ignored:
                # a [hidden_sf, rows] allocation viewed transposed.
                esf = torch.empty((hidden_sf, rows), dtype=torch.float32, device=x.device).t()
                sf_rs, sf_cs = 1, rows
            else:
                esf = torch.empty((rows, hidden_sf), dtype=torch.float32, device=x.device)
                sf_rs, sf_cs = hidden_sf, 1

        lib.ep_expand_scatter(self._h, 1 if use_fp8 else 0,
                              rx.data_ptr(), _ptr(rsf), rtw.data_ptr(),
                              row_map.data_ptr(), n,
                              ex.data_ptr(), _ptr(esf), sf_rs, sf_cs, ew.data_ptr(),
                              1 if do_zero_padding else 0, expert_alignment,
                              counts.data_ptr(), offsets.data_ptr(), num_sms, stream)

        h.expanded = True
        h.row_map, h.expert_offsets, h.num_expanded_rows = row_map, offsets, rows
        self._expert_metadata(h, counts, expert_alignment)
        meta = torch.empty((n, 2 + num_topk), dtype=torch.int32, device=x.device)
        meta[:, 0] = rsrc
        meta[:, 1] = rsrc // self.num_max_tokens_per_rank
        meta[:, 2:] = row_map
        h.recv_src_metadata = meta
        if cumulative_local_expert_recv_stats is not None:
            cumulative_local_expert_recv_stats += counts
        out_x = (ex, esf) if use_fp8 else ex
        # Expanded dispatch returns no per-row top-k indices: a row IS an
        # (expert, token) pair, so the index would be a constant per run.
        return out_x, None, ew, h, EventOverlap()

    def combine(self, x, handle, topk_weights=None, bias=None,
                num_sms=0, num_qps=0, async_with_compute_stream=0,
                allocate_on_comm_stream=0, previous_event=None, **kwargs):
        if handle is None:
            raise ValueError("combine requires the handle returned by dispatch")
        lib = _lib()
        # See dispatch(): the comm-stream arguments are accepted and ignored.
        if previous_event is not None:
            previous_event.current_stream_wait()
        stream = torch.cuda.current_stream().cuda_stream
        num_tokens = handle.num_tokens
        x = x.contiguous()
        b0, b1 = (bias, None)
        if isinstance(bias, tuple):
            b0, b1 = bias
        in_w = topk_weights.contiguous().float() if topk_weights is not None else None

        out = torch.empty((num_tokens, self.hidden), dtype=torch.bfloat16, device=x.device)
        out_w = (torch.empty((num_tokens, self.num_topk), dtype=torch.float32, device=x.device)
                 if in_w is not None else None)

        # What each rank writes into the window:
        #   not expanded                      -> 1, one reduced row per token
        #   expanded, multiple reduction off  -> 0, one row per (slot, k)
        #   expanded, multiple reduction on   -> 1, reduced from the expanded
        #                                        input inside the kernel
        grouped = 1 if (not handle.expanded or self.allow_multiple_reduction) else 0
        rc = lib.ep_combine(self._h, x.data_ptr(), _ptr(in_w),
                            _ptr(handle.row_map) if handle.expanded else 0,
                            handle.recv_topk_idx.data_ptr(),
                            handle.recv_src_metadata[:, 0].contiguous().data_ptr(),
                            handle.num_recv,
                            handle.topk_idx_i32.data_ptr(), num_tokens,
                            _ptr(b0), _ptr(b1), grouped,
                            out.data_ptr(), _ptr(out_w), num_sms, stream)
        if rc != 0:
            raise RuntimeError("ep_combine failed")
        return out, out_w, EventOverlap()
