/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * C ABI for the rccl_ep intranode kernels, consumed from Python via ctypes
 * rather than a torch extension, which keeps the binding independent of the
 * torch version. Owns the communicator and the symmetric window; every
 * per-call structure belongs to the caller, which is what makes a cached
 * replay exact.
 *
 * The stream belongs to the caller too. Every data-path entry point takes one
 * and launches on it: torch creates its streams non-blocking, so the legacy
 * null stream does NOT implicitly order against them, and a null-stream launch
 * would read inputs still being produced on the caller's stream. Setup
 * (ep_create, ep_configure) stays on the null stream -- it runs once, before
 * any caller work exists to race with.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include "device/combine.h"
#include "device/dispatch.h"
#include "device/ep_common.h"
#include "device/expand.h"
#include "device/hip_prims.h"
#include "include/ep_arch.h"
#include "include/ep_layout.h"

using namespace rccl_ep;

namespace {

struct EpHandle {
  EpConfig cfg{};
  ncclComm_t comm{};
  ncclWindow_t win{};
  void* win_buf{};
  size_t win_bytes{};
  EpWindowLayout lay{};
  ncclDevComm devComm{};
  WindowView* d_views{};
  WindowView self_view{};
  int32_t *d_recvoff{}, *d_total{};
  bool configured{false};
};

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) {          \
    fprintf(stderr, "[rccl_ep] %s:%d %s\n", __FILE__, __LINE__,          \
            hipGetErrorString(e_)); return -1; } } while (0)

inline int sms_or(int num_sms, int dflt) { return num_sms > 0 ? num_sms : dflt; }

// `num_sms` is a TOTAL CTA budget, but the dispatch grid is
// (num_ranks, ctas_per_rank), so it has to be divided down. Treating it as a
// per-rank count would quietly take num_ranks times the CUs the caller asked
// for, which defeats the entire point of the knob.
inline int ctas_per_rank(int num_sms, int dflt, int num_ranks) {
  const int total = sms_or(num_sms, dflt);
  const int per = total / num_ranks;
  return per > 0 ? per : 1;
}

// Release the symmetric window. ep_configure's failure paths run with
// `configured` still false, so ep_destroy would never reach a window they had
// already taken; they have to hand it back themselves.
inline void release_window(EpHandle* h, bool registered) {
  if (registered) ncclCommWindowDeregister(h->comm, h->win);
  if (h->win_buf) {
    ncclMemFree(h->win_buf);
    h->win_buf = nullptr;
  }
}

// Waves per block for every kernel that moves payload. Keep this equal to
// RCCL's 256-thread collective block on AMD so a fixed CTA budget also fixes
// the resident thread budget. The token loops below must provide enough work
// per wave; increasing this constant would make a nominally equal-CTA
// comparison consume more hardware than RCCL.
//
// Widening changes which wave owns which token and which lane owns which
// element, never the order of additions applied to one element, so results
// stay bit-exact.
//
constexpr int kEpWaves = 4;

}  // namespace

__global__ void k_capi_fill_views(ncclWindow_t w, EpWindowLayout l, int n, WindowView* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int p = 0; p < n; ++p) { out[p].base = (uint8_t*)ncclGetLsaPointer(w, 0, p); out[p].l = l; }
  }
}

extern "C" {

int ep_unique_id_size() { return (int)sizeof(ncclUniqueId); }

int ep_threads_per_cta() { return kEpWaves * kWarpSize; }

int ep_waves_per_cta() { return kEpWaves; }

int ep_bf16_lanes_per_token() { return kBf16TokenGroupSize; }

int ep_get_unique_id(char* out) {
  ncclUniqueId id;
  if (ncclGetUniqueId(&id) != ncclSuccess) return -1;
  memcpy(out, &id, sizeof(id));
  return 0;
}

// unique_id must be ncclUniqueId bytes broadcast from rank 0 by the caller.
void* ep_create(int rank, int num_ranks, const char* unique_id,
                int max_tokens_per_rank, int hidden, int device) {
  // Bind explicitly. The caller's torch context may have selected a different
  // device than this thread's current one.
  hipError_t he = hipSetDevice(device);
  if (he != hipSuccess) {
    fprintf(stderr, "[rccl_ep] hipSetDevice(%d): %s\n", device, hipGetErrorString(he));
    return nullptr;
  }
  // Payload copies move 16B per lane from a 256B-aligned region base, so the
  // vector paths are only aligned when hidden is a multiple of 8. The scalar
  // tails cover a leftover element count, not a misaligned base, so reject an
  // odd hidden here rather than reach a misaligned dwordx4.
  if (hidden % 8 != 0) {
    fprintf(stderr, "[rccl_ep] hidden %d must be a multiple of 8 (16B vector alignment)\n", hidden);
    return nullptr;
  }
  // kMaxRanks bounds the scale-up domain the window layout is written for, and
  // ctas_per_rank divides by num_ranks, so zero is a division by zero.
  if (num_ranks < 1 || num_ranks > kMaxRanks) {
    fprintf(stderr, "[rccl_ep] num_ranks %d must be in [1, kMaxRanks=%d]\n", num_ranks, kMaxRanks);
    return nullptr;
  }

  auto* h = new EpHandle();
  h->cfg.num_ranks = num_ranks;
  h->cfg.rank = rank;
  h->cfg.hidden = hidden;
  h->cfg.num_max_tokens_per_rank = max_tokens_per_rank;

  ncclUniqueId id;
  memcpy(&id, unique_id, sizeof(id));
  ncclResult_t nr = ncclCommInitRank(&h->comm, num_ranks, id, rank);
  if (nr != ncclSuccess) {
    fprintf(stderr, "[rccl_ep] ncclCommInitRank: %s\n", ncclGetErrorString(nr));
    delete h;
    return nullptr;
  }
  // Unchecked, a failure here returns a handle of null device pointers that
  // faults inside a kernel later, leaking the comm.
  const bool alloced =
      hipMalloc(&h->d_recvoff, num_ranks * sizeof(int32_t)) == hipSuccess &&
      hipMalloc(&h->d_total, sizeof(int32_t)) == hipSuccess &&
      hipMalloc(&h->d_views, num_ranks * sizeof(WindowView)) == hipSuccess;
  if (!alloced) {
    fprintf(stderr, "[rccl_ep] hipMalloc for scratch failed\n");
    hipFree(h->d_recvoff);
    hipFree(h->d_total);
    hipFree(h->d_views);
    ncclCommDestroy(h->comm);
    delete h;
    return nullptr;
  }
  return h;
}

// Size and register the symmetric window. Separate from ep_create because the
// window size depends on num_topk and num_experts, which the API only
// reveals at the first dispatch, not at buffer construction.
int ep_configure(void* handle, int num_experts, int num_topk) {
  auto* h = static_cast<EpHandle*>(handle);
  if (h->configured && h->cfg.num_experts == num_experts && h->cfg.num_topk == num_topk)
    return 0;
  // Validated before the teardown below, so a rejected reconfigure leaves the
  // existing window intact rather than freed.
  if (num_topk < 1 || num_topk > kMaxTopk) {
    fprintf(stderr, "[rccl_ep] num_topk %d must be in [1, kMaxTopk=%d]\n", num_topk, kMaxTopk);
    return -1;
  }
  // experts_per_rank() truncates, so a non-multiple leaves the tail experts
  // owned by no rank: k_ep_plan never routes their tokens, and the epilogue's
  // owner = e / experts_per_rank() reaches num_ranks, indexing one whole
  // rank-region past the end of `y`. num_experts <= 0 is the epr == 0 case.
  if (num_experts <= 0 || num_experts % h->cfg.num_ranks != 0) {
    fprintf(stderr, "[rccl_ep] num_experts %d must be a positive multiple of num_ranks %d\n",
            num_experts, h->cfg.num_ranks);
    return -1;
  }
  if (h->configured) {
    release_window(h, true);
    h->configured = false;
  }
  h->cfg.num_experts = num_experts;
  h->cfg.num_topk = num_topk;
  h->lay = EpWindowLayout::make(h->cfg, sizeof(bf16_t));
  h->win_bytes = h->lay.total_bytes;

  ncclResult_t nr = ncclMemAlloc(&h->win_buf, h->win_bytes);
  if (nr != ncclSuccess) {
    fprintf(stderr, "[rccl_ep] ncclMemAlloc(%zu): %s\n", h->win_bytes, ncclGetErrorString(nr));
    h->win_buf = nullptr;
    return -1;
  }
  hipMemset(h->win_buf, 0, h->win_bytes);
  nr = ncclCommWindowRegister(h->comm, h->win_buf, h->win_bytes, &h->win, NCCL_WIN_COLL_SYMMETRIC);
  if (nr != ncclSuccess) {
    fprintf(stderr, "[rccl_ep] WindowRegister: %s (NCCL_CUMEM_ENABLE=1?)\n", ncclGetErrorString(nr));
    release_window(h, false);
    return -1;
  }

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 8;   // LSA only: no GIN contexts on the intranode path
  nr = ncclDevCommCreate(h->comm, &reqs, &h->devComm);
  if (nr != ncclSuccess) {
    fprintf(stderr, "[rccl_ep] ncclDevCommCreate: %s\n", ncclGetErrorString(nr));
    release_window(h, true);
    return -1;
  }

  hipLaunchKernelGGL(k_capi_fill_views, dim3(1), dim3(1), 0, 0,
                     h->win, h->lay, h->cfg.num_ranks, h->d_views);
  // Spelled out rather than wrapped in HC: the macro returns before the window
  // can be handed back.
  hipError_t he = hipDeviceSynchronize();
  if (he != hipSuccess) {
    fprintf(stderr, "[rccl_ep] view fill: %s\n", hipGetErrorString(he));
    release_window(h, true);
    return -1;
  }
  h->self_view = WindowView{(uint8_t*)h->win_buf, h->lay};

  // Before any payload moves: a mixed-architecture communicator has no correct
  // behaviour, only a slower way to be wrong. This runs here rather than in
  // ep_create because it travels over the LSA window, which does not exist
  // until now.
  char err[256];
  if (check_uniform_arch(h->cfg, h->cfg.rank, h->d_views, h->self_view,
                         h->devComm, err, sizeof(err)) != 0) {
    fprintf(stderr, "[rccl_ep] %s\n", err);
    release_window(h, true);
    return -1;
  }

  h->configured = true;
  return 0;
}

size_t ep_window_bytes(void* handle) {
  return static_cast<EpHandle*>(handle)->win_bytes;
}

void ep_destroy(void* handle) {
  auto* h = static_cast<EpHandle*>(handle);
  if (!h) return;
  if (h->configured) release_window(h, true);
  hipFree(h->d_recvoff);
  hipFree(h->d_total);
  hipFree(h->d_views);
  ncclCommDestroy(h->comm);
  delete h;
}

int ep_barrier(void* handle, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  hipStream_t s = (hipStream_t)stream;
  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(8), dim3(kWarpSize), 0, s, h->devComm);
  HC(hipStreamSynchronize(s));
  return 0;
}

// Routing plan. Written into caller-owned tensors so a cached dispatch replays
// the identical plan rather than depending on library state.
int ep_plan(int num_ranks, int num_experts, int num_topk,
            uintptr_t topk_idx, int num_tokens,
            uintptr_t out_slot, uintptr_t out_list, uintptr_t out_sendc,
            uintptr_t out_expertc,
            int deterministic, int num_sms,
            uintptr_t stream) {
  if (num_ranks < 1 || num_ranks > kMaxRanks ||
      num_topk < 1 || num_topk > kMaxTopk ||
      num_experts <= 0 || num_experts % num_ranks != 0) {
    fprintf(stderr, "[rccl_ep] invalid plan shape: ranks=%d experts=%d topk=%d\n",
            num_ranks, num_experts, num_topk);
    return -1;
  }
  EpConfig cfg{};
  cfg.num_ranks = num_ranks;
  cfg.num_experts = num_experts;
  cfg.num_topk = num_topk;
  hipStream_t s = (hipStream_t)stream;
  HC(hipMemsetAsync((void*)out_expertc, 0, num_experts * sizeof(int32_t), s));
  if (deterministic) {
    hipLaunchKernelGGL(k_ep_plan, dim3(num_ranks), dim3(kWarpSize), 0, s,
                       cfg, (const int32_t*)topk_idx, num_tokens,
                       (int32_t*)out_slot, (int32_t*)out_list, (int32_t*)out_sendc,
                       (int32_t*)out_expertc);
  } else {
    HC(hipMemsetAsync((void*)out_sendc, 0, num_ranks * sizeof(int32_t), s));
    const int needed = (num_tokens + kEpWaves - 1) / kEpWaves;
    const int blocks = std::max(1, std::min(sms_or(num_sms, 128), needed));
    hipLaunchKernelGGL(k_ep_plan_parallel, dim3(blocks), dim3(kEpWaves * kWarpSize), 0, s,
                       cfg, (const int32_t*)topk_idx, num_tokens,
                       (int32_t*)out_slot, (int32_t*)out_list, (int32_t*)out_sendc,
                       (int32_t*)out_expertc);
  }
  return 0;
}

// Returns the number of received tokens, or -1 on error.
int ep_dispatch(void* handle,
                uintptr_t x, uintptr_t x_sf,
                uintptr_t topk_idx, uintptr_t topk_w, int num_tokens,
                uintptr_t send_list, uintptr_t sendc, int expected_num_recv,
                uintptr_t cached_recv_counts, uintptr_t cached_recv_offsets,
                int use_fp8, int num_sms,
                uintptr_t out_x, uintptr_t out_sf,
                uintptr_t out_topk, uintptr_t out_tw, uintptr_t out_src,
                uintptr_t out_expertc,
                uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  const EpConfig& c = h->cfg;
  hipStream_t s = (hipStream_t)stream;
  // The C ABI is reachable without the Python layer -- the unit tests include
  // these headers directly -- so an unconfigured handle has to say so rather
  // than fault on a null window.
  if (!h->configured) {
    fprintf(stderr, "[rccl_ep] ep_dispatch before a successful ep_configure\n");
    return -1;
  }
  // fp8 payload is 1 byte/element, so the 16B vector copies need hidden % 16 == 0;
  // ep_create's hidden % 8 guard only covers bf16 (2 bytes/element).
  if (use_fp8 && c.hidden % 16 != 0) {
    fprintf(stderr, "[rccl_ep] hidden %d must be a multiple of 16 for fp8 (16B vector alignment)\n", c.hidden);
    return -1;
  }

  // Slot index is rank * num_max_tokens_per_rank + i, with i bounded only by
  // num_tokens in k_ep_plan. Past capacity the push runs into the next rank's
  // region of the peer's window, and combine's src_tok = g % T then aliases
  // two tokens onto one slot.
  if (num_tokens > c.num_max_tokens_per_rank) {
    fprintf(stderr, "[rccl_ep] num_tokens %d exceeds num_max_tokens_per_rank %d\n",
            num_tokens, c.num_max_tokens_per_rank);
    return -1;
  }

  // 128 total CTAs is the measured bandwidth peak on gfx950; sweeping the CTA
  // budget shows dispatch flat from 32 upward on 8x MI350X.
  const int cta = ctas_per_rank(num_sms, 128, c.num_ranks);

  HC(hipMemsetAsync((uint8_t*)h->win_buf + h->lay.off_counts, 0, c.num_ranks * sizeof(int32_t), s));

  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(8), dim3(kWarpSize), 0, s, h->devComm);
  const bool known_recv_layout =
      expected_num_recv >= 0 && cached_recv_counts != 0 &&
      cached_recv_offsets != 0;
  if (use_fp8) {
    hipLaunchKernelGGL(dispatch_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, (const fp8_t*)x, (const float*)x_sf, (const int32_t*)topk_idx,
                       (const float*)topk_w, num_tokens, (const int32_t*)send_list,
                       (const int32_t*)sendc, h->d_views);
  } else {
    if (c.num_topk == 8 && c.hidden == 2048) {
      hipLaunchKernelGGL(dispatch_bf16_8x2048_impl,
                         dim3(c.num_ranks, cta),
                         dim3(kEpWaves * kWarpSize),
                         0,
                         s,
                         c,
                         (const bf16_t*)x,
                         (const int32_t*)topk_idx,
                         (const float*)topk_w,
                         num_tokens,
                         (const int32_t*)send_list,
                         (const int32_t*)sendc,
                         h->d_views);
    } else {
      hipLaunchKernelGGL(dispatch_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                         c, (const bf16_t*)x, (const int32_t*)topk_idx, (const float*)topk_w,
                         num_tokens, (const int32_t*)send_list, (const int32_t*)sendc, h->d_views);
    }
  }
  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(8), dim3(kWarpSize), 0, s, h->devComm);
  const int32_t* recv_offsets = h->d_recvoff;
  if (known_recv_layout) {
    recv_offsets = reinterpret_cast<const int32_t*>(cached_recv_offsets);
  } else {
    hipLaunchKernelGGL(k_ep_scan_counts, dim3(1), dim3(1), 0, s,
                       c, h->self_view, h->d_recvoff, h->d_total);
  }

  // Cached replay already knows the receive extent from its handle. New plans
  // still read it back to size the Python-visible tensors.
  int total = expected_num_recv;
  if (total < 0) {
    HC(hipMemcpyAsync(&total, h->d_total, sizeof(int32_t), hipMemcpyDeviceToHost, s));
    HC(hipStreamSynchronize(s));
  }

  const bool copy_payload_in_kernel = !known_recv_layout || use_fp8;
  if (!copy_payload_in_kernel) {
    if (!cached_recv_counts) {
      fprintf(stderr, "[rccl_ep] cached bf16 dispatch requires receive counts\n");
      return -1;
    }
    const auto* recv_counts = reinterpret_cast<const int32_t*>(cached_recv_counts);
    int dst_offset = 0;
    for (int src = 0; src < c.num_ranks; ++src) {
      const int count = recv_counts[src];
      const size_t src_slot = EpWindowLayout::slot(c, src, 0);
      HC(hipMemcpyAsync(
          (bf16_t*)out_x + (size_t)dst_offset * c.hidden,
          (bf16_t*)((uint8_t*)h->win_buf + h->lay.off_x) + src_slot * c.hidden,
          (size_t)count * c.hidden * sizeof(bf16_t),
          hipMemcpyDeviceToDevice, s));
      dst_offset += count;
    }
  }

  if (use_fp8) {
    hipLaunchKernelGGL(dispatch_copy_epilogue_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, h->self_view, recv_offsets, (fp8_t*)out_x, (float*)out_sf,
                       (int32_t*)out_topk, (float*)out_tw, (int32_t*)out_src, true);
  } else {
    if (copy_payload_in_kernel) {
      hipLaunchKernelGGL(dispatch_copy_epilogue_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                         c, h->self_view, recv_offsets, (bf16_t*)out_x,
                         (int32_t*)out_topk, (float*)out_tw, (int32_t*)out_src, true);
    } else {
      hipLaunchKernelGGL(dispatch_metadata_epilogue_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                         c, h->self_view, recv_offsets,
                         (int32_t*)out_topk, (float*)out_tw, (int32_t*)out_src);
    }
  }
  if (out_expertc) {
    hipLaunchKernelGGL(k_expand_count, dim3(c.experts_per_rank()), dim3(256), 0, s,
                       c, (const int32_t*)out_topk, total, (int32_t*)out_expertc);
  }
  HC(hipGetLastError());
  return total;
}

// Per-source received counts, so the caller can report the per-rank prefix sum.
int ep_recv_counts(void* handle, uintptr_t out_counts, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  hipStream_t s = (hipStream_t)stream;
  // No sync: the destination is a caller tensor read back on the same stream.
  HC(hipMemcpyAsync((void*)out_counts, (uint8_t*)h->win_buf + h->lay.off_counts,
                    h->cfg.num_ranks * sizeof(int32_t), hipMemcpyDeviceToDevice, s));
  return 0;
}

int ep_expert_counts(void* handle, uintptr_t recv_topk, int num_recv, uintptr_t out_counts,
                     uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  hipStream_t s = (hipStream_t)stream;
  hipLaunchKernelGGL(k_expand_count, dim3(h->cfg.experts_per_rank()), dim3(256), 0, s,
                     h->cfg, (const int32_t*)recv_topk, num_recv, (int32_t*)out_counts);
  HC(hipStreamSynchronize(s));
  return 0;
}

// Build the expanded-layout row map. Returns total rows (including padding).
int ep_expand_build(void* handle, uintptr_t recv_topk, int num_recv, int expert_alignment,
                    uintptr_t counts, uintptr_t offsets, uintptr_t psum,
                    uintptr_t hist, uintptr_t row_map, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  const EpConfig& c = h->cfg;
  hipStream_t s = (hipStream_t)stream;
  const int epr = c.experts_per_rank();
  const int blocks = num_recv > 0 ? (num_recv + 255) / 256 : 1;

  hipLaunchKernelGGL(k_expand_count, dim3(epr), dim3(256), 0, s,
                     c, (const int32_t*)recv_topk, num_recv, (int32_t*)counts);
  hipLaunchKernelGGL(k_expand_scan, dim3(1), dim3(1), 0, s,
                     c, expert_alignment, (const int32_t*)counts,
                     (int32_t*)offsets, (int32_t*)psum, h->d_total);
  hipLaunchKernelGGL(k_expand_hist, dim3(blocks), dim3(256), 0, s,
                     c, (const int32_t*)recv_topk, num_recv, (int32_t*)hist);
  hipLaunchKernelGGL(k_expand_base, dim3(epr), dim3(kWarpSize), 0, s,
                     c, num_recv, (int32_t*)hist);
  hipLaunchKernelGGL(k_expand_map, dim3(blocks), dim3(256), 0, s,
                     c, (const int32_t*)recv_topk, num_recv,
                     (const int32_t*)offsets, (const int32_t*)hist, (int32_t*)row_map);

  // Row count reaches the host to size the expanded tensors, so this sync is
  // structural. The copy is already ordered behind k_expand_scan on `s`.
  int total = 0;
  HC(hipMemcpyAsync(&total, h->d_total, sizeof(int32_t), hipMemcpyDeviceToHost, s));
  HC(hipStreamSynchronize(s));
  return total;
}

int ep_expand_scatter(void* handle, int use_fp8,
                      uintptr_t src_x, uintptr_t src_sf, uintptr_t src_w,
                      uintptr_t row_map, int num_recv,
                      uintptr_t out_x, uintptr_t out_sf, int sf_row_stride, int sf_col_stride,
                      uintptr_t out_w,
                      int zero_pad, int expert_alignment,
                      uintptr_t counts, uintptr_t offsets, int num_sms, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  const EpConfig& c = h->cfg;
  hipStream_t s = (hipStream_t)stream;
  const int cta = sms_or(num_sms, 64);

  if (use_fp8) {
    hipLaunchKernelGGL(k_expand_scatter_fp8, dim3(cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, (const fp8_t*)src_x, (const float*)src_sf, (const float*)src_w,
                       (const int32_t*)row_map, num_recv, (fp8_t*)out_x, (float*)out_sf,
                       sf_row_stride, sf_col_stride, (float*)out_w);
  } else {
    hipLaunchKernelGGL(k_expand_scatter, dim3(cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, (const bf16_t*)src_x, (const float*)src_w,
                       (const int32_t*)row_map, num_recv, (bf16_t*)out_x, (float*)out_w);
  }
  if (zero_pad) {
    hipLaunchKernelGGL(k_expand_zero_pad, dim3(c.experts_per_rank()), dim3(kEpWaves * kWarpSize), 0, s,
                       c, expert_alignment, (const int32_t*)counts, (const int32_t*)offsets,
                       use_fp8 ? 1 : (int)sizeof(bf16_t), (void*)out_x,
                       (float*)out_sf, sf_row_stride, sf_col_stride, (float*)out_w);
  }
  HC(hipGetLastError());
  return 0;
}

// `grouped` selects what each rank writes into the window: 1 for one reduced
// row per received token, with this rank's owned rows summed here in ascending
// k; 0 for one row per (slot, k). In either mode a non-null `row_map` means the
// caller's input is the expanded layout and is addressed through it.
int ep_combine(void* handle,
               uintptr_t y, uintptr_t in_w, uintptr_t row_map,
               uintptr_t recv_topk, uintptr_t recv_src, int num_recv,
               uintptr_t topk_idx, int num_tokens,
               uintptr_t bias0, uintptr_t bias1, int grouped,
               uintptr_t out, uintptr_t out_w, int num_sms, uintptr_t stream) {
  auto* h = static_cast<EpHandle*>(handle);
  const EpConfig& c = h->cfg;
  hipStream_t s = (hipStream_t)stream;
  if (!h->configured) {
    fprintf(stderr, "[rccl_ep] ep_combine before a successful ep_configure\n");
    return -1;
  }
  // Two different grids. The transport is (num_ranks, per_rank) so a CTA owns
  // one peer; the reduce epilogue is purely local and 1-D over tokens, so it
  // takes the whole budget. Sharing one value silently starved the epilogue of
  // CTAs and cost it 7x.
  const int cta = ctas_per_rank(num_sms, 128, c.num_ranks);
  const int cta_reduce = sms_or(num_sms, 128);

  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(8), dim3(kWarpSize), 0, s, h->devComm);
  const bool production_grouped =
      grouped && row_map == 0 && c.num_topk == 8 && c.hidden == 2048;
  if (production_grouped) {
    hipLaunchKernelGGL(combine_grouped_bf16_8x2048_impl,
                       dim3(c.num_ranks, cta),
                       dim3(kEpWaves * kWarpSize),
                       0,
                       s,
                       c,
                       (const bf16_t*)y,
                       (const float*)in_w,
                       (const int32_t*)recv_src,
                       num_recv,
                       h->self_view,
                       h->d_views);
  } else {
    hipLaunchKernelGGL(combine_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, s,
                       c, (const bf16_t*)y, (const float*)in_w, (const int32_t*)row_map,
                       (const int32_t*)recv_topk, (const int32_t*)recv_src, num_recv,
                       grouped, h->self_view, h->d_views);
  }
  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(8), dim3(kWarpSize), 0, s, h->devComm);
  if (production_grouped && bias0 == 0 && bias1 == 0) {
    hipLaunchKernelGGL(combine_reduce_grouped_bf16_8x2048_impl,
                       dim3(cta_reduce),
                       dim3(kEpWaves * kWarpSize),
                       0,
                       s,
                       c,
                       h->self_view,
                       (const int32_t*)topk_idx,
                       num_tokens,
                       (bf16_t*)out,
                       (float*)out_w);
  } else {
    hipLaunchKernelGGL(combine_reduce_epilogue_impl, dim3(cta_reduce), dim3(kEpWaves * kWarpSize), 0, s,
                       c, h->self_view, (const int32_t*)topk_idx, num_tokens,
                       (const bf16_t*)bias0, (const bf16_t*)bias1, grouped,
                       (bf16_t*)out, (float*)out_w);
  }
  HC(hipGetLastError());
  return 0;
}

}  // extern "C"
