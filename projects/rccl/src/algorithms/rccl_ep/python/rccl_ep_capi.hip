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
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <nccl.h>
#include <nccl_device.h>

#include "device/hip_prims.h"
#include "device/ep_common.h"
#include "device/dispatch.h"
#include "device/combine.h"
#include "device/expand.h"
#include "include/ep_layout.h"
#include "include/ep_arch.h"

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

// Waves per block for every kernel that moves payload. The parallelism these
// kernels get is CTAs x waves, and only the CTA count follows the caller's
// num_sms budget, so a narrow block starves them at every budget below the
// default. 16 waves is 1024 threads, the gfx950 ceiling.
//
// Widening changes which wave owns which token and which lane owns which
// element, never the order of additions applied to one element, so results
// stay bit-exact.
//
// Caveat: the combine reduce is 1-D over tokens at one wave per token, so a
// wide block over-subscribes when tokens are few -- the decode regime, which
// this path does not serve.
constexpr int kEpWaves = 16;

}  // namespace

__global__ void k_capi_fill_views(ncclWindow_t w, EpWindowLayout l, int n, WindowView* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    for (int p = 0; p < n; ++p) { out[p].base = (uint8_t*)ncclGetLsaPointer(w, 0, p); out[p].l = l; }
}

extern "C" {

int ep_unique_id_size() { return (int)sizeof(ncclUniqueId); }

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
  if (h->configured) {
    ncclCommWindowDeregister(h->comm, h->win);
    ncclMemFree(h->win_buf);
    h->configured = false;
  }
  if (num_topk > kMaxTopk) {
    fprintf(stderr, "[rccl_ep] num_topk %d exceeds kMaxTopk %d\n", num_topk, kMaxTopk);
    return -1;
  }
  h->cfg.num_experts = num_experts;
  h->cfg.num_topk = num_topk;
  h->lay = EpWindowLayout::make(h->cfg, sizeof(bf16_t));
  h->win_bytes = h->lay.total_bytes;

  ncclResult_t nr = ncclMemAlloc(&h->win_buf, h->win_bytes);
  if (nr != ncclSuccess) {
    fprintf(stderr, "[rccl_ep] ncclMemAlloc(%zu): %s\n", h->win_bytes, ncclGetErrorString(nr));
    return -1;
  }
  hipMemset(h->win_buf, 0, h->win_bytes);
  nr = ncclCommWindowRegister(h->comm, h->win_buf, h->win_bytes, &h->win, NCCL_WIN_COLL_SYMMETRIC);
  if (nr != ncclSuccess) {
    fprintf(stderr, "[rccl_ep] WindowRegister: %s (NCCL_CUMEM_ENABLE=1?)\n", ncclGetErrorString(nr));
    return -1;
  }

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount = 8;   // LSA only: no GIN contexts on the intranode path
  nr = ncclDevCommCreate(h->comm, &reqs, &h->devComm);
  if (nr != ncclSuccess) {
    fprintf(stderr, "[rccl_ep] ncclDevCommCreate: %s\n", ncclGetErrorString(nr));
    return -1;
  }

  hipLaunchKernelGGL(k_capi_fill_views, dim3(1), dim3(1), 0, 0,
                     h->win, h->lay, h->cfg.num_ranks, h->d_views);
  HC(hipDeviceSynchronize());
  h->self_view = WindowView{(uint8_t*)h->win_buf, h->lay};

  // Before any payload moves: a mixed-architecture communicator has no correct
  // behaviour, only a slower way to be wrong. This runs here rather than in
  // ep_create because it travels over the LSA window, which does not exist
  // until now.
  char err[256];
  if (check_uniform_arch(h->cfg, h->cfg.rank, h->d_views, h->self_view,
                         h->devComm, err, sizeof(err)) != 0) {
    fprintf(stderr, "[rccl_ep] %s\n", err);
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
  if (h->configured) {
    ncclCommWindowDeregister(h->comm, h->win);
    ncclMemFree(h->win_buf);
  }
  hipFree(h->d_recvoff);
  hipFree(h->d_total);
  hipFree(h->d_views);
  ncclCommDestroy(h->comm);
  delete h;
}

int ep_barrier(void* handle) {
  auto* h = static_cast<EpHandle*>(handle);
  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(8), dim3(kWarpSize), 0, 0, h->devComm);
  HC(hipDeviceSynchronize());
  return 0;
}

// Routing plan. Written into caller-owned tensors so a cached dispatch replays
// the identical plan rather than depending on library state.
int ep_plan(void* handle, uintptr_t topk_idx, int num_tokens,
            uintptr_t out_slot, uintptr_t out_list, uintptr_t out_sendc) {
  auto* h = static_cast<EpHandle*>(handle);
  hipLaunchKernelGGL(k_ep_plan, dim3(h->cfg.num_ranks), dim3(kWarpSize), 0, 0,
                     h->cfg, (const int32_t*)topk_idx, num_tokens,
                     (int32_t*)out_slot, (int32_t*)out_list, (int32_t*)out_sendc);
  HC(hipDeviceSynchronize());
  return 0;
}

// Returns the number of received tokens, or -1 on error.
int ep_dispatch(void* handle,
                uintptr_t x, uintptr_t x_sf,
                uintptr_t topk_idx, uintptr_t topk_w, int num_tokens,
                uintptr_t send_list, uintptr_t sendc, int use_fp8, int num_sms,
                uintptr_t out_x, uintptr_t out_sf,
                uintptr_t out_topk, uintptr_t out_tw, uintptr_t out_src) {
  auto* h = static_cast<EpHandle*>(handle);
  const EpConfig& c = h->cfg;
  // 128 total CTAs is the measured bandwidth peak on gfx950; sweeping the CTA
  // budget shows dispatch flat from 32 upward on 8x MI350X.
  const int cta = ctas_per_rank(num_sms, 128, c.num_ranks);

  HC(hipMemset((uint8_t*)h->win_buf + h->lay.off_counts, 0, c.num_ranks * sizeof(int32_t)));

  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(8), dim3(kWarpSize), 0, 0, h->devComm);
  if (use_fp8) {
    hipLaunchKernelGGL(dispatch_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, 0,
                       c, (const fp8_t*)x, (const float*)x_sf, (const int32_t*)topk_idx,
                       (const float*)topk_w, num_tokens, (const int32_t*)send_list,
                       (const int32_t*)sendc, h->d_views);
  } else {
    hipLaunchKernelGGL(dispatch_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, 0,
                       c, (const bf16_t*)x, (const int32_t*)topk_idx, (const float*)topk_w,
                       num_tokens, (const int32_t*)send_list, (const int32_t*)sendc, h->d_views);
  }
  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(8), dim3(kWarpSize), 0, 0, h->devComm);
  hipLaunchKernelGGL(k_ep_scan_counts, dim3(1), dim3(1), 0, 0,
                     c, h->self_view, h->d_recvoff, h->d_total);
  HC(hipDeviceSynchronize());

  int total = 0;
  HC(hipMemcpy(&total, h->d_total, sizeof(int32_t), hipMemcpyDeviceToHost));

  if (use_fp8) {
    hipLaunchKernelGGL(dispatch_copy_epilogue_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, 0,
                       c, h->self_view, h->d_recvoff, (fp8_t*)out_x, (float*)out_sf,
                       (int32_t*)out_topk, (float*)out_tw, (int32_t*)out_src);
  } else {
    hipLaunchKernelGGL(dispatch_copy_epilogue_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, 0,
                       c, h->self_view, h->d_recvoff, (bf16_t*)out_x,
                       (int32_t*)out_topk, (float*)out_tw, (int32_t*)out_src);
  }
  HC(hipDeviceSynchronize());
  return total;
}

// Per-source received counts, so the caller can report the per-rank prefix sum.
int ep_recv_counts(void* handle, uintptr_t out_counts) {
  auto* h = static_cast<EpHandle*>(handle);
  HC(hipMemcpy((void*)out_counts, (uint8_t*)h->win_buf + h->lay.off_counts,
               h->cfg.num_ranks * sizeof(int32_t), hipMemcpyDeviceToDevice));
  return 0;
}

int ep_expert_counts(void* handle, uintptr_t recv_topk, int num_recv, uintptr_t out_counts) {
  auto* h = static_cast<EpHandle*>(handle);
  hipLaunchKernelGGL(k_expand_count, dim3(h->cfg.experts_per_rank()), dim3(256), 0, 0,
                     h->cfg, (const int32_t*)recv_topk, num_recv, (int32_t*)out_counts);
  HC(hipDeviceSynchronize());
  return 0;
}

// Build the expanded-layout row map. Returns total rows (including padding).
int ep_expand_build(void* handle, uintptr_t recv_topk, int num_recv, int expert_alignment,
                    uintptr_t counts, uintptr_t offsets, uintptr_t psum,
                    uintptr_t hist, uintptr_t row_map) {
  auto* h = static_cast<EpHandle*>(handle);
  const EpConfig& c = h->cfg;
  const int epr = c.experts_per_rank();
  const int blocks = num_recv > 0 ? (num_recv + 255) / 256 : 1;

  hipLaunchKernelGGL(k_expand_count, dim3(epr), dim3(256), 0, 0,
                     c, (const int32_t*)recv_topk, num_recv, (int32_t*)counts);
  hipLaunchKernelGGL(k_expand_scan, dim3(1), dim3(1), 0, 0,
                     c, expert_alignment, (const int32_t*)counts,
                     (int32_t*)offsets, (int32_t*)psum, h->d_total);
  hipLaunchKernelGGL(k_expand_hist, dim3(blocks), dim3(256), 0, 0,
                     c, (const int32_t*)recv_topk, num_recv, (int32_t*)hist);
  hipLaunchKernelGGL(k_expand_base, dim3(epr), dim3(kWarpSize), 0, 0,
                     c, num_recv, (int32_t*)hist);
  hipLaunchKernelGGL(k_expand_map, dim3(blocks), dim3(256), 0, 0,
                     c, (const int32_t*)recv_topk, num_recv,
                     (const int32_t*)offsets, (const int32_t*)hist, (int32_t*)row_map);
  HC(hipDeviceSynchronize());

  int total = 0;
  HC(hipMemcpy(&total, h->d_total, sizeof(int32_t), hipMemcpyDeviceToHost));
  return total;
}

int ep_expand_scatter(void* handle, int use_fp8,
                      uintptr_t src_x, uintptr_t src_sf, uintptr_t src_w,
                      uintptr_t row_map, int num_recv,
                      uintptr_t out_x, uintptr_t out_sf, int sf_row_stride, int sf_col_stride,
                      uintptr_t out_w,
                      int zero_pad, int expert_alignment,
                      uintptr_t counts, uintptr_t offsets, int num_sms) {
  auto* h = static_cast<EpHandle*>(handle);
  const EpConfig& c = h->cfg;
  const int cta = sms_or(num_sms, 64);

  if (use_fp8) {
    hipLaunchKernelGGL(k_expand_scatter_fp8, dim3(cta), dim3(kEpWaves * kWarpSize), 0, 0,
                       c, (const fp8_t*)src_x, (const float*)src_sf, (const float*)src_w,
                       (const int32_t*)row_map, num_recv, (fp8_t*)out_x, (float*)out_sf,
                       sf_row_stride, sf_col_stride, (float*)out_w);
  } else {
    hipLaunchKernelGGL(k_expand_scatter, dim3(cta), dim3(kEpWaves * kWarpSize), 0, 0,
                       c, (const bf16_t*)src_x, (const float*)src_w,
                       (const int32_t*)row_map, num_recv, (bf16_t*)out_x, (float*)out_w);
  }
  if (zero_pad) {
    hipLaunchKernelGGL(k_expand_zero_pad, dim3(c.experts_per_rank()), dim3(kEpWaves * kWarpSize), 0, 0,
                       c, expert_alignment, (const int32_t*)counts, (const int32_t*)offsets,
                       use_fp8 ? 1 : (int)sizeof(bf16_t), (void*)out_x,
                       (float*)out_sf, sf_row_stride, sf_col_stride, (float*)out_w);
  }
  HC(hipDeviceSynchronize());
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
               uintptr_t out, uintptr_t out_w, int num_sms) {
  auto* h = static_cast<EpHandle*>(handle);
  const EpConfig& c = h->cfg;
  // Two different grids. The transport is (num_ranks, per_rank) so a CTA owns
  // one peer; the reduce epilogue is purely local and 1-D over tokens, so it
  // takes the whole budget. Sharing one value silently starved the epilogue of
  // CTAs and cost it 7x.
  const int cta = ctas_per_rank(num_sms, 128, c.num_ranks);
  const int cta_reduce = sms_or(num_sms, 128);

  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(8), dim3(kWarpSize), 0, 0, h->devComm);
  hipLaunchKernelGGL(combine_impl, dim3(c.num_ranks, cta), dim3(kEpWaves * kWarpSize), 0, 0,
                     c, (const bf16_t*)y, (const float*)in_w, (const int32_t*)row_map,
                     (const int32_t*)recv_topk, (const int32_t*)recv_src, num_recv,
                     grouped, h->self_view, h->d_views);
  hipLaunchKernelGGL(k_ep_lsa_barrier, dim3(8), dim3(kWarpSize), 0, 0, h->devComm);
  hipLaunchKernelGGL(combine_reduce_epilogue_impl, dim3(cta_reduce), dim3(kEpWaves * kWarpSize), 0, 0,
                     c, h->self_view, (const int32_t*)topk_idx, num_tokens,
                     (const bf16_t*)bias0, (const bf16_t*)bias1, grouped,
                     (bf16_t*)out, (float*)out_w);
  HC(hipDeviceSynchronize());
  return 0;
}

}  // extern "C"
