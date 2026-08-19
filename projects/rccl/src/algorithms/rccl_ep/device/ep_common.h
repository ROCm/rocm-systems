/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Shared device types and primitives: the window view, the wave-wide copies,
 * and the cross-rank rendezvous.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_EP_COMMON_H_
#define RCCL_EP_COMMON_H_

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <cstdint>

#include "device/hip_prims.h"
#include <nccl_device.h>

#include "include/ep_layout.h"

namespace rccl_ep {

using bf16_t = __hip_bfloat16;
// OCP e4m3 on gfx950. gfx942 uses the fnuz variant, which is a DIFFERENT wire
// format: the two are mutually exclusive per arch in HIP's FP8 API, so a
// mixed-arch communicator must be rejected at construction rather than
// silently exchanging incompatible bytes.
using fp8_t = __hip_fp8_e4m3;

// Byte view of a rank's symmetric window plus the offsets into it.
struct WindowView {
  uint8_t* base;
  EpWindowLayout l;

  __device__ __forceinline__ int32_t* counts() const {
    return (int32_t*)(base + l.off_counts);
  }
  __device__ __forceinline__ uint32_t* flags() const {
    return (uint32_t*)(base + l.off_flags);
  }
  __device__ __forceinline__ bf16_t* x() const {
    return (bf16_t*)(base + l.off_x);
  }
  __device__ __forceinline__ int32_t* topk_idx() const {
    return (int32_t*)(base + l.off_topk_idx);
  }
  __device__ __forceinline__ float* topk_w() const {
    return (float*)(base + l.off_topk_w);
  }
  __device__ __forceinline__ int32_t* src_idx() const {
    return (int32_t*)(base + l.off_src_idx);
  }
  __device__ __forceinline__ bf16_t* y() const {
    return (bf16_t*)(base + l.off_y);
  }
  __device__ __forceinline__ float* cw() const {
    return (float*)(base + l.off_cw);
  }
  __device__ __forceinline__ fp8_t* x_fp8() const {
    return (fp8_t*)(base + l.off_x);
  }
  __device__ __forceinline__ float* sf() const {
    return (float*)(base + l.off_sf);
  }
  __device__ __forceinline__ char* arch() const {
    return (char*)(base + l.off_arch);
  }
};

// Largest top-k position of token `i` that this rank owns, or -1 if none. It is
// the position at which the origin folds this rank's partial sum, which is what
// keeps that accumulation in strict top-k index order.
__device__ __forceinline__ int last_local_slot(const int32_t* __restrict__ recv_topk, size_t i, int num_topk) {
  for (int k = num_topk - 1; k >= 0; --k)
    if (recv_topk[i * num_topk + k] >= 0) return k;
  return -1;
}

// Vectorised copy of `n` bf16 elements, one wave cooperating. Uses dwordx4
// (8 bf16) per lane, the widest store AMDGCN offers and the bulk-copy path for
// token payloads.
//
// Needs both pointers 16B aligned, which reduces to hidden being a multiple of
// 8; ep_create enforces it. The scalar loop below is a tail for a leftover
// element count, NOT a fallback for a misaligned base.
__device__ __forceinline__ void wave_copy_bf16(bf16_t* __restrict__ dst, const bf16_t* __restrict__ src, int n) {
  const int lane = get_lane_idx();
  constexpr int kPerVec = 8;  // 8 * 2B = 16B = dwordx4
  const int nvec = n / kPerVec;

  const auto* s4 = reinterpret_cast<const uint4*>(src);
  auto* d4 = reinterpret_cast<uint4*>(dst);
  for (int i = lane; i < nvec; i += kWarpSize) d4[i] = s4[i];

  // scalar tail
  for (int i = nvec * kPerVec + lane; i < n; i += kWarpSize) dst[i] = src[i];
}

// Cross-rank rendezvous, device side. Must be launched with exactly
// `lsaBarrierCount` CTAs so every rank presents the same shape; a mismatch
// deadlocks. Replaces the host barrier.
__global__ void k_ep_lsa_barrier(ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier,
                                         (uint32_t)blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);
}

// Exclusive prefix sum over the per-source counts, on device, so the receive
// offsets never round-trip through the host.
__global__ void k_ep_scan_counts(EpConfig cfg, WindowView self, int32_t* __restrict__ recv_offsets,
                                 int32_t* __restrict__ total_recv) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int acc = 0;
  for (int r = 0; r < cfg.num_ranks; ++r) {
    recv_offsets[r] = acc;
    acc += ld_acquire_sys<int32_t>(&self.counts()[r]);
  }
  *total_recv = acc;
}

// FP8 dispatch. Same routing as bf16; payload is 1 byte per element and each
// token additionally carries hidden_sf per-block scales.
__device__ __forceinline__ void wave_copy_bytes(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src, int n) {
  const int lane = get_lane_idx();
  const int nvec = n / 16;
  const auto* s4 = reinterpret_cast<const uint4*>(src);
  auto* d4 = reinterpret_cast<uint4*>(dst);
  for (int i = lane; i < nvec; i += kWarpSize) d4[i] = s4[i];
  for (int i = nvec * 16 + lane; i < n; i += kWarpSize) dst[i] = src[i];
}

}  // namespace rccl_ep

#endif  // RCCL_EP_COMMON_H_
