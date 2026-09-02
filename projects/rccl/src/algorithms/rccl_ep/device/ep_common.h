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

#include <cstdint>

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <nccl_device.h>

#include "device/hip_prims.h"
#include "include/ep_layout.h"

namespace rccl_ep {

using bf16_t = __hip_bfloat16;
// OCP e4m3 on gfx950. gfx942 uses the fnuz variant, which is a DIFFERENT wire
// format: the two are mutually exclusive per arch in HIP's FP8 API, so a
// mixed-arch communicator must be rejected at construction rather than
// silently exchanging incompatible bytes.
using fp8_t = __hip_fp8_e4m3;

// DeepEP V2 schedules 32 independent CUDA warps per CTA at hidden=2048. A
// gfx950 wave has 64 lanes, so splitting it into eight-lane copy groups keeps
// 32 tokens in flight from a 256-thread CTA without increasing the CTA's
// hardware footprint to DeepEP's 1024 threads.
constexpr int kBf16TokenGroupSize = 8;

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
  constexpr int kUnroll = 4;
  constexpr int kLoopStride = kWarpSize * kUnroll;
  const int nvec = n / kPerVec;

  const auto* s4 = reinterpret_cast<const uint4*>(src);
  auto* d4 = reinterpret_cast<uint4*>(dst);
  int i = lane;
  for (; i < (nvec / kLoopStride) * kLoopStride; i += kLoopStride) {
    const uint4 value0 = s4[i];
    const uint4 value1 = s4[i + kWarpSize];
    const uint4 value2 = s4[i + 2 * kWarpSize];
    const uint4 value3 = s4[i + 3 * kWarpSize];
    d4[i] = value0;
    d4[i + kWarpSize] = value1;
    d4[i + 2 * kWarpSize] = value2;
    d4[i + 3 * kWarpSize] = value3;
  }
  for (; i < nvec; i += kWarpSize) d4[i] = s4[i];

  // scalar tail
  for (int i = nvec * kPerVec + lane; i < n; i += kWarpSize) dst[i] = src[i];
}

// Peer-window payloads are streaming writes: no later work on the sender
// reuses these cache lines. Bypassing the sender's caches matches RCCL's gfx950
// copy path and prevents four-wave blocks from stalling behind L2 writeback.
__device__ __forceinline__ void store16_peer(uint4* dst, uint4 value) {
  auto* p = reinterpret_cast<int32_t*>(dst);
  __builtin_nontemporal_store(static_cast<int32_t>(value.x), p + 0);
  __builtin_nontemporal_store(static_cast<int32_t>(value.y), p + 1);
  __builtin_nontemporal_store(static_cast<int32_t>(value.z), p + 2);
  __builtin_nontemporal_store(static_cast<int32_t>(value.w), p + 3);
}

__device__ __forceinline__ uint4 load16_peer(const uint4* src) {
  const auto* p = reinterpret_cast<const int32_t*>(src);
  uint4 value;
  value.x = __builtin_nontemporal_load(p + 0);
  value.y = __builtin_nontemporal_load(p + 1);
  value.z = __builtin_nontemporal_load(p + 2);
  value.w = __builtin_nontemporal_load(p + 3);
  return value;
}

__device__ __forceinline__ void wave_copy_bf16_peer(
    bf16_t* __restrict__ dst, const bf16_t* __restrict__ src, int n) {
  const int lane = get_lane_idx();
  constexpr int kPerVec = 8;
  constexpr int kUnroll = 4;
  constexpr int kLoopStride = kWarpSize * kUnroll;
  const int nvec = n / kPerVec;

  const auto* s4 = reinterpret_cast<const uint4*>(src);
  auto* d4 = reinterpret_cast<uint4*>(dst);
  int i = lane;
  for (; i < (nvec / kLoopStride) * kLoopStride; i += kLoopStride) {
    const uint4 value0 = s4[i];
    const uint4 value1 = s4[i + kWarpSize];
    const uint4 value2 = s4[i + 2 * kWarpSize];
    const uint4 value3 = s4[i + 3 * kWarpSize];
    store16_peer(d4 + i, value0);
    store16_peer(d4 + i + kWarpSize, value1);
    store16_peer(d4 + i + 2 * kWarpSize, value2);
    store16_peer(d4 + i + 3 * kWarpSize, value3);
  }
  for (; i < nvec; i += kWarpSize) store16_peer(d4 + i, s4[i]);

  for (int i = nvec * kPerVec + lane; i < n; i += kWarpSize) dst[i] = src[i];
}

template <int kGroupSize>
__device__ __forceinline__ void group_copy_bf16_peer(
    bf16_t* __restrict__ dst, const bf16_t* __restrict__ src, int n,
    int group_lane) {
  constexpr int kPerVec = 8;
  constexpr int kUnroll = 4;
  constexpr int kLoopStride = kGroupSize * kUnroll;
  const int nvec = n / kPerVec;

  const auto* s4 = reinterpret_cast<const uint4*>(src);
  auto* d4 = reinterpret_cast<uint4*>(dst);
  int i = group_lane;
  for (; i < (nvec / kLoopStride) * kLoopStride; i += kLoopStride) {
    const uint4 value0 = s4[i];
    const uint4 value1 = s4[i + kGroupSize];
    const uint4 value2 = s4[i + 2 * kGroupSize];
    const uint4 value3 = s4[i + 3 * kGroupSize];
    store16_peer(d4 + i, value0);
    store16_peer(d4 + i + kGroupSize, value1);
    store16_peer(d4 + i + 2 * kGroupSize, value2);
    store16_peer(d4 + i + 3 * kGroupSize, value3);
  }
  for (; i < nvec; i += kGroupSize) store16_peer(d4 + i, s4[i]);

  for (int i = nvec * kPerVec + group_lane; i < n; i += kGroupSize) {
    dst[i] = src[i];
  }
}

__device__ __forceinline__ void wave_copy_bf16_from_peer(
    bf16_t* __restrict__ dst, const bf16_t* __restrict__ src, int n) {
  const int lane = get_lane_idx();
  constexpr int kPerVec = 8;
  constexpr int kUnroll = 4;
  constexpr int kLoopStride = kWarpSize * kUnroll;
  const int nvec = n / kPerVec;

  const auto* s4 = reinterpret_cast<const uint4*>(src);
  auto* d4 = reinterpret_cast<uint4*>(dst);
  int i = lane;
  for (; i < (nvec / kLoopStride) * kLoopStride; i += kLoopStride) {
    const uint4 value0 = load16_peer(s4 + i);
    const uint4 value1 = load16_peer(s4 + i + kWarpSize);
    const uint4 value2 = load16_peer(s4 + i + 2 * kWarpSize);
    const uint4 value3 = load16_peer(s4 + i + 3 * kWarpSize);
    d4[i] = value0;
    d4[i + kWarpSize] = value1;
    d4[i + 2 * kWarpSize] = value2;
    d4[i + 3 * kWarpSize] = value3;
  }
  for (; i < nvec; i += kWarpSize) d4[i] = load16_peer(s4 + i);

  for (int tail = nvec * kPerVec + lane; tail < n; tail += kWarpSize) {
    dst[tail] = src[tail];
  }
}

template <int kGroupSize>
__device__ __forceinline__ void group_copy_bf16_from_peer(
    bf16_t* __restrict__ dst, const bf16_t* __restrict__ src, int n,
    int group_lane) {
  constexpr int kPerVec = 8;
  constexpr int kUnroll = 4;
  constexpr int kLoopStride = kGroupSize * kUnroll;
  const int nvec = n / kPerVec;

  const auto* s4 = reinterpret_cast<const uint4*>(src);
  auto* d4 = reinterpret_cast<uint4*>(dst);
  int i = group_lane;
  for (; i < (nvec / kLoopStride) * kLoopStride; i += kLoopStride) {
    const uint4 value0 = load16_peer(s4 + i);
    const uint4 value1 = load16_peer(s4 + i + kGroupSize);
    const uint4 value2 = load16_peer(s4 + i + 2 * kGroupSize);
    const uint4 value3 = load16_peer(s4 + i + 3 * kGroupSize);
    d4[i] = value0;
    d4[i + kGroupSize] = value1;
    d4[i + 2 * kGroupSize] = value2;
    d4[i + 3 * kGroupSize] = value3;
  }
  for (; i < nvec; i += kGroupSize) d4[i] = load16_peer(s4 + i);

  for (int tail = nvec * kPerVec + group_lane; tail < n;
       tail += kGroupSize) {
    dst[tail] = src[tail];
  }
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

__device__ __forceinline__ void wave_copy_bytes_peer(
    uint8_t* __restrict__ dst, const uint8_t* __restrict__ src, int n) {
  const int lane = get_lane_idx();
  constexpr int kUnroll = 4;
  constexpr int kLoopStride = kWarpSize * kUnroll;
  const int nvec = n / 16;
  const auto* s4 = reinterpret_cast<const uint4*>(src);
  auto* d4 = reinterpret_cast<uint4*>(dst);
  int i = lane;
  for (; i < (nvec / kLoopStride) * kLoopStride; i += kLoopStride) {
    const uint4 value0 = s4[i];
    const uint4 value1 = s4[i + kWarpSize];
    const uint4 value2 = s4[i + 2 * kWarpSize];
    const uint4 value3 = s4[i + 3 * kWarpSize];
    store16_peer(d4 + i, value0);
    store16_peer(d4 + i + kWarpSize, value1);
    store16_peer(d4 + i + 2 * kWarpSize, value2);
    store16_peer(d4 + i + 3 * kWarpSize, value3);
  }
  for (; i < nvec; i += kWarpSize) store16_peer(d4 + i, s4[i]);
  for (int i = nvec * 16 + lane; i < n; i += kWarpSize) {
    __builtin_nontemporal_store(src[i], dst + i);
  }
}

__device__ __forceinline__ void wave_copy_bytes_from_peer(
    uint8_t* __restrict__ dst, const uint8_t* __restrict__ src, int n) {
  const int lane = get_lane_idx();
  constexpr int kUnroll = 4;
  constexpr int kLoopStride = kWarpSize * kUnroll;
  const int nvec = n / 16;
  const auto* s4 = reinterpret_cast<const uint4*>(src);
  auto* d4 = reinterpret_cast<uint4*>(dst);
  int i = lane;
  for (; i < (nvec / kLoopStride) * kLoopStride; i += kLoopStride) {
    const uint4 value0 = load16_peer(s4 + i);
    const uint4 value1 = load16_peer(s4 + i + kWarpSize);
    const uint4 value2 = load16_peer(s4 + i + 2 * kWarpSize);
    const uint4 value3 = load16_peer(s4 + i + 3 * kWarpSize);
    d4[i] = value0;
    d4[i + kWarpSize] = value1;
    d4[i + 2 * kWarpSize] = value2;
    d4[i + 3 * kWarpSize] = value3;
  }
  for (; i < nvec; i += kWarpSize) d4[i] = load16_peer(s4 + i);
  for (int tail = nvec * 16 + lane; tail < n; tail += kWarpSize) {
    dst[tail] = src[tail];
  }
}

}  // namespace rccl_ep

#endif  // RCCL_EP_COMMON_H_
