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
#include <type_traits>

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

// The two payload types a dispatch moves; an fp8 payload carries one fp32 scale per 128 elements.
template <typename T>
constexpr bool kIsFp8 = std::is_same_v<T, fp8_t>;
static_assert(sizeof(bf16_t) == 2 && sizeof(fp8_t) == 1, "payload element sizes");

// Byte view of a rank's symmetric window plus the offsets into it.
struct WindowView {
  uint8_t* base;
  EpWindowLayout l;

  __device__ __forceinline__ int32_t* counts() const {
    return (int32_t*)(base + l.off_counts);
  }
  // The staged token payload, as the dispatch's payload type T (bf16_t or fp8_t).
  template <typename T>
  __device__ __forceinline__ T* payload() const {
    return (T*)(base + l.off_x);
  }
  __device__ __forceinline__ bf16_t* x() const {
    return payload<bf16_t>();
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
  __device__ __forceinline__ float* sf() const {
    return (float*)(base + l.off_sf);
  }
  __device__ __forceinline__ char* arch() const {
    return (char*)(base + l.off_arch);
  }
};

// Debug-only device check of a precondition the host already enforces; compiled in only with
// -DRCCL_EP_DEBUG (independent of NDEBUG), so no production build pays for it.
#if defined(RCCL_EP_DEBUG)
#define EP_DEBUG_ASSERT(cond) \
  do { \
    if (!(cond)) { \
      __builtin_trap(); \
    } \
  } while (0)
#else
#define EP_DEBUG_ASSERT(cond) ((void)0)
#endif

// Vectorised copy of `n` elements of T, one wave cooperating: dwordx4 (16 B) per lane, the widest
// store AMDGCN offers and the bulk-copy path for token payloads, then an element-wise tail for a
// leftover count. The tail is not a fallback for a misaligned base: `dst` and `src` must be 16B
// aligned. A payload row satisfies both, since ep_create requires hidden % 8 == 0 (bf16) and the
// fp8 entry points hidden % 16 == 0, so no row reaches the tail.
//
// Pushes into a peer's window use this copy: nontemporal stores are slower there.
template <typename T>
__device__ __forceinline__ void wave_copy(T* __restrict__ dst, const T* __restrict__ src, int n) {
  static_assert(16 % sizeof(T) == 0, "a dwordx4 holds whole elements");
  constexpr int kPerVec = 16 / (int)sizeof(T);
  const int lane = get_lane_idx();
  const int nvec = n / kPerVec;
  const auto* s4 = reinterpret_cast<const uint4*>(src);
  auto* d4 = reinterpret_cast<uint4*>(dst);
  for (int i = lane; i < nvec; i += kWarpSize) d4[i] = s4[i];
  for (int i = nvec * kPerVec + lane; i < n; i += kWarpSize) dst[i] = src[i];
}

// wave_copy for a LOCAL copy, in bytes, with nontemporal stores since the output is not read
// again by the kernel. kNtLoad also streams the source past L2, for a row read exactly once.
// kUnroll issues that many loads per lane before the first store; otherwise the compiler emits
// load / wait / store per vector.
using ep_v4i = int __attribute__((ext_vector_type(4)));
template <bool kNtLoad>
__device__ __forceinline__ ep_v4i ld_v4(const ep_v4i* p) {
  if constexpr (kNtLoad) return __builtin_nontemporal_load(p);
  else return *p;
}
template <bool kNtLoad, int kUnroll>
__device__ __forceinline__ void wave_copy_local_bytes(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src,
                                                      int n) {
  const int lane = get_lane_idx();
  const int nvec = n / 16;
  const auto* s4 = reinterpret_cast<const ep_v4i*>(src);
  auto* d4 = reinterpret_cast<ep_v4i*>(dst);
  int i = lane;
  if constexpr (kUnroll > 1) {
    for (; i + (kUnroll - 1) * kWarpSize < nvec; i += kUnroll * kWarpSize) {
      ep_v4i v[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) v[u] = ld_v4<kNtLoad>(s4 + i + u * kWarpSize);
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) __builtin_nontemporal_store(v[u], d4 + i + u * kWarpSize);
    }
  }
  for (; i < nvec; i += kWarpSize) __builtin_nontemporal_store(ld_v4<kNtLoad>(s4 + i), d4 + i);
  for (int j = nvec * 16 + lane; j < n; j += kWarpSize) dst[j] = src[j];
}

// wave_copy_local_bytes of `n` elements of T.
template <bool kNtLoad, int kUnroll, typename T>
__device__ __forceinline__ void wave_copy_local(T* __restrict__ dst, const T* __restrict__ src, int n) {
  wave_copy_local_bytes<kNtLoad, kUnroll>(reinterpret_cast<uint8_t*>(dst), reinterpret_cast<const uint8_t*>(src),
                                          n * (int)sizeof(T));
}

// CTA count for k_ep_lsa_barrier. Must equal the ncclDevCommRequirements::
// lsaBarrierCount set in ep_configure and be used at every launch site; a
// mismatch deadlocks the rendezvous rather than failing to compile.
constexpr int kEpBarrierCtas = 8;

// Cross-rank rendezvous, device side. Replaces the host barrier. Barrier A, before a
// push, keeps it out of a peer's window until the peer has finished reading it;
// barrier B, after the push, makes the pushed rows visible before the epilogue.
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

}  // namespace rccl_ep

#endif  // RCCL_EP_COMMON_H_
