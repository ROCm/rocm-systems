// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Checked-in gfx1250 kernels launched with AMD's extended clustered-dispatch
/// packet.  The companion host test deliberately bypasses an ordinary HIP
/// launch: cluster_count/cluster_size are part of the behavior under test.

#include <hip/hip_runtime.h>

#include <cstdint>

#define RJ_CONSAN_GFX1250_CLUSTER_CAVE                                                             \
  ".rept 512\n\t"                                                                                  \
  "s_nop 0\n\t"                                                                                    \
  ".endr\n\t"

namespace {

__device__ __forceinline__ void store_lds_word(volatile uint32_t *address, uint32_t value) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  asm volatile("ds_write_b32 %0, %1\n\t" RJ_CONSAN_GFX1250_CLUSTER_CAVE
               :
               : "v"(lds_address), "v"(value)
               : "memory");
}

__device__ __forceinline__ uint32_t load_lds_word(volatile uint32_t *address) {
  const uint32_t lds_address =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) & 0xffffu);
  uint32_t value = 0;
  asm volatile("ds_read_b32 %0, %1\n\t" RJ_CONSAN_GFX1250_CLUSTER_CAVE "s_wait_dscnt 0\n\t"
               : "=v"(value)
               : "v"(lds_address)
               : "memory");
  return value;
}

template <bool Synchronize> __device__ __forceinline__ void cluster_barrier() {
  if constexpr (Synchronize) {
    // Cluster barrier id -3. Keep the architectural instructions visible
    // rather than relying on a compiler builtin or host-side mnemonic check.
    asm volatile(".long 0xbe804ec3\n\t" // s_barrier_signal -3
                 ".long 0xbf94fffd\n\t" // s_barrier_wait -3
                 RJ_CONSAN_GFX1250_CLUSTER_CAVE ::
                     : "memory");
  } else {
    asm volatile(RJ_CONSAN_GFX1250_CLUSTER_CAVE ::: "memory");
  }
}

template <bool Synchronize>
__device__ void local_cluster_publication(uint32_t *observed, uint32_t *control,
                                          uint32_t marker_base) {
  __shared__ uint32_t payload;
  const uint32_t tid = threadIdx.x;
  const uint32_t group = blockIdx.x;
  if (tid == 0)
    store_lds_word(&payload, marker_base | group);
  cluster_barrier<Synchronize>();
  if (tid == 32)
    observed[group] = load_lds_word(&payload);
  control[group * blockDim.x + tid] = marker_base | (group << 8u) | tid;
}

__device__ __forceinline__ void cluster_load_async_b32(uint32_t lds_address, uint32_t global_offset,
                                                       const uint32_t *global_base,
                                                       uint32_t multicast_mask) {
  asm volatile("s_mov_b32 m0, %3\n\t"
               "cluster_load_async_to_lds_b32 %0, %1, %2\n\t"
               :
               : "v"(lds_address), "v"(global_offset), "s"(global_base), "s"(multicast_mask)
               : "memory");
}

__device__ __forceinline__ void wait_async_loads() {
  asm volatile("s_wait_asynccnt 0\n\t" RJ_CONSAN_GFX1250_CLUSTER_CAVE ::: "memory");
}

template <bool Synchronize>
__device__ void tdm_publication(const uint32_t *input, uint32_t *observed, uint32_t *control,
                                bool multicast) {
  __shared__ uint32_t staged[32];
  __shared__ uint32_t publication;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t group = blockIdx.x;

  if (tid < 32u) {
    const uint32_t lds_address =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&staged[lane]) & 0xffffu);
    const uint32_t *source = multicast ? input : input + group * 32u;
    cluster_load_async_b32(lds_address, lane * sizeof(uint32_t), source, multicast ? 0x3u : 0u);
    wait_async_loads();
    // Validate the direct-to-LDS transfer in the issuing wave. The cluster
    // publication below separately checks cross-wave/CTA ordering.
    observed[group * 32u + lane] = load_lds_word(&staged[lane]);
  }
  if (tid == 0u)
    store_lds_word(&publication, 0x7d000000u | group);
  cluster_barrier<Synchronize>();
  uint32_t control_value = (multicast ? 0x5c000000u : 0x5d000000u) | (group << 8u) | tid;
  if (tid == 32u) {
    const uint32_t published = load_lds_word(&publication);
    if constexpr (Synchronize) {
      control_value ^= published ^ (0x7d000000u | group);
    } else {
      // Keep the unguarded read in the incorrect workload without making its
      // intentionally nondeterministic value part of the numerical oracle.
      asm volatile("" : : "v"(published) : "memory");
    }
  }
  control[group * blockDim.x + tid] = control_value;
}

} // namespace

extern "C" __global__ void consan_gfx1250_cluster_sync_correct(const uint32_t *, uint32_t *observed,
                                                               uint32_t *control) {
  local_cluster_publication<true>(observed, control, 0x51000000u);
}

extern "C" __global__ void
consan_gfx1250_cluster_sync_incorrect(const uint32_t *, uint32_t *observed, uint32_t *control) {
  local_cluster_publication<false>(observed, control, 0x51000000u);
}

extern "C" __global__ void
consan_gfx1250_multi_cluster_correct(const uint32_t *, uint32_t *observed, uint32_t *control) {
  local_cluster_publication<true>(observed, control, 0x52000000u);
}

extern "C" __global__ void
consan_gfx1250_multi_cluster_incorrect(const uint32_t *, uint32_t *observed, uint32_t *control) {
  local_cluster_publication<false>(observed, control, 0x52000000u);
}

extern "C" __global__ void consan_gfx1250_tdm_wait_correct(const uint32_t *input,
                                                           uint32_t *observed, uint32_t *control) {
  tdm_publication<true>(input, observed, control, false);
}

extern "C" __global__ void
consan_gfx1250_tdm_wait_incorrect(const uint32_t *input, uint32_t *observed, uint32_t *control) {
  tdm_publication<false>(input, observed, control, false);
}

extern "C" __global__ void consan_gfx1250_cluster_multicast_correct(const uint32_t *input,
                                                                    uint32_t *observed,
                                                                    uint32_t *control) {
  tdm_publication<true>(input, observed, control, true);
}

extern "C" __global__ void consan_gfx1250_cluster_multicast_incorrect(const uint32_t *input,
                                                                      uint32_t *observed,
                                                                      uint32_t *control) {
  tdm_publication<false>(input, observed, control, true);
}
