/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_SDMA_POLICY_HPP_
#define LIBRARY_SRC_SDMA_POLICY_HPP_

#include <hip/hip_runtime.h>

#include <functional>
#include <vector>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "util.hpp"

#if defined(USE_SDMA)
#include "sdma/anvil.hpp"
#include "sdma/anvil_device.hpp"
#endif

namespace rocshmem {

#if defined(USE_SDMA)
class SdmaImpl {
 public:
  static constexpr int kMaxPoolQueues = 64;   // one sdmaDirty bit per pool queue
  static constexpr int kMaxPoolEngines = 32;

  // Configuration (set from environment variables during init)
  // True only after the IPC pod agreed to use SDMA; init_constant_memory() derives
  // constmem.ipc_sdma_threshold from it. Stays false if sdmaHostInit never runs.
  bool sdmaEnabled{false};
  // Bit q set = pool queue q has a pending non-blocking SDMA op. Tracked per queue rather than per
  // (pe, channel) so it fits one word for any pod size.
  uint64_t sdmaDirty{0};
  size_t sdmaThreshold{256};  // Use SDMA for transfers >= 256B
  int numChannels{1};
  int sdmaChannel{0};       // Per-context base channel (= ctx_id % numChannels)
  int sdmaChannelStride{0}; // 0 = use sdmaChannel as-is; 1 = add wf_id offset

  // Slot table [pod_size * numChannels], indexed by the pod position isIpcAvailable hands every
  // caller, on- or off-node: deviceHandles_d[pod_pe * numChannels + channel]. Every slot points at
  // a queue of this rank's pool, which spreads over all usable engines of its GPU. A queue can
  // write any address the GPU maps, so off-node fabric peers use SDMA too.
  sdma_anvil::SdmaQueueDeviceHandle** deviceHandles_d{nullptr};
  uint8_t* slotQueue_d{nullptr};       // pool queue index of each slot
  uint64_t* peQueueMask_d{nullptr};    // [pod_size] pool queues any slot of that PE uses
  sdma_anvil::SdmaQueueDeviceHandle** poolHandles_d{nullptr};  // [poolSize], by sdmaDirty bit

  int shm_size{0};    // node-local ranks
  int pod_size{0};    // IPC pod ranks; sizes the slot table and the agreement
  int my_pe{0};
  int local_rank{0};
  int pod_rank{0};
  // Host-side pool layout: numPoolLayers queues on each of poolEngines[0, numPoolEngines). Pool
  // queue q sits on poolEngines[q % numPoolEngines]. poolId 0 means no pool is held.
  int poolId{0};
  int numPoolEngines{0};
  int numPoolLayers{0};
  int poolEngines[kMaxPoolEngines]{};

  // Gathers one int from every rank of the IPC pod into values[0, pod_size), indexed by pod rank.
  // The caller has already written its own entry at values[pod_rank].
  using PodAllGather = std::function<void(int* values)>;

  // Host initialization (called from IpcSdmaImpl::ipcHostInit). Collective over the IPC pod, so
  // every rank that may pick the SDMA alltoall agrees on sdmaEnabled. local_pod_index[i] is the
  // pod position of node-local rank i, which runs on HIP device i.
  __host__ void sdmaHostInit(int pe, int local_size, int local_rank, int pod_size, int pod_rank,
                             const std::vector<int>& local_pod_index,
                             const PodAllGather& allGather);
  __host__ void sdmaHostStop();
  // Fill the slot table from the pool, or with nulls when SDMA was dropped. local_pod_index is
  // passed through rather than stored so SdmaImpl stays trivially device-constructible.
  __host__ void sdmaPublishHandles(int deviceId, bool useQueues,
                                   const std::vector<int>& local_pod_index);
  __host__ bool sdmaBuildPool(int deviceId);
  __host__ void sdmaReleasePool();

  // Device-side copy with optional wavefront-affine channel spreading.
  //
  // sdmaChannelStride controls the mode (set host-side in assignSdmaChannel):
  //
  // - 0 (default for per-WG contexts, ctx_id>=1): effective_channel = sdmaChannel.
  //   wg_ctx_create already distributes WGs across channels via ctx_id % numChannels;
  //   adding a wf_id offset would only reshuffle contention without reducing it.
  //
  // - 1 (default context ctx_id=0, or ROCSHMEM_SDMA_SPREAD_CHANNELS=1):
  //   effective_channel = (sdmaChannel + wf_id) % numChannels.
  //   When all WGs share one context, the offset spreads N*W wavefronts across
  //   W channels, reducing per-channel CAS contention from N*W to N.
  template <MemcpyKind Kind = MemcpyKind::Put>
  __device__ sdma_anvil::SdmaQueueDeviceHandle* sdmaCopy(void* dst, void* src,
                                                    size_t size, int local_pe) {
    int effective_channel = (sdmaChannel +
        sdmaChannelStride * (get_flat_block_id() / WF_SIZE)) % numChannels;
    int idx = local_pe * numChannels + effective_channel;
    sdma_anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[idx];
    if (handle != nullptr) {
      // Flush GL0/GL1 → GL2 before submitting the SDMA descriptor.
      // Fine-grain memory on AMD CDNA is CC (cache-coherent, cached in GL2):
      // the SDMA engine reads from GL2, but __syncthreads() in the caller only
      // drains stores to GL0 without flushing to GL2.  Agent scope is sufficient
      // because SDMA probes GL2 via the coherence protocol on the same die.
      atomic::threadfence<atomic::memory_scope::device,
                          atomic::memory_order::release>();
      sdma_anvil::put(*handle, dst, src, size);
      // Mark the pool queue dirty so a later fence/quiet drains it. Blocking copies drain inline
      // via quietAll, so the dirty bit would only cause a redundant poll in a later fence.
      if constexpr (!is_blocking(Kind)) {
        uint64_t bit = 1ULL << slotQueue_d[idx];
        atomic::fetch_or<atomic::memory_scope::device,
                         atomic::memory_order::relaxed>(&sdmaDirty, bit);
      }
    }
    return handle;
  }

  // Pool queues that copies to local_pe can have used.
  __device__ __forceinline__ uint64_t sdmaPeMask(int local_pe) const {
    return peQueueMask_d[local_pe];
  }

  // Drain the pool queues in mask that have pending work, clearing their dirty bits. A queue can
  // be shared with other PEs; draining it fully is still correct, just conservative.
  // The caller must use rocshmem_ctx_fence(ctx, pe) for it to read sdmaDirty from
  // the same context (and SdmaImpl) that submitted the puts.
  __device__ void sdmaQuietMask(uint64_t mask) {
    uint64_t was_dirty =
      atomic::fetch_and<atomic::memory_scope::device,
                        atomic::memory_order::relaxed>(&sdmaDirty, ~mask) & mask;
    while (was_dirty) {
      int bit = __builtin_ffsll(was_dirty) - 1;
      sdma_anvil::quiet(*poolHandles_d[bit]);
      was_dirty &= ~(1ULL << bit);
    }
  }

  __device__ void sdmaQuiet(int local_pe) { sdmaQuietMask(sdmaPeMask(local_pe)); }

  // Wait for all SDMA completions across every pool queue with pending work.
  __device__ void sdmaQuietAll() {
    uint64_t dirty =
      atomic::exchange<atomic::memory_scope::device,
                       atomic::memory_order::relaxed>(&sdmaDirty, 0ULL);
    while (dirty) {
      int bit = __builtin_ffsll(dirty) - 1;
      sdma_anvil::quiet(*poolHandles_d[bit]);
      dirty &= ~(1ULL << bit);
    }
  }
};
#endif  // USE_SDMA

}  // namespace rocshmem

#endif  // LIBRARY_SRC_SDMA_POLICY_HPP_
