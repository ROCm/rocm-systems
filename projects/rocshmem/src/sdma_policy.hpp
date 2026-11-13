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
  // Configuration (set from environment variables during init)
  // True only after the node-local group agreed to use SDMA; init_constant_memory() derives
  // constmem.ipc_sdma_threshold from it. Stays false if sdmaHostInit never runs.
  bool sdmaEnabled{false};
  // Dirty bitmask: bit (slot & 63) set = some slot with that residue has a pending SDMA op, where
  // slot = local_pe * numChannels + ch. Up to 64 slots this is one bit per slot; past that, slots
  // share bits and a quiet drains every slot of a set bit, which is conservative but correct.
  uint64_t sdmaDirty{0};
  size_t sdmaThreshold{256};  // Use SDMA for transfers >= 256B
  int numChannels{1};
  int sdmaChannel{0};       // Per-context base channel (= ctx_id % numChannels)
  int sdmaChannelStride{0}; // 0 = use sdmaChannel as-is; 1 = add wf_id offset

  // Device resources - 2D array: [shm_size * numChannels], indexed by the IPC pod position that
  // isIpcAvailable hands every caller, so a pod spanning several hosts stays in bounds.
  // Index as: deviceHandles_d[local_pe * numChannels + sdmaChannel]
  // A peer on this host uses Anvil's per-device queues; a peer on another host of the pod uses
  // queues of its own (Anvil remote-peer queues), written through its fabric-mapped memory.
  sdma_anvil::SdmaQueueDeviceHandle** deviceHandles_d{nullptr};
  int shm_size{0};     // IPC pod size: the slot count per channel and the agreement group
  int my_pe{0};
  int pod_rank{0};     // this rank's position in the pod
  int local_size{0};   // pod ranks on this host
  int local_rank{0};   // this rank's position among them
  // Host-side key of the queues this init connected (sdma_policy.cpp); 0 = none.
  int wiringId{0};

  // Gathers bytes_per_rank bytes from every pod rank into buf, indexed by pod rank. The caller has
  // already written its own entry at buf + pod_rank * bytes_per_rank.
  using PodAllGather = std::function<void(void* buf, int bytes_per_rank)>;

  // Host initialization (called from IpcSdmaImpl::ipcHostInit). Collective over the IPC pod: ranks
  // exchange device and host ids, then agree on the connect result, so a failure on any rank drops
  // SDMA on all of them (ranks that disagree on sdmaEnabled run incompatible alltoalls and hang).
  __host__ void sdmaHostInit(int pe, int pod_size, int pod_rank, const PodAllGather& allGather);
  __host__ void sdmaHostStop();
  // Copy this rank's queue handles to deviceHandles_d, or nulls when SDMA was dropped.
  __host__ void sdmaPublishHandles(bool useQueues);
  __host__ void sdmaReleasePeers();

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
      // Mark (local_pe, effective_channel) dirty so sdmaQuiet drains the right
      // channel.  Blocking copies drain inline via quietAll, so the dirty bit
      // is unnecessary and would only cause a redundant poll in a later fence.
      if constexpr (!is_blocking(Kind)) {
        uint64_t bit = 1ULL << (idx & 63);
        atomic::fetch_or<atomic::memory_scope::device,
                         atomic::memory_order::relaxed>(&sdmaDirty, bit);
      }
    }
    return handle;
  }

  // Dirty bits that copies to local_pe can have set.
  __device__ __forceinline__ uint64_t sdmaPeMask(int local_pe) const {
    uint64_t mask = 0;
    for (int ch = 0; ch < numChannels; ch++) mask |= 1ULL << ((local_pe * numChannels + ch) & 63);
    return mask;
  }

  // Drain every slot whose dirty bit is set in `bits`.
  __device__ void sdmaDrainBits(uint64_t bits) {
    const int numSlots = shm_size * numChannels;
    while (bits) {
      int bit = __builtin_ffsll(bits) - 1;
      for (int slot = bit; slot < numSlots; slot += 64) {
        sdma_anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[slot];
        if (handle != nullptr) sdma_anvil::quiet(*handle);
      }
      bits &= ~(1ULL << bit);
    }
  }

  // Wait for pending SDMA ops in `mask`, clearing those dirty bits.
  // The caller must use rocshmem_ctx_fence(ctx, pe) for it to read sdmaDirty from
  // the same context (and SdmaImpl) that submitted the puts.
  __device__ void sdmaQuietMask(uint64_t mask) {
    sdmaDrainBits(atomic::fetch_and<atomic::memory_scope::device,
                                    atomic::memory_order::relaxed>(&sdmaDirty, ~mask) & mask);
  }

  // Wait for SDMA completions for a specific PE across all channels.
  __device__ void sdmaQuiet(int local_pe) { sdmaQuietMask(sdmaPeMask(local_pe)); }

  // Wait for all SDMA completions across all PEs and channels.
  __device__ void sdmaQuietAll() {
    sdmaDrainBits(atomic::exchange<atomic::memory_scope::device,
                                   atomic::memory_order::relaxed>(&sdmaDirty, 0ULL));
  }
};
#endif  // USE_SDMA

}  // namespace rocshmem

#endif  // LIBRARY_SRC_SDMA_POLICY_HPP_
