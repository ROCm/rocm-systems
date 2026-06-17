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
  bool sdmaEnabled{true};
  // Dirty bitmask: bit [local_pe * numChannels + ch] set = channel ch has a
  // pending SDMA op to local_pe.  With up to 8 PEs × 8 channels = 64 bits.
  uint64_t sdmaDirty{0};
  size_t sdmaThreshold{256};  // Use SDMA for transfers >= 256B
  int numChannels{1};
  int sdmaChannel{0};       // Per-context base channel (= ctx_id % numChannels)
  int sdmaChannelStride{0}; // 0 = use sdmaChannel as-is; 1 = add wf_id offset

  // Device resources - 2D array: [shm_size * numChannels]
  // Index as: deviceHandles_d[local_pe * numChannels + sdmaChannel]
  anvil::SdmaQueueDeviceHandle** deviceHandles_d{nullptr};
  int shm_size{0};
  int my_pe{0};
  int local_rank{0};

  // Host initialization (called from IpcOnImpl::ipcHostInit)
  __host__ void sdmaHostInit(int pe, int num_pes, int local_rank);
  __host__ void sdmaHostStop();

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
  __device__ anvil::SdmaQueueDeviceHandle* sdmaCopy(void* dst, void* src,
                                                    size_t size, int local_pe) {
    int effective_channel = (sdmaChannel +
        sdmaChannelStride * (get_flat_block_id() / WF_SIZE)) % numChannels;
    int idx = local_pe * numChannels + effective_channel;
    anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[idx];
    if (handle != nullptr) {
      // Flush GL0/GL1 → GL2 before submitting the SDMA descriptor.
      // Fine-grain memory on AMD CDNA is CC (cache-coherent, cached in GL2):
      // the SDMA engine reads from GL2, but __syncthreads() in the caller only
      // drains stores to GL0 without flushing to GL2.  Agent scope is sufficient
      // because SDMA probes GL2 via the coherence protocol on the same die.
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      anvil::put(*handle, dst, src, size);
      // Mark (local_pe, effective_channel) dirty so sdmaQuiet drains the right
      // channel.  Blocking copies drain inline via quietAll, so the dirty bit
      // is unnecessary and would only cause a redundant poll in a later fence.
      if constexpr (!is_blocking(Kind)) {
        uint64_t bit = 1ULL << (local_pe * numChannels + effective_channel);
        __hip_atomic_fetch_or(&sdmaDirty, bit, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    return handle;
  }

  // Wait for SDMA completions for a specific PE across all channels.
  // Atomically clears all (local_pe, ch) dirty bits and drains only the
  // channels that had pending work on that context.
  // The caller must use rocshmem_ctx_fence(ctx, pe) for it to read sdmaDirty from
  // the same context (and SdmaImpl) that submitted the puts.
  __device__ void sdmaQuiet(int local_pe) {
    // Build mask covering all channels for this PE.
    uint64_t pe_mask = ((1ULL << numChannels) - 1) << (local_pe * numChannels);
    uint64_t was_dirty = __hip_atomic_fetch_and(&sdmaDirty, ~pe_mask,
                                                __ATOMIC_RELAXED,
                                                __HIP_MEMORY_SCOPE_AGENT) & pe_mask;
    if (!was_dirty) return;
    // Drain only the channels that were marked dirty.
    for (int ch = 0; ch < numChannels; ch++) {
      if (was_dirty & (1ULL << (local_pe * numChannels + ch))) {
        anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[local_pe * numChannels + ch];
        if (handle != nullptr) anvil::quiet(*handle);
      }
    }
  }

  // Wait for all SDMA completions across all PEs and channels.
  // Iterates the sdmaDirty bitmask where each set bit corresponds to a
  // (pe, channel) pair that has a pending SDMA op.
  __device__ void sdmaQuietAll() {
    uint64_t dirty = __hip_atomic_exchange(&sdmaDirty, 0ULL, __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT);
    while (dirty) {
      int bit = __builtin_ffsll(dirty) - 1;  // bit = pe * numChannels + ch
      anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[bit];
      if (handle != nullptr) {
        anvil::quiet(*handle);
      }
      dirty &= ~(1ULL << bit);
    }
  }
};
#endif  // USE_SDMA

}  // namespace rocshmem

#endif  // LIBRARY_SRC_SDMA_POLICY_HPP_
