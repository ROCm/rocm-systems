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
#include "mpi_instance.hpp"
#include "bootstrap/bootstrap.hpp"
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
  uint64_t sdmaDirtyPECh{0};
  size_t sdmaThreshold{256};  // Use SDMA for transfers >= 256B
  int numChannels{1};
  int sdmaChannel{0};  // Per-context channel index (fallback / quietAll path)

  // Device resources - 2D array: [shm_size * numChannels]
  // Index as: deviceHandles_d[local_pe * numChannels + sdmaChannel]
  anvil::SdmaQueueDeviceHandle** deviceHandles_d{nullptr};
  int shm_size{0};
  int my_pe{0};
  int local_rank{0};

  // Host initialization (called from IpcOnImpl::ipcHostInit)
  __host__ void sdmaHostInit(int pe, int num_pes, MPI_Comm comm);
  __host__ void sdmaHostInit(int pe, int num_pes, TcpBootstrap* bootstrap);
  __host__ void sdmaHostStop();

  // Device-side copy with warp-affine channel selection to minimise CAS contention.
  //
  // The ctx-level sdmaChannel (= ctx_id % numChannels) already distributes blocks
  // across channels.  However, each block's warp_group has kNumWarpsPerGroup warps
  // that all call put_nbi_warp for the same destination PE — each wavefront fires
  // its own CAS into the same (sdmaChannel, dest_PE) ring, causing up to
  // kNumWarpsPerGroup-way intra-block CAS contention on top of the inter-block load.
  //
  // Offsetting by the warp index within the workgroup gives each wavefront its own
  // dedicated ring: (sdmaChannel + warp_id_in_wg) % numChannels.  With 8 warps and
  // 8 channels the intra-block contention drops to zero; inter-block contention on
  // any single ring drops from (numBlocks/numChannels * kNumWarpsPerGroup) to just
  // (numBlocks/numChannels).
  __device__ anvil::SdmaQueueDeviceHandle* sdmaCopy(void* dst, void* src,
                                                     size_t size, int local_pe) {
    // AMD wavefront is 64 threads wide (gfx9/gfx10/gfx11 with wave64 mode).
    constexpr int kWavefrontSize = 64;
    int warp_id_in_wg = static_cast<int>(threadIdx.x / kWavefrontSize);
    int effective_channel = (sdmaChannel + warp_id_in_wg) % numChannels;
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
      // Mark (local_pe, effective_channel) dirty so sdmaQuiet drains the right channel.
      uint64_t bit = 1ULL << (local_pe * numChannels + effective_channel);
      __hip_atomic_fetch_or(&sdmaDirtyPECh, bit, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    return handle;
  }

  // Wait for SDMA completions for a specific PE across all channels.
  // Atomically clears all (local_pe, ch) dirty bits and drains only the
  // channels that had pending work.  No DeepEP changes required: the
  // count-sending thread (which calls rocshmem_fence(pe)) is in the same CTA
  // as the data-sending warp, so both see the same blockIdx — but we drain all
  // channels here anyway to be safe against any channel-selection scheme.
  __device__ void sdmaQuiet(int local_pe) {
    // Build mask covering all channels for this PE.
    uint64_t pe_mask = ((1ULL << numChannels) - 1) << (local_pe * numChannels);
    uint64_t was_dirty = __hip_atomic_fetch_and(&sdmaDirtyPECh, ~pe_mask,
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
  // Iterates the sdmaDirtyPECh bitmask where each set bit corresponds to a
  // (pe, channel) pair that has a pending SDMA op.
  __device__ void sdmaQuietAll() {
    uint64_t dirty = __hip_atomic_exchange(&sdmaDirtyPECh, 0ULL, __ATOMIC_RELAXED,
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
