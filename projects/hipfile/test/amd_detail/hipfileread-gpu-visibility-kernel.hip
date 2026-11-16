/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfileread-gpu-visibility-kernel.h"

#include <hip/hip_runtime.h>

namespace hipFileTest {

/// @brief Checks a workgroup's share of the buffer before and after hipFileRead
///
/// See VisibilityKernelArgs. When the kernel waits for hipFileRead, it orders its reads of the buffer
/// after hipFileRead returns with a system-scope acquire of the ReadState the host stores with release
/// semantics. That ordering guarantees visibility only for fine-grained memory: a kernel that is
/// running when another agent writes coarse-grained memory may see the old data.
///
/// @param args The kernel's arguments
static __global__ void
checkVisibility(VisibilityKernelArgs args)
{
    __shared__ uint32_t           previous_mismatches;
    __shared__ uint32_t           stale;
    __shared__ uint32_t           unexpected;
    __shared__ unsigned long long first_bad;
    __shared__ uint32_t           read_state;

    if (threadIdx.x == 0) {
        previous_mismatches = 0;
        stale               = 0;
        unexpected          = 0;
        first_bad           = UINT64_MAX;
        read_state          = static_cast<uint32_t>(ReadState::Succeeded);
    }
    __syncthreads();

    const size_t begin{blockIdx.x * args.word_count / gridDim.x};
    const size_t end{(blockIdx.x + 1) * args.word_count / gridDim.x};

    if (args.check_previous) {
        for (size_t i{begin + threadIdx.x}; i < end; i += blockDim.x) {
            if (args.buffer[i] != args.previous_base + static_cast<uint32_t>(i)) {
                atomicAdd(&previous_mismatches, 1U);
            }
        }
    }

    if (args.ready) {
        // Report ready only once every thread has read its values
        __syncthreads();
        if (threadIdx.x == 0) {
            __hip_atomic_store(&args.ready[blockIdx.x], 1U, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }

    if (args.read_state) {
        if (threadIdx.x == 0) {
            const uint64_t deadline{static_cast<uint64_t>(wall_clock64()) + args.timeout_ticks};
            for (;;) {
                read_state = __hip_atomic_load(args.read_state, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                if (read_state != static_cast<uint32_t>(ReadState::InProgress) ||
                    static_cast<uint64_t>(wall_clock64()) >= deadline) {
                    break;
                }
                __builtin_amdgcn_s_sleep(127);
            }
        }
        __syncthreads();
        // Every thread acquires for itself, so none of its reads below can be served by data cached
        // before hipFileRead returned
        (void)__hip_atomic_load(args.read_state, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    }

    if (args.check_current && read_state == static_cast<uint32_t>(ReadState::Succeeded)) {
        for (size_t i{begin + threadIdx.x}; i < end; i += blockDim.x) {
            const uint32_t observed{args.buffer[i]};
            if (observed == args.current_base + static_cast<uint32_t>(i)) {
                continue;
            }
            if (observed == args.previous_base + static_cast<uint32_t>(i)) {
                atomicAdd(&stale, 1U);
            }
            else {
                atomicAdd(&unexpected, 1U);
            }
            // The buffer has fewer than 2^32 words, so the index fits in the upper half
            atomicMin(&first_bad, (static_cast<unsigned long long>(i) << 32) | observed);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        args.results[blockIdx.x] = VisibilityResult{
            .previous_mismatches = previous_mismatches,
            .stale               = stale,
            .unexpected          = unexpected,
            .timed_out           = read_state == static_cast<uint32_t>(ReadState::InProgress) ? 1U : 0U,
            .first_bad           = first_bad,
        };
    }
}

hipError_t
launchVisibilityKernel(const VisibilityKernelArgs &args, unsigned int workgroups, hipStream_t stream)
{
    // Clear any error an earlier HIP call left behind, so that only the launch's
    // own error is returned. A successful hipFileRead can leave one.
    (void)hipGetLastError();
    checkVisibility<<<workgroups, VISIBILITY_KERNEL_THREADS, 0, stream>>>(args);
    return hipGetLastError();
}

}
