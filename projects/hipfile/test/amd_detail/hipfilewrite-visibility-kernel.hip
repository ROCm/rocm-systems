/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfilewrite-visibility-kernel.h"

#include <hip/hip_runtime.h>

namespace hipFileTest {

/// @brief Stores a generation in a workgroup's share of the buffer
///
/// See ProducerKernelArgs. When the kernel reports to the host through args.ready, every work-item
/// first releases its stores at system scope. That makes them visible to other agents only in
/// fine-grained memory: stores to coarse-grained memory are guaranteed visible outside the GPU
/// only once the kernel has finished.
///
/// @param args The kernel's arguments
static __global__ void
producePattern(ProducerKernelArgs args)
{
    __shared__ uint32_t write_state;

    if (threadIdx.x == 0) {
        write_state = static_cast<uint32_t>(WriteState::Finished);
    }

    const size_t begin{blockIdx.x * args.word_count / gridDim.x};
    const size_t end{(blockIdx.x + 1) * args.word_count / gridDim.x};

    for (size_t i{begin + threadIdx.x}; i < end; i += blockDim.x) {
        args.buffer[i] = patternValue(args.generation, i);
    }

    if (args.ready) {
        // Every work-item releases its own stores
        __threadfence_system();
        // Report ready only once every work-item has stored and released its values
        __syncthreads();
        if (threadIdx.x == 0) {
            __hip_atomic_store(&args.ready[blockIdx.x], 1U, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }

    if (args.write_state) {
        if (threadIdx.x == 0) {
            const uint64_t deadline{static_cast<uint64_t>(wall_clock64()) + args.timeout_ticks};
            for (;;) {
                write_state =
                    __hip_atomic_load(args.write_state, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                if (write_state != static_cast<uint32_t>(WriteState::InProgress) ||
                    static_cast<uint64_t>(wall_clock64()) >= deadline) {
                    break;
                }
                __builtin_amdgcn_s_sleep(127);
            }
            if (write_state == static_cast<uint32_t>(WriteState::InProgress)) {
                __hip_atomic_store(&args.timed_out[blockIdx.x], 1U, __ATOMIC_RELEASE,
                                   __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
        // Keep every work-item of the workgroup running until hipFileWrite returns
        __syncthreads();
    }
}

hipError_t
launchProducerKernel(const ProducerKernelArgs &args, unsigned int workgroups, hipStream_t stream)
{
    // Clear any error an earlier HIP call left behind, so that only the launch's
    // own error is returned. A successful hipFileRead or hipFileWrite can leave one.
    (void)hipGetLastError();
    producePattern<<<workgroups, PRODUCER_KERNEL_WORK_ITEMS, 0, stream>>>(args);
    return hipGetLastError();
}

}
