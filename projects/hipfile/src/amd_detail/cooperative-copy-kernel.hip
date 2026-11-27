/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#include "cooperative-copy.h"

#include <hip/hip_runtime.h>

namespace hipFile {

template <typename T>
__device__ static T
acquire(T *address)
{
    return __hip_atomic_load(address, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T>
__device__ static void
release(T *address, T value)
{
    __hip_atomic_store(address, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Give each lane a contiguous vector of bytes, including on unaligned file tails or
// buffer offsets. A fixed-size memcpy allows the compiler to select legal device accesses.
static __device__ void
copyBytes(unsigned char *destination, const unsigned char *source, size_t count)
{
    constexpr size_t Width   = 16;
    const size_t     vectors = count / Width;
    for (size_t i = threadIdx.x; i < vectors; i += blockDim.x) {
        __builtin_memcpy(destination + i * Width, source + i * Width, Width);
    }
    for (size_t i = vectors * Width + threadIdx.x; i < count; i += blockDim.x) {
        destination[i] = source[i];
    }
}

// Exactly one workgroup. No part of the handshake waits for an unscheduled workgroup.
static __global__ void
copyChunks(CopyExchange *exchange, unsigned char *buffer, bool read, uint64_t ticket)
{
    if (acquire(&exchange->ticket) != ticket) {
        return;
    }
    if (threadIdx.x == 0) {
        release(&exchange->resident, ticket);
    }
    __shared__ bool stop;
    for (unsigned index = 0;; index ^= 1U) {
        auto &slot = exchange->slots[index];
        if (threadIdx.x == 0) {
            while (!acquire(&slot.requested) && !acquire(&exchange->stop)) {
                __builtin_amdgcn_s_sleep(1);
            }
            stop = acquire(&exchange->stop) != 0;
        }
        __syncthreads();
        if (stop) {
            break;
        }
        auto *payload = buffer + slot.offset;
        if (read) {
            copyBytes(payload, slot.data.data(), slot.count);
        }
        else {
            copyBytes(slot.data.data(), payload, slot.count);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            release(&slot.requested, 0U);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        release(&exchange->stopped, 1U);
        while (!acquire(&exchange->finished)) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
}

const void *
cooperativeCopyKernel()
{
    return reinterpret_cast<const void *>(copyChunks);
}

}
