/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// A kernel stores to a hipMalloc buffer and tells the host it is done through
// a flag in host memory. The host waits for the flag, sleeps another 50 ms,
// and calls a fastpath hipFileWrite: the file gets what the buffer held BEFORE
// the kernel. The host never asks HIP to synchronize.
//
//   usage: 03-write-after-kernel-without-sync DIRECTORY [FIX]
//
// Why: hipMalloc memory is coarse-grained, which the GPU caches in its L2
// cache. The kernel's stores stay in L2 until a queue packet with a
// system-scope release writes them back to VRAM, which hipStreamSynchronize
// enqueues. Fastpath's NVMe read of VRAM bypasses L2. A sleep does not help,
// however long: the data is not on its way.
//
// By HIP's rules this is the caller's bug (coarse-grained memory is only up to
// date for other agents at synchronized kernel boundaries), but nothing in
// hipFileWrite's documentation says the caller must synchronize first, and
// the fallback backend does not need it.
//
// Measured: iterations of 50 with wrong data, in two runs
//
//   FIX           RX 6800 XT (gfx1030)   MI300X (gfx942)
//   none          50, 50                 0, 0
//   stream-sync   0, 0                   0, 0
//   uncached      0, 0                   0, 0
//   hdp-flush     50, 50                 0, 0   (not a fix)

#include "repro-common.h"

#include <chrono>
#include <thread>

// Stores a value in every word of the buffer, then sets a flag in host memory
static __global__ void
storeKernel(uint32_t *buffer, size_t word_count, uint32_t value, uint32_t *done)
{
    for (size_t i{threadIdx.x}; i < word_count; i += blockDim.x) {
        buffer[i] = value;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __hip_atomic_store(done, 1U, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

int
main(int argc, char **argv)
{
    const repro::Args args{argc, argv, {"stream-sync", "uncached", "hdp-flush"}};
    repro::fastpathOnly();

    const size_t      size{1 * repro::MiB};
    const repro::File file{repro::createFile(args.directory, size)};
    uint32_t         *from_file{repro::hostWords(size)};
    void             *buffer{nullptr};
    void             *done_memory{nullptr};
    hipStream_t       stream{nullptr};
    if (args.fix == "uncached") {
        // FIX: memory that bypasses the GPU's L2 cache.
        HIP_OK(hipExtMallocWithFlags(&buffer, size, hipDeviceMallocUncached));
    }
    else {
        HIP_OK(hipMalloc(&buffer, size));
    }
    HIP_OK(hipHostMalloc(&done_memory, sizeof(uint32_t), hipHostMallocCoherent));
    HIP_OK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    auto *done{static_cast<uint32_t *>(done_memory)};

    repro::Tally tally;
    for (uint32_t iteration{1}; iteration <= 50; iteration++) {
        *done = 0;
        storeKernel<<<1, 256, 0, stream>>>(static_cast<uint32_t *>(buffer), size / sizeof(uint32_t),
                                           iteration, done);

        // Wait for the kernel without telling HIP, then long after it
        while (std::atomic_ref<uint32_t>{*done}.load(std::memory_order_acquire) == 0) {
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{50});

        if (args.fix == "stream-sync") {
            // FIX: ask HIP to wait for the kernel's stream, which also writes
            // L2 back to VRAM. hipFile could do this at the start of a
            // fastpath write, if it knew the stream; hipDeviceSynchronize
            // covers every stream.
            HIP_OK(hipStreamSynchronize(stream));
        }
        if (args.fix == "hdp-flush") {
            // NOT A FIX: the data is in L2, not in the HDP.
            repro::flushHdp();
        }

        repro::transferred(hipFileWrite(file.handle, buffer, size, 0, 0), size, "hipFileWrite");

        repro::readFromStorage(file, from_file, size);
        tally.check(from_file, size, iteration);
        HIP_OK(hipStreamSynchronize(stream));
    }
    return tally.finish(args);
}
