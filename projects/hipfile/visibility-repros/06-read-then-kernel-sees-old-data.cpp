/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// A kernel reads a hipMalloc buffer and tells the host it is done through a
// flag in host memory. The host then calls a fastpath hipFileRead into the
// buffer, and, after it returns, launches a second kernel that reads the
// buffer: the second kernel sees what the buffer held BEFORE hipFileRead.
//
//   usage: 06-read-then-kernel-sees-old-data DIRECTORY [FIX]
//
// Why: hipMalloc memory is coarse-grained, which the GPU caches in its L2
// cache. The first kernel leaves the buffer's old contents in L2. Fastpath's
// NVMe write goes straight to VRAM, and nothing tells the GPU to drop its
// copy: no queue packet with a system-scope acquire runs between the two
// kernels, because the host never synchronizes with HIP between them. The
// fallback backend is not affected, as the GPU itself writes the buffer.
//
// The second reader does not have to be an application's kernel. hipMemcpy to
// another GPU buffer, which HIP does with a kernel, copies the old data too:
// hipfileread-gpu-visibility --consumer copy --prime --prime-wait flag.
//
// Measured: iterations of 50 with wrong data, in two runs
//
//   FIX                      RX 6800 XT (gfx1030)   MI300X (gfx942)
//   none                     50, 50                 0, 0
//   sync-before-read         0, 0                   0, 0
//   sync-after-read          0, 0                   0, 0
//   device-sync-after-read   0, 0                   0, 0
//   uncached                 0, 0                   0, 0
//   hdp-flush                50, 50                 0, 0   (not a fix)

#include "repro-common.h"

#include <thread>

// Counts the words of the buffer that differ from a value, then sets a flag in host memory
static __global__ void
checkKernel(const uint32_t *buffer, size_t word_count, uint32_t expected, uint32_t *wrong, uint32_t *done)
{
    uint32_t count{0};
    for (size_t i{0}; i < word_count; i++) {
        count += buffer[i] != expected;
    }
    *wrong = count;
    __hip_atomic_store(done, 1U, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

int
main(int argc, char **argv)
{
    const repro::Args args{
        argc,
        argv,
        {"sync-before-read", "sync-after-read", "device-sync-after-read", "uncached", "hdp-flush"}};
    repro::fastpathOnly();

    const size_t      size{1 * repro::MiB};
    const size_t      word_count{size / sizeof(uint32_t)};
    const repro::File file{repro::createFile(args.directory, size)};
    uint32_t         *host{repro::hostWords(size)};
    void             *buffer{nullptr};
    void             *wrong_memory{nullptr};
    void             *done_memory{nullptr};
    hipStream_t       stream{nullptr};
    if (args.fix == "uncached") {
        // FIX: memory that bypasses the GPU's L2 cache.
        HIP_OK(hipExtMallocWithFlags(&buffer, size, hipDeviceMallocUncached));
    }
    else {
        HIP_OK(hipMalloc(&buffer, size));
    }
    HIP_OK(hipMalloc(&wrong_memory, 2 * sizeof(uint32_t)));
    HIP_OK(hipHostMalloc(&done_memory, sizeof(uint32_t), hipHostMallocCoherent));
    HIP_OK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    auto *wrong{static_cast<uint32_t *>(wrong_memory)};
    auto *done{static_cast<uint32_t *>(done_memory)};

    // The buffer starts with value 0
    HIP_OK(hipMemset(buffer, 0, size));
    HIP_OK(hipDeviceSynchronize());

    repro::Tally tally;
    for (uint32_t iteration{1}; iteration <= 50; iteration++) {
        // This iteration's value goes into the file; the buffer holds the previous one
        repro::fill(host, size, iteration);
        repro::writeToStorage(file, host, size);

        // The first kernel reads the buffer. The host waits for it without telling HIP.
        *done = 0;
        checkKernel<<<1, 1, 0, stream>>>(static_cast<const uint32_t *>(buffer), word_count, iteration - 1,
                                         &wrong[1], done);
        while (std::atomic_ref<uint32_t>{*done}.load(std::memory_order_acquire) == 0) {
            std::this_thread::yield();
        }

        if (args.fix == "sync-before-read") {
            // FIX: synchronize with the first kernel's stream before reading,
            // which makes the GPU drop what the kernel left in L2.
            HIP_OK(hipStreamSynchronize(stream));
        }

        repro::transferred(hipFileRead(file.handle, buffer, size, 0, 0), size, "hipFileRead");

        if (args.fix == "sync-after-read") {
            // FIX: synchronize with the first kernel's stream after reading.
            HIP_OK(hipStreamSynchronize(stream));
        }
        if (args.fix == "device-sync-after-read") {
            // FIX: synchronize with every stream after reading. hipFile could
            // do this at the end of a fastpath read.
            HIP_OK(hipDeviceSynchronize());
        }
        if (args.fix == "hdp-flush") {
            // NOT A FIX: the old data is in L2, not in the HDP.
            repro::flushHdp();
        }

        // The second kernel reads the buffer, and this time the host synchronizes
        uint32_t wrong_count{0};
        checkKernel<<<1, 1, 0, stream>>>(static_cast<const uint32_t *>(buffer), word_count, iteration,
                                         &wrong[0], done);
        HIP_OK(hipStreamSynchronize(stream));
        HIP_OK(hipMemcpy(&wrong_count, &wrong[0], sizeof(wrong_count), hipMemcpyDeviceToHost));
        tally.record(wrong_count, word_count, iteration - 1, iteration);
    }
    return tally.finish(args);
}
