/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// A kernel stores to a hipMalloc buffer and tells the host it is done through
// a flag in host memory. The host waits for the flag, then stores new data in
// the buffer itself, through the pointer hipMalloc returned, and calls a
// fallback hipFileWrite: the file gets the KERNEL's data, not the host's.
//
//   usage: 08-write-after-cpu-stores-over-kernel-stores DIRECTORY [FIX]
//
// This is hipfilewrite-visibility --producer cpu --prime store --wait flag
// with fallback. It needs a GPU whose VRAM the CPU can map (large BAR).
//
// Why (cause B): hipMalloc memory is coarse-grained, which the GPU caches in
// its L2 cache. The kernel's stores are still in L2, dirty, when it sets the
// flag. The host's stores go through the BAR straight to VRAM, past L2. The
// next queue packet with a system-scope release writes L2 back to VRAM, over
// the host's data: the buffer itself is corrupted, and VRAM, read back through
// the BAR, holds the kernel's data. After each wrong iteration the program
// prints what VRAM holds.
//
// Without a fix only the FIRST iteration fails. hipFile's first fallback
// hipFileWrite in the process does something that enqueues such a release
// (with one fallback hipFileWrite of another buffer before the loop, all 50
// iterations pass), and the stress test stops at its first iteration. Any
// other release does it too: sync-after-cpu-stores, a hipStreamSynchronize
// after the host's stores, makes EVERY iteration fail. So the data is in
// danger whatever hipFile does, until something writes L2 back before the
// host stores (sync-before-cpu-stores) or the buffer bypasses L2 (uncached).
//
// By HIP's rules this is the caller's bug: it stores to coarse-grained memory
// from the host while the kernel's stores to it are not yet synchronized.
// Fastpath hipFileWrite passed this in the stress test; why its first write
// does not trigger the write-back has not been looked into. In the iterations
// that pass, the buffer is presumably corrupted afterwards, by the
// hipStreamSynchronize at the end of the iteration (inferred from
// sync-after-cpu-stores, not checked).
//
// Measured: iterations of 50 with wrong data, ROCm 10.1
//
//   FIX                      MI100 (gfx908)
//   none                     1   (the first iteration)
//   sync-before-cpu-stores   0
//   uncached                 0
//   sync-after-cpu-stores    50  (makes it worse; see above)
//   hdp-flush                1   (not a fix)

#include "repro-common.h"

#include <atomic>
#include <cstring>
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
    const repro::Args args{
        argc, argv, {"sync-before-cpu-stores", "sync-after-cpu-stores", "uncached", "hdp-flush"}};
    repro::fallbackOnly();

    const size_t      size{1 * repro::MiB};
    const size_t      word_count{size / sizeof(uint32_t)};
    const repro::File file{repro::createFile(args.directory, size)};
    uint32_t         *host{repro::hostWords(size)};
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
    auto *words{static_cast<uint32_t *>(buffer)};

    repro::Tally tally;
    for (uint32_t iteration{1}; iteration <= 50; iteration++) {
        // The kernel stores an odd value; the host then stores the even one after it
        const uint32_t kernel_value{2 * iteration - 1};
        const uint32_t cpu_value{2 * iteration};

        *done = 0;
        storeKernel<<<1, 256, 0, stream>>>(words, word_count, kernel_value, done);
        while (std::atomic_ref<uint32_t>{*done}.load(std::memory_order_acquire) == 0) {
            std::this_thread::yield();
        }

        if (args.fix == "sync-before-cpu-stores") {
            // FIX: ask HIP to wait for the kernel, which writes L2 back to VRAM
            // before the host stores.
            HIP_OK(hipStreamSynchronize(stream));
        }

        // The host stores its data through the BAR, and fences the stores
        repro::fill(host, size, cpu_value);
        std::memcpy(buffer, host, size);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        if (args.fix == "sync-after-cpu-stores") {
            // NOT A FIX: its release writes L2 back over the host's data.
            HIP_OK(hipStreamSynchronize(stream));
        }
        if (args.fix == "hdp-flush") {
            // NOT A FIX: the old data is in L2, not in the HDP.
            repro::flushHdp();
        }

        repro::transferred(hipFileWrite(file.handle, buffer, size, 0, 0), size, "hipFileWrite");
        HIP_OK(hipStreamSynchronize(stream));

        repro::readFromStorage(file, from_file, size);
        const size_t failed_before{tally.failed};
        tally.check(from_file, size, cpu_value);
        if (tally.failed != failed_before && tally.failed <= 3) {
            // What VRAM holds, read by the CPU through the BAR
            size_t cpu_words{0};
            size_t kernel_words{0};
            for (size_t i{0}; i < word_count; i++) {
                const uint32_t value{static_cast<volatile uint32_t *>(words)[i]};
                cpu_words += value == cpu_value;
                kernel_words += value == kernel_value;
            }
            printf("    VRAM, read through the BAR: %zu words hold the host's value, %zu the kernel's\n",
                   cpu_words, kernel_words);
        }
    }
    return tally.finish(args);
}
