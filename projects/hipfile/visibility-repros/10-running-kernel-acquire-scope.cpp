/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// A kernel that stays alive across transfers reads the buffer, waits for a
// flag the host stores in host memory once new data is in the buffer, and
// reads the buffer again. Unless it does an acquire of a wide enough scope
// between the flag and the read, it reads the OLD data, from the GPU's caches.
//
//   usage: 10-fastpath-read-seen-by-running-kernel DIRECTORY [FIX]
//          10-cpu-stores-seen-by-running-kernel    DIRECTORY [FIX]
//          10-fallback-read-seen-by-running-kernel DIRECTORY [FIX]
//          and each of the three with -uncached, -coarse, or -16M after its
//          name
//
// This is hipfileread-gpu-watcher-visibility --acquire wavefront, and FIX
// chooses the acquire: acquire-workgroup, acquire-agent, or acquire-system.
// The kernel and the host's side of it are in long-running-kernel.h.
//
// PRODUCER chooses what puts the new data in the buffer: 0, a fastpath
// hipFileRead; 1, the CPU, storing through the BAR; 2, a fallback
// hipFileRead. MEMORY chooses the buffer: 0, fine-grained; 1, uncached; 2,
// coarse-grained (hipMalloc). SIZE_MIB is the size of the buffer and of the
// transfer.
//
// Why: this is HIP's memory model at work, not a hipFile bug. A kernel may
// keep what it has read in the GPU's caches until it acquires, and data that
// did not come from the kernel's own GPU needs a system-scope acquire.
// Nothing can acquire on a running kernel's behalf: hipFile cannot, and a
// queue packet reaches only kernels launched after it. Which narrower
// acquire happens to be enough depends on which cache holds the old line,
// and so on the GPU:
//
// - no acquire: FAILS on both GPUs, for every producer and kind of memory.
// - acquire-workgroup: enough on RX 6800 XT in fine-grained and uncached
//   memory. NOT enough on MI300X, where it compiles to no instruction.
// - acquire-agent, acquire-system: enough on both GPUs, except in
//   coarse-grained memory on RX 6800 XT (below).
// - Uncached memory is NOT a fix: it bypasses L2, and the kernel still holds
//   the old line in a cache nearer to it.
// - Coarse-grained memory on RX 6800 XT, with a fastpath read or CPU stores:
//   NO acquire is enough, and the system-scope acquire that starts the next
//   iteration does not show the data either. This is cause B: L2 holds the
//   old lines of coarse-grained memory, and only a queue packet invalidates
//   it. HIP does not promise that a running kernel sees writes to
//   coarse-grained memory from outside the GPU. A fallback read is seen with
//   acquire-agent, as the GPU itself does the copy.
//
// When an iteration fails, the program also prints how many of the stale
// watchers read the new data on a later read with the same acquire, and how
// many watchers still read old data after the system-scope acquire that
// starts the next iteration. Old data that a system-scope acquire shows was
// in VRAM all along, hidden by a cache.
//
// A pass without a wide enough acquire is luck, and two kinds of luck are
// known:
//
// - pinned-copy, unpinned-copy (fallback programs only, with no acquire): on
//   MI300X, a fallback read of more than 1 MiB is seen WITHOUT any acquire.
//   HIP pins the host buffer of a hipMemcpy that large, and we infer that
//   pinning changes the process's GPU page tables, which drops what the
//   kernel has cached. pinned-copy makes HIP pin for every size
//   (GPU_PINNED_MIN_XFER_SIZE=0), and the 1 MiB read passes; unpinned-copy
//   stops it (GPU_PINNED_MIN_XFER_SIZE=64), and the 16 MiB read fails. On RX
//   6800 XT neither changes the verdict.
// - SPACING_WORDS: with the watchers' words a power of two apart, many
//   watchers read the new data without an acquire: on RX 6800 XT, with CPU
//   stores of 1 MiB, all but 1 to 42 of the 1024. See spacingOf().
//
// The host sleeps 1 ms between the data and the flag, so that data that is
// merely late (09 and 11) is not what fails here. The sleep changes no count:
// 0, 1 and 10 ms gave the same results.
//
// Measured: see README.md.

#include "long-running-kernel.h"

#include <chrono>
#include <cstring>
#include <thread>

// 0: fastpath hipFileRead, 1: CPU stores, 2: fallback hipFileRead
#ifndef PRODUCER
#define PRODUCER 0
#endif
// 0: fine-grained, 1: uncached, 2: coarse-grained
#ifndef MEMORY
#define MEMORY 0
#endif
#ifndef SIZE_MIB
#define SIZE_MIB 1
#endif
#ifndef WORKGROUPS
#define WORKGROUPS 32
#endif
#ifndef WATCHERS_PER_WORKGROUP
#define WATCHERS_PER_WORKGROUP 32
#endif

constexpr size_t   BUFFER_SIZE{SIZE_MIB * repro::MiB};
constexpr size_t   WORD_COUNT{BUFFER_SIZE / sizeof(uint32_t)};
constexpr uint32_t WATCHERS{WORKGROUPS * WATCHERS_PER_WORKGROUP};
constexpr uint32_t ITERATIONS{20};
// Every iteration has data no other iteration has, so that a watcher that
// never sees any new data is wrong in every iteration
constexpr uint32_t GENERATIONS{ITERATIONS + 2};
// Distance between two neighbouring watchers' words, in 32-bit words
#ifdef SPACING_WORDS
constexpr size_t SPACING{SPACING_WORDS};
#else
constexpr size_t SPACING{repro::spacingOf(WORD_COUNT, WATCHERS)};
#endif
static_assert(SPACING >= 16 && SPACING * (WATCHERS - 1) < WORD_COUNT,
              "every watcher needs a line of its own, inside the buffer");
static_assert(WORKGROUPS <= repro::MAX_WORKGROUPS);

int
main(int argc, char **argv)
{
    const repro::Args args{argc,
                           argv,
                           {"acquire-workgroup", "acquire-agent", "acquire-system",
#if PRODUCER == 2
                            "pinned-copy", "unpinned-copy"
#endif
                           }};
    const int scope{args.fix == "acquire-workgroup" ? repro::ACQUIRE_WORKGROUP
                    : args.fix == "acquire-agent"   ? repro::ACQUIRE_AGENT
                    : args.fix == "acquire-system"  ? repro::ACQUIRE_SYSTEM
                                                    : repro::ACQUIRE_NONE};
    // Without this, ROCr gives PCIe GPUs coarse-grained memory for
    // hipDeviceMallocFinegrained. Set before the first HIP call.
    setenv("HSA_FORCE_FINE_GRAIN_PCIE", "1", 1);
    if (args.fix == "pinned-copy" || args.fix == "unpinned-copy") {
        // NOT fixes. HIP pins the host buffer of a hipMemcpy larger than this
        // many MiB. Set before the first HIP call.
        setenv("GPU_PINNED_MIN_XFER_SIZE", args.fix == "pinned-copy" ? "0" : "64", 1);
    }
#if PRODUCER == 0
    repro::fastpathOnly();
#elif PRODUCER == 2
    repro::fallbackOnly();
    repro::refuseFallbackWhereItHung();
#endif

    void       *buffer{nullptr};
    void       *host_memory{nullptr};
    hipStream_t stream{nullptr};
#if MEMORY == 2
    HIP_OK(hipMalloc(&buffer, BUFFER_SIZE));
#else
    HIP_OK(hipExtMallocWithFlags(&buffer, BUFFER_SIZE,
                                 MEMORY == 1 ? hipDeviceMallocUncached : hipDeviceMallocFinegrained));
#endif
    HIP_OK(hipHostMalloc(&host_memory, sizeof(repro::Shared), hipHostMallocCoherent));
    HIP_OK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    auto     *shared{new (host_memory) repro::Shared{}};
    uint32_t *host{repro::hostWords(BUFFER_SIZE)};
#if PRODUCER != 1
    const repro::File file{repro::createFile(args.directory, BUFFER_SIZE)};
#endif

    // Puts an iteration's data in every word of the buffer
    const auto produce{[&](uint32_t iteration) {
        repro::fill(host, BUFFER_SIZE, repro::dataOf(iteration, GENERATIONS));
#if PRODUCER == 1
        std::memcpy(buffer, host, BUFFER_SIZE);
        std::atomic_thread_fence(std::memory_order_seq_cst);
#else
        repro::writeToStorage(file, host, BUFFER_SIZE);
        repro::transferred(hipFileRead(file.handle, buffer, BUFFER_SIZE, 0, 0), BUFFER_SIZE, "hipFileRead");
#endif
    }};

    // The data iteration 1 replaces, in the buffer before the kernel starts
    produce(0);
    HIP_OK(hipDeviceSynchronize());
    repro::watch<<<WORKGROUPS, WATCHERS_PER_WORKGROUP, 0, stream>>>(
        static_cast<uint32_t *>(buffer), SPACING, GENERATIONS, shared, &shared->arrived, scope);

    // One start more than there are iterations: the watchers' system-scope
    // acquire at the start of iteration i + 1 tells what became of iteration
    // i's data.
    repro::Tally tally;
    bool         failed{false};
    for (uint32_t iteration{1}; iteration <= ITERATIONS + 1; iteration++) {
        repro::tell(&shared->started, iteration);
        repro::waitForWorkgroups(shared, WORKGROUPS, &repro::Workgroup::primed, iteration);
        if (failed && tally.failed <= 3) {
            printf(
                "    after the system-scope acquire that starts the next iteration, %u watchers still read "
                "old data\n",
                repro::found(shared, WORKGROUPS).unseen);
        }

        if (iteration <= ITERATIONS) {
            // The new data arrives while the kernel runs
            produce(iteration);
            // Not a fix: it only rules out data that is late (see 11)
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        repro::tell(&shared->arrived, iteration);
        repro::waitForWorkgroups(shared, WORKGROUPS, &repro::Workgroup::finished, iteration);
        if (iteration > ITERATIONS) {
            break;
        }

        const repro::Found found{repro::found(shared, WORKGROUPS)};
        tally.record(found.stale, WATCHERS, found.observed, repro::dataOf(iteration, GENERATIONS));
        failed = found.stale != 0;
        if (failed && tally.failed <= 3) {
            printf("    %u of the %u watchers that read old data read the new data on a later read\n",
                   found.late, found.stale);
        }
    }
    repro::tell(&shared->started, repro::STOP);
    HIP_OK(hipStreamSynchronize(stream));
    return tally.finish(args);
}
