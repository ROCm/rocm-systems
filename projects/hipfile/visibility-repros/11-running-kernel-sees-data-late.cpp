/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// A kernel that stays alive across transfers does everything HIP asks of it:
// its buffer is fine-grained, the host stores a flag in host memory once new
// data is in the buffer, and the kernel acquires at system scope between the
// flag and its read. A watcher still reads the OLD data, and reads the new
// data on a later read: once in a few hundred iterations when the transfer is
// 1 MiB, and in nearly every iteration when the CPU stores 64 bytes.
//
//   usage: 11-fastpath-read-late-for-running-kernel        DIRECTORY [FIX]
//          11-cpu-stores-late-for-running-kernel           DIRECTORY [FIX]
//          11-cpu-stores-64-bytes-late-for-running-kernel  DIRECTORY [FIX]
//          11-cpu-stores-4096-bytes-seen-by-running-kernel DIRECTORY [FIX]
//
// This is hipfileread-gpu-watcher-visibility --workgroups 1, with fastpath
// and with --producer bar. The kernel and the host's side of it are in
// long-running-kernel.h.
//
// PRODUCER chooses what puts the new data in the buffer: 0, a fastpath
// hipFileRead of 1 MiB; 1, the CPU, storing through the BAR and then fencing,
// with no hipFile and no drive. STORE_BYTES makes the CPU store only that
// many bytes at the start of the buffer, with a single watcher on the first
// word: the last two programs store 64 and 4096 bytes.
//
// Why (RX 6800 XT; the behavior is measured, where it happens is not): on
// the way into VRAM, something holds back a small amount of written data
// until about 3.5 us have passed or about 4 KiB have been written, as a write
// combiner with a timer would. The 64-byte program shows it alone: the store
// is late in 10 of 10 iterations. The 4096-byte program stores enough to push
// its own data through, and passes. With 1 MiB, only what is left over at the
// end is held, so the first two programs fail once in hundreds or thousands
// of iterations. The data is late, not cached: the stale watcher reads the
// new data on a later read, and the acquire is already at system scope.
//
// This is the failure 09 shows. 09's kernel is launched for each transfer,
// sleeps between its polls of the flag and loads the flag a second time
// before it reads. Here the kernel stays alive and every watcher reads its
// word right after its own read of the flag, so it fails far more often.
//
// With 1 MiB, each watcher reads the first word of one 4 KiB page of the
// buffer, in page order. Which watcher is stale follows what is written last.
// With a fastpath read it is the watcher of the last page of a 16 KiB piece
// of the transfer, between watchers 127 and 255. With CPU stores it is
// watcher 0, at the start of the buffer; we infer that memcpy stores the head
// of a large copy last.
//
// The fixes have 09's names:
//
// - bar-read: FIX. The CPU reads one word of the buffer through the BAR
//   before it stores the flag, which drains what is held. With the 64-byte
//   store a few iterations in 100000 are still late.
// - vram-flag: FIX. The flag is in VRAM, stored through the BAR after the
//   data, so it is held with the data.
// - wait-5us: the host spins 5 us before the flag. FIX for the fastpath read;
//   CPU stores are still late a few times in 20000 iterations.
// - wait-100us: FIX. The host spins 100 us before the flag. 10 us is enough.
// - uncached: NOT A FIX. The buffer bypasses the GPU's L2 cache.
// - hdp-flush: NOT A FIX. The host writes the HDP's memory flush register
//   before the flag. It does not drain what is held.
//
// The sfence after the CPU's stores rules out the CPU's own write-combining
// buffers. It changes nothing.
//
// MI300X does not fail: there a 4-byte store reaches the kernel as soon as a
// 4096-byte one. Measured: see README.md.

#include "long-running-kernel.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <immintrin.h>

// 0: fastpath hipFileRead, 1: CPU stores
#ifndef PRODUCER
#define PRODUCER 0
#endif

constexpr size_t BUFFER_SIZE{1 * repro::MiB};
// With CPU stores: the bytes at the start of the buffer the CPU stores in each
// iteration
#ifndef STORE_BYTES
#define STORE_BYTES BUFFER_SIZE
#endif
constexpr size_t STORED{STORE_BYTES};
// The data alternates between 1 and 2
constexpr uint32_t GENERATIONS{2};
constexpr uint32_t MAX_ITERATIONS{20000};
// Stop after this many wrong iterations or seconds
constexpr size_t       MAX_FAILED{10};
constexpr auto         MAX_TIME{std::chrono::seconds{60}};
constexpr unsigned int WORKGROUPS{1};
// One watcher for each 4 KiB page, or a single watcher on the buffer's first
// word when the CPU stores less than the whole buffer
constexpr unsigned int WATCHERS{STORED == BUFFER_SIZE ? 256 : 1};
static_assert(STORED >= sizeof(uint32_t) && STORED <= BUFFER_SIZE);

int
main(int argc, char **argv)
{
    const repro::Args args{
        argc, argv, {"bar-read", "vram-flag", "wait-5us", "wait-100us", "uncached", "hdp-flush"}};
    // Without this, ROCr gives PCIe GPUs coarse-grained memory for
    // hipDeviceMallocFinegrained. Set before the first HIP call.
    setenv("HSA_FORCE_FINE_GRAIN_PCIE", "1", 1);
#if PRODUCER == 0
    repro::fastpathOnly();
#endif

    const size_t word_count{BUFFER_SIZE / sizeof(uint32_t)};
    void        *buffer{nullptr};
    void        *host_memory{nullptr};
    void        *vram_flag{nullptr};
    hipStream_t  stream{nullptr};
    HIP_OK(hipExtMallocWithFlags(
        &buffer, BUFFER_SIZE, args.fix == "uncached" ? hipDeviceMallocUncached : hipDeviceMallocFinegrained));
    HIP_OK(hipHostMalloc(&host_memory, sizeof(repro::Shared), hipHostMallocCoherent));
    // vram-flag: the flag in VRAM, which the CPU stores through the BAR
    HIP_OK(hipExtMallocWithFlags(&vram_flag, sizeof(uint32_t), hipDeviceMallocUncached));
    HIP_OK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    auto *shared{new (host_memory) repro::Shared{}};
    auto *flag{args.fix == "vram-flag" ? static_cast<uint32_t *>(vram_flag) : &shared->arrived};
    *static_cast<volatile uint32_t *>(flag) = 0;

    // The data alternates between 1 and 2 in every word. The file holds the
    // 1s at offset 0 and the 2s after them.
    uint32_t *host{repro::hostWords(2 * BUFFER_SIZE)};
    repro::fill(host, BUFFER_SIZE, 1);
    repro::fill(host + word_count, BUFFER_SIZE, 2);
#if PRODUCER == 0
    const repro::File file{repro::createFile(args.directory, 2 * BUFFER_SIZE)};
    repro::writeToStorage(file, host, 2 * BUFFER_SIZE);
#endif

    // Puts data, 1 or 2, in every word of the buffer
    const auto produce{[&](uint32_t data) {
#if PRODUCER == 1
        std::memcpy(buffer, host + (data - 1) * word_count, STORED);
        // The BAR is mapped write-combining: push the stores out of the CPU
        _mm_sfence();
        std::atomic_thread_fence(std::memory_order_seq_cst);
#else
        repro::transferred(hipFileRead(file.handle, buffer, BUFFER_SIZE, (data - 1) * BUFFER_SIZE, 0),
                           BUFFER_SIZE, "hipFileRead");
#endif
    }};

    // The data iteration 1 replaces, in the buffer before the kernel starts
    produce(repro::dataOf(0, GENERATIONS));
    HIP_OK(hipDeviceSynchronize());
    repro::watch<<<WORKGROUPS, WATCHERS, 0, stream>>>(static_cast<uint32_t *>(buffer), word_count / WATCHERS,
                                                      GENERATIONS, shared, flag, repro::ACQUIRE_SYSTEM);

    const auto   start{std::chrono::steady_clock::now()};
    repro::Tally tally;
    for (uint32_t iteration{1}; iteration <= MAX_ITERATIONS && tally.failed < MAX_FAILED &&
                                std::chrono::steady_clock::now() - start < MAX_TIME;
         iteration++) {
        repro::tell(&shared->started, iteration);
        repro::waitForWorkgroups(shared, WORKGROUPS, &repro::Workgroup::primed, iteration);

        // The new data arrives while the kernel runs
        produce(repro::dataOf(iteration, GENERATIONS));

        if (args.fix == "bar-read") {
            (void)static_cast<volatile uint32_t *>(buffer)[0];
        }
        if (args.fix == "wait-5us" || args.fix == "wait-100us") {
            const auto until{std::chrono::steady_clock::now() +
                             std::chrono::microseconds{args.fix == "wait-5us" ? 5 : 100}};
            while (std::chrono::steady_clock::now() < until) {
            }
        }
        if (args.fix == "hdp-flush") {
            repro::flushHdp();
        }

        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (args.fix == "vram-flag") {
            *static_cast<volatile uint32_t *>(flag) = iteration;
        }
        else {
            repro::tell(flag, iteration);
        }
        repro::waitForWorkgroups(shared, WORKGROUPS, &repro::Workgroup::finished, iteration);

        const repro::Found found{repro::found(shared, WORKGROUPS)};
        tally.record(found.stale, WATCHERS, found.observed, repro::dataOf(iteration, GENERATIONS));
        if (found.stale != 0 && tally.failed <= 3) {
            printf(
                "    the first of them is watcher %u of %u; %u of them read the new data on a later read\n",
                found.lowest, WATCHERS, found.late);
        }
    }
    repro::tell(&shared->started, repro::STOP);
    HIP_OK(hipStreamSynchronize(stream));
    printf("ran for %.1f s\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return tally.finish(args);
}
