/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// A kernel is already running when new data arrives in its fine-grained
// buffer. The host stores a flag in host memory once the data is in, and the
// kernel waits for the flag, acquires it at system scope, and reads the
// buffer: some 128-byte lines still hold the OLD data.
//
//   usage: 09-fastpath-read-during-kernel DIRECTORY [FIX]
//          09-cpu-stores-during-kernel    DIRECTORY [FIX]
//          09-fallback-read-during-kernel DIRECTORY [FIX]
//
// PRODUCER chooses what puts the new data in the buffer:
//
// 0 (09-fastpath-read-during-kernel): a fastpath hipFileRead of 4 KiB. This is
//   hipfileread-gpu-visibility --kernel during --prime --buffer-size 4K.
// 1 (09-cpu-stores-during-kernel): the CPU, storing through the BAR and then
//   fencing. No hipFile, no drive and no peer-to-peer: this fails sooner than
//   the drive does.
// 2 (09-fallback-read-during-kernel): a fallback hipFileRead of 1 MiB, which
//   is the stress test's default size. On RX 9070 (gfx1201) this HANGS the
//   amdgpu driver, after which TTM workers stay in D state, rocminfo hangs,
//   and the node needs a reboot: the stress test did it on two nodes with
//   ROCm 7.2, and this program on a third with ROCm 10.1 (amdgpu 6.16.13):
//   it never finished, and was killed after 900 s. No fix is known. The
//   program refuses to run unless REPRO_MAY_HANG_DRIVER=1 is in the
//   environment, and run-all.sh skips it. The stress test passed this on every
//   other GPU tested; this program has only run on the RX 9070.
//
// Why (PRODUCER 0 and 1, strong suspicion): on RX 9070 (gfx1201) in
// Threadripper PRO 7955WX (Zen 4) hosts, within a few iterations; the same
// GPU and driver pass in a Threadripper PRO 5975WX (Zen 3) host. Also, far
// less often (about 1 iteration in 5000 with a fastpath read), on RX 6800 XT
// (gfx1030) in a Threadripper 3960X (Zen 2) host, with the same fixes; 11 is
// the form that fails quickly there. On that GPU what follows does NOT fit
// the measurements: 11 shows data of less than 4 KiB being held for 3 to 4 us
// on its way into VRAM, as by a write combiner (see 11 and README.md). The
// suspicion below has not been checked against that on the RX 9070.
// The kernel learns of the flag by
// reading host memory, and the completion of that read reaches the GPU ahead
// of posted writes into VRAM that were sent before it. PCIe forbids that
// unless the completion has Relaxed Ordering or ID-Based Ordering set
// (PCIe 6.0, table 2-42, rules D2a and D2b). The stale lines become right
// about a microsecond later with no new acquire, so the data is late, not
// cached. Anything that makes the posted writes land before the flag cures it:
//
// - bar-read: FIX. The CPU reads one word of the buffer through the BAR. A
//   read request may not pass posted writes, so its completion pushes them
//   ahead of it. Reading a different VRAM allocation works too.
// - vram-flag: FIX. The flag is in VRAM, stored through the BAR after the
//   data. Data and flag are both posted writes to the GPU, which stay in
//   order.
// - wait-5us, wait-100us: the host spins 5 or 100 us before the flag. See
//   README.md for which producer each cures.
// - uncached: NOT A FIX. The buffer bypasses the GPU's L2 cache.
// - hdp-flush: NOT A FIX. The host writes the HDP's memory flush register
//   before the flag.
//
// Kernels launched after hipFileRead returns are not affected, so this needs a
// kernel that is running during the transfer and learns of it through host
// memory. Any HIP program that writes VRAM from the CPU and signals a running
// kernel that way is affected, so hipFile cannot fix it alone.
//
// Measured: see README.md.

#include "repro-common.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

// 0: fastpath hipFileRead, 1: CPU stores, 2: fallback hipFileRead
#ifndef PRODUCER
#define PRODUCER 0
#endif

#if PRODUCER == 2
constexpr size_t BUFFER_SIZE{1 * repro::MiB};
constexpr size_t MAX_ITERATIONS{2000};
#else
constexpr size_t BUFFER_SIZE{4 * repro::KiB};
constexpr size_t MAX_ITERATIONS{200000};
#endif
// Stop after this many wrong iterations or seconds
constexpr size_t MAX_FAILED{10};
constexpr auto   MAX_TIME{std::chrono::seconds{60}};
constexpr int    WORKGROUPS{32};
constexpr int    THREADS{256};

// Each workgroup reads its slice of the buffer (as the stress test's --prime
// does), reports that it is running, waits for the flag, and counts the words
// of its slice that differ from the new value
static __global__ void
waitThenCheck(const uint32_t *buffer, size_t word_count, const uint32_t *flag, uint32_t *ready,
              uint32_t expected, uint32_t *wrong, uint32_t *first_wrong)
{
    __shared__ uint32_t seen;
    const size_t        begin{blockIdx.x * word_count / gridDim.x};
    const size_t        end{(blockIdx.x + 1) * word_count / gridDim.x};

    uint32_t primed{0};
    for (size_t i{begin + threadIdx.x}; i < end; i += blockDim.x) {
        primed += buffer[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __hip_atomic_store(&ready[blockIdx.x], 1U, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        while (__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == 0) {
            __builtin_amdgcn_s_sleep(127);
        }
        seen = 0;
    }
    __syncthreads();
    // Every thread acquires the flag for itself before it reads
    (void)__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);

    uint32_t count{0};
    for (size_t i{begin + threadIdx.x}; i < end; i += blockDim.x) {
        const uint32_t value{buffer[i]};
        if (value != expected) {
            count++;
            (void)atomicCAS_system(first_wrong, UINT32_MAX, value);
        }
    }
    atomicAdd(&seen, count);
    __syncthreads();
    if (threadIdx.x == 0) {
        // Keeps the priming loads from being optimized away
        atomicAdd_system(wrong, seen + (primed == 0xdeadbeef ? 1U : 0U));
    }
}

int
main(int argc, char **argv)
{
    const repro::Args args{
        argc, argv, {"bar-read", "vram-flag", "wait-5us", "wait-100us", "uncached", "hdp-flush"}};
#if PRODUCER == 0
    repro::fastpathOnly();
#elif PRODUCER == 2
    repro::fallbackOnly();
    if (!getenv("REPRO_MAY_HANG_DRIVER")) {
        repro::die("this hung the amdgpu driver on RX 9070 (gfx1201); set REPRO_MAY_HANG_DRIVER=1 to run it");
    }
#endif
    // Without this, ROCr gives PCIe GPUs coarse-grained memory for
    // hipDeviceMallocFinegrained. Set before the first HIP call.
    setenv("HSA_FORCE_FINE_GRAIN_PCIE", "1", 1);

    const size_t word_count{BUFFER_SIZE / sizeof(uint32_t)};
    void        *buffer{nullptr};
    void        *host_memory{nullptr};
    void        *vram_flag{nullptr};
    hipStream_t  stream{nullptr};
    HIP_OK(hipExtMallocWithFlags(
        &buffer, BUFFER_SIZE, args.fix == "uncached" ? hipDeviceMallocUncached : hipDeviceMallocFinegrained));
    // The host flag, the ready flags and the wrong count, in host memory both sides see
    HIP_OK(hipHostMalloc(&host_memory, 4 * repro::KiB, hipHostMallocCoherent));
    // FIX vram-flag: the flag in VRAM instead, which the CPU stores through the BAR
    HIP_OK(hipExtMallocWithFlags(&vram_flag, sizeof(uint32_t), hipDeviceMallocUncached));
    HIP_OK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    auto *words{static_cast<uint32_t *>(buffer)};
    auto *host_flag{static_cast<uint32_t *>(host_memory)};
    auto *ready{host_flag + 64};
    auto *wrong{host_flag + 128};
    auto *first_wrong{host_flag + 192};
    auto *flag{args.fix == "vram-flag" ? static_cast<uint32_t *>(vram_flag) : host_flag};

    // Two generations, 1 and 2, alternate. The file holds generation 1 at
    // offset 0 and generation 2 after it.
    uint32_t *host{repro::hostWords(2 * BUFFER_SIZE)};
    repro::fill(host, BUFFER_SIZE, 1);
    repro::fill(host + word_count, BUFFER_SIZE, 2);
#if PRODUCER != 1
    const repro::File file{repro::createFile(args.directory, 2 * BUFFER_SIZE)};
    repro::writeToStorage(file, host, 2 * BUFFER_SIZE);
#endif
    std::memcpy(buffer, host + word_count, BUFFER_SIZE);
    HIP_OK(hipDeviceSynchronize());

    const auto   start{std::chrono::steady_clock::now()};
    repro::Tally tally;
    for (size_t iteration{0}; iteration < MAX_ITERATIONS && tally.failed < MAX_FAILED &&
                              std::chrono::steady_clock::now() - start < MAX_TIME;
         iteration++) {
        const uint32_t generation{static_cast<uint32_t>(iteration % 2 + 1)};
        *wrong       = 0;
        *first_wrong = UINT32_MAX;
        for (int i{0}; i < WORKGROUPS; i++) {
            ready[i] = 0;
        }
        *static_cast<volatile uint32_t *>(flag) = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);

        waitThenCheck<<<WORKGROUPS, THREADS, 0, stream>>>(words, word_count, flag, ready, generation, wrong,
                                                          first_wrong);
        for (int i{0}; i < WORKGROUPS; i++) {
            while (std::atomic_ref<uint32_t>{ready[i]}.load(std::memory_order_acquire) == 0) {
            }
        }

        // The new data arrives while the kernel runs
#if PRODUCER == 1
        std::memcpy(buffer, host + (generation - 1) * word_count, BUFFER_SIZE);
        std::atomic_thread_fence(std::memory_order_seq_cst);
#else
        repro::transferred(hipFileRead(file.handle, buffer, BUFFER_SIZE, (generation - 1) * BUFFER_SIZE, 0),
                           BUFFER_SIZE, "hipFileRead");
#endif

        if (args.fix == "bar-read") {
            // FIX: a read through the BAR pushes the posted writes ahead of it.
            (void)static_cast<volatile uint32_t *>(buffer)[0];
        }
        if (args.fix == "wait-5us" || args.fix == "wait-100us") {
            // Gives the posted writes time to land
            const auto until{std::chrono::steady_clock::now() +
                             std::chrono::microseconds{args.fix == "wait-5us" ? 5 : 100}};
            while (std::chrono::steady_clock::now() < until) {
            }
        }
        if (args.fix == "hdp-flush") {
            // NOT A FIX: the data is not waiting in the HDP.
            repro::flushHdp();
        }

        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (args.fix == "vram-flag") {
            *static_cast<volatile uint32_t *>(flag) = 1;
        }
        else {
            std::atomic_ref<uint32_t>{*flag}.store(1, std::memory_order_release);
        }
        HIP_OK(hipStreamSynchronize(stream));
        tally.record(*wrong, word_count, *first_wrong, generation);
    }
    printf("ran for %.1f s\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return tally.finish(args);
}
