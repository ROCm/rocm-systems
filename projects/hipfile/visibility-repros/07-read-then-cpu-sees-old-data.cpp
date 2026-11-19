/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// A fallback hipFileRead into a hipMalloc buffer, while other CPU threads keep
// reading the buffer through the pointer hipMalloc returned. After hipFileRead
// returns, the CPU reads the buffer: some words still hold what the buffer
// held BEFORE hipFileRead. This is what hipfileread-cpu-visibility checks.
//
//   usage: 07-read-then-cpu-sees-old-data DIRECTORY [FIX]
//
// Why: the CPU reads VRAM through the GPU's BAR and its host data path (HDP),
// which has a small read cache. The reader threads keep old lines in that
// cache. The fallback hipFileRead ends in a hipMemcpy, which the GPU does
// itself (the SDMA engine, or a kernel for 16 KiB or less), writing VRAM
// directly, and nothing invalidates the HDP's cache afterwards. The old data
// stays visible to the CPU for tens of milliseconds. The fastpath backend is
// not affected: the NVMe drive's writes arrive through the HDP, which
// refreshes the lines they write.
//
// It needs two or more CPUs reading at once, and a GPU whose VRAM the CPU can
// map (hipDeviceAttributeIsLargeBar).
//
// Measured: iterations of 300 with wrong data, in two runs
//
//   FIX           RX 6800 XT (gfx1030)   MI300X (gfx942)
//   none          226, 178               0, 0
//   hdp-flush     0, 0                   0, 0
//   kernel        0, 0                   0, 0
//   device-sync   226, 215               0, 0   (not a fix)

#include "repro-common.h"

#include <thread>

int
main(int argc, char **argv)
{
    const repro::Args args{argc, argv, {"hdp-flush", "kernel", "device-sync"}};
    repro::fallbackOnly();

    const size_t      size{64 * repro::KiB};
    const size_t      word_count{size / sizeof(uint32_t)};
    const repro::File file{repro::createFile(args.directory, size)};
    uint32_t         *host{repro::hostWords(size)};
    void             *buffer{nullptr};
    void             *scratch{nullptr};
    HIP_OK(hipMalloc(&buffer, size));
    HIP_OK(hipMalloc(&scratch, 4 * repro::KiB));

    int device{};
    int cpu_can_map_vram{};
    HIP_OK(hipGetDevice(&device));
    HIP_OK(hipDeviceGetAttribute(&cpu_can_map_vram, hipDeviceAttributeIsLargeBar, device));
    if (!cpu_can_map_vram) {
        repro::die("the CPU cannot map this GPU's VRAM");
    }
    // The buffer, as the CPU sees it through the BAR
    const auto *vram{static_cast<const volatile uint32_t *>(buffer)};

    // Seven threads, each reading one word of the buffer over and over
    const size_t             reader_count{7};
    std::atomic<bool>        stop{false};
    std::vector<std::thread> readers;
    for (size_t reader{0}; reader < reader_count; reader++) {
        readers.emplace_back([&stop, vram, index = reader * word_count / reader_count]() {
            while (!stop.load(std::memory_order_relaxed)) {
                (void)vram[index];
            }
        });
    }

    repro::Tally tally;
    for (uint32_t iteration{1}; iteration <= 300; iteration++) {
        repro::fill(host, size, iteration);
        repro::writeToStorage(file, host, size);

        repro::transferred(hipFileRead(file.handle, buffer, size, 0, 0), size, "hipFileRead");

        if (args.fix == "hdp-flush") {
            // FIX: flush the HDP, which also invalidates its read cache.
            // hipFile could do this at the end of a fallback read, after its
            // hipMemcpy, and ROCr after any copy into VRAM.
            repro::flushHdp();
        }
        if (args.fix == "kernel") {
            // FIX: run any kernel and wait for it. The queue packet's acquire
            // fence makes the GPU invalidate the HDP's read cache.
            HIP_OK(hipMemset(scratch, 0, 4 * repro::KiB));
            HIP_OK(hipDeviceSynchronize());
        }
        if (args.fix == "device-sync") {
            // NOT A FIX: with nothing queued, HIP enqueues nothing.
            HIP_OK(hipDeviceSynchronize());
        }

        // The CPU reads every word through the BAR
        std::atomic_thread_fence(std::memory_order_seq_cst);
        tally.check(vram, size, iteration);
    }
    stop = true;
    for (std::thread &reader : readers) {
        reader.join();
    }
    return tally.finish(args);
}
