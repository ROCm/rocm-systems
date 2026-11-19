/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// hipMemcpy of 1 MiB from host memory into a GPU buffer, then a fastpath
// hipFileWrite of the buffer's first 4 KiB: the file gets what those 4 KiB
// held BEFORE hipMemcpy. Single-threaded, every call synchronous.
//
//   usage: 04-write-small-after-large-hipmemcpy DIRECTORY [FIX]
//
// Why: fastpath's NVMe read of VRAM goes through the GPU's host data path
// (HDP), which has a small read cache (about 16 KiB on gfx1030). The previous
// iteration's hipFileWrite left these 4 KiB in that cache. HIP copies more
// than 16 KiB with the SDMA engine, which writes VRAM directly, and nothing
// invalidates the HDP's cache afterwards, so the next read through the HDP is
// served the old data. VRAM itself is right: copying the buffer back shows
// the new data.
//
// It needs the copy to be done by SDMA (more than 16 KiB) and the write to be
// small enough to still be in the cache (16 KiB or less). Writing 32 KiB or
// more of the same buffer passes, because the write itself replaces the cache.
//
// Measured: iterations of 200 with wrong data, in two runs
//
//   FIX           RX 6800 XT (gfx1030)   MI300X (gfx942)
//   none          186, 185               0, 0
//   hdp-flush     0, 0                   0, 0
//   kernel        0, 0                   0, 0
//   sleep-5ms     17, 18                 0, 0   (not a fix)
//   write-32k     0, 0                   0, 0   (not a fix: a different write)
//
// Setting HSA_ENABLE_SDMA=0 or GPU_FORCE_BLIT_COPY_SIZE=1048576 in the
// environment, which make a kernel do the copy, also makes it pass.

#include "repro-common.h"

#include <chrono>
#include <thread>

int
main(int argc, char **argv)
{
    const repro::Args args{argc, argv, {"hdp-flush", "kernel", "sleep-5ms", "write-32k"}};
    repro::fastpathOnly();

    const size_t      copy_size{1 * repro::MiB};
    const size_t      write_size{args.fix == "write-32k" ? 32 * repro::KiB : 4 * repro::KiB};
    const repro::File file{repro::createFile(args.directory, copy_size)};
    uint32_t         *host{repro::hostWords(copy_size)};
    uint32_t         *from_file{repro::hostWords(copy_size)};
    void             *buffer{nullptr};
    void             *scratch{nullptr};
    HIP_OK(hipMalloc(&buffer, copy_size));
    HIP_OK(hipMalloc(&scratch, 4 * repro::KiB));

    repro::Tally tally;
    for (uint32_t iteration{1}; iteration <= 200; iteration++) {
        repro::fill(host, copy_size, iteration);
        HIP_OK(hipMemcpy(buffer, host, copy_size, hipMemcpyHostToDevice));

        if (args.fix == "hdp-flush") {
            // FIX: flush the HDP, which also invalidates its read cache. This
            // is what hipDeviceFlushGPUDirectRDMAWrites does where HIP has it,
            // and what ROCr could do after an SDMA copy, or hipFile before a
            // fastpath write.
            repro::flushHdp();
        }
        if (args.fix == "kernel") {
            // FIX: run any kernel and wait for it. The queue packet's acquire
            // fence makes the GPU invalidate the HDP's read cache.
            HIP_OK(hipMemset(scratch, 0, 4 * repro::KiB));
            HIP_OK(hipDeviceSynchronize());
        }
        if (args.fix == "sleep-5ms") {
            // NOT A FIX, but it hides the failure: the cache is refilled or
            // invalidated by something else within a few milliseconds.
            std::this_thread::sleep_for(std::chrono::milliseconds{5});
        }

        repro::transferred(hipFileWrite(file.handle, buffer, write_size, 0, 0), write_size, "hipFileWrite");

        repro::readFromStorage(file, from_file, write_size);
        tally.check(from_file, write_size, iteration);
    }
    return tally.finish(args);
}
