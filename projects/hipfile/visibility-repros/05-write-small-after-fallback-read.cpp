/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// The failure of 04-write-small-after-large-hipmemcpy.cpp with no HIP call in
// the application: hipFileRead of 64 KiB from a file where hipFile uses its
// fallback backend (here, in /dev/shm), then hipFileWrite of the buffer's
// first 4 KiB to a file where it uses fastpath. The second file gets what
// those 4 KiB held BEFORE hipFileRead.
//
//   usage: 05-write-small-after-fallback-read DIRECTORY [FIX]
//
// Why: the fallback hipFileRead is a pread into host memory and a hipMemcpy to
// the GPU, which the SDMA engine does. The rest is as in 04: the HDP's read
// cache still holds what the previous fastpath hipFileWrite read.
//
// PRODUCER below chooses what stores the 64 KiB, to show that the fallback
// hipFileRead fails because of its hipMemcpy and for no other reason. With 0,
// it is the fallback hipFileRead. With 1, it is the application's own
// hipMemcpy from host memory, and the program calls no hipFileRead.
//
// A fastpath hipFileRead of the same 64 KiB, from a second file in DIRECTORY,
// does not fail: 0 of 200 in three runs on the RX 6800 XT. The drive's writes
// reach VRAM through the HDP, which keeps its read cache right.
//
// With PRODUCER 0, hipFile chooses the backend for each file, so the program
// cannot force one. If it passes, check that PRODUCER 1 fails in the same
// DIRECTORY: if it does not, hipFile is not using fastpath there.
//
// Measured: iterations of 200 with wrong data, in three runs on the RX 6800 XT
// (gfx1030). On MI300X (gfx942) the fallback hipFileRead had 0 in two runs.
//
//   FIX           fallback hipFileRead   hipMemcpy
//   none          175, 175, 176          177, 179, 177
//   hdp-flush     0, 0, 0                0, 0, 0
//   kernel        0, 0, 0                0, 0, 0

#include "repro-common.h"

// 0: fallback hipFileRead, 1: hipMemcpy from host memory
#ifndef PRODUCER
#define PRODUCER 0
#endif

int
main(int argc, char **argv)
{
    const repro::Args args{argc, argv, {"hdp-flush", "kernel"}};
    if (PRODUCER == 1) {
        // With no fallback hipFileRead, nothing may fall back
        repro::fastpathOnly();
    }

    const size_t      read_size{64 * repro::KiB};
    const size_t      write_size{4 * repro::KiB};
    const repro::File fallback_file{repro::createFile("/dev/shm", read_size)};
    const repro::File fastpath_file{repro::createFile(args.directory, read_size)};
    uint32_t         *host{repro::hostWords(read_size)};
    uint32_t         *from_file{repro::hostWords(read_size)};
    void             *buffer{nullptr};
    void             *scratch{nullptr};
    HIP_OK(hipMalloc(&buffer, read_size));
    HIP_OK(hipMalloc(&scratch, 4 * repro::KiB));

    repro::Tally tally;
    for (uint32_t iteration{1}; iteration <= 200; iteration++) {
        repro::fill(host, read_size, iteration);

        if (PRODUCER == 0) {
            // The fallback hipFileRead: a pread into host memory, then a hipMemcpy
            repro::writeToStorage(fallback_file, host, read_size);
            repro::transferred(hipFileRead(fallback_file.handle, buffer, read_size, 0, 0), read_size,
                               "hipFileRead");
        }
        else {
            // The same hipMemcpy, without hipFile
            HIP_OK(hipMemcpy(buffer, host, read_size, hipMemcpyHostToDevice));
        }

        if (args.fix == "hdp-flush") {
            // FIX: flush the HDP, which also invalidates its read cache.
            // hipFile could do this at the end of a fallback read, after its
            // hipMemcpy, or at the start of a fastpath write.
            repro::flushHdp();
        }
        if (args.fix == "kernel") {
            // FIX: run any kernel and wait for it; see 04.
            HIP_OK(hipMemset(scratch, 0, 4 * repro::KiB));
            HIP_OK(hipDeviceSynchronize());
        }

        repro::transferred(hipFileWrite(fastpath_file.handle, buffer, write_size, 0, 0), write_size,
                           "hipFileWrite");

        repro::readFromStorage(fastpath_file, from_file, write_size);
        tally.check(from_file, write_size, iteration);
    }
    return tally.finish(args);
}
