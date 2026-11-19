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
// hipFile chooses the backend for each file, so this reproducer cannot force
// one. If it passes, check that 04 fails in the same DIRECTORY: if it does
// not, hipFile is not using fastpath there.
//
// Measured: iterations of 200 with wrong data, in two runs
//
//   FIX           RX 6800 XT (gfx1030)   MI300X (gfx942)
//   none          177, 175               0, 0
//   hdp-flush     0, 0                   0, 0
//   kernel        0, 0                   0, 0

#include "repro-common.h"

int
main(int argc, char **argv)
{
    const repro::Args args{argc, argv, {"hdp-flush", "kernel"}};

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
        repro::writeToStorage(fallback_file, host, read_size);

        repro::transferred(hipFileRead(fallback_file.handle, buffer, read_size, 0, 0), read_size,
                           "hipFileRead");

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
