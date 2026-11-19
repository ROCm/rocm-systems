/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// hipMemset of a 64 MiB GPU buffer, then a fastpath hipFileWrite of the
// buffer's last 4 KiB: the file gets what those 4 KiB held BEFORE hipMemset.
//
//   usage: 01-write-after-hipmemset DIRECTORY [FIX]
//
// Why: hipMemset is a synchronous call, but HIP fills the buffer with a kernel
// and returns without waiting for it when the destination is the pointer
// hipMalloc returned (clr hip_memory.cpp, by design). hipFile's fallback
// backend is not affected, as its hipMemcpy queues behind that kernel. The
// fastpath backend asks the NVMe drive to read VRAM directly, and nothing
// orders that read after the kernel.
//
// The buffer is large and the write is of its end only to make the kernel lose
// the race every time. With a 4 KiB buffer the same happens less often: about
// one write in 100 on MI300X in hipfilewrite-visibility --producer memset.
//
// hipMemsetD8, hipMemsetD16, and hipMemsetD32 behave the same (set CALL below).
//
// Measured: iterations of 200 with wrong data, in two runs
//
//   FIX           RX 6800 XT (gfx1030)   MI300X (gfx942)
//   none          199, 200               199, 199
//   stream-sync   0, 0                   0, 0
//   device-sync   0, 0                   0, 0
//   async         0, 0                   0, 0
//   hdp-flush     199, 200               199, 199   (not a fix)

#include "repro-common.h"

// 0: hipMemset, 8: hipMemsetD8, 16: hipMemsetD16, 32: hipMemsetD32
#ifndef CALL
#define CALL 0
#endif

int
main(int argc, char **argv)
{
    const repro::Args args{argc, argv, {"stream-sync", "device-sync", "async", "hdp-flush"}};
    repro::fastpathOnly();

    const size_t      size{64 * repro::MiB};
    const size_t      write_size{4 * repro::KiB};
    const auto        write_offset{static_cast<hoff_t>(size - write_size)};
    const repro::File file{repro::createFile(args.directory, write_size)};
    uint32_t         *from_file{repro::hostWords(write_size)};
    void             *buffer{nullptr};
    hipStream_t       stream{nullptr};
    HIP_OK(hipMalloc(&buffer, size));
    HIP_OK(hipStreamCreate(&stream));

    repro::Tally tally;
    for (uint32_t iteration{1}; iteration <= 200; iteration++) {
        // A byte that differs from the one the previous iteration stored
        const auto     byte{static_cast<unsigned char>(iteration % 255 + 1)};
        const uint32_t word{0x01010101U * byte};

        if (args.fix == "async") {
            // FIX: the asynchronous call, and wait for its stream.
            HIP_OK(hipMemsetAsync(buffer, byte, size, stream));
            HIP_OK(hipStreamSynchronize(stream));
        }
        else {
            // The call under test
#if CALL == 0
            HIP_OK(hipMemset(buffer, byte, size));
#elif CALL == 8
            HIP_OK(hipMemsetD8(buffer, byte, size));
#elif CALL == 16
            HIP_OK(hipMemsetD16(buffer, static_cast<unsigned short>(word), size / 2));
#else
            HIP_OK(hipMemsetD32(buffer, static_cast<int>(word), size / 4));
#endif
        }

        if (args.fix == "stream-sync") {
            // FIX: wait for the null stream, which hipMemset's kernel runs on.
            // hipFile could do this at the start of a fastpath write.
            HIP_OK(hipStreamSynchronize(nullptr));
        }
        if (args.fix == "device-sync") {
            // FIX: wait for every stream.
            HIP_OK(hipDeviceSynchronize());
        }
        if (args.fix == "hdp-flush") {
            // NOT A FIX: flushing the HDP does not wait for the kernel.
            repro::flushHdp();
        }

        // Write the last 4 KiB of the buffer to the start of the file
        repro::transferred(hipFileWrite(file.handle, buffer, write_size, 0, write_offset), write_size,
                           "hipFileWrite");

        repro::readFromStorage(file, from_file, write_size);
        tally.check(from_file, write_size, word);
        // Let the kernel finish before the next iteration
        HIP_OK(hipDeviceSynchronize());
    }
    return tally.finish(args);
}
