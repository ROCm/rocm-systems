/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// hipMemcpy of 64 MiB from one GPU buffer to another, then a hipFileWrite of
// the destination's last 4 KiB: the file gets what those 4 KiB held BEFORE
// hipMemcpy.
//
//   usage: 02-write-after-device-to-device-hipmemcpy DIRECTORY [FIX]
//
// Why: hipMemcpy is a synchronous call, but HIP copies between GPU buffers
// with a kernel and returns without waiting for it (clr hip_memory.cpp). The
// rest is as in 01-write-after-hipmemset.cpp: nothing orders fastpath's read
// of VRAM after that kernel.
//
// hipMemcpyDtoD behaves the same (CALL=2). So does hipMemcpyWithStream
// (CALL=1), and with it even hipFile's fallback backend writes the old data
// (BACKEND=1): the copy's kernel runs on the application's stream, and
// fallback's own hipMemcpy, on the null stream, does not queue behind a
// non-blocking stream.
//
// The buffers are large and the write is of the end only to make the kernel
// lose the race every time; with 1 MiB or 4 KiB buffers it loses it sometimes.
//
// Measured: iterations of 100 with wrong data, in two runs
//
//   FIX           RX 6800 XT (gfx1030)   MI300X (gfx942)
//   none          99, 99                 97, 98
//   stream-sync   0, 0                   0, 0
//   device-sync   0, 0                   0, 0
//   async         0, 0                   0, 0
//   hdp-flush     99, 99                 96, 97   (not a fix)
//
// Without a fix, hipMemcpyDtoD: 99, 99 and 97, 97. hipMemcpyWithStream with
// fastpath: 100, 100 and 95, 97; with fallback: 99, 99 and 98, 97. The fixes
// work the same way for each.

#include "repro-common.h"

// 0: hipMemcpy, 1: hipMemcpyWithStream, 2: hipMemcpyDtoD
#ifndef CALL
#define CALL 0
#endif

// 0: fastpath, 1: fallback
#ifndef BACKEND
#define BACKEND 0
#endif

int
main(int argc, char **argv)
{
    const repro::Args args{argc, argv, {"stream-sync", "device-sync", "async", "hdp-flush"}};
    if (BACKEND == 0) {
        repro::fastpathOnly();
    }
    else {
        repro::fallbackOnly();
    }

    const size_t      size{64 * repro::MiB};
    const size_t      write_size{4 * repro::KiB};
    const auto        write_offset{static_cast<hoff_t>(size - write_size)};
    const repro::File file{repro::createFile(args.directory, write_size)};
    uint32_t         *host{repro::hostWords(size)};
    uint32_t         *from_file{repro::hostWords(write_size)};
    void             *source{nullptr};
    void             *buffer{nullptr};
    hipStream_t       stream{nullptr};
    HIP_OK(hipMalloc(&source, size));
    HIP_OK(hipMalloc(&buffer, size));
    HIP_OK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    repro::Tally tally;
    for (uint32_t iteration{1}; iteration <= 100; iteration++) {
        // Put this iteration's value in the source buffer, and wait for it
        repro::fill(host, size, iteration);
        HIP_OK(hipMemcpy(source, host, size, hipMemcpyHostToDevice));
        HIP_OK(hipDeviceSynchronize());

        if (args.fix == "async") {
            // FIX: the asynchronous call, and wait for its stream.
            HIP_OK(hipMemcpyAsync(buffer, source, size, hipMemcpyDeviceToDevice, stream));
            HIP_OK(hipStreamSynchronize(stream));
        }
        else {
            // The call under test
#if CALL == 0
            HIP_OK(hipMemcpy(buffer, source, size, hipMemcpyDeviceToDevice));
#elif CALL == 1
            HIP_OK(hipMemcpyWithStream(buffer, source, size, hipMemcpyDeviceToDevice, stream));
#else
            HIP_OK(hipMemcpyDtoD(buffer, source, size));
#endif
        }

        if (args.fix == "stream-sync") {
            // FIX: wait for the stream the copy's kernel runs on: the null
            // stream for hipMemcpy and hipMemcpyDtoD, the application's for
            // hipMemcpyWithStream.
            HIP_OK(hipStreamSynchronize(CALL == 1 ? stream : nullptr));
        }
        if (args.fix == "device-sync") {
            // FIX: wait for every stream. hipFile could do this at the start
            // of a write, with either backend.
            HIP_OK(hipDeviceSynchronize());
        }
        if (args.fix == "hdp-flush") {
            // NOT A FIX: flushing the HDP does not wait for the kernel.
            repro::flushHdp();
        }

        // Write the last 4 KiB of the destination to the start of the file
        repro::transferred(hipFileWrite(file.handle, buffer, write_size, 0, write_offset), write_size,
                           "hipFileWrite");

        repro::readFromStorage(file, from_file, write_size);
        tally.check(from_file, write_size, iteration);
        // Let the kernel finish before the next iteration
        HIP_OK(hipDeviceSynchronize());
    }
    return tally.finish(args);
}
