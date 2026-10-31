// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <stdio.h>
#include <stdlib.h>

#include "hip/hip_runtime.h"

#define HIP_ASSERT(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        hipError_t gpuErr = call;                                                                  \
        if(hipSuccess != gpuErr)                                                                   \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "GPU API Error - %s:%d: '%s'\n",                                               \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(gpuErr));                                                    \
            exit(1);                                                                               \
        }                                                                                          \
    } while(0)

__global__ void
scale_kernel(float* data, int n, float factor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) data[tid] = data[tid] * factor;
}

// Occupies the GPU for at least the given number of wall clock ticks.
__global__ void
spin_kernel(long long ticks)
{
    const auto start = wall_clock64();
    while(wall_clock64() - start < ticks)
    {}
}

int
main(int argc, char** argv)
{
    // Defaults reproduce the record and wait counts the accompanying validate.py expects,
    // so the integration test invokes this with no arguments. The knobs exist so other
    // integration tests can drive the same binary at a different problem size or depth,
    // per the tests/bin guidance on parameterizing the shared application suite.
    int n              = 1024;
    int iterations     = 4;
    int big_multiplier = 256;

    for(int i = 1; i < argc; ++i)
    {
        auto arg = std::string_view{argv[i]};
        auto val = [&]() { return (i + 1 < argc) ? atoi(argv[++i]) : 0; };

        if(arg == "--size")
            n = val();
        else if(arg == "--iterations")
            iterations = val();
        else if(arg == "--big-multiplier")
            big_multiplier = val();
        else if(arg == "--help" || arg == "-h")
        {
            printf("usage: %s [--size N] [--iterations N] [--big-multiplier N]\n", argv[0]);
            printf("  --size            elements per kernel (default 1024)\n");
            printf("  --iterations      record/wait loop iterations (default 4)\n");
            printf("  --big-multiplier  grid scale for the long-running kernels that keep a\n");
            printf("                    record barrier in flight (default 256)\n");
            return 0;
        }
        // Positional form kept for compatibility with existing invocations.
        else if(i == 1)
            n = atoi(argv[1]);
        else if(i == 2)
            iterations = atoi(argv[2]);
    }

    if(n <= 0 || iterations <= 0 || big_multiplier <= 0)
    {
        printf("error: --size, --iterations and --big-multiplier must all be positive\n");
        return 1;
    }

    hipStream_t stream0 = nullptr;
    hipStream_t stream1 = nullptr;
    HIP_ASSERT(hipStreamCreate(&stream0));
    HIP_ASSERT(hipStreamCreate(&stream1));

    hipEvent_t event0 = nullptr;
    hipEvent_t event1 = nullptr;
    HIP_ASSERT(hipEventCreate(&event0));
    HIP_ASSERT(hipEventCreate(&event1));

    float* d_data = nullptr;
    HIP_ASSERT(hipMalloc(&d_data, n * sizeof(float)));
    HIP_ASSERT(hipMemset(d_data, 0, n * sizeof(float)));

    auto block_size = 256;
    auto grid_size  = (n + block_size - 1) / block_size;

    for(int i = 0; i < iterations; ++i)
    {
        // Launch on stream0, then record event
        scale_kernel<<<grid_size, block_size, 0, stream0>>>(d_data, n, 1.01f);
        HIP_ASSERT(hipEventRecord(event0, stream0));

        // stream1 waits for stream0's event, then launches
        HIP_ASSERT(hipStreamWaitEvent(stream1, event0, 0));
        scale_kernel<<<grid_size, block_size, 0, stream1>>>(d_data, n, 0.99f);
        HIP_ASSERT(hipEventRecord(event1, stream1));

        // stream0 waits for stream1's event
        HIP_ASSERT(hipStreamWaitEvent(stream0, event1, 0));
    }

    // Exercise hipEventRecordWithFlags to confirm both API entry points are traced
    scale_kernel<<<grid_size, block_size, 0, stream0>>>(d_data, n, 1.01f);
    HIP_ASSERT(hipEventRecordWithFlags(event0, stream0, hipEventRecordDefault));
    HIP_ASSERT(hipStreamWaitEvent(stream1, event0, 0));

    // Exercise same-stream wait: CLR short-circuits when the event was recorded
    // on the same stream. No barrier is dispatched, no WAIT record should appear.
    HIP_ASSERT(hipStreamWaitEvent(stream0, event0, 0));

    // Exercise deferred wait: launch a large kernel so the event is unlikely to
    // have completed by the time hipStreamWaitEvent checks ready(). CLR will add
    // the event's signal as a dep_signal on the next dispatch's barrier rather
    // than emitting a standalone wait barrier. The profiler must detect the
    // dependency when it sees the signal in the subsequent kernel's barrier.
    scale_kernel<<<grid_size * big_multiplier, block_size, 0, stream0>>>(d_data, n, 1.01f);
    HIP_ASSERT(hipEventRecord(event0, stream0));
    HIP_ASSERT(hipStreamWaitEvent(stream1, event0, 0));
    scale_kernel<<<grid_size, block_size, 0, stream1>>>(d_data, n, 0.99f);
    HIP_ASSERT(hipStreamSynchronize(stream1));

    // Exercise duplicate wait: two streams wait on the same event before the
    // first wait is consumed. Both should produce WAIT completion records
    // without leaking correlation ID ref counts (which would hang on shutdown).
    hipStream_t stream2 = nullptr;
    HIP_ASSERT(hipStreamCreate(&stream2));

    scale_kernel<<<grid_size * big_multiplier, block_size, 0, stream0>>>(d_data, n, 1.01f);
    HIP_ASSERT(hipEventRecord(event0, stream0));
    HIP_ASSERT(hipStreamWaitEvent(stream1, event0, 0));
    HIP_ASSERT(hipStreamWaitEvent(stream2, event0, 0));
    scale_kernel<<<grid_size, block_size, 0, stream1>>>(d_data, n, 0.99f);
    scale_kernel<<<grid_size, block_size, 0, stream2>>>(d_data, n, 0.99f);
    HIP_ASSERT(hipStreamSynchronize(stream1));
    HIP_ASSERT(hipStreamSynchronize(stream2));
    HIP_ASSERT(hipStreamDestroy(stream2));

    // Exercise destroy with pending wait: record an event, register a wait on
    // another stream, then destroy the event before the wait is consumed. The
    // profiler must clean up the pending wait and release correlation ID ref
    // counts without leaking or crashing. No WAIT record should appear for this
    // event since it was destroyed before consumption.
    hipEvent_t destroy_event = nullptr;
    HIP_ASSERT(hipEventCreate(&destroy_event));
    scale_kernel<<<grid_size * big_multiplier, block_size, 0, stream0>>>(d_data, n, 1.01f);
    HIP_ASSERT(hipEventRecord(destroy_event, stream0));
    HIP_ASSERT(hipStreamWaitEvent(stream1, destroy_event, 0));
    HIP_ASSERT(hipEventDestroy(destroy_event));
    scale_kernel<<<grid_size, block_size, 0, stream1>>>(d_data, n, 0.99f);
    HIP_ASSERT(hipDeviceSynchronize());

    // Exercise graph capture exclusion: recording an event during graph capture
    // should not produce a RECORD completion. The is_stream_capturing guard must
    // prevent check_coalesced_record from fabricating a record.
    hipStream_t capture_stream = nullptr;
    hipEvent_t  capture_event  = nullptr;
    HIP_ASSERT(hipStreamCreate(&capture_stream));
    HIP_ASSERT(hipEventCreate(&capture_event));
    HIP_ASSERT(hipStreamBeginCapture(capture_stream, hipStreamCaptureModeGlobal));
    HIP_ASSERT(hipEventRecord(capture_event, capture_stream));
    hipGraph_t graph = nullptr;
    HIP_ASSERT(hipStreamEndCapture(capture_stream, &graph));
    HIP_ASSERT(hipGraphDestroy(graph));
    HIP_ASSERT(hipEventDestroy(capture_event));
    HIP_ASSERT(hipStreamDestroy(capture_stream));

    // Exercise the default stream. Passing nullptr selects the legacy default stream here
    // and the per-thread default stream through the _spt entry points, and hipStreamLegacy
    // and hipStreamPerThread name those two explicitly. The profiler must treat a wait on
    // the same default stream as same-stream (no completion) while still tracing a wait
    // that genuinely crosses from the default stream to a created one.
    hipEvent_t default_event = nullptr;
    HIP_ASSERT(hipEventCreate(&default_event));

    // Same default stream on both sides, spelled the same way: must be short-circuited.
    scale_kernel<<<grid_size * big_multiplier, block_size, 0, nullptr>>>(d_data, n, 1.01f);
    HIP_ASSERT(hipEventRecord(default_event, nullptr));
    HIP_ASSERT(hipStreamWaitEvent(nullptr, default_event, 0));

    // Default stream to a created stream: a real cross-stream dependency that must trace.
    scale_kernel<<<grid_size * big_multiplier, block_size, 0, nullptr>>>(d_data, n, 1.01f);
    HIP_ASSERT(hipEventRecord(default_event, nullptr));
    HIP_ASSERT(hipStreamWaitEvent(stream1, default_event, 0));
    scale_kernel<<<grid_size, block_size, 0, stream1>>>(d_data, n, 0.99f);
    HIP_ASSERT(hipDeviceSynchronize());

    // Exercise same-stream wait while the record barrier is still in flight. CLR
    // short-circuits Event::streamWait when the wait targets the recording stream, so
    // no GPU dependency is ever created and nothing will claim a pending wait
    // registered for it. register_deferred_wait must decline to register one.
    //
    // The large kernel keeps the record barrier in flight so the completed-generation
    // guard does not reject the wait first -- that guard is what makes the same-stream
    // wait earlier in this file harmless, and it hides this case when the event has
    // already retired.
    //
    // same_stream_event is deliberately NOT destroyed: hipEventDestroy releases any
    // stranded pending wait, which would mask a regression. Left outstanding, a stranded
    // entry keeps a correlation ID reference and finalization aborts with "retired
    // dangling correlation IDs", failing this test loudly.
    hipEvent_t same_stream_event = nullptr;
    HIP_ASSERT(hipEventCreate(&same_stream_event));
    scale_kernel<<<grid_size * big_multiplier, block_size, 0, stream0>>>(d_data, n, 1.01f);
    HIP_ASSERT(hipEventRecord(same_stream_event, stream0));
    HIP_ASSERT(hipStreamWaitEvent(stream0, same_stream_event, 0));
    HIP_ASSERT(hipDeviceSynchronize());

    // Exercise a wait whose call submits another barrier ahead of its own. stream1 is a
    // blocking stream, so while the null stream has work in flight, CLR's getStream enqueues a
    // marker on stream1 that waits for the null stream before the wait's own barrier. The
    // record is on a non-blocking stream behind a longer kernel, so that marker retires well
    // before the record does, and only the wait's own barrier ends after the record. A WAIT
    // timed by the marker ends before its RECORD, which validate.py checks.
    //
    // Then the same for a record: stream0 is a blocking stream, so while the null stream has
    // work in flight, CLR's getStream enqueues a marker on stream0 that waits for the null
    // stream before the record's own barrier. The wait is from the non-blocking stream, which
    // gets no marker, and depends on the record's own barrier only. A RECORD timed by the
    // marker leaves that wait matching nothing, so no WAIT is reported, which validate.py
    // checks. validate.py finds both scenarios by their position after hipStreamCreateWithFlags,
    // which the app calls only here.
    //
    // ordered_event and marker_event are destroyed only at the end: freeing one here would let
    // a later event reuse its address, and its WAIT would then hide that event's record-only
    // handle from the handle-keyed checks in validate_no_hip_api.py.
    hipEvent_t ordered_event = nullptr;
    hipEvent_t marker_event  = nullptr;
    {
        auto wall_clock_khz = 0;
        HIP_ASSERT(hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, 0));
        const auto ticks_per_msec = static_cast<long long>(wall_clock_khz);

        hipStream_t nonblocking_stream = nullptr;
        HIP_ASSERT(hipStreamCreateWithFlags(&nonblocking_stream, hipStreamNonBlocking));
        HIP_ASSERT(hipEventCreate(&ordered_event));
        HIP_ASSERT(hipEventCreate(&marker_event));

        spin_kernel<<<1, 1, 0, nonblocking_stream>>>(50 * ticks_per_msec);
        HIP_ASSERT(hipEventRecord(ordered_event, nonblocking_stream));
        spin_kernel<<<1, 1, 0, nullptr>>>(5 * ticks_per_msec);
        HIP_ASSERT(hipStreamWaitEvent(stream1, ordered_event, 0));
        scale_kernel<<<grid_size, block_size, 0, stream1>>>(d_data, n, 0.99f);
        HIP_ASSERT(hipDeviceSynchronize());

        spin_kernel<<<1, 1, 0, nullptr>>>(5 * ticks_per_msec);
        HIP_ASSERT(hipEventRecord(marker_event, stream0));
        HIP_ASSERT(hipStreamWaitEvent(nonblocking_stream, marker_event, 0));
        scale_kernel<<<grid_size, block_size, 0, nonblocking_stream>>>(d_data, n, 0.99f);
        HIP_ASSERT(hipDeviceSynchronize());

        HIP_ASSERT(hipStreamDestroy(nonblocking_stream));
    }

    // Exercise an interprocess event recorded and waited on within this process. The wait
    // depends on the event's interprocess signal, so it is reported only if the RECORD is
    // timed by the barrier that completes on that signal. The record is behind a long kernel
    // so the event is still pending when the wait is made. validate.py finds the record and
    // the wait as the last hipEventRecord and hipStreamWaitEvent before hipIpcGetEventHandle,
    // which the app calls only here.
    //
    // ipc_event is destroyed only at the end, for the same address-reuse reason as
    // ordered_event and marker_event.
    hipEvent_t ipc_event = nullptr;
    {
        auto wall_clock_khz = 0;
        HIP_ASSERT(hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, 0));
        const auto ticks_per_msec = static_cast<long long>(wall_clock_khz);

        HIP_ASSERT(
            hipEventCreateWithFlags(&ipc_event, hipEventInterprocess | hipEventDisableTiming));

        spin_kernel<<<1, 1, 0, stream0>>>(20 * ticks_per_msec);
        HIP_ASSERT(hipEventRecord(ipc_event, stream0));
        HIP_ASSERT(hipStreamWaitEvent(stream1, ipc_event, 0));
        scale_kernel<<<grid_size, block_size, 0, stream1>>>(d_data, n, 0.99f);
        HIP_ASSERT(hipDeviceSynchronize());

        hipIpcEventHandle_t ipc_handle = {};
        HIP_ASSERT(hipIpcGetEventHandle(&ipc_handle, ipc_event));
    }

    // Exercise same-stream waits that name the recording stream differently. Each record is
    // behind a spin kernel, so it is still in flight when its wait is issued.
    //
    // A wait on hipStreamLegacy for an event recorded on nullptr, and a hipStreamWaitEvent_spt
    // on nullptr for an event recorded on hipStreamPerThread, name the recording stream, so CLR
    // creates no dependency and no WAIT may be reported. legacy_event and per_thread_event are
    // not destroyed, for the same reason as same_stream_event above: a wait registered for either
    // is never claimed, and the execute test fails on its dangling correlation ID. stream1 is a
    // blocking stream, so the kernel launched on it after the legacy wait makes CLR enqueue a
    // marker on stream1 that depends on the record, which must not report the wait either.
    //
    // validate.py finds the _spt wait as the app's only hipStreamWaitEvent_spt call, and the
    // legacy wait as the last hipStreamWaitEvent ahead of it.
    hipEvent_t legacy_event     = nullptr;
    hipEvent_t per_thread_event = nullptr;
    {
        auto wall_clock_khz = 0;
        HIP_ASSERT(hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, 0));
        const auto ticks_per_msec = static_cast<long long>(wall_clock_khz);

        HIP_ASSERT(hipEventCreate(&legacy_event));
        HIP_ASSERT(hipEventCreate(&per_thread_event));

        spin_kernel<<<1, 1, 0, nullptr>>>(20 * ticks_per_msec);
        HIP_ASSERT(hipEventRecord(legacy_event, nullptr));
        HIP_ASSERT(hipStreamWaitEvent(hipStreamLegacy, legacy_event, 0));
        scale_kernel<<<grid_size, block_size, 0, stream1>>>(d_data, n, 0.99f);
        HIP_ASSERT(hipDeviceSynchronize());

        spin_kernel<<<1, 1, 0, hipStreamPerThread>>>(20 * ticks_per_msec);
        HIP_ASSERT(hipEventRecord(per_thread_event, hipStreamPerThread));
        HIP_ASSERT(hipStreamWaitEvent_spt(nullptr, per_thread_event, 0));
        HIP_ASSERT(hipDeviceSynchronize());
    }

    // Exercise coalescing: record a hipEventDisableTiming event multiple times
    // on the same stream with no intervening work. CLR's ShouldCoalesceMarker
    // may skip dispatching a barrier for the 2nd and 3rd records, reusing the
    // first barrier's HwEvent. The profiler's coalescing handler should still
    // produce completion records for all three.
    hipEvent_t coalesce_event = nullptr;
    HIP_ASSERT(hipEventCreateWithFlags(&coalesce_event, hipEventDisableTiming));

    scale_kernel<<<grid_size, block_size, 0, stream0>>>(d_data, n, 1.01f);
    HIP_ASSERT(hipEventRecord(coalesce_event, stream0));
    HIP_ASSERT(hipEventRecord(coalesce_event, stream0));
    HIP_ASSERT(hipEventRecord(coalesce_event, stream0));

    HIP_ASSERT(hipDeviceSynchronize());

    HIP_ASSERT(hipEventDestroy(default_event));
    HIP_ASSERT(hipEventDestroy(coalesce_event));
    HIP_ASSERT(hipEventDestroy(ordered_event));
    HIP_ASSERT(hipEventDestroy(marker_event));
    HIP_ASSERT(hipEventDestroy(ipc_event));
    HIP_ASSERT(hipFree(d_data));
    HIP_ASSERT(hipEventDestroy(event0));
    HIP_ASSERT(hipEventDestroy(event1));
    HIP_ASSERT(hipStreamDestroy(stream0));
    HIP_ASSERT(hipStreamDestroy(stream1));

    return 0;
}
