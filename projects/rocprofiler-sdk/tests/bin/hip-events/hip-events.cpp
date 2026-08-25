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

int
main(int argc, char** argv)
{
    int n          = 1024;
    int iterations = 4;

    if(argc > 1) n = atoi(argv[1]);
    if(argc > 2) iterations = atoi(argv[2]);

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
    scale_kernel<<<grid_size * 256, block_size, 0, stream0>>>(d_data, n, 1.01f);
    HIP_ASSERT(hipEventRecord(event0, stream0));
    HIP_ASSERT(hipStreamWaitEvent(stream1, event0, 0));
    scale_kernel<<<grid_size, block_size, 0, stream1>>>(d_data, n, 0.99f);
    HIP_ASSERT(hipStreamSynchronize(stream1));

    // Exercise duplicate wait: two streams wait on the same event before the
    // first wait is consumed. Both should produce WAIT completion records
    // without leaking correlation ID ref counts (which would hang on shutdown).
    hipStream_t stream2 = nullptr;
    HIP_ASSERT(hipStreamCreate(&stream2));

    scale_kernel<<<grid_size * 256, block_size, 0, stream0>>>(d_data, n, 1.01f);
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
    scale_kernel<<<grid_size * 256, block_size, 0, stream0>>>(d_data, n, 1.01f);
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

    HIP_ASSERT(hipEventDestroy(coalesce_event));
    HIP_ASSERT(hipFree(d_data));
    HIP_ASSERT(hipEventDestroy(event0));
    HIP_ASSERT(hipEventDestroy(event1));
    HIP_ASSERT(hipStreamDestroy(stream0));
    HIP_ASSERT(hipStreamDestroy(stream1));

    return 0;
}
