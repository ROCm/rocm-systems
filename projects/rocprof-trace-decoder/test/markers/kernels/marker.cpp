// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Test manual SQTT markers (named and numeric).
//
// Named markers require the pass plugin. Numeric markers work standalone.
//
// Build (enabled + pass plugin for named markers):
//   hipcc -DSQTT_ENABLED=1 -fpass-plugin=../../../build/lib/libsqttinstrumentpass.so \
//         -I ../../include/ marker.cpp -o marker

#include <hip/hip_runtime.h>
#include "markers.hpp"

__global__ void test_kernel(float *out, const float *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Scoped named markers (require pass plugin, auto-assigned IDs)
    sqtt_marker_enter("load_phase");
    float val = in[idx];
    sqtt_marker_exit("load_phase");

    // Nested scoped markers
    sqtt_marker_enter("compute_phase");
    val = val * val + 1.0f;

    sqtt_marker_enter("inner_step");
    val = val / (val + 1.0f);
    sqtt_marker_exit("inner_step");

    sqtt_marker_exit("compute_phase");

    // Point marker (no scope change)
    sqtt_marker_point("checkpoint");
    sqtt_marker_data("payload_value", static_cast<uint32_t>(idx));

    // Numeric enter/exit markers (work without pass plugin)
    sqtt_marker_point(static_cast<uint32_t>(blockIdx.x));
    sqtt_marker_enter(10u);
    out[idx] = val;
    sqtt_marker_exit(10u);
}

int main() {
    const int N = 304*256;
    float *d_in, *d_out;
    hipMalloc(&d_in, N * sizeof(float));
    hipMalloc(&d_out, N * sizeof(float));

    std::vector<float> h_in(N, 1.0f);
    hipMemcpy(d_in, h_in.data(), N * sizeof(float), hipMemcpyHostToDevice);

    test_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    hipDeviceSynchronize();

    hipFree(d_in);
    hipFree(d_out);
    return 0;
}
