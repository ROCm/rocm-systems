// Test manual SQTT markers (named and numeric).
//
// Named markers require the pass plugin. Numeric markers work standalone.
//
// Build (enabled + pass plugin for named markers):
//   hipcc -DSQTT_ENABLED=1 -fpass-plugin=../../build/SQTTInstrumentPass.so \
//         -I ../../include/ marker.hip -o marker

#include <hip/hip_runtime.h>
#include "sqtt_trace.hpp"

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

    // Numeric enter/exit markers (work without pass plugin)
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
