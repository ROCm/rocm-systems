// Background GPU load for the lateness-distribution comparison (item 6). Runs for SECS seconds.
// MODE=kernels   : back-to-back short kernels with hipStreamSynchronize (each completes with a
//                  system-scope release)
// MODE=longkernel: one kernel that spins on the GPU for SECS seconds (GPU busy/clocked up, but no
//                  packet completes while it runs)
// MODE=sdma      : back-to-back 1 MiB D2D copies on SDMA engine 0 via ROCr (no compute packets)
// Build: amdclang++ -x hip --offload-arch=gfx1030 -O2 -std=c++20 -DA1_WITH_HSA bgload.cpp -o bgload
//        -L/opt/rocm/lib -lhsa-runtime64 -Wl,-rpath,/opt/rocm/lib
#include "interventions.h"
#include <string>

__global__ void spin_kernel(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}
__global__ void small_kernel(uint32_t *p) { p[threadIdx.x] += 1; }

int main() {
    const std::string mode = std::getenv("MODE") ? std::getenv("MODE") : "kernels";
    const double secs = std::getenv("SECS") ? std::atof(std::getenv("SECS")) : 10.0;
    a1::Ctx c;
    a1::init(c, false);
    const uint64_t end = a1::now_ns() + uint64_t(secs * 1e9);
    long n = 0;
    if (mode == "kernels") {
        while (a1::now_ns() < end) { small_kernel<<<1, 64>>>(static_cast<uint32_t *>(c.other)); (void)hipStreamSynchronize(0); n++; }
    } else if (mode == "longkernel") {
        int khz = 0;
        (void)hipDeviceGetAttribute(&khz, hipDeviceAttributeClockRate, 0); // clock64 runs at shader clock
        // clock64 rate varies with DPM; loop launching ~0.5 s kernels until time is up
        while (a1::now_ns() < end) { spin_kernel<<<1, 1>>>((long long)khz * 500); (void)hipStreamSynchronize(0); n++; }
    } else if (mode == "sdma") {
        void *a{}, *b{};
        (void)hipMalloc(&a, 1 << 20);
        (void)hipMalloc(&b, 1 << 20);
        while (a1::now_ns() < end) { a1::hsa_sdma_copy(c, b, a, 1 << 20, true); n++; }
    }
    fprintf(stderr, "bgload %s: %ld submissions in %.1f s\n", mode.c_str(), n, secs);
}
