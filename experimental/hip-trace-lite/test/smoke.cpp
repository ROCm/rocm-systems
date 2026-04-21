// test/smoke.cpp — exercises one async HIP op so libhiptracelite (LD_PRELOADed)
// captures a HIP_OPS record. Uses runtime APIs only so plain g++ + libamdhip64
// is enough to build (no hipcc needed for the experimental scaffold).
#include "htl_record.hpp"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    const char* out = std::getenv("HTL_OUTPUT_FILE");
    if (!out) out = "./hiptrace.bin";

    void* d = nullptr;
    if (hipMalloc(&d, 1024) != hipSuccess) {
        std::fprintf(stderr, "smoke: hipMalloc failed\n");
        return 2;
    }
    if (hipMemset(d, 0xAB, 1024) != hipSuccess) {
        std::fprintf(stderr, "smoke: hipMemset failed\n");
        return 2;
    }
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::fprintf(stderr, "smoke: hipDeviceSynchronize failed: %s\n",
                     hipGetErrorString(err));
        return 2;
    }
    hipFree(d);

    if (argc > 1 && std::strcmp(argv[1], "--no-verify") == 0) return 0;

    std::fprintf(stderr, "smoke: hipMemset issued ok; check %s after exit\n", out);
    return 0;
}
