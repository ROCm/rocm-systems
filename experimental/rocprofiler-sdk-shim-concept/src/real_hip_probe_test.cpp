/*
 * real_hip_probe_test.cpp
 *
 * Standalone helper for validating the real HIP path through rocprofiler-
 * register + shim wrappers. The public HIP APIs do route through the real
 * registered table on the tested ROCm stack, so this loop is enough to prove
 * the wrapped hot path end-to-end.
 */
#include <cstdio>
#include <unistd.h>

#include <hip/hip_runtime.h>

int main()
{
    for(int i = 0; i < 4000; ++i) {
        int count = 0;
        if(hipGetDeviceCount(&count) != hipSuccess) {
            fprintf(stderr, "hipGetDeviceCount failed\n");
            return 10;
        }

        void* ptr = nullptr;
        if(hipMalloc(&ptr, 4096) != hipSuccess) {
            fprintf(stderr, "hipMalloc failed\n");
            return 11;
        }

        if(hipFree(ptr) != hipSuccess) {
            fprintf(stderr, "hipFree failed\n");
            return 12;
        }

        usleep(1000);
    }

    fprintf(stderr, "hip probe loop done\n");
    sleep(8);
    return 0;
}
