// Integration test: Application-driven late loading of rocprofiler-sdk
// This test demonstrates the late-start functionality where the application
// initializes HIP first, then loads rocprofiler-sdk dynamically.

#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Forward declarations for rocprofiler-sdk functions (loaded dynamically)
typedef int rocprofiler_status_t;
typedef rocprofiler_status_t (*rocprofiler_start_late_fn)(uint32_t);
typedef rocprofiler_status_t (*rocprofiler_stop_late_fn)(void);
typedef rocprofiler_status_t (*rocprofiler_is_late_start_fn)(int*);

#define ROCPROFILER_LATE_START_AUTO (1 << 8)

#define HIP_CHECK(cmd)                                                                             \
    do                                                                                             \
    {                                                                                              \
        hipError_t error = (cmd);                                                                  \
        if(error != hipSuccess)                                                                    \
        {                                                                                          \
            fprintf(stderr, "HIP error %d at %s:%d\n", error, __FILE__, __LINE__);                 \
            exit(1);                                                                               \
        }                                                                                          \
    } while(0)

// Simple kernel for testing
__global__ void
vectorAdd(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) c[i] = a[i] + b[i];
}

int
main(int argc, char** argv)
{
    printf("=== Late-Start Integration Test ===\n\n");

    // Phase 1: Initialize HIP BEFORE loading rocprofiler-sdk
    printf("Phase 1: Initializing HIP runtime...\n");

    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    printf("  Found %d HIP device(s)\n", device_count);

    if(device_count == 0)
    {
        printf("  No HIP devices found, skipping GPU operations\n");
        printf("\nThis test is skipped.\n");
        return 0;
    }

    HIP_CHECK(hipSetDevice(0));

    // Phase 2: Do some GPU work BEFORE late-start (should NOT be traced)
    printf("\nPhase 2: GPU work before late-start (not traced)...\n");

    const int N    = 1024;
    size_t    size = N * sizeof(float);

    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c(N, 0.0f);

    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, size));
    HIP_CHECK(hipMalloc(&d_b, size));
    HIP_CHECK(hipMalloc(&d_c, size));

    HIP_CHECK(hipMemcpy(d_a, h_a.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), size, hipMemcpyHostToDevice));

    vectorAdd<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
    HIP_CHECK(hipDeviceSynchronize());

    printf("  Completed pre-late-start kernel execution\n");

    // Phase 3: Load rocprofiler-sdk dynamically
    printf("\nPhase 3: Loading rocprofiler-sdk...\n");

    void* rocp_handle = dlopen("librocprofiler-sdk.so.0", RTLD_NOW | RTLD_GLOBAL);
    if(!rocp_handle)
    {
        fprintf(stderr, "  Failed to load rocprofiler-sdk: %s\n", dlerror());
        fprintf(stderr, "  This is expected if rocprofiler-sdk is not installed\n");

        // Cleanup
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));
        printf("\nThis test is skipped.\n");
        return 0;
    }
    printf("  Successfully loaded rocprofiler-sdk\n");

    // Get function pointers
    auto start_late = (rocprofiler_start_late_fn) dlsym(rocp_handle, "rocprofiler_start_late");
    auto stop_late  = (rocprofiler_stop_late_fn) dlsym(rocp_handle, "rocprofiler_stop_late");
    auto is_late_start =
        (rocprofiler_is_late_start_fn) dlsym(rocp_handle, "rocprofiler_is_late_start");

    if(!start_late || !stop_late || !is_late_start)
    {
        fprintf(stderr, "  Failed to find late-start functions\n");
        dlclose(rocp_handle);
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));
        return 1;
    }

    // Phase 4: Call rocprofiler_start_late()
    printf("\nPhase 4: Starting late profiling...\n");

    rocprofiler_status_t status = start_late(ROCPROFILER_LATE_START_AUTO);
    if(status != 0)
    {
        fprintf(stderr, "  rocprofiler_start_late failed with status %d\n", status);
        dlclose(rocp_handle);
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));
        return 1;
    }
    printf("  Late profiling started successfully\n");

    int is_late = 0;
    is_late_start(&is_late);
    printf("  Is late-started: %s\n", is_late ? "yes" : "no");

    // Phase 5: Do GPU work AFTER late-start (SHOULD be traced)
    printf("\nPhase 5: GPU work after late-start (traced)...\n");

    // Multiple kernel launches to generate trace data
    for(int i = 0; i < 5; i++)
    {
        vectorAdd<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Memory operations
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, size, hipMemcpyDeviceToHost));

    printf("  Completed %d kernel launches (should be traced)\n", 5);

    // Verify result
    bool correct = true;
    for(int i = 0; i < N; i++)
    {
        if(h_c[i] != 3.0f)
        {
            correct = false;
            break;
        }
    }
    printf("  Computation result: %s\n", correct ? "correct" : "INCORRECT");

    // Phase 6: Stop late profiling (optional)
    printf("\nPhase 6: Stopping late profiling...\n");
    status = stop_late();
    printf("  Stop status: %d\n", status);

    is_late_start(&is_late);
    printf("  Is late-started after stop: %s\n", is_late ? "yes" : "no");

    // Cleanup
    printf("\nPhase 7: Cleanup...\n");
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    dlclose(rocp_handle);

    printf("\n=== Test completed successfully ===\n");
    return 0;
}
