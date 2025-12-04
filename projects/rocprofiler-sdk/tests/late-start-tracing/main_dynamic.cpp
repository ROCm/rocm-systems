// True late-start integration test using dynamic loading
// This test does NOT link to rocprofiler-sdk at build time.
// It initializes HIP first, THEN loads rocprofiler-sdk dynamically.
//
// This test validates that late-start profiling works for API tracing.
//
// SUPPORTED in late-start:
//   - HIP API tracing (via API table wrapping)
//   - HSA API tracing (via API table wrapping)
//   - ROCTx tracing (via API table wrapping)
//
// NOT SUPPORTED in late-start:
//   - Kernel dispatch tracing (requires queue interception at queue creation)
//   - PC sampling (requires queue-level setup)
//   - Counter collection (requires queue-level setup)
//
// This is an architectural limitation: queue-based features require
// infrastructure to be installed when queues are created. Queues that
// already exist before profiling starts cannot be retroactively intercepted.

#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

// Global counter for HIP API calls (will be set by tool callbacks after SDK loads)
// Note: Kernel dispatch tracing is NOT supported in late-start scenarios
extern std::atomic<uint64_t> g_hip_api_calls;

std::atomic<uint64_t> g_hip_api_calls{0};

int
main(int argc, char** argv)
{
    printf("=== True Late-Start Test (Dynamic Loading) ===\n\n");

    // Phase 1: Initialize HIP BEFORE rocprofiler-sdk is loaded
    printf("Phase 1: Initializing HIP (rocprofiler-sdk NOT loaded yet)...\n");

    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    printf("  Found %d HIP device(s)\n", device_count);

    if(device_count == 0)
    {
        printf("  No HIP devices found\n");
        printf("\nTest SKIPPED (no GPU available)\n");
        return 0;
    }

    HIP_CHECK(hipSetDevice(0));
    printf("  HIP runtime initialized successfully\n");
    printf("  IMPORTANT: rocprofiler-sdk is NOT loaded yet!\n");

    // Phase 2: Do GPU work BEFORE late-start (should NOT be traced)
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

    // Launch kernel (not traced)
    vectorAdd<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
    HIP_CHECK(hipDeviceSynchronize());

    printf("  Completed pre-late-start GPU work\n");
    printf("  API calls traced: %lu (should be 0)\n",
           g_hip_api_calls.load(std::memory_order_relaxed));

    // Phase 3: Dynamically load rocprofiler-sdk
    printf("\nPhase 3: Dynamically loading rocprofiler-sdk...\n");

    void* sdk_handle = dlopen("librocprofiler-sdk.so.0", RTLD_NOW | RTLD_GLOBAL);
    if(!sdk_handle)
    {
        sdk_handle = dlopen("librocprofiler-sdk.so", RTLD_NOW | RTLD_GLOBAL);
    }

    if(!sdk_handle)
    {
        fprintf(stderr, "  Failed to load rocprofiler-sdk: %s\n", dlerror());
        fprintf(stderr, "  Make sure librocprofiler-sdk.so is in LD_LIBRARY_PATH\n");

        // Cleanup
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));

        printf("\nTest SKIPPED (SDK not available)\n");
        return 0;
    }

    printf("  Successfully loaded rocprofiler-sdk\n");

    // Phase 4: Load tool configuration library
    printf("\nPhase 4: Loading tool configuration...\n");

    // Try multiple possible paths for the tool library
    void* tool_handle = dlopen("./liblate-start-dynamic-client.so", RTLD_NOW | RTLD_GLOBAL);
    if(!tool_handle) tool_handle = dlopen("liblate-start-dynamic-client.so", RTLD_NOW | RTLD_GLOBAL);
    if(!tool_handle)
    {
        fprintf(stderr, "  Failed to load tool: %s\n", dlerror());
        dlclose(sdk_handle);

        // Cleanup
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));

        printf("\nTest FAILED (tool library not found)\n");
        return 1;
    }

    printf("  Loaded tool configuration library\n");

    // Get the tool's rocprofiler_configure function
    using configure_fn_t = void* (*)(uint32_t, const char*, uint32_t, void*);
    auto tool_configure  = (configure_fn_t) dlsym(tool_handle, "rocprofiler_configure");

    if(!tool_configure)
    {
        fprintf(stderr, "  Failed to find rocprofiler_configure in tool\n");
        dlclose(tool_handle);
        dlclose(sdk_handle);

        // Cleanup
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));

        printf("\nTest FAILED\n");
        return 1;
    }

    // Phase 5: Call rocprofiler_force_configure - TRIGGER LATE-START!
    printf("\nPhase 5: Triggering late-start via rocprofiler_force_configure()...\n");

    using force_configure_fn_t = int (*)(void*);
    auto force_configure = (force_configure_fn_t) dlsym(sdk_handle, "rocprofiler_force_configure");

    if(!force_configure)
    {
        fprintf(stderr, "  Failed to find rocprofiler_force_configure\n");
        dlclose(tool_handle);
        dlclose(sdk_handle);

        // Cleanup
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));

        printf("\nTest FAILED\n");
        return 1;
    }

    int status = force_configure((void*) tool_configure);

    if(status != 0)
    {
        fprintf(stderr, "  rocprofiler_force_configure() failed with status %d\n", status);
        dlclose(tool_handle);
        dlclose(sdk_handle);

        // Cleanup
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));

        printf("\nTest FAILED\n");
        return 1;
    }

    printf("  Late-start profiling activated!\n");

    // Phase 6: Do GPU work AFTER late-start (SHOULD be traced)
    printf("\nPhase 6: GPU work after late-start (should be traced)...\n");

    // Reset result buffer
    std::fill(h_c.begin(), h_c.end(), 0.0f);
    HIP_CHECK(hipMemcpy(d_c, h_c.data(), size, hipMemcpyHostToDevice));

    // Record API call count before
    uint64_t api_calls_before = g_hip_api_calls.load(std::memory_order_relaxed);

    // Execute traced operations
    const int num_kernel_launches = 5;
    for(int i = 0; i < num_kernel_launches; i++)
    {
        vectorAdd<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_c.data(), d_c, size, hipMemcpyDeviceToHost));

    // Record API call count after
    uint64_t api_calls_after = g_hip_api_calls.load(std::memory_order_relaxed);
    uint64_t api_calls_delta = api_calls_after - api_calls_before;

    printf("  Traced HIP API calls: %lu\n", api_calls_delta);

    // Verify computation result
    bool correct = true;
    for(int i = 0; i < N; i++)
    {
        if(h_c[i] != 3.0f)
        {
            correct = false;
            break;
        }
    }

    // Phase 7: Validation
    printf("\nPhase 7: Validation...\n");

    bool test_passed = true;

    // We expect at least num_kernel_launches API calls (kernel launches)
    // plus potentially some additional HIP API calls (memcpy, sync, etc.)
    if(api_calls_delta < num_kernel_launches)
    {
        fprintf(stderr,
                "  FAIL: Expected at least %d HIP API calls, got %lu\n",
                num_kernel_launches,
                api_calls_delta);
        test_passed = false;
    }
    else
    {
        printf("  PASS: HIP API calls traced (%lu >= %d expected)\n",
               api_calls_delta,
               num_kernel_launches);
    }

    // Note: Kernel dispatch tracing is NOT validated because it's not supported
    // in late-start scenarios (requires queue interception at queue creation time)

    if(!correct)
    {
        fprintf(stderr, "  FAIL: Computation produced incorrect results\n");
        test_passed = false;
    }
    else
    {
        printf("  PASS: Computation correct\n");
    }

    // Cleanup
    printf("\nPhase 8: Cleanup...\n");
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    dlclose(tool_handle);
    dlclose(sdk_handle);

    // Final result
    printf("\n=== Test %s ===\n", test_passed ? "PASSED" : "FAILED");
    return test_passed ? 0 : 1;
}
