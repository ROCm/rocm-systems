/*
 * Copyright (c) Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * SDMA Test - exercises memory copy operations that use SDMA engines
 *
 * This example performs Host-to-Device, Device-to-Device, and Device-to-Host
 * memory transfers to exercise the SDMA (System DMA) engines on AMD GPUs.
 *
 * Usage: sdma_test [OPTIONS]
 *   -s, --size <MB>        Transfer size in MB (default: 512)
 *   -n, --iterations <N>   Number of iterations (default: 10)
 *   -c, --copies <N>       Number of copies per iteration (default: 10)
 *   -d, --device <ID>      GPU device ID (default: 0)
 *   -h, --help             Show this help message
 *
 * Run with profiler:
 *   rocprof-sys-run -e ROCPROFSYS_AMD_SMI_METRICS=sdma_usage -- ./sdma_test
 */

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd)                                                                   \
    do                                                                                   \
    {                                                                                    \
        hipError_t error = cmd;                                                          \
        if(error != hipSuccess)                                                          \
        {                                                                                \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(error),         \
                    __FILE__, __LINE__);                                                 \
            exit(1);                                                                     \
        }                                                                                \
    } while(0)

// Default configuration
static constexpr size_t DEFAULT_SIZE_MB    = 512;
static constexpr int    DEFAULT_ITERATIONS = 10;
static constexpr int    DEFAULT_COPIES     = 10;
static constexpr int    DEFAULT_DEVICE_ID  = 0;

volatile bool g_running = true;

void
signal_handler(int signum)
{
    (void) signum;
    g_running = false;
    printf("\nStopping...\n");
}

void
print_usage(const char* prog_name)
{
    printf("SDMA Test - exercises memory copy operations using SDMA engines\n\n");
    printf("Usage: %s [OPTIONS]\n\n", prog_name);
    printf("Options:\n");
    printf("  -s, --size <MB>        Transfer size in MB (default: %zu)\n",
           DEFAULT_SIZE_MB);
    printf("  -n, --iterations <N>   Number of iterations, 0=infinite (default: %d)\n",
           DEFAULT_ITERATIONS);
    printf("  -c, --copies <N>       Number of copies per iteration (default: %d)\n",
           DEFAULT_COPIES);
    printf("  -d, --device <ID>      GPU device ID (default: %d)\n", DEFAULT_DEVICE_ID);
    printf("  -h, --help             Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s                     # Run with defaults (512MB, 10 iterations)\n",
           prog_name);
    printf("  %s -s 1024 -n 5        # 1GB transfers, 5 iterations\n", prog_name);
    printf("  %s -s 256 -n 0         # 256MB transfers, run until Ctrl+C\n", prog_name);
    printf("\nRun with profiler:\n");
    printf("  rocprof-sys-run -e ROCPROFSYS_AMD_SMI_METRICS=sdma_usage -- %s\n",
           prog_name);
}

size_t
parse_size(const char* str)
{
    size_t value = std::strtoull(str, nullptr, 10);
    size_t len   = std::strlen(str);
    if(len > 0)
    {
        char suffix = str[len - 1];
        switch(suffix)
        {
            case 'G':
            case 'g': value *= 1024;  // fall through
            case 'M':
            case 'm': break;  // already in MB
            case 'K':
            case 'k': value /= 1024; break;
            default: break;
        }
    }
    return value;
}

int
main(int argc, char** argv)
{
    // Configuration with defaults
    size_t size_mb    = DEFAULT_SIZE_MB;
    int    iterations = DEFAULT_ITERATIONS;
    int    num_copies = DEFAULT_COPIES;
    int    device_id  = DEFAULT_DEVICE_ID;

    // Parse command line arguments
    for(int i = 1; i < argc; i++)
    {
        if(std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        else if((std::strcmp(argv[i], "-s") == 0 ||
                 std::strcmp(argv[i], "--size") == 0) &&
                i + 1 < argc)
        {
            size_mb = parse_size(argv[++i]);
        }
        else if((std::strcmp(argv[i], "-n") == 0 ||
                 std::strcmp(argv[i], "--iterations") == 0) &&
                i + 1 < argc)
        {
            iterations = std::atoi(argv[++i]);
        }
        else if((std::strcmp(argv[i], "-c") == 0 ||
                 std::strcmp(argv[i], "--copies") == 0) &&
                i + 1 < argc)
        {
            num_copies = std::atoi(argv[++i]);
        }
        else if((std::strcmp(argv[i], "-d") == 0 ||
                 std::strcmp(argv[i], "--device") == 0) &&
                i + 1 < argc)
        {
            device_id = std::atoi(argv[++i]);
        }
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    size_t size_bytes = size_mb * 1024ULL * 1024ULL;

    // Setup signal handler for Ctrl+C
    signal(SIGINT, signal_handler);

    printf("SDMA Test Configuration:\n");
    printf("  Transfer size: %zu MB (%zu bytes)\n", size_mb, size_bytes);
    printf("  Iterations: %d%s\n", iterations, iterations == 0 ? " (infinite)" : "");
    printf("  Copies per iteration: %d\n", num_copies);
    printf("  Device ID: %d\n", device_id);
    if(iterations == 0)
    {
        printf("  Press Ctrl+C to stop\n");
    }
    printf("\n");

    // Set device
    HIP_CHECK(hipSetDevice(device_id));

    // Get device properties
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    printf("Using GPU: %s\n\n", props.name);

    // Allocate host memory (pinned for better DMA performance)
    void* h_src = nullptr;
    void* h_dst = nullptr;
    HIP_CHECK(hipHostMalloc(&h_src, size_bytes, hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_dst, size_bytes, hipHostMallocDefault));

    // Initialize source data
    memset(h_src, 0xAB, size_bytes);
    memset(h_dst, 0x00, size_bytes);

    // Allocate device memory
    void* d_buf1 = nullptr;
    void* d_buf2 = nullptr;
    HIP_CHECK(hipMalloc(&d_buf1, size_bytes));
    HIP_CHECK(hipMalloc(&d_buf2, size_bytes));

    // Create stream for async operations
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    int    round         = 0;
    double total_h2d_bw  = 0.0;
    double total_d2d_bw  = 0.0;
    double total_d2h_bw  = 0.0;
    bool   infinite_mode = (iterations == 0);
    int    target_rounds = infinite_mode ? INT32_MAX : iterations;

    while(g_running && round < target_rounds)
    {
        round++;
        printf("=== Iteration %d", round);
        if(!infinite_mode)
        {
            printf("/%d", iterations);
        }
        printf(" ===\n");

        // H2D
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < num_copies && g_running; i++)
        {
            HIP_CHECK(
                hipMemcpyAsync(d_buf1, h_src, size_bytes, hipMemcpyHostToDevice, stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        auto   end     = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        double bw      = (double) (size_bytes * num_copies) / elapsed / 1e9;
        total_h2d_bw += bw;
        printf("  H2D: %d x %zu MB in %.3f s (%.2f GB/s)\n", num_copies, size_mb, elapsed,
               bw);

        if(!g_running) break;

        // D2D
        start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < num_copies && g_running; i++)
        {
            HIP_CHECK(hipMemcpyAsync(d_buf2, d_buf1, size_bytes, hipMemcpyDeviceToDevice,
                                     stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        end     = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        bw      = (double) (size_bytes * num_copies) / elapsed / 1e9;
        total_d2d_bw += bw;
        printf("  D2D: %d x %zu MB in %.3f s (%.2f GB/s)\n", num_copies, size_mb, elapsed,
               bw);

        if(!g_running) break;

        // D2H
        start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < num_copies && g_running; i++)
        {
            HIP_CHECK(
                hipMemcpyAsync(h_dst, d_buf2, size_bytes, hipMemcpyDeviceToHost, stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        end     = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        bw      = (double) (size_bytes * num_copies) / elapsed / 1e9;
        total_d2h_bw += bw;
        printf("  D2H: %d x %zu MB in %.3f s (%.2f GB/s)\n\n", num_copies, size_mb,
               elapsed, bw);
    }

    // Print summary
    printf("=== Summary ===\n");
    printf("Completed %d iteration(s)\n", round);
    if(round > 0)
    {
        printf("Average bandwidth:\n");
        printf("  H2D: %.2f GB/s\n", total_h2d_bw / round);
        printf("  D2D: %.2f GB/s\n", total_d2d_bw / round);
        printf("  D2H: %.2f GB/s\n", total_d2h_bw / round);
    }

    // Verify data integrity
    printf("\nData verification: ");
    if(memcmp(h_src, h_dst, size_bytes) == 0)
    {
        printf("PASSED\n");
    }
    else
    {
        printf("FAILED\n");
    }

    // Cleanup
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_buf1));
    HIP_CHECK(hipFree(d_buf2));
    HIP_CHECK(hipHostFree(h_src));
    HIP_CHECK(hipHostFree(h_dst));

    return 0;
}
