// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

/**
 * AMD GPU Mega Kernel Unit Test - Host Code
 *
 * This file contains the host-side code to run the GPU mega kernel
 * and report test results for various AMD GPU architectures.
 *
 * Supported architectures:
 *   - gfx90a  (MI250/CDNA2)
 *   - gfx942  (MI300/CDNA3)
 *   - gfx950  (MI350/CDNA4)
 *   - gfx1150 (Strix Point / RDNA 3.5)
 *   - gfx1151 (Strix Halo / RDNA 3.5)
 *   - gfx1152 (Krackan / RDNA 3.5)
 *   - gfx1200 (RX 9060/RDNA4)
 *   - gfx1201 (RX 9070 XT/RDNA4)
 *
 * Build: make (default GPU_ARCH=gfx1151; override e.g. GPU_ARCH=gfx1201 make) or manual:
 *   hipcc -O2 -g -std=c++17 -Wall -Wextra -fPIC --offload-arch=gfx1151 -I. -o
 * mega_kernel_test mega_kernel.hip main.cpp
 *
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <hip/hip_runtime.h>
#include <vector>

#include "atomic_global_buffers.h"
#include "mega_kernel_host_arch.hpp"
#include "mega_kernel_types.h"

// Default configuration
#define DEFAULT_BATCH_SIZE     1024
#define DEFAULT_BLOCK_SIZE     256
#define DEFAULT_NUM_ITERATIONS 10
#define MIN_BATCH_SIZE         64
#define MAX_BATCH_SIZE         (1024 * 1024 * 16)  // 16M elements max
#define MAX_NUM_ITERATIONS     10000

// Command-line options
struct TestConfig
{
    int      batch_size;
    int      block_size;
    int      num_iterations;
    MfmaMode mfma_mode;
    bool     verbose;
    bool     help;
};

void
print_usage(const char* prog_name)
{
    printf("Usage: %s [OPTIONS]\n\n", prog_name);
    printf("AMD GPU Mega Kernel Unit Test\n\n");
    printf("Options:\n");
    printf("  -b, --batch-size <N>   Set batch/problem size (default: %d)\n",
           DEFAULT_BATCH_SIZE);
    printf("                         Valid range: %d to %d\n", MIN_BATCH_SIZE,
           MAX_BATCH_SIZE);
    printf("  -t, --block-size <N>   Set thread block size (default: %d)\n",
           DEFAULT_BLOCK_SIZE);
    printf("                         Must be power of 2, 64-256 (max 256 due to kernel "
           "shared memory)\n");
    printf("  -n, --num-iterations <N> Number of kernel iterations (default: %d)\n",
           DEFAULT_NUM_ITERATIONS);
    printf("                         Valid range: 1 to %d\n", MAX_NUM_ITERATIONS);
    printf("  -m, --mfma-mode <MODE> Set MFMA test mode (default: both)\n");
    printf("                         Modes: both, asm, builtin\n");
    printf("  -v, --verbose          Enable verbose output\n");
    printf("  -h, --help             Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s                     # Run with default settings\n", prog_name);
    printf("  %s -b 4096             # Run with 4096 elements\n", prog_name);
    printf("  %s -b 65536 -t 512     # Run with 64K elements, 512 threads/block\n",
           prog_name);
    printf("  %s -n 100              # Run kernel 100 times\n", prog_name);
    printf("  %s --mfma-mode asm     # Test MFMA using inline assembly only\n",
           prog_name);
    printf("  %s -m builtin          # Test MFMA using HIP builtins only\n", prog_name);
}

TestConfig
parse_args(int argc, char** argv)
{
    TestConfig config;
    config.batch_size     = DEFAULT_BATCH_SIZE;
    config.block_size     = DEFAULT_BLOCK_SIZE;
    config.num_iterations = DEFAULT_NUM_ITERATIONS;
    config.mfma_mode      = MFMA_MODE_BOTH;
    config.verbose        = false;
    config.help           = false;

    static struct option long_options[] = { { "batch-size", required_argument, 0, 'b' },
                                            { "block-size", required_argument, 0, 't' },
                                            { "num-iterations", required_argument, 0,
                                              'n' },
                                            { "mfma-mode", required_argument, 0, 'm' },
                                            { "verbose", no_argument, 0, 'v' },
                                            { "help", no_argument, 0, 'h' },
                                            { 0, 0, 0, 0 } };

    int opt;
    int option_index = 0;
    while((opt = getopt_long(argc, argv, "b:t:n:m:vh", long_options, &option_index)) !=
          -1)
    {
        switch(opt)
        {
            case 'b':
                config.batch_size = atoi(optarg);
                if(config.batch_size < MIN_BATCH_SIZE)
                {
                    fprintf(stderr, "Warning: batch_size too small, using minimum: %d\n",
                            MIN_BATCH_SIZE);
                    config.batch_size = MIN_BATCH_SIZE;
                }
                if(config.batch_size > MAX_BATCH_SIZE)
                {
                    fprintf(stderr, "Warning: batch_size too large, using maximum: %d\n",
                            MAX_BATCH_SIZE);
                    config.batch_size = MAX_BATCH_SIZE;
                }
                break;
            case 't':
                config.block_size = atoi(optarg);
                // Validate block size (must be power of 2, 64-256)
                // NOTE: Max 256 due to fixed-size __shared__ arrays in kernel (e.g.,
                // vmem_lds_test_buffer[256])
                if(config.block_size < 64 || config.block_size > 256 ||
                   (config.block_size & (config.block_size - 1)) != 0)
                {
                    fprintf(stderr,
                            "Warning: invalid block_size (must be 64-256, power of 2), "
                            "using default: %d\n",
                            DEFAULT_BLOCK_SIZE);
                    config.block_size = DEFAULT_BLOCK_SIZE;
                }
                break;
            case 'n':
                config.num_iterations = atoi(optarg);
                if(config.num_iterations < 1)
                {
                    fprintf(stderr,
                            "Warning: num_iterations too small, using minimum: 1\n");
                    config.num_iterations = 1;
                }
                if(config.num_iterations > MAX_NUM_ITERATIONS)
                {
                    fprintf(stderr,
                            "Warning: num_iterations too large, using maximum: %d\n",
                            MAX_NUM_ITERATIONS);
                    config.num_iterations = MAX_NUM_ITERATIONS;
                }
                break;
            case 'm':
                if(strcmp(optarg, "asm") == 0)
                {
                    config.mfma_mode = MFMA_MODE_ASM;
                }
                else if(strcmp(optarg, "builtin") == 0)
                {
                    config.mfma_mode = MFMA_MODE_BUILTIN;
                }
                else if(strcmp(optarg, "both") == 0)
                {
                    config.mfma_mode = MFMA_MODE_BOTH;
                }
                else
                {
                    fprintf(stderr, "Warning: invalid mfma-mode '%s', using 'both'\n",
                            optarg);
                    config.mfma_mode = MFMA_MODE_BOTH;
                }
                break;
            case 'v': config.verbose = true; break;
            case 'h': config.help = true; break;
            default: config.help = true; break;
        }
    }

    return config;
}

// Macro for HIP error checking
#define HIP_CHECK(cmd)                                                                   \
    do                                                                                   \
    {                                                                                    \
        hipError_t error = (cmd);                                                        \
        if(error != hipSuccess)                                                          \
        {                                                                                \
            fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(error),        \
                    __FILE__, __LINE__);                                                 \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while(0)

void
print_separator()
{
    printf("============================================================================="
           "===\n");
}

void
print_test_result(const char* test_name, int passed, int expected = 1)
{
    // Width of 48 to accommodate longest test name: "Async LDS Store
    // (global_store_async_from_lds)"
    if(passed < 0)
    {
        // Negative value indicates test was bypassed (not applicable to this arch)
        printf("  %-48s : \033[33mBYPASSED\033[0m (N/A)\n", test_name);
    }
    else if(passed >= expected)
    {
        printf("  %-48s : \033[32mPASS\033[0m (%d/%d)\n", test_name, passed, expected);
    }
    else
    {
        printf("  %-48s : \033[31mFAIL\033[0m (%d/%d)\n", test_name, passed, expected);
    }
}

// Global variable to store detected architecture for later use
static char g_arch_name[64] = "";
// 0=unknown, 1=gfx90a, 2=gfx942, 3=gfx950, 4=gfx1200, 5=gfx1201, 6=gfx115x
static int g_arch_type = 0;

void
print_device_info()
{
    int             device;
    hipDeviceProp_t props;

    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    mega_kernel_host::detect_architecture(props.gcnArchName, g_arch_name, sizeof(g_arch_name),
                                         &g_arch_type);

    print_separator();
    printf("AMD GPU MEGA KERNEL UNIT TEST\n");
    print_separator();
    printf("Device Information:\n");
    printf("  Name:                  %s\n", props.name);
    printf("  GCN Architecture:      %s\n", props.gcnArchName);
    printf("  Compute Units:         %d\n", props.multiProcessorCount);
    printf("  Max Threads/Block:     %d\n", props.maxThreadsPerBlock);
    printf("  Warp Size:             %d\n", props.warpSize);
    printf("  Global Memory:         %.2f GB\n",
           props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared Memory/Block:   %zu KB\n", props.sharedMemPerBlock / 1024);
    printf("  Clock Rate:            %d MHz\n", props.clockRate / 1000);
    print_separator();
    printf("Detected Platform:       %s\n",
           mega_kernel_host::arch_description(g_arch_type));
    print_separator();
}

int
main(int argc, char** argv)
{
    // Parse command-line arguments
    TestConfig config = parse_args(argc, argv);

    if(config.help)
    {
        print_usage(argv[0]);
        return 0;
    }

    // Print device info
    print_device_info();

    // Configuration from command-line or defaults
    const int BUFFER_SIZE = config.batch_size;
    const int BLOCK_SIZE  = config.block_size;
    const int NUM_BLOCKS  = (BUFFER_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Global atomic scratch: sizes and index layout — see atomic_global_buffers.h
    const int GLOBAL_INT_SIZE    = MEGA_KERNEL_GLOBAL_INT_ELEMENTS;
    const int GLOBAL_FLOAT_SIZE  = MEGA_KERNEL_GLOBAL_FLOAT_ELEMENTS;
    const int GLOBAL_DOUBLE_SIZE = MEGA_KERNEL_GLOBAL_DOUBLE_ELEMENTS;

    printf("Test Configuration:\n");
    printf("  Batch/Buffer Size:     %d elements\n", BUFFER_SIZE);
    printf("  Block Size:            %d threads\n", BLOCK_SIZE);
    printf("  Number of Blocks:      %d\n", NUM_BLOCKS);
    printf("  Total Threads:         %d\n", NUM_BLOCKS * BLOCK_SIZE);
    const char* mfma_mode_str = (config.mfma_mode == MFMA_MODE_ASM)       ? "asm"
                                : (config.mfma_mode == MFMA_MODE_BUILTIN) ? "builtin"
                                                                          : "both";
    printf("  MFMA Test Mode:        %s\n", mfma_mode_str);
    printf("  Num Iterations:        %d\n", config.num_iterations);
    if(config.verbose)
    {
        printf("  Verbose Mode:          enabled\n");
    }
    print_separator();

    // Allocate host memory
    TestResults h_results;
    memset(&h_results, 0, sizeof(TestResults));

    float* h_input  = (float*) malloc(BUFFER_SIZE * sizeof(float));
    float* h_output = (float*) malloc(BUFFER_SIZE * sizeof(float));

    // Initialize input buffer
    for(int i = 0; i < BUFFER_SIZE; i++)
    {
        h_input[i]  = (float) i;
        h_output[i] = 0.0f;
    }

    // Allocate device memory
    TestResults* d_results;
    float*       d_global_float;
    double*      d_global_double;
    int*         d_global_int;
    float*       d_input;
    float*       d_output;
    float*       d_async_lds_src;
    float*       d_async_lds_dst;

    const int ASYNC_LDS_SIZE = 256;  // Size for async LDS test buffers

    HIP_CHECK(hipMalloc(&d_results, sizeof(TestResults)));
    HIP_CHECK(hipMalloc(&d_global_float, GLOBAL_FLOAT_SIZE * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_global_double, GLOBAL_DOUBLE_SIZE * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_global_int, GLOBAL_INT_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_input, BUFFER_SIZE * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, BUFFER_SIZE * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_async_lds_src, ASYNC_LDS_SIZE * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_async_lds_dst, ASYNC_LDS_SIZE * sizeof(float)));

    // Host-prepared zeros for global atomic buffers (kernel CAS etc. expect initial 0).
    std::vector<int>    h_global_int(static_cast<size_t>(GLOBAL_INT_SIZE), 0);
    std::vector<float>  h_global_float(static_cast<size_t>(GLOBAL_FLOAT_SIZE), 0.0f);
    std::vector<double> h_global_double(static_cast<size_t>(GLOBAL_DOUBLE_SIZE), 0.0);

    // Initialize device memory
    HIP_CHECK(hipMemset(d_results, 0, sizeof(TestResults)));
    HIP_CHECK(hipMemcpy(d_global_int, h_global_int.data(),
                        h_global_int.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_global_float, h_global_float.data(),
                        h_global_float.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_global_double, h_global_double.data(),
                        h_global_double.size() * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_input, h_input, BUFFER_SIZE * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_output, 0, BUFFER_SIZE * sizeof(float)));
    HIP_CHECK(hipMemset(d_async_lds_src, 0, ASYNC_LDS_SIZE * sizeof(float)));
    HIP_CHECK(hipMemset(d_async_lds_dst, 0, ASYNC_LDS_SIZE * sizeof(float)));

    // Create texture and surface objects for Strix (gfx1150/1151/1152) to exercise
    // INSTS_TEX_LOAD (tex1Dfetch) and INSTS_TEX_STORE (surf1Dwrite).
    hipTextureObject_t tex_obj       = 0;
    hipSurfaceObject_t surf_obj      = 0;
    float*             d_surf_buffer = nullptr;
    if(g_arch_type == 6)
    {
        hipResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType                = hipResourceTypeLinear;
        resDesc.res.linear.sizeInBytes = BUFFER_SIZE * sizeof(float);
        resDesc.res.linear.desc =
            hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);

        resDesc.res.linear.devPtr = d_input;
        hipTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.normalizedCoords = 0;
        texDesc.filterMode       = hipFilterModePoint;
        texDesc.addressMode[0]   = hipAddressModeClamp;
        HIP_CHECK(hipCreateTextureObject(&tex_obj, &resDesc, &texDesc, nullptr));

        HIP_CHECK(hipMalloc(&d_surf_buffer, BUFFER_SIZE * sizeof(float)));
        HIP_CHECK(hipMemset(d_surf_buffer, 0, BUFFER_SIZE * sizeof(float)));
        resDesc.res.linear.devPtr = d_surf_buffer;
        HIP_CHECK(hipCreateSurfaceObject(&surf_obj, &resDesc));
    }

    printf("Running GPU Mega Kernel for %s...\n",
           mega_kernel_host::arch_description(g_arch_type));
    if(config.num_iterations > 1)
    {
        printf("  Running %d iterations...\n", config.num_iterations);
    }
    print_separator();

    // Create events for timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Compute dynamic shared memory size (must match kernel expectations)
    // Layout in test_lds_operations:
    //   - lds_float: BLOCK_SIZE * sizeof(float)
    //   - lds_int: BLOCK_SIZE * sizeof(int)
    //   - lds_double: (BLOCK_SIZE / 2) * sizeof(double)
    //   - lds_atomic_counter: sizeof(int) for LDS atomic test
    const size_t sharedMemBytes =
        static_cast<size_t>(BLOCK_SIZE) * sizeof(float) +       // lds_float
        static_cast<size_t>(BLOCK_SIZE) * sizeof(int) +         // lds_int
        static_cast<size_t>(BLOCK_SIZE / 2) * sizeof(double) +  // lds_double
        sizeof(int);                                            // lds_atomic_counter

    // Record start time
    HIP_CHECK(hipEventRecord(start, 0));

    // Run kernel for the specified number of iterations
    for(int iter = 0; iter < config.num_iterations; iter++)
    {
        // Reset device memory for each iteration (except first)
        // NOTE: d_input must also be reset because test_vmem_operations modifies it
        if(iter > 0)
        {
            HIP_CHECK(hipMemset(d_results, 0, sizeof(TestResults)));
            HIP_CHECK(hipMemcpy(d_global_int, h_global_int.data(),
                                h_global_int.size() * sizeof(int),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_global_float, h_global_float.data(),
                                h_global_float.size() * sizeof(float),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_global_double, h_global_double.data(),
                                h_global_double.size() * sizeof(double),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_input, h_input, BUFFER_SIZE * sizeof(float),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemset(d_output, 0, BUFFER_SIZE * sizeof(float)));
        }

        // Launch kernel
        hipLaunchKernelGGL(mega_kernel, dim3(NUM_BLOCKS), dim3(BLOCK_SIZE),
                           sharedMemBytes, 0, d_results, d_global_float, d_global_double,
                           d_global_int, d_input, d_output, d_async_lds_src,
                           d_async_lds_dst, BUFFER_SIZE, (int) config.mfma_mode, tex_obj,
                           surf_obj);

        // Check for launch errors
        HIP_CHECK(hipGetLastError());
    }

    // Wait for all iterations to complete
    HIP_CHECK(hipDeviceSynchronize());

    // Record stop time
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_elapsed_ms;
    HIP_CHECK(hipEventElapsedTime(&total_elapsed_ms, start, stop));
    float avg_elapsed_ms = total_elapsed_ms / config.num_iterations;

    if(tex_obj != 0)
    {
        HIP_CHECK(hipDestroyTextureObject(tex_obj));
    }
    if(surf_obj != 0)
    {
        HIP_CHECK(hipDestroySurfaceObject(surf_obj));
    }
    if(d_surf_buffer != nullptr)
    {
        HIP_CHECK(hipFree(d_surf_buffer));
    }

    // Copy results back
    HIP_CHECK(
        hipMemcpy(&h_results, d_results, sizeof(TestResults), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_output, d_output, BUFFER_SIZE * sizeof(float),
                        hipMemcpyDeviceToHost));

    // Verify memory operations output
    bool memory_verified = true;
    for(int i = 0; i < BUFFER_SIZE && memory_verified; i++)
    {
        float expected = h_input[i] * 2.0f + 1.0f;
        if(fabsf(h_output[i] - expected) > 0.001f)
        {
            memory_verified = false;
            printf("Memory verification failed at index %d: expected %.3f, got %.3f\n", i,
                   expected, h_output[i]);
        }
    }

    // Print results
    printf("TEST RESULTS\n");
    print_separator();

    printf("\n[Category: Warp/Wave Operations]\n");
    print_test_result("Warp Shuffle (shfl/shfl_xor/shfl_up/down)",
                      h_results.warp_shuffle_passed);
    print_test_result("Warp Ballot (__ballot/__any/__all)", h_results.warp_ballot_passed);
    print_test_result("Warp Permute (ds_bpermute/ds_permute)",
                      h_results.warp_permute_passed);
    print_test_result("Warp Reduce (shuffle-based reduction)",
                      h_results.warp_reduce_passed);

    printf("\n[Category: FP8/BF8 Conversions (gfx942+/RDNA4)]\n");
    print_test_result("FP8 E4M3 Conversion", h_results.fp8_convert_passed);
    print_test_result("BF8 E5M2 Conversion", h_results.bf8_convert_passed);

    printf("\n[Category: FP16/BF16 Operations]\n");
    print_test_result("FP16 Arithmetic", h_results.fp16_convert_passed);
    print_test_result("BF16 Conversion", h_results.bf16_convert_passed);

    printf("\n[Category: FP32/FP64 Arithmetic]\n");
    print_test_result("FP32 Arithmetic (add/sub/mul/div/fma)",
                      h_results.fp32_arith_passed);
    print_test_result("FP64 Arithmetic (add/sub/mul/div/fma)",
                      h_results.fp64_arith_passed);

    printf("\n[Category: Integer Operations]\n");
    print_test_result("INT32 Arithmetic & Bit Ops", h_results.int32_arith_passed);
    print_test_result("INT64 Arithmetic", h_results.int64_arith_passed);

    printf("\n[Category: Packed Operations]\n");
    print_test_result("Packed FP16 (half2)", h_results.pk_f16_passed);
    print_test_result("Packed BF16", h_results.pk_bf16_passed);
    print_test_result("Packed Add Operations", h_results.pk_add_passed);

    printf("\n[Category: Atomic Operations]\n");
    print_test_result("Atomic Add FP32", h_results.atomic_add_f32_passed);
    print_test_result("Atomic Add FP64 (HW atomic on CDNA)",
                      h_results.atomic_add_f64_passed);
    print_test_result("Atomic Min FP32", h_results.atomic_min_f32_passed);
    print_test_result("Atomic Max FP32", h_results.atomic_max_f32_passed);
    print_test_result("Atomic Integer Operations", h_results.atomic_int_passed);
    print_test_result("LDS Atomic Operations", h_results.lds_atomic_passed);

    printf("\n[Category: Transcendental Functions]\n");
    print_test_result("Sine (sinf)", h_results.trans_sin_passed);
    print_test_result("Cosine (cosf)", h_results.trans_cos_passed);
    print_test_result("Exponential (expf)", h_results.trans_exp_passed);
    print_test_result("Logarithm (logf)", h_results.trans_log_passed);
    print_test_result("Square Root (sqrtf)", h_results.trans_sqrt_passed);
    print_test_result("Reciprocal (1/x)", h_results.trans_rcp_passed);
    print_test_result("Reciprocal Square Root (rsqrtf)", h_results.trans_rsq_passed);

    printf("\n[Category: Memory Operations]\n");
    print_test_result("Global Memory Load", h_results.global_load_passed);
    print_test_result("Global Memory Store", h_results.global_store_passed);
    print_test_result("LDS Load", h_results.lds_load_passed);
    print_test_result("LDS Store", h_results.lds_store_passed);
    print_test_result("Memory Output Verification", memory_verified ? 1 : 0);

    printf("\n[Category: DOT Product Operations]\n");
    print_test_result("DOT4 (4-element INT8 dot)", h_results.dot4_passed);
    print_test_result("DOT8 (8-element INT4 dot)", h_results.dot8_passed);

    printf("\n[Category: MFMA Operations (gfx90a+/CDNA)]\n");
    print_test_result("MFMA FP32 (v_mfma_f32_32x32x2f32)", h_results.mfma_f32_passed);
    print_test_result("MFMA FP64 (v_mfma_f64_16x16x4f64)", h_results.mfma_f64_passed);
    print_test_result("MFMA FP16 (v_mfma_f32_32x32x8f16)", h_results.mfma_f16_passed);
    print_test_result("MFMA BF16 (v_mfma_f32_32x32x*bf16)", h_results.mfma_bf16_passed);
    print_test_result("MFMA INT8 (v_mfma_i32_32x32x*i8)", h_results.mfma_i8_passed);
    print_test_result("MFMA FP8 (v_mfma_f32_32x32x16_fp8)", h_results.mfma_f8_passed);

    printf("\n[Category: Dual-Issue VALU (instruction patterns only)]\n");
    print_test_result("Dual-Issue Patterns (not scheduler-verified)",
                      h_results.dual_issue_passed);

    printf("\n[Category: Lane/Thread Information]\n");
    print_test_result("Lane ID Builtin", h_results.lane_id_passed);
    print_test_result("Wavefront Size", h_results.wavefront_passed);

    printf("\n[Category: Async LDS Operations (gfx950+/RDNA4)]\n");
    print_test_result("Async LDS Load (global_load_async_to_lds)",
                      h_results.async_lds_load_passed);
    print_test_result("Async LDS Store (global_store_async_from_lds)",
                      h_results.async_lds_store_passed);

    printf("\n[Category: WMMA Operations (RDNA4/gfx12 - Wave32)]\n");
    print_test_result("WMMA FP16 → FP32 (wmma_f32_16x16x16_f16)",
                      h_results.wmma_f16_passed);
    print_test_result("WMMA BF16 → FP32 (wmma_f32_16x16x16_bf16)",
                      h_results.wmma_bf16_passed);
    print_test_result("WMMA INT8 → INT32 (wmma_i32_16x16x16_iu8)",
                      h_results.wmma_i8_passed);

    printf("\n[Category: VMEM (Vector Memory) Operations - Inline ASM]\n");
    print_test_result("Flat Load/Store (flat_load_b32/flat_store_b32)",
                      h_results.vmem_flat_passed);
    print_test_result("Global Load/Store (global_load_b32/global_store_b32)",
                      h_results.vmem_global_passed);
    print_test_result("Buffer Load/Store (buffer_load_dword)",
                      h_results.vmem_buffer_passed);
    print_test_result("Scratch Memory (scratch_load/scratch_store)",
                      h_results.vmem_scratch_passed);
    print_test_result("LDS Operations (ds_read_b32/ds_write_b32)",
                      h_results.vmem_lds_passed);
    print_test_result("Texture Load (tex1Dfetch / INSTS_TEX_LOAD)",
                      h_results.vmem_tex_load_passed);
    print_test_result("Texture Store (surf1Dwrite / INSTS_TEX_STORE)",
                      h_results.vmem_tex_store_passed);

    printf("\n[Category: Atomic Operations - Inline ASM]\n");
    print_test_result("Global INT Atomics (global_atomic_add/sub/min/max)",
                      h_results.atomic_global_int_passed);
    print_test_result("Global FP32 Atomics (global_atomic_add_f32)",
                      h_results.atomic_global_f32_passed);
    print_test_result("Global FP64 Atomics (global_atomic_add_f64/min/max)",
                      h_results.atomic_global_f64_passed);
    print_test_result("Flat INT Atomics (flat_atomic_add)",
                      h_results.atomic_flat_int_passed);
    print_test_result("DS INT Atomics (ds_add_u32/min_i32/max_u32)",
                      h_results.atomic_ds_int_passed);
    print_test_result("DS FP32 Atomics (ds_add_f32/min_f32)",
                      h_results.atomic_ds_f32_passed);
    print_test_result("DS FP64 Atomics (ds_add_f64)", h_results.atomic_ds_f64_passed);
    print_test_result("Packed FP16 Atomics (global_atomic_pk_add_f16)",
                      h_results.atomic_pk_f16_passed);
    print_test_result("Packed BF16 Atomics (global_atomic_pk_add_bf16)",
                      h_results.atomic_pk_bf16_passed);
    print_test_result("CAS Atomics (global_atomic_cmpswap)", h_results.atomic_cas_passed);

    print_separator();

    // Summary - count passed, failed, and bypassed tests
    int passed_categories   = 0;
    int bypassed_categories = 0;
    int failed_categories   = 0;

// Helper macro to count test results
// > 0 = passed, < 0 = bypassed, == 0 = failed
#define COUNT_TEST(result)                                                               \
    if((result) > 0)                                                                     \
        passed_categories++;                                                             \
    else if((result) < 0)                                                                \
        bypassed_categories++;                                                           \
    else                                                                                 \
        failed_categories++;

    COUNT_TEST(h_results.warp_shuffle_passed);
    COUNT_TEST(h_results.warp_ballot_passed);
    COUNT_TEST(h_results.warp_permute_passed);
    COUNT_TEST(h_results.warp_reduce_passed);
    COUNT_TEST(h_results.fp8_convert_passed);
    COUNT_TEST(h_results.bf8_convert_passed);
    COUNT_TEST(h_results.fp16_convert_passed);
    COUNT_TEST(h_results.bf16_convert_passed);
    COUNT_TEST(h_results.fp32_arith_passed);
    COUNT_TEST(h_results.fp64_arith_passed);
    COUNT_TEST(h_results.int32_arith_passed);
    COUNT_TEST(h_results.int64_arith_passed);
    COUNT_TEST(h_results.pk_f16_passed);
    COUNT_TEST(h_results.pk_bf16_passed);
    COUNT_TEST(h_results.pk_add_passed);
    COUNT_TEST(h_results.atomic_add_f32_passed);
    COUNT_TEST(h_results.atomic_add_f64_passed);
    COUNT_TEST(h_results.atomic_min_f32_passed);
    COUNT_TEST(h_results.atomic_max_f32_passed);
    COUNT_TEST(h_results.atomic_int_passed);
    COUNT_TEST(h_results.lds_atomic_passed);
    COUNT_TEST(h_results.trans_sin_passed);
    COUNT_TEST(h_results.trans_cos_passed);
    COUNT_TEST(h_results.trans_exp_passed);
    COUNT_TEST(h_results.trans_log_passed);
    COUNT_TEST(h_results.trans_sqrt_passed);
    COUNT_TEST(h_results.trans_rcp_passed);
    COUNT_TEST(h_results.trans_rsq_passed);
    COUNT_TEST(h_results.global_load_passed);
    COUNT_TEST(h_results.global_store_passed);
    COUNT_TEST(h_results.lds_load_passed);
    COUNT_TEST(h_results.lds_store_passed);
    COUNT_TEST(h_results.dot4_passed);
    COUNT_TEST(h_results.dot8_passed);
    COUNT_TEST(h_results.mfma_f32_passed);
    COUNT_TEST(h_results.mfma_f64_passed);
    COUNT_TEST(h_results.mfma_f16_passed);
    COUNT_TEST(h_results.mfma_bf16_passed);
    COUNT_TEST(h_results.mfma_i8_passed);
    COUNT_TEST(h_results.mfma_f8_passed);
    COUNT_TEST(h_results.dual_issue_passed);
    COUNT_TEST(h_results.lane_id_passed);
    COUNT_TEST(h_results.wavefront_passed);
    COUNT_TEST(h_results.async_lds_load_passed);
    COUNT_TEST(h_results.async_lds_store_passed);
    COUNT_TEST(h_results.wmma_f16_passed);
    COUNT_TEST(h_results.wmma_bf16_passed);
    COUNT_TEST(h_results.wmma_i8_passed);
    COUNT_TEST(h_results.vmem_flat_passed);
    COUNT_TEST(h_results.vmem_global_passed);
    COUNT_TEST(h_results.vmem_buffer_passed);
    COUNT_TEST(h_results.vmem_scratch_passed);
    COUNT_TEST(h_results.vmem_lds_passed);
    COUNT_TEST(h_results.vmem_tex_load_passed);
    COUNT_TEST(h_results.vmem_tex_store_passed);
    COUNT_TEST(h_results.atomic_global_int_passed);
    COUNT_TEST(h_results.atomic_global_f32_passed);
    COUNT_TEST(h_results.atomic_global_f64_passed);
    COUNT_TEST(h_results.atomic_flat_int_passed);
    COUNT_TEST(h_results.atomic_ds_int_passed);
    COUNT_TEST(h_results.atomic_ds_f32_passed);
    COUNT_TEST(h_results.atomic_ds_f64_passed);
    COUNT_TEST(h_results.atomic_pk_f16_passed);
    COUNT_TEST(h_results.atomic_pk_bf16_passed);
    COUNT_TEST(h_results.atomic_cas_passed);
    if(memory_verified)
        passed_categories++;
    else
        failed_categories++;

#undef COUNT_TEST

    int total_tests =
        passed_categories + failed_categories;  // Exclude bypassed from total

    printf("SUMMARY\n");
    print_separator();
    printf("Test Categories Passed:   %d / %d\n", passed_categories, total_tests);
    printf("Test Categories Failed:   %d\n", failed_categories);
    printf("Test Categories Bypassed: %d (N/A for this architecture)\n",
           bypassed_categories);
    printf("Kernel Iterations:        %d\n", config.num_iterations);
    printf("Total Execution Time:     %.3f ms\n", total_elapsed_ms);
    if(config.num_iterations > 1)
    {
        printf("Average Time/Iteration:   %.3f ms\n", avg_elapsed_ms);
    }

    if(failed_categories == 0)
    {
        printf("\n\033[32m*** OVERALL: PASSED (%d/%d, %d bypassed) ***\033[0m\n",
               passed_categories, total_tests, bypassed_categories);
    }
    else
    {
        printf(
            "\n\033[31m*** OVERALL: FAILED (%d/%d, %d failed, %d bypassed) ***\033[0m\n",
            passed_categories, total_tests, failed_categories, bypassed_categories);
    }
    print_separator();

    // Print architecture-specific feature summary
    printf("\n%s SPECIFIC FEATURES TESTED:\n",
           mega_kernel_host::arch_description(g_arch_type));

    switch(g_arch_type)
    {
        case 1:  // MI250/CDNA2
            printf("  - FP32/FP64/FP16/BF16 arithmetic\n");
            printf("  - Hardware FP64 atomics (add/min/max)\n");
            printf("  - DOT4/DOT8 integer dot products (v_dot4_u32_u8)\n");
            printf("  - MFMA (Matrix Fused Multiply-Add) instructions\n");
            printf("  - Wave64 operations\n");
            printf("  - LDS operations (32KB per CU)\n");
            break;

        case 2:  // MI300/CDNA3
            printf("  - All MI250 features plus:\n");
            printf("  - FP8/BF8 conversions (FNUZ format)\n");
            printf("  - TF32 (TensorFloat-32) support\n");
            printf("  - Sparse MFMA instructions\n");
            printf("  - Same-wave dual-issue via VOPD/VOPD3 encoding\n");
            printf("      - Two operations per instruction (X op + Y op)\n");
            printf("  - Enhanced Wave64 operations\n");
            break;

        case 3:  // MI350/CDNA4
            printf("  - All MI300 features plus:\n");
            printf("  - FP8/BF8 scaled conversions (OCP format)\n");
            printf("  - FP6/FP4 conversions\n");
            printf("  - Cross-wave dual-issue VALU (in addition to VOPD)\n");
            printf("      - 2 instructions/cycle from different waves\n");
            printf("      - Requires Wave32, single-cycle ops, different MACC units\n");
            printf("  - Async LDS (global_load_async_to_lds, "
                   "global_store_async_from_lds)\n");
            printf("  - Wave32 optimized operations\n");
            break;

        case 4:  // RX 9060/RDNA4
        case 5:  // RX 9070 XT/RDNA4
            printf("  - FP8/BF8 conversions (OCP format)\n");
            printf("  - FP16/BF16 optimized arithmetic\n");
            printf("  - WMMA (Wave Matrix Multiply Accumulate) instructions\n");
            printf("      - 16x16x16 matrix operations on Wave32\n");
            printf("      - FP16/BF16 → FP32, INT8 → INT32\n");
            printf("      - 2x throughput vs RDNA3 (1024 FLOPS/clock for FP16/BF16)\n");
            printf("  - Async LDS operations\n");
            printf("  - Wave32 native execution\n");
            printf("  - WGP (Workgroup Processor) architecture\n");
            printf("  - Limited FP64 support (consumer GPU)\n");
            break;

        case 6:  // Strix/Strix Halo (RDNA 3.5)
            printf("  - RDNA 3.5 APU iGPU: Strix Point (gfx1150) / Strix Halo (gfx1151) / "
                   "Krackan (gfx1152)\n");
            printf("  - Wave32 native, WMMA (FP16/BF16/INT8) 16x16x16\n");
            printf("  - VOPD same-wave dual-issue VALU\n");
            printf("  - No CDNA-style FP8/BF8 cvt builtins; scalar FP; s_singleuse_vdst\n");
            printf("  - Async LDS, WGP architecture\n");
            break;

        default:
            printf("  - Basic GPU operations\n");
            printf("  - FP32/FP64 arithmetic\n");
            printf("  - Integer operations\n");
            printf("  - LDS operations\n");
            break;
    }
    print_separator();

    // Cleanup
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_results));
    HIP_CHECK(hipFree(d_global_float));
    HIP_CHECK(hipFree(d_global_double));
    HIP_CHECK(hipFree(d_global_int));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_async_lds_src));
    HIP_CHECK(hipFree(d_async_lds_dst));
    free(h_input);
    free(h_output);

    return (failed_categories == 0) ? 0 : 1;
}
