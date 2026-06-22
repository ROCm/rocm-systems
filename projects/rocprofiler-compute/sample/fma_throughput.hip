// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// fma_throughput - a fused-multiply-add throughput microbenchmark.
//
// Launches one compute-bound FMA kernel per numeric precision the target GPU
// supports, so that profiling produces a distinct arithmetic-intensity point
// per datatype. Each thread runs a register-resident accumulator chain
// (acc = acc * b + a) with no memory traffic in the hot loop; only the final
// accumulator is stored, which keeps the loop from being optimized away.
//
// Precision support is decided two ways that must agree:
//   - Compile time: each low-precision kernel body is wrapped in
//     #if defined(__gfx*__). Those macros are defined ONLY in the device
//     compilation pass, so the body emits real device code only for the offload
//     arch(es) that support the type and compiles to an empty no-op elsewhere.
//     This keeps the file building for every --offload-arch.
//   - Run time: the host resolves the live GPU arch to the supported-datatype
//     set (mirroring SUPPORTED_DATATYPES in src/utils/roofline_calc.py) and only
//     launches + reports the precisions that arch supports. Builds are native
//     (no --offload-arch override), so the runtime arch equals the compiled one.
//
// The base precisions (FP16/BF16/FP32/FP64/I8/I32/I64) are supported on every
// roofline arch and need no gating. FP8/FP6/FP4 are best-effort: HIP exposes
// them as storage/convert types with no native arithmetic, so those kernels
// round-trip the accumulator through float each step.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp6.h>
#include <hip/hip_fp4.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t err = call;                                               \
        if (err != hipSuccess) {                                             \
            std::cerr << "HIP error: " << hipGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                     \
    } while (0)

// Tuning: enough iterations to be compute-bound, small enough to finish fast.
static constexpr int kDefaultIters = 4096;
static constexpr int kUnroll = 64;
static constexpr int kThreadsPerBlock = 256;
static constexpr int kBlocksPerCU = 32;

// ----------------------------------------------------------------------------
// Base-precision kernel: native arithmetic for float and integer types.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void fmaThroughput(T* out, T a, T b, int iters)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    T acc = static_cast<T>(tid & 0x7);

    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            acc = acc * b + a;
        }
    }

    out[tid] = acc;
}

// ----------------------------------------------------------------------------
// Low-precision kernels (FP8/FP6/FP4): storage types with no native arithmetic,
// so the accumulator chain runs in float and converts back to the storage type
// each iteration. The struct types are format-specific per arch (FP8 is FNUZ on
// CDNA3 and OCP on CDNA4), so the host launches over a raw 1-byte storage buffer
// and the arch-correct struct type is selected inside the device body via #if.
// On an unsupported arch the body is empty and the launch is a harmless no-op
// (the host also avoids launching it - see run-time gating).
// ----------------------------------------------------------------------------
template <typename FP>
__device__ inline void fma_low_prec_body(unsigned char* out, float a, float b, int iters)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = static_cast<float>(FP(static_cast<float>(tid & 0x3)));

    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            // Round-trip through the low-precision storage type each step so the
            // conversion HW is exercised and the value stays representable.
            acc = static_cast<float>(FP(acc * b + a));
        }
    }

    out[tid] = FP(acc).__x;
}

__global__ void fmaThroughputFP8(unsigned char* out, float a, float b, int iters)
{
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    fma_low_prec_body<__hip_fp8_e4m3_fnuz>(out, a, b, iters);  // CDNA3: FNUZ
#elif defined(__gfx950__)
    fma_low_prec_body<__hip_fp8_e4m3>(out, a, b, iters);  // CDNA4: OCP
#else
    (void)out;
    (void)a;
    (void)b;
    (void)iters;
#endif
}

__global__ void fmaThroughputFP6(unsigned char* out, float a, float b, int iters)
{
#if defined(__gfx950__)
    fma_low_prec_body<__hip_fp6_e2m3>(out, a, b, iters);
#else
    (void)out;
    (void)a;
    (void)b;
    (void)iters;
#endif
}

__global__ void fmaThroughputFP4(unsigned char* out, float a, float b, int iters)
{
#if defined(__gfx950__)
    fma_low_prec_body<__hip_fp4_e2m1>(out, a, b, iters);
#else
    (void)out;
    (void)a;
    (void)b;
    (void)iters;
#endif
}

// ----------------------------------------------------------------------------
// Host helpers.
// ----------------------------------------------------------------------------
static std::string detect_arch(int device)
{
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    std::string arch(props.gcnArchName);
    // Strip feature suffix, e.g. "gfx942:sramecc+:xnack-" -> "gfx942".
    size_t colon = arch.find(':');
    if (colon != std::string::npos) {
        arch = arch.substr(0, colon);
    }
    return arch;
}

// Mirrors SUPPORTED_DATATYPES in src/utils/roofline_calc.py.
static std::vector<std::string> supported_dtypes(const std::string& arch)
{
    std::vector<std::string> base = {"FP16", "BF16", "FP32", "FP64", "I8", "I32", "I64"};
    if (arch == "gfx940" || arch == "gfx941" || arch == "gfx942") {
        base.push_back("FP8");
    } else if (arch == "gfx950") {
        base.push_back("FP8");
        base.push_back("FP6");
        base.push_back("FP4");
    }
    return base;
}

static int launch_dims(int device, int& blocks)
{
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    blocks = props.multiProcessorCount * kBlocksPerCU;
    return blocks * kThreadsPerBlock;
}

template <typename T>
static void run_base(const std::string& name, int device, int iters)
{
    int blocks = 0;
    int total_threads = launch_dims(device, blocks);

    T* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(T) * total_threads));
    fmaThroughput<T><<<blocks, kThreadsPerBlock>>>(
        d_out, static_cast<T>(1), static_cast<T>(1), iters);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_out));

    std::cout << "  [ran]     " << name << std::endl;
}

// All FP8/FP6/FP4 storage types are a single byte, so the host launches over a
// raw byte buffer; the kernel reinterprets per the compiled arch.
using LowPrecKernel = void (*)(unsigned char*, float, float, int);

static void run_low_prec(const std::string& name, int device, int iters, LowPrecKernel kernel)
{
    int blocks = 0;
    int total_threads = launch_dims(device, blocks);

    unsigned char* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(unsigned char) * total_threads));
    kernel<<<blocks, kThreadsPerBlock>>>(d_out, 1.0f, 1.0f, iters);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_out));

    std::cout << "  [ran]     " << name << std::endl;
}

// Datatype identifiers, in the order SUPPORTED_DATATYPES lists them.
static const std::vector<std::string> kAllDtypes = {
    "FP4", "FP6", "FP8", "FP16", "BF16", "FP32", "FP64", "I8", "I32", "I64"};

int main(int argc, char** argv)
{
    int device = 0;
    int iters = kDefaultIters;
    std::string only;
    bool all = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = std::atoi(argv[++i]);
        } else if (arg == "--dtype" && i + 1 < argc) {
            only = argv[++i];
            all = false;
        } else if (arg == "--all") {
            all = true;
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " [--device <id>] [--dtype <NAME>] [--all] [--iters <N>]\n"
                      << "  Datatypes: FP4 FP6 FP8 FP16 BF16 FP32 FP64 I8 I32 I64\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    if (!all) {
        bool known = false;
        for (const auto& d : kAllDtypes) {
            if (d == only) known = true;
        }
        if (!known) {
            std::cerr << "Unknown datatype: " << only << std::endl;
            return 1;
        }
    }

    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device < 0 || device >= device_count) {
        std::cerr << "Invalid device " << device << " (found " << device_count
                  << " device(s))" << std::endl;
        return 1;
    }
    HIP_CHECK(hipSetDevice(device));

    std::string arch = detect_arch(device);
    std::vector<std::string> supported = supported_dtypes(arch);

    std::cout << "fma_throughput on device " << device << " (" << arch
              << "), iters=" << iters << std::endl;

    auto wanted = [&](const std::string& dtype) { return all || dtype == only; };
    auto is_supported = [&](const std::string& dtype) {
        for (const auto& d : supported) {
            if (d == dtype) return true;
        }
        return false;
    };
    auto skip = [&](const std::string& dtype) {
        std::cout << "  [skipped] " << dtype << " (unsupported on " << arch << ")"
                  << std::endl;
    };

    // Base precisions - supported on every roofline arch.
    if (wanted("FP16")) run_base<_Float16>("FP16", device, iters);
    if (wanted("BF16")) run_base<__hip_bfloat16>("BF16", device, iters);
    if (wanted("FP32")) run_base<float>("FP32", device, iters);
    if (wanted("FP64")) run_base<double>("FP64", device, iters);
    if (wanted("I8")) run_base<int8_t>("I8", device, iters);
    if (wanted("I32")) run_base<int32_t>("I32", device, iters);
    if (wanted("I64")) run_base<int64_t>("I64", device, iters);

    // Low precisions - launched only where the runtime arch supports them; the
    // kernel bodies are additionally compile-time gated to the same archs.
    if (wanted("FP8")) {
        if (is_supported("FP8"))
            run_low_prec("FP8", device, iters, fmaThroughputFP8);
        else
            skip("FP8");
    }
    if (wanted("FP6")) {
        if (is_supported("FP6"))
            run_low_prec("FP6", device, iters, fmaThroughputFP6);
        else
            skip("FP6");
    }
    if (wanted("FP4")) {
        if (is_supported("FP4"))
            run_low_prec("FP4", device, iters, fmaThroughputFP4);
        else
            skip("FP4");
    }

    return 0;
}
