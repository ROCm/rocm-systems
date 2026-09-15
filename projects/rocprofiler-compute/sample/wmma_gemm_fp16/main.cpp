// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../common.h"
#include "wmma_gemm_fp16.hpp"

using namespace wmma_gemm_fp16;

namespace {

struct Options {
    uint32_t m     = kDefaultM;
    uint32_t n     = kDefaultN;
    uint32_t k     = kDefaultK;
    uint32_t iters = kDefaultIters;
    uint32_t warmup = kDefaultWarmup;
    int device   = 0;
};

void usage()
{
    std::cout
        << "Usage: wmma_gemm_fp16 [OPTIONS]\n"
        << "  FP16 WMMA GEMM for gfx1250 WMMA/XDL metric health profiling.\n\n"
        << "Optional:\n"
        << "  -m, --m <value>       Rows of A / D [default: " << kDefaultM << "]\n"
        << "  -n, --n <value>       Cols of B / D [default: " << kDefaultN << "]\n"
        << "  -k, --k <value>       Inner dimension K (multiple of 32) [default: "
        << kDefaultK << "]\n"
        << "  -i, --iter <value>    Kernel iterations [default: " << kDefaultIters
        << "]\n"
        << "  -w, --warmup <value>  Warmup iterations [default: " << kDefaultWarmup
        << "]\n"
        << "  -d, --device <id>     HIP device id [default: 0]\n"
        << "  -h, --help            Show this help\n";
    std::exit(EXIT_SUCCESS);
}

Options parse_args(int argc, char* argv[])
{
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value       = [&](const char* flag) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                usage();
            }
            return std::string(argv[++i]);
        };

        if (arg == "-h" || arg == "--help") {
            usage();
        } else if (arg == "-m" || arg == "--m") {
            opts.m = static_cast<uint32_t>(std::stoul(need_value(arg.c_str())));
        } else if (arg == "-n" || arg == "--n") {
            opts.n = static_cast<uint32_t>(std::stoul(need_value(arg.c_str())));
        } else if (arg == "-k" || arg == "--k") {
            opts.k = static_cast<uint32_t>(std::stoul(need_value(arg.c_str())));
        } else if (arg == "-i" || arg == "--iter") {
            opts.iters = static_cast<uint32_t>(std::stoul(need_value(arg.c_str())));
        } else if (arg == "-w" || arg == "--warmup") {
            opts.warmup = static_cast<uint32_t>(std::stoul(need_value(arg.c_str())));
        } else if (arg == "-d" || arg == "--device") {
            opts.device = std::stoi(need_value(arg.c_str()));
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            usage();
        }
    }
    return opts;
}

void fill_matrix(std::vector<_Float16>& mat, uint32_t rows, uint32_t cols)
{
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(0.5f, 2.0f);
    mat.resize(static_cast<size_t>(rows) * cols);
    for (auto& value : mat) {
        value = static_cast<_Float16>(dist(rng));
    }
}

bool smoke_check(const std::vector<_Float16>& d, uint32_t m, uint32_t n)
{
    for (uint32_t row = 0; row < std::min(m, 64u); ++row) {
        for (uint32_t col = 0; col < std::min(n, 64u); ++col) {
            const float value = static_cast<float>(d[static_cast<size_t>(row) * n + col]);
            if (value != 0.0f && std::isfinite(value)) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

int main(int argc, char* argv[])
{
    const Options opts = parse_args(argc, argv);

    if (opts.m == 0 || opts.n == 0 || opts.k == 0) {
        std::cerr << "M, N, and K must be positive.\n";
        return EXIT_FAILURE;
    }
    if (opts.k % kBlockK != 0) {
        std::cerr << "K must be a multiple of " << kBlockK << " for 16x16x32 WMMA tiles.\n";
        return EXIT_FAILURE;
    }
    if (opts.m % kBlockM != 0 || opts.n % kBlockN != 0) {
        std::cerr << "M and N must be multiples of " << kBlockM << ".\n";
        return EXIT_FAILURE;
    }

    hipCheck(hipSetDevice(opts.device));

    hipDeviceProp_t props{};
    hipCheck(hipGetDeviceProperties(&props, opts.device));
    std::cout << "Device: " << props.name << " (" << props.gcnArchName << ")\n";
    std::cout << "Problem: M=" << opts.m << " N=" << opts.n << " K=" << opts.k
              << " iters=" << opts.iters << "\n";

    std::vector<_Float16> h_a;
    std::vector<_Float16> h_b;
    std::vector<_Float16> h_d(static_cast<size_t>(opts.m) * opts.n, _Float16(0));

    fill_matrix(h_a, opts.m, opts.k);
    fill_matrix(h_b, opts.n, opts.k);

    _Float16 *d_a = nullptr;
    _Float16 *d_b = nullptr;
    _Float16 *d_d = nullptr;

    const size_t bytes_a = h_a.size() * sizeof(_Float16);
    const size_t bytes_b = h_b.size() * sizeof(_Float16);
    const size_t bytes_d = h_d.size() * sizeof(_Float16);

    hipCheck(hipMalloc(&d_a, bytes_a));
    hipCheck(hipMalloc(&d_b, bytes_b));
    hipCheck(hipMalloc(&d_d, bytes_d));

    hipCheck(hipMemcpy(d_a, h_a.data(), bytes_a, hipMemcpyHostToDevice));
    hipCheck(hipMemcpy(d_b, h_b.data(), bytes_b, hipMemcpyHostToDevice));
    hipCheck(hipMemcpy(d_d, h_d.data(), bytes_d, hipMemcpyHostToDevice));

    const dim3 block_dim(kThreadsX, kThreadsY);
    const dim3 grid_dim(DivUp(static_cast<int>(opts.m), static_cast<int>(kBlockM)),
                        DivUp(static_cast<int>(opts.n), static_cast<int>(kBlockN)),
                        1);

    std::cout << "Launch grid=(" << grid_dim.x << "," << grid_dim.y << ") block=("
              << block_dim.x << "," << block_dim.y << ")\n";

    auto launch = [&]() {
        hipLaunchKernelGGL(gemm_wmma_fp16,
                           grid_dim,
                           block_dim,
                           0,
                           0,
                           opts.m,
                           opts.n,
                           opts.k,
                           d_a,
                           d_b,
                           d_d);
        hipCheck(hipGetLastError());
    };

    for (uint32_t i = 0; i < opts.warmup; ++i) {
        launch();
    }
    hipCheck(hipDeviceSynchronize());

    const auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < opts.iters; ++i) {
        launch();
    }
    hipCheck(hipDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    const double gflops =
        2.0 * static_cast<double>(opts.m) * opts.n * opts.k * opts.iters * 1e-9;
    const double tflops_per_s = gflops / elapsed_ms;

    hipCheck(hipMemcpy(h_d.data(), d_d, bytes_d, hipMemcpyDeviceToHost));

    const bool passed = smoke_check(h_d, opts.m, opts.n);
    std::cout << "Elapsed: " << elapsed_ms << " ms (" << opts.iters << " iterations)\n";
    std::cout << "Throughput: " << tflops_per_s << " TFLOP/s (FP16 GEMM convention)\n";
    std::cout << "Smoke check: " << (passed ? "PASS (non-zero output)" : "FAIL") << "\n";

    hipCheck(hipFree(d_a));
    hipCheck(hipFree(d_b));
    hipCheck(hipFree(d_d));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
