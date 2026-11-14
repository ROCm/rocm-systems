// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "bench_hooks.h"
#include "common.hpp"
#include <cstdint>

#include <algorithm>
#include <vector>

namespace
{
using matrix = std::vector<double>;

matrix
random_matrix(std::uint64_t n, std::uint64_t& rng)
{
    matrix m(n * n);
    for(auto& x : m)
        x = static_cast<double>(workload::splitmix64(rng) % 1000) / 1000.0;
    return m;
}

void
multiply(const matrix& a, const matrix& b, matrix& c, std::uint64_t n)
{
    for(std::uint64_t i = 0; i < n; ++i)
        for(std::uint64_t k = 0; k < n; ++k)
        {
            const double aik = a[i * n + k];
            for(std::uint64_t j = 0; j < n; ++j)
                c[i * n + j] += aik * b[k * n + j];
        }
}

std::uint64_t
run(const workload::args& a)
{
    std::uint64_t rng = a.seed;
    matrix        x = random_matrix(a.size, rng), y = random_matrix(a.size, rng);
    matrix        z(a.size * a.size);
    for(std::uint64_t it = 0; it < a.iters; ++it)
    {
        std::fill(z.begin(), z.end(), 0.0);
        multiply(x, y, z, a.size);
        x.swap(z);  // feed result back so iterations depend on each other
        for(auto& v : x)
            v = v - static_cast<std::uint64_t>(v);
    }
    std::uint64_t sum = 0;
    for(double v : x)
        sum = workload::mix(sum, static_cast<std::uint64_t>(v * 1e9));
    return sum;
}
}  // namespace

int
main(int argc, char** argv)
{
    const auto a = workload::parse_args(
        argc, argv, { 256, 280, 1 }, "cpu_compute",
        "size = matrix dimension N, iters = number of NxN multiplications.");
    bench_hook(BENCH_HOOK_WORK_BEGIN);
    const std::uint64_t sum = run(a);
    bench_hook(BENCH_HOOK_WORK_END);
    workload::print_checksum(sum);
    return 0;
}
