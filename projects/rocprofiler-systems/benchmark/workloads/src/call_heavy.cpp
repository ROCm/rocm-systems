// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "bench_hooks.h"
#include "common.hpp"
#include <cstdint>

namespace
{
__attribute__((noinline)) std::uint64_t
                          chain(std::uint64_t x, std::uint64_t depth)
{
    if(depth == 0) return x * 0x9e3779b97f4a7c15ULL + 1;
    const std::uint64_t r = chain(x ^ depth, depth - 1) + depth;
    asm volatile(""
                 : "+r"(x));  // defeat tail-call/inlining so every level is a real call
    return r + (x & 1);
}

std::uint64_t
run(const workload::args& a)
{
    std::uint64_t sum = a.seed;
    for(std::uint64_t i = 0; i < a.iters; ++i)
        sum = workload::mix(sum, chain(sum + i, a.size));
    return sum;
}
}  // namespace

int
main(int argc, char** argv)
{
    const auto a =
        workload::parse_args(argc, argv, { 16, 77000000, 1 }, "call_heavy",
                             "size = call depth per chain, iters = number of chains "
                             "(total calls = size * iters).");
    bench_hook(BENCH_HOOK_WORK_BEGIN);
    const std::uint64_t sum = run(a);
    bench_hook(BENCH_HOOK_WORK_END);
    workload::print_checksum(sum);
    return 0;
}
