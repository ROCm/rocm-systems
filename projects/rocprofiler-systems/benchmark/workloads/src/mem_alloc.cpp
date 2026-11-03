// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "bench_hooks.h"
#include "common.hpp"
#include <cstdint>

#include <vector>

namespace
{
std::uint64_t
run(const workload::args& a)
{
    std::vector<std::uint64_t> buf(a.size * 1024 * 1024 / sizeof(std::uint64_t));
    for(std::uint64_t i = 0; i < buf.size(); i += 512)
        buf[i] = i;  // touch every 4 KiB page
    std::uint64_t rng = a.seed, sum = 0;
    for(std::uint64_t i = 0; i < a.iters; ++i)
    {
        auto& slot = buf[workload::splitmix64(rng) % buf.size()];
        slot += i;
        sum = workload::mix(sum, slot);
    }
    return sum;
}
}  // namespace

int
main(int argc, char** argv)
{
    const auto a =
        workload::parse_args(argc, argv, { 256, 60000000, 1 }, "mem_alloc",
                             "size = buffer size in MB (allocated and page-touched), "
                             "iters = number of random read-modify-write accesses.");
    if(a.size == 0) return 2;
    bench_hook(BENCH_HOOK_WORK_BEGIN);
    const std::uint64_t sum = run(a);
    bench_hook(BENCH_HOOK_WORK_END);
    workload::print_checksum(sum);
    return 0;
}
