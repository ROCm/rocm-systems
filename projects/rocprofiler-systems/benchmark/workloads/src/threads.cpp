// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "bench_hooks.h"
#include "common.hpp"
#include <cstdint>

#include <mutex>
#include <thread>
#include <vector>

namespace
{
constexpr std::uint64_t lock_every = 4;  // 1 in 4 iterations takes the shared lock

struct shared_state
{
    std::mutex    mtx;
    std::uint64_t counter = 0;
};

std::uint64_t
worker(std::uint64_t iters, std::uint64_t seed, shared_state& st)
{
    std::uint64_t rng = seed, local = 0;
    for(std::uint64_t i = 0; i < iters; ++i)
    {
        local = workload::mix(local, workload::splitmix64(rng));
        if(i % lock_every == 0)
        {
            std::lock_guard<std::mutex> lk(st.mtx);
            ++st.counter;
        }
    }
    return local;
}

std::uint64_t
run(const workload::args& a)
{
    shared_state               st;
    std::vector<std::uint64_t> results(a.size);
    std::vector<std::thread>   pool;
    for(std::uint64_t t = 0; t < a.size; ++t)
        pool.emplace_back([&, t] { results[t] = worker(a.iters, a.seed + t, st); });
    for(auto& th : pool)
        th.join();
    std::uint64_t sum = st.counter;
    for(std::uint64_t r : results)
        sum = workload::mix(sum, r);  // fixed order: schedule-independent
    return sum;
}
}  // namespace

int
main(int argc, char** argv)
{
    const auto a = workload::parse_args(
        argc, argv, { 4, 25000000, 1 }, "threads",
        "size = number of threads (>=1), iters = iterations per thread "
        "(every 4th takes a shared mutex).");
    if(a.size == 0) return 2;
    bench_hook(BENCH_HOOK_WORK_BEGIN);
    const std::uint64_t sum = run(a);
    bench_hook(BENCH_HOOK_WORK_END);
    workload::print_checksum(sum);
    return 0;
}
