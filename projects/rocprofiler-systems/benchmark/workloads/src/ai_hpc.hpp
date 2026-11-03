// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "bench_hooks.h"
#include "common.hpp"
#include <cstdint>

#include <cmath>
#include <memory>
#include <thread>
#include <vector>

// Inference-serving shape: each thread is a worker that pushes activations through a
// tanh MLP layer, act = tanh(act x W) with N x N matrices (2*N^3 flops per layer, the
// dense-GEMM core of both AI and HPC codes). The layer is reached through `depth` nested
// calls, like module.forward() nesting in an AI framework, so profilers that unwind the
// call stack see a stack of configurable depth while the GEMM runs.
//
// A Backend provides `worker(args, seed)` with `void layer()` (one GEMM + tanh) and
// `std::uint64_t checksum()`.
namespace ai_hpc
{
inline constexpr std::uint64_t default_depth   = 64;
inline constexpr std::uint64_t default_threads = 4;

inline const char* help =
    "size = matrix dimension N (GEMM cost 2*N^3 per layer), iters = layers per thread, "
    "depth = nested calls above each layer, threads = worker threads.";

// He-style init keeps the activations of a tanh MLP at O(1) for any number of layers.
inline void
init_matrices(std::vector<float>& act, std::vector<float>& weights, std::uint64_t n,
              std::uint64_t seed)
{
    const float scale = std::sqrt(6.0f / static_cast<float>(n));
    auto        unit  = [&] {
        return static_cast<float>(workload::splitmix64(seed) % 2001) / 1000.0f - 1.0f;
    };
    for(auto& x : act)
        x = unit();
    for(auto& w : weights)
        w = unit() * scale;
}

// noinline + work after the call: every level is a real stack frame (no tail call).
template <class Worker>
__attribute__((noinline)) std::uint64_t
                          forward(Worker& w, std::uint64_t depth)
{
    if(depth == 0)
    {
        w.layer();
        return 0;
    }
    std::uint64_t r = forward(w, depth - 1);
    asm volatile("" : "+r"(r));
    return r + depth;
}

template <class Worker>
std::uint64_t
run_worker(Worker& w, const workload::args& a)
{
    std::uint64_t acc = 0;
    for(std::uint64_t i = 0; i < a.iters; ++i)
        acc = workload::mix(acc, forward(w, a.depth));
    return workload::mix(acc, w.checksum());
}

template <class Worker>
int
run(int argc, char** argv, const char* name, workload::args defaults,
    const char* help_text = help)
{
    defaults.depth   = default_depth;
    defaults.threads = default_threads;
    const auto a     = workload::parse_args(argc, argv, defaults, name, help_text);
    if(a.size == 0 || a.iters == 0 || a.threads == 0) return 2;

    // Allocation and initialization stay outside the measured region.
    std::vector<std::unique_ptr<Worker>> workers;
    for(std::uint64_t t = 0; t < a.threads; ++t)
        workers.push_back(std::make_unique<Worker>(a, a.seed + t));

    std::vector<std::uint64_t> results(a.threads);
    std::vector<std::thread>   pool;
    bench_hook(BENCH_HOOK_WORK_BEGIN);
    for(std::uint64_t t = 0; t < a.threads; ++t)
        pool.emplace_back([&, t] { results[t] = run_worker(*workers[t], a); });
    for(auto& th : pool)
        th.join();
    bench_hook(BENCH_HOOK_WORK_END);

    std::uint64_t sum = a.seed;
    for(std::uint64_t r : results)
        sum = workload::mix(sum, r);  // fixed order: schedule-independent
    workload::print_checksum(sum);
    return 0;
}
}  // namespace ai_hpc
