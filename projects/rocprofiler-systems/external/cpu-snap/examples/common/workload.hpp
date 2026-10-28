#pragma once

/**
 * @file workload.hpp
 * @brief Deterministic synthetic workload shared by every example +
 *        bench. Eight noinline frames so the kernel callchain walker
 *        and post-processed flame-graph have stack depth to chew on:
 *
 *          leaf_compute -> hot_loop -> matmul_inner -> matmul_outer
 *          -> workload_step -> workload_iteration -> workload_dispatch
 *          -> run_workload
 *
 *        Callers pick an entry-point at whatever depth fits their
 *        demo. The leaf does a periodic *= 0.9999999 rescale to keep
 *        the accumulator from overflowing during long runs.
 */

#include <cstdint>

namespace cs_examples
{

[[gnu::noinline]] inline double
leaf_compute(double seed, std::uint64_t iters) noexcept
{
    double acc = seed;
    for(std::uint64_t i = 0; i < iters; ++i)
    {
        acc = acc * 1.0000001 + static_cast<double>(i & 0xFFFFU);
        if((i & 0xFFFU) == 0) acc *= 0.9999999;
    }
    return acc;
}

[[gnu::noinline]] inline double
hot_loop(double seed, std::uint64_t iters) noexcept
{
    return leaf_compute(seed + 1.0, iters);
}

[[gnu::noinline]] inline double
matmul_inner(double seed, std::uint64_t iters) noexcept
{
    double acc = 0.0;
    for(int i = 0; i < 4; ++i)
    {
        acc += hot_loop(seed + i, iters / 4);
    }
    return acc;
}

[[gnu::noinline]] inline double
matmul_outer(double seed, std::uint64_t iters) noexcept
{
    return matmul_inner(seed * 1.1, iters);
}

[[gnu::noinline]] inline double
workload_step(double seed, std::uint64_t iters) noexcept
{
    return matmul_outer(seed + 0.5, iters);
}

[[gnu::noinline]] inline double
workload_iteration(double seed, std::uint64_t iters) noexcept
{
    return workload_step(seed, iters);
}

[[gnu::noinline]] inline double
workload_dispatch(double seed, std::uint64_t iters) noexcept
{
    return workload_iteration(seed, iters);
}

[[gnu::noinline]] inline double
run_workload(double seed, std::uint64_t iters) noexcept
{
    return workload_dispatch(seed, iters);
}

}  // namespace cs_examples
