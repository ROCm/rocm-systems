// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "bench_hooks.h"
#include "common.hpp"
#include <cstdint>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <time.h>
#include <vector>

namespace gpu_workload
{
struct config
{
    const char*    name;
    const char*    help;
    workload::args defaults;
    int            fma_steps;
};

namespace
{
constexpr std::uint64_t max_timed_kernels = 2048;
constexpr int           block_size        = 256;

void
check(hipError_t err, const char* what)
{
    if(err == hipSuccess) return;
    std::fprintf(stderr, "gpu workload: %s failed: %s\n", what, hipGetErrorString(err));
    std::exit(1);
}

#define HIP_CHECK(call) check((call), #call)

// Adds 1 to every element per launch, so the final buffer depends on every launch.
// The dependent FMA chain only burns time; (y - y) keeps it from being optimized away.
__global__ void
accumulate_kernel(float* buf, std::uint64_t n, int steps)
{
    const std::uint64_t i =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= n) return;
    const float x = buf[i];
    float       y = x;
    for(int k = 0; k < steps; ++k)
        y = fmaf(y, 0.999f, 0.001f);
    buf[i] = x + 1.0f + (y - y);
}

struct metrics
{
    double interval_ms = 0.0;
    double kernel_us   = 0.0;
    double gap_us      = 0.0;
    double launch_us   = 0.0;
};

struct event_pair
{
    hipEvent_t start{};
    hipEvent_t stop{};

    event_pair()
    {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }
    ~event_pair()
    {
        (void) hipEventDestroy(start);
        (void) hipEventDestroy(stop);
    }
    event_pair(const event_pair&)            = delete;
    event_pair& operator=(const event_pair&) = delete;
};

struct device_buffer
{
    float* ptr = nullptr;

    explicit device_buffer(size_t bytes) { HIP_CHECK(hipMalloc(&ptr, bytes)); }
    ~device_buffer() { (void) hipFree(ptr); }
    device_buffer(const device_buffer&)            = delete;
    device_buffer& operator=(const device_buffer&) = delete;
};

std::uint64_t
now_ns()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ULL +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

float
elapsed_ms(hipEvent_t from, hipEvent_t to)
{
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, from, to));
    return ms;
}

std::vector<float>
random_input(std::uint64_t n, std::uint64_t seed)
{
    std::vector<float> v(n);
    for(auto& x : v)
        x = static_cast<float>(workload::splitmix64(seed) % 1000) / 1000.0f;
    return v;
}

void
launch(const device_buffer& buf, std::uint64_t n, int steps)
{
    const unsigned blocks = static_cast<unsigned>((n + block_size - 1) / block_size);
    hipLaunchKernelGGL(accumulate_kernel, dim3(blocks), dim3(block_size), 0, 0, buf.ptr,
                       n, steps);
}

void
warm_up(const device_buffer& buf, std::uint64_t n, int steps)
{
    launch(buf, n, steps);
    HIP_CHECK(hipDeviceSynchronize());
}

std::uint64_t
timed_launch(const device_buffer& buf, std::uint64_t n, int steps)
{
    const std::uint64_t t0 = now_ns();
    launch(buf, n, steps);
    return now_ns() - t0;
}

metrics
collect(const std::vector<event_pair>& timed, const event_pair& loop,
        std::uint64_t launch_ns_total, std::uint64_t iters)
{
    metrics m;
    m.interval_ms = elapsed_ms(loop.start, loop.stop);
    m.launch_us = static_cast<double>(launch_ns_total) / 1e3 / static_cast<double>(iters);
    double kernel_ms = 0.0, gap_ms = 0.0;
    for(size_t i = 0; i < timed.size(); ++i)
    {
        kernel_ms += elapsed_ms(timed[i].start, timed[i].stop);
        if(i + 1 < timed.size()) gap_ms += elapsed_ms(timed[i].stop, timed[i + 1].start);
    }
    m.kernel_us = kernel_ms * 1e3 / static_cast<double>(timed.size());
    if(timed.size() > 1) m.gap_us = gap_ms * 1e3 / static_cast<double>(timed.size() - 1);
    return m;
}

metrics
launch_loop(const device_buffer& buf, std::uint64_t n, std::uint64_t iters, int steps)
{
    const std::uint64_t     timed_count = std::min(iters, max_timed_kernels);
    std::vector<event_pair> timed(timed_count);
    event_pair              loop;
    std::uint64_t           launch_ns = 0;

    HIP_CHECK(hipEventRecord(loop.start, nullptr));
    for(std::uint64_t i = 0; i < iters; ++i)
    {
        if(i < timed_count) HIP_CHECK(hipEventRecord(timed[i].start, nullptr));
        launch_ns += timed_launch(buf, n, steps);
        if(i < timed_count) HIP_CHECK(hipEventRecord(timed[i].stop, nullptr));
    }
    HIP_CHECK(hipEventRecord(loop.stop, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    return collect(timed, loop, launch_ns, iters);
}

std::uint64_t
checksum_of(const std::vector<float>& v)
{
    std::uint64_t sum = 0;
    for(float x : v)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &x, sizeof(bits));
        sum = workload::mix(sum, bits);
    }
    return sum;
}

void
print_metric(const char* name, double value)
{
    std::printf("metric %s %.3f\n", name, value);
}

void
print_metrics(const metrics& m)
{
    print_metric("gpu_interval_ms", m.interval_ms);
    print_metric("gpu_kernel_us", m.kernel_us);
    print_metric("gpu_gap_us", m.gap_us);
    print_metric("gpu_launch_us", m.launch_us);
}
}  // namespace

inline int
run(int argc, char** argv, const config& cfg)
{
    const auto a = workload::parse_args(argc, argv, cfg.defaults, cfg.name, cfg.help);
    if(a.size == 0 || a.iters == 0) return 2;

    const size_t       bytes = a.size * sizeof(float);
    std::vector<float> host  = random_input(a.size, a.seed);
    device_buffer      buf(bytes);
    HIP_CHECK(hipMemcpy(buf.ptr, host.data(), bytes, hipMemcpyHostToDevice));
    warm_up(buf, a.size, cfg.fma_steps);

    bench_hook(BENCH_HOOK_WORK_BEGIN);
    const metrics m = launch_loop(buf, a.size, a.iters, cfg.fma_steps);
    HIP_CHECK(hipMemcpy(host.data(), buf.ptr, bytes, hipMemcpyDeviceToHost));
    bench_hook(BENCH_HOOK_WORK_END);

    print_metrics(m);
    workload::print_checksum(checksum_of(host));
    return 0;
}
}  // namespace gpu_workload
