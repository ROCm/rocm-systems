// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Exercises application-owned completion signals and queries timing across their lifecycle.

#include <rocprofiler-sdk-roctx/roctx.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#define HSA_CHECK(...)                                                                             \
    do                                                                                             \
    {                                                                                              \
        hsa_status_t _status = (__VA_ARGS__);                                                      \
        if(_status != HSA_STATUS_SUCCESS)                                                          \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "%s:%d :: %s failed with %d\n",                                                \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    #__VA_ARGS__,                                                                  \
                    static_cast<int>(_status));                                                    \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while(0)

namespace
{
constexpr size_t copy_bytes = 4 << 20;
constexpr int    iterations = 8;

hsa_agent_t           cpu_agent = {};
hsa_agent_t           gpu_agent = {};
hsa_amd_memory_pool_t cpu_pool  = {};
hsa_amd_memory_pool_t gpu_pool  = {};

hsa_status_t
find_agents(hsa_agent_t agent, void*)
{
    auto type = hsa_device_type_t{};
    HSA_CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type));
    if(type == HSA_DEVICE_TYPE_CPU && cpu_agent.handle == 0) cpu_agent = agent;
    if(type == HSA_DEVICE_TYPE_GPU && gpu_agent.handle == 0) gpu_agent = agent;
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
find_pool(hsa_amd_memory_pool_t pool, void* data)
{
    auto segment       = hsa_amd_segment_t{};
    auto alloc_allowed = false;
    HSA_CHECK(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment));
    HSA_CHECK(hsa_amd_memory_pool_get_info(
        pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed));

    auto* found = static_cast<hsa_amd_memory_pool_t*>(data);
    if(segment == HSA_AMD_SEGMENT_GLOBAL && alloc_allowed && found->handle == 0) *found = pool;
    return HSA_STATUS_SUCCESS;
}
uint64_t clock_frequency = 0;
FILE*    output          = stdout;

void
wait_for(hsa_signal_t signal, hsa_signal_value_t threshold)
{
    if(hsa_signal_wait_scacquire(signal,
                                 HSA_SIGNAL_CONDITION_LT,
                                 threshold,
                                 5 * clock_frequency,
                                 HSA_WAIT_STATE_BLOCKED) >= threshold)
    {
        fprintf(stderr, "Timed out waiting for completion signal\n");
        exit(EXIT_FAILURE);
    }
}

hsa_amd_profiling_async_copy_time_t
query_time(hsa_signal_t signal, const char* name, bool traced = true, uint64_t after = 0)
{
    auto time = hsa_amd_profiling_async_copy_time_t{};
    HSA_CHECK(hsa_amd_profiling_get_async_copy_time(signal, &time));
    if(time.end <= time.start || time.end <= after)
    {
        fprintf(stderr,
                "%s: invalid copy time (%lu, %lu), lower bound %lu\n",
                name,
                time.start,
                time.end,
                after);
        exit(EXIT_FAILURE);
    }
    fprintf(output, "%s,%lu,%lu,%lu,%d\n", name, time.start, time.end, clock_frequency, traced);
    return time;
}
}  // namespace

int
main(int argc, char** argv)
{
    if(argc > 1)
    {
        output = fopen(argv[1], "w");
        if(!output) return EXIT_FAILURE;
    }
    fprintf(output, "Case,Start,End,Frequency,Traced\n");
    HSA_CHECK(hsa_init());
    HSA_CHECK(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &clock_frequency));
    HSA_CHECK(hsa_iterate_agents(find_agents, nullptr));
    HSA_CHECK(hsa_amd_agent_iterate_memory_pools(cpu_agent, find_pool, &cpu_pool));
    HSA_CHECK(hsa_amd_agent_iterate_memory_pools(gpu_agent, find_pool, &gpu_pool));
    HSA_CHECK(hsa_amd_profiling_async_copy_enable(true));

    void* host = nullptr;
    void* dev  = nullptr;
    HSA_CHECK(hsa_amd_memory_pool_allocate(cpu_pool, copy_bytes, 0, &host));
    HSA_CHECK(hsa_amd_memory_pool_allocate(gpu_pool, copy_bytes, 0, &dev));
    HSA_CHECK(hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, host));
    memset(host, 0x5a, copy_bytes);

    for(int i = 0; i < 2 * iterations; ++i)
    {
        auto to_device = (i % 2 == 0);
        auto signal    = hsa_signal_t{};
        HSA_CHECK(hsa_signal_create(1, 0, nullptr, &signal));

        if(to_device)
            HSA_CHECK(hsa_amd_memory_async_copy(
                dev, gpu_agent, host, cpu_agent, copy_bytes, 0, nullptr, signal));
        else
            HSA_CHECK(hsa_amd_memory_async_copy(
                host, cpu_agent, dev, gpu_agent, copy_bytes, 0, nullptr, signal));

        wait_for(signal, 1);
        query_time(signal, to_device ? "host-to-device" : "device-to-host");

        HSA_CHECK(hsa_signal_destroy(signal));
    }

    void* other_host = nullptr;
    void* other_dev  = nullptr;
    HSA_CHECK(hsa_amd_memory_pool_allocate(cpu_pool, copy_bytes, 0, &other_host));
    HSA_CHECK(hsa_amd_memory_pool_allocate(gpu_pool, copy_bytes, 0, &other_dev));
    HSA_CHECK(hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, other_host));
    memset(other_host, 0xa5, copy_bytes);

    auto signal = hsa_signal_t{};
    auto gate   = hsa_signal_t{};
    HSA_CHECK(hsa_signal_create(1, 0, nullptr, &signal));
    HSA_CHECK(hsa_signal_create(1, 0, nullptr, &gate));
    HSA_CHECK(hsa_amd_memory_async_copy(
        other_dev, gpu_agent, other_host, cpu_agent, copy_bytes, 0, nullptr, signal));
    wait_for(signal, 1);
    auto previous = query_time(signal, "reuse-initial");

    for(int i = 0; i < iterations; ++i)
    {
        hsa_signal_store_screlease(signal, 1);
        HSA_CHECK(hsa_amd_memory_async_copy(
            dev, gpu_agent, host, cpu_agent, copy_bytes, 0, nullptr, signal));
        wait_for(signal, 1);
        previous = query_time(signal, "reuse", true, previous.end);

        // The gate separates completion order from submission order. Opposite directions
        // use independent copy engines and separate buffers.
        for(int mode = 0; mode < 3; ++mode)
        {
            hsa_signal_store_screlease(signal, 2);
            hsa_signal_store_screlease(gate, 1);
            if(mode == 1) roctxProfilerPause(0);
            HSA_CHECK(hsa_amd_memory_async_copy(
                dev, gpu_agent, host, cpu_agent, copy_bytes, 1, &gate, signal));
            if(mode == 1) roctxProfilerResume(0);
            if(mode == 2) roctxProfilerPause(0);
            HSA_CHECK(hsa_amd_memory_async_copy(
                other_host, cpu_agent, other_dev, gpu_agent, copy_bytes, 0, nullptr, signal));
            if(mode == 2) roctxProfilerResume(0);
            wait_for(signal, 2);
            auto first = query_time(signal, "overlap-first", mode != 2, previous.end);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            hsa_signal_store_screlease(gate, 0);
            wait_for(signal, 1);
            previous = query_time(signal, "overlap-last", mode != 1, first.end);
        }

        hsa_signal_store_screlease(signal, 1);
        hsa_signal_store_screlease(gate, 1);
        HSA_CHECK(hsa_amd_memory_async_copy(
            dev, gpu_agent, host, cpu_agent, copy_bytes, 1, &gate, signal));
        auto rejected = hsa_amd_memory_async_copy(
            other_host, cpu_agent, other_dev, gpu_agent, copy_bytes, 1, nullptr, signal);
        if(rejected == HSA_STATUS_SUCCESS) return EXIT_FAILURE;
        hsa_signal_store_screlease(gate, 0);
        wait_for(signal, 1);
        previous = query_time(signal, "rejected-submission", true, previous.end);
    }

    HSA_CHECK(hsa_signal_destroy(gate));
    HSA_CHECK(hsa_signal_destroy(signal));
    HSA_CHECK(hsa_signal_create(1, 0, nullptr, &signal));
    auto empty_time = hsa_amd_profiling_async_copy_time_t{};
    if(hsa_amd_profiling_get_async_copy_time(signal, &empty_time) != HSA_STATUS_ERROR)
        return EXIT_FAILURE;
    HSA_CHECK(hsa_signal_destroy(signal));
    HSA_CHECK(hsa_amd_memory_pool_free(other_host));
    HSA_CHECK(hsa_amd_memory_pool_free(other_dev));
    HSA_CHECK(hsa_amd_memory_pool_free(host));
    HSA_CHECK(hsa_amd_memory_pool_free(dev));
    HSA_CHECK(hsa_shut_down());

    if(output != stdout) fclose(output);
    return EXIT_SUCCESS;
}
