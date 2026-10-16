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

// Submits async copies with application-owned completion signals and reads
// hsa_amd_profiling_get_async_copy_time() on those signals after waiting, the way libomptarget
// does. Exits with failure if any copy time is unavailable.

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
}  // namespace

int
main()
{
    HSA_CHECK(hsa_init());
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

    int failures = 0;
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

        while(hsa_signal_wait_scacquire(
                  signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED) >= 1)
        {}

        auto time   = hsa_amd_profiling_async_copy_time_t{};
        auto status = hsa_amd_profiling_get_async_copy_time(signal, &time);
        if(status != HSA_STATUS_SUCCESS || time.end <= time.start)
        {
            fprintf(stderr,
                    "%s copy %d: hsa_amd_profiling_get_async_copy_time returned %d (start=%lu "
                    "end=%lu)\n",
                    (to_device) ? "host-to-device" : "device-to-host",
                    i,
                    static_cast<int>(status),
                    static_cast<unsigned long>(time.start),
                    static_cast<unsigned long>(time.end));
            ++failures;
        }

        HSA_CHECK(hsa_signal_destroy(signal));
    }

    HSA_CHECK(hsa_amd_memory_pool_free(host));
    HSA_CHECK(hsa_amd_memory_pool_free(dev));
    HSA_CHECK(hsa_shut_down());

    return (failures == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
