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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// End-to-end check that a dispatch counter collection context restricted to one GPU leaves
// the other GPUs alone -- neither instrumented nor serialized.
//
// The unit tests in source/lib/rocprofiler-sdk/counters/tests/per_agent_scoping_test.cpp cover
// the internal state directly. This test covers the property a user actually cares about: a
// concurrent multi-stream workload on a GPU outside the context's agent set must keep running
// concurrently. Serialization forces one dispatch at a time per agent, so the wall time of the
// workload is a direct proxy for whether the agent is serialized.
//
// Absolute times are meaningless across machines, so every timing assertion is a ratio against
// a baseline measured in the same process. The unrestricted context doubles as a positive
// control: if it does not visibly serialize the same workload, this machine cannot demonstrate
// stream concurrency at all and the timing assertions are skipped rather than guessed at. The
// record and callback assertions are exact and always run.

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[%s:%d] %s failed: %s\n",                                                     \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    msg,                                                                           \
                    rocprofiler_get_status_string(CHECKSTATUS));                                   \
            exit(1);                                                                               \
        }                                                                                          \
    }

#define HIP_CALL(cmd)                                                                              \
    {                                                                                              \
        hipError_t e = cmd;                                                                        \
        if(e != hipSuccess)                                                                        \
        {                                                                                          \
            fprintf(stderr, "[%s:%d] HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(e));  \
            exit(1);                                                                               \
        }                                                                                          \
    }

namespace
{
// Contexts are created during tool_init and started/stopped from main().
rocprofiler_context_id_t ctx_scoped_gpu0 = {.handle = 0};
rocprofiler_context_id_t ctx_scoped_gpu1 = {.handle = 0};
rocprofiler_context_id_t ctx_all         = {.handle = 0};
rocprofiler_context_id_t ctx_all_second  = {.handle = 0};

// gpu_agents[i] is the rocprofiler agent for HIP device i. Populated in tool_init by
// matching each HIP device's PCI domain/BDF to the rocprofiler agent with the same
// coordinates so that the contexts and assertions both refer to the same physical device
// regardless of the order rocprofiler enumerates agents or how visible-device variables
// reorder HIP ordinals.
std::vector<rocprofiler_agent_id_t> gpu_agents = {};

// Intermediate storage used to build the HIP-to-rocprofiler mapping: keeps the
// location_id (BDF) and PCI domain alongside the agent id until the mapping is done.
struct agent_info_t
{
    rocprofiler_agent_id_t id;
    uint32_t               location_id;  // (bus<<8)|(device<<3)|function
    uint32_t               domain;
};
std::vector<agent_info_t> all_gpu_agent_info = {};

std::mutex                 observed_mutex     = {};
std::map<uint64_t, size_t> dispatch_per_agent = {};
std::map<uint64_t, size_t> records_per_agent  = {};

void
reset_observations()
{
    auto lk = std::unique_lock{observed_mutex};
    dispatch_per_agent.clear();
    records_per_agent.clear();
}

size_t
dispatches_on(rocprofiler_agent_id_t agent)
{
    auto lk  = std::unique_lock{observed_mutex};
    auto itr = dispatch_per_agent.find(agent.handle);
    return (itr == dispatch_per_agent.end()) ? 0 : itr->second;
}

size_t
records_on(rocprofiler_agent_id_t agent)
{
    auto lk  = std::unique_lock{observed_mutex};
    auto itr = records_per_agent.find(agent.handle);
    return (itr == records_per_agent.end()) ? 0 : itr->second;
}

void
record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                rocprofiler_counter_record_t*,
                size_t record_count,
                rocprofiler_user_data_t,
                void*)
{
    if(record_count == 0) return;
    auto lk = std::unique_lock{observed_mutex};
    records_per_agent[dispatch_data.dispatch_info.agent_id.handle]++;
}

void
dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                  rocprofiler_counter_config_id_t*             config,
                  rocprofiler_user_data_t*,
                  void*)
{
    const auto agent_id = dispatch_data.dispatch_info.agent_id;

    {
        auto lk = std::unique_lock{observed_mutex};
        dispatch_per_agent[agent_id.handle]++;
    }

    static std::shared_mutex                                             cache_mutex = {};
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> cache       = {};

    auto search = [&]() {
        if(auto pos = cache.find(agent_id.handle); pos != cache.end())
        {
            *config = pos->second;
            return true;
        }
        return false;
    };

    {
        auto rlock = std::shared_lock{cache_mutex};
        if(search()) return;
    }

    auto wlock = std::unique_lock{cache_mutex};
    if(search()) return;

    auto supported = std::vector<rocprofiler_counter_id_t>{};
    ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
                         agent_id,
                         [](rocprofiler_agent_id_t,
                            rocprofiler_counter_id_t* counters,
                            size_t                    num_counters,
                            void*                     user_data) {
                             auto* vec =
                                 static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
                             for(size_t i = 0; i < num_counters; i++)
                                 vec->push_back(counters[i]);
                             return ROCPROFILER_STATUS_SUCCESS;
                         },
                         static_cast<void*>(&supported)),
                     "iterate supported counters");

    auto collect = std::vector<rocprofiler_counter_id_t>{};
    for(auto& counter : supported)
    {
        rocprofiler_counter_info_v0_t info;
        ROCPROFILER_CALL(rocprofiler_query_counter_info(
                             counter, ROCPROFILER_COUNTER_INFO_VERSION_0, (void*) &info),
                         "query counter info");
        if(strcmp(info.name, "SQ_WAVES") == 0) collect.push_back(counter);
    }

    if(collect.empty()) return;

    rocprofiler_counter_config_id_t profile = {.handle = 0};
    ROCPROFILER_CALL(
        rocprofiler_create_counter_config(agent_id, collect.data(), collect.size(), &profile),
        "create counter config");

    cache.emplace(agent_id.handle, profile);
    *config = profile;
}

void
configure_cc(rocprofiler_context_id_t* ctx, const rocprofiler_agent_id_t* agent)
{
    ROCPROFILER_CALL(rocprofiler_create_context(ctx), "create context");
    ROCPROFILER_CALL(rocprofiler_configure_callback_dispatch_counting_service(
                         *ctx, dispatch_callback, nullptr, record_callback, nullptr),
                     "configure dispatch counting service");
    if(agent)
        ROCPROFILER_CALL(rocprofiler_dispatch_counting_service_set_agents(*ctx, agent, 1),
                         "restrict counting service to agent");
}

rocprofiler_status_t
collect_gpu_agents(rocprofiler_agent_version_t, const void** agents, size_t num_agents, void*)
{
    for(size_t i = 0; i < num_agents; ++i)
    {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
        if(agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            all_gpu_agent_info.push_back({agent->id, agent->location_id, agent->domain});
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    ROCPROFILER_CALL(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                                        collect_gpu_agents,
                                                        sizeof(rocprofiler_agent_v0_t),
                                                        nullptr),
                     "query available agents");

    // Nothing to configure when the machine cannot express the scenario; main() reports the
    // skip so the reason is visible in the test output.
    if(all_gpu_agent_info.size() < 2) return 0;

    // Map each HIP device ordinal to its rocprofiler agent using PCI domain/BDF so that
    // gpu_agents[i] is the agent for HIP device i, regardless of the order rocprofiler
    // enumerates agents or how ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES reorders ordinals.
    int device_count = 0;
    if(hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) return 0;

    for(int dev = 0; dev < device_count; ++dev)
    {
        hipDeviceProp_t prop{};
        if(hipGetDeviceProperties(&prop, dev) != hipSuccess) return 0;
        const auto bus = static_cast<uint32_t>(prop.pciBusID);
        const auto did = static_cast<uint32_t>(prop.pciDeviceID);
        const auto dom = static_cast<uint32_t>(prop.pciDomainID);
        for(const auto& info : all_gpu_agent_info)
        {
            if(info.domain == dom && (info.location_id >> 8) == bus &&
               ((info.location_id >> 3) & 0x1Fu) == did)
            {
                gpu_agents.push_back(info.id);
                break;
            }
        }
    }

    if(gpu_agents.size() < 2) return 0;

    configure_cc(&ctx_scoped_gpu0, &gpu_agents[0]);
    configure_cc(&ctx_scoped_gpu1, &gpu_agents[1]);
    configure_cc(&ctx_all, nullptr);
    configure_cc(&ctx_all_second, nullptr);

    return 0;
}

void
tool_fini(void*)
{}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id)
{
    id->name        = "PerAgentScopingTest";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}

// -------------------------------------------------------------------------------------------
// Workload
// -------------------------------------------------------------------------------------------

__global__ void
spin_kernel(long long cycles, float* out)
{
    long long start = clock64();
    float     acc   = 0.0F;
    while(clock64() - start < cycles)
        acc += 1.0F;
    // Never true for a 64-thread block; keeps the loop from being optimized away without
    // adding a real store to the timed path.
    if(threadIdx.x == 1024) *out = acc;
}

namespace
{
constexpr int  kNumStreams = 8;
constexpr int  kKernelsPer = 4;
constexpr long kSpinCycles = 2000000;

// Launches kKernelsPer kernels on each of kNumStreams streams and waits for all of them. When
// the agent is not serialized the streams overlap and the wall time is roughly kKernelsPer
// kernels long; when it is serialized every dispatch runs alone and the wall time is roughly
// kNumStreams * kKernelsPer kernels long.
double
time_concurrent_workload(int device)
{
    HIP_CALL(hipSetDevice(device));

    hipStream_t streams[kNumStreams];
    for(auto& s : streams)
        HIP_CALL(hipStreamCreate(&s));

    float* out = nullptr;
    HIP_CALL(hipMalloc(&out, sizeof(float)));

    // Warm up so queue creation and code object loading are not inside the measurement.
    for(auto& s : streams)
        hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(64), 0, s, kSpinCycles / 10, out);
    HIP_CALL(hipDeviceSynchronize());

    auto t0 = std::chrono::steady_clock::now();
    for(int k = 0; k < kKernelsPer; ++k)
        for(auto& s : streams)
            hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(64), 0, s, kSpinCycles, out);
    HIP_CALL(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();

    for(auto& s : streams)
        HIP_CALL(hipStreamDestroy(s));
    HIP_CALL(hipFree(out));

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int failures = 0;

void
check(bool cond, const std::string& what)
{
    printf("[%s] %s\n", cond ? "PASS" : "FAIL", what.c_str());
    if(!cond) ++failures;
}
}  // namespace

int
main()
{
    int device_count = 0;
    HIP_CALL(hipGetDeviceCount(&device_count));

    if(device_count < 2 || gpu_agents.size() < 2)
    {
        printf("SKIP: per-agent scoping requires at least 2 GPUs (found %d)\n", device_count);
        return 0;
    }

    const auto agent_0 = gpu_agents[0];
    const auto agent_1 = gpu_agents[1];

    // Phase 1: no counter collection anywhere. Reference for an unserialized GPU-0.
    const double baseline_ms = time_concurrent_workload(0);

    // Phase 2: unrestricted context. This is what every tool got before per-agent scoping, and
    // it doubles as the positive control for the timing measurement.
    reset_observations();
    ROCPROFILER_CALL(rocprofiler_start_context(ctx_all), "start unrestricted context");
    const double unrestricted_ms = time_concurrent_workload(0);
    ROCPROFILER_CALL(rocprofiler_stop_context(ctx_all), "stop unrestricted context");
    const size_t unrestricted_dispatches_gpu0 = dispatches_on(agent_0);

    // Phase 3: context restricted to GPU-1. GPU-1 must still be collected on; GPU-0 must be
    // untouched.
    reset_observations();
    ROCPROFILER_CALL(rocprofiler_start_context(ctx_scoped_gpu1), "start scoped context");
    time_concurrent_workload(1);
    const double scoped_ms = time_concurrent_workload(0);
    ROCPROFILER_CALL(rocprofiler_stop_context(ctx_scoped_gpu1), "stop scoped context");
    const size_t scoped_records_gpu1    = records_on(agent_1);
    const size_t scoped_dispatches_gpu0 = dispatches_on(agent_0);
    const size_t scoped_records_gpu0    = records_on(agent_0);

    printf("baseline=%.2fms unrestricted=%.2fms scoped=%.2fms (%d streams x %d kernels)\n",
           baseline_ms,
           unrestricted_ms,
           scoped_ms,
           kNumStreams,
           kKernelsPer);

    // Exact assertions. These hold regardless of how fast the machine is.
    check(unrestricted_dispatches_gpu0 > 0,
          "unrestricted context sees dispatches on GPU-0 (control)");
    check(scoped_records_gpu1 > 0, "scoped context still collects counters on its own agent");
    check(scoped_dispatches_gpu0 == 0,
          "scoped context never receives a dispatch callback for GPU-0");
    check(scoped_records_gpu0 == 0, "scoped context produces no records for GPU-0");

    // Timing assertions, but only when the positive control shows this machine can actually
    // demonstrate the difference. Otherwise the workload never overlapped to begin with and a
    // ratio would be measuring noise.
    const double control_ratio = unrestricted_ms / baseline_ms;
    if(control_ratio < 2.0)
    {
        printf("SKIP timing checks: unrestricted context only slowed GPU-0 by %.2fx, so this "
               "machine does not show stream concurrency clearly enough to measure\n",
               control_ratio);
    }
    else
    {
        const double scoped_ratio = scoped_ms / baseline_ms;
        printf("serialization slowdown: unrestricted=%.2fx scoped=%.2fx\n",
               control_ratio,
               scoped_ratio);
        check(scoped_ratio < 2.0,
              "a context scoped to GPU-1 leaves GPU-0 running at baseline speed");
    }

    // Conflict semantics: disjoint agent sets may run together, overlapping ones may not.
    {
        auto status_a = rocprofiler_start_context(ctx_scoped_gpu0);
        auto status_b = rocprofiler_start_context(ctx_scoped_gpu1);
        check(status_a == ROCPROFILER_STATUS_SUCCESS && status_b == ROCPROFILER_STATUS_SUCCESS,
              "two counter collection contexts on disjoint agents can both be active");
        ROCPROFILER_CALL(rocprofiler_stop_context(ctx_scoped_gpu1), "stop gpu1 context");
        ROCPROFILER_CALL(rocprofiler_stop_context(ctx_scoped_gpu0), "stop gpu0 context");
    }

    {
        ROCPROFILER_CALL(rocprofiler_start_context(ctx_all), "start unrestricted context");
        auto status = rocprofiler_start_context(ctx_all_second);
        check(status == ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT,
              "two unrestricted counter collection contexts still conflict");
        ROCPROFILER_CALL(rocprofiler_stop_context(ctx_all), "stop unrestricted context");
    }

    if(failures > 0)
    {
        printf("%d check(s) failed\n", failures);
        return 1;
    }

    printf("all per-agent scoping checks passed\n");
    return 0;
}
