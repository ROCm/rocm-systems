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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Regression test for counter-record coverage in rocprofiler_synchronize_contexts.
//
// A dispatch's counter record is handed to a consumer thread and written there, so a tool that
// synchronizes on its own dispatch and flushes immediately can get a buffer with the dispatch's
// counter record still missing. The record then lands in the following window.
//
// Every window is asserted, never an aggregate: a late record arrives in the next window, so
// totals reconcile even when every individual window was short.

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/synchronize_contexts.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define CHECK_ROCP(...)                                                                            \
    {                                                                                              \
        auto _status = (__VA_ARGS__);                                                              \
        if(_status != ROCPROFILER_STATUS_SUCCESS)                                                  \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[counter-window] %s failed at line %i: %s\n",                                 \
                    #__VA_ARGS__,                                                                  \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_status));                                       \
            std::abort();                                                                          \
        }                                                                                          \
    }

#define CHECK_HIP(...)                                                                             \
    {                                                                                              \
        auto _err = (__VA_ARGS__);                                                                 \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[counter-window] %s failed at line %i: %s\n",                                 \
                    #__VA_ARGS__,                                                                  \
                    __LINE__,                                                                      \
                    hipGetErrorString(_err));                                                      \
            std::abort();                                                                          \
        }                                                                                          \
    }

namespace
{
// Enough windows that passing without the drain is not a realistic outcome. A single window is
// only sometimes short, so a small count can pass against a broken build.
constexpr uint64_t kWindows             = 1000;
constexpr uint64_t kDispatchesPerWindow = 2;

rocprofiler_context_id_t g_ctx    = {};
rocprofiler_buffer_id_t  g_buffer = {};

std::atomic<uint64_t> g_headers_this_window{0};
std::atomic<uint64_t> g_headers_total{0};

void
buffer_callback(rocprofiler_context_id_t,
                rocprofiler_buffer_id_t,
                rocprofiler_record_header_t** headers,
                size_t                        num_headers,
                void*,
                uint64_t)
{
    for(size_t i = 0; i < num_headers; ++i)
    {
        if(headers[i] == nullptr) continue;
        if(headers[i]->category != ROCPROFILER_BUFFER_CATEGORY_COUNTERS) continue;
        if(headers[i]->kind != ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER)
            continue;

        g_headers_this_window.fetch_add(1, std::memory_order_relaxed);
        g_headers_total.fetch_add(1, std::memory_order_relaxed);
    }
}

void
dispatch_callback(rocprofiler_dispatch_counting_service_data_t data,
                  rocprofiler_counter_config_id_t*             config,
                  rocprofiler_user_data_t*,
                  void*)
{
    static auto _mutex = std::mutex{};
    static auto _cache = std::unordered_map<uint64_t, rocprofiler_counter_config_id_t>{};

    const auto _agent = data.dispatch_info.agent_id;
    {
        auto _lk = std::lock_guard<std::mutex>{_mutex};
        if(auto itr = _cache.find(_agent.handle); itr != _cache.end())
        {
            *config = itr->second;
            return;
        }
    }

    auto _all = std::vector<rocprofiler_counter_id_t>{};
    CHECK_ROCP(rocprofiler_iterate_agent_supported_counters(
        _agent,
        [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* cs, size_t n, void* ud) {
            auto* _v = static_cast<std::vector<rocprofiler_counter_id_t>*>(ud);
            for(size_t i = 0; i < n; ++i)
                _v->push_back(cs[i]);
            return ROCPROFILER_STATUS_SUCCESS;
        },
        &_all));

    auto _want = std::vector<rocprofiler_counter_id_t>{};
    for(auto citr : _all)
    {
        auto _info = rocprofiler_counter_info_v0_t{};
        CHECK_ROCP(
            rocprofiler_query_counter_info(citr, ROCPROFILER_COUNTER_INFO_VERSION_0, &_info));
        if(_info.name && std::string{_info.name} == "SQ_WAVES") _want.push_back(citr);
    }

    if(_want.empty())
    {
        fprintf(stderr, "[counter-window] SQ_WAVES is not available on this agent\n");
        std::abort();
    }

    auto _cfg = rocprofiler_counter_config_id_t{.handle = 0};
    CHECK_ROCP(rocprofiler_create_counter_config(_agent, _want.data(), _want.size(), &_cfg));
    {
        auto _lk = std::lock_guard<std::mutex>{_mutex};
        _cache.emplace(_agent.handle, _cfg);
    }
    *config = _cfg;
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    CHECK_ROCP(rocprofiler_create_context(&g_ctx));
    CHECK_ROCP(rocprofiler_create_buffer(g_ctx,
                                         512 * 1024,
                                         256 * 1024,
                                         ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                         buffer_callback,
                                         nullptr,
                                         &g_buffer));
    CHECK_ROCP(rocprofiler_configure_buffer_dispatch_counting_service(
        g_ctx, g_buffer, dispatch_callback, nullptr));
    CHECK_ROCP(rocprofiler_start_context(g_ctx));
    return 0;
}

void
tool_fini(void*)
{}

__global__ void
touch_kernel(float* ptr)
{
    ptr[threadIdx.x] = static_cast<float>(threadIdx.x);
}
}  // namespace

extern "C" __attribute__((visibility("default"))) rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id)
{
    id->name         = "synchronize-contexts-counter-window";
    static auto _cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &_cfg;
}

int
main()
{
    float* _dev = nullptr;
    CHECK_HIP(hipMalloc(&_dev, 64 * sizeof(float)));

    auto     _deficient       = uint64_t{0};
    auto     _first_bad       = uint64_t{0};
    uint64_t _first_bad_count = 0;

    for(uint64_t w = 0; w < kWindows; ++w)
    {
        g_headers_this_window.store(0, std::memory_order_relaxed);

        for(uint64_t k = 0; k < kDispatchesPerWindow; ++k)
        {
            hipLaunchKernelGGL(touch_kernel, dim3(1), dim3(64), 0, nullptr, _dev);
            CHECK_HIP(hipGetLastError());
        }
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_ROCP(rocprofiler_synchronize_contexts(&g_ctx, 1));
        CHECK_ROCP(rocprofiler_flush_buffer(g_buffer));

        const auto _got = g_headers_this_window.load(std::memory_order_relaxed);
        if(_got < kDispatchesPerWindow)
        {
            if(_deficient == 0)
            {
                _first_bad       = w;
                _first_bad_count = _got;
            }
            ++_deficient;
        }
    }

    CHECK_HIP(hipFree(_dev));

    // A run that produced no counter records at all would report every window deficient for a
    // reason unrelated to the ordering this test is about.
    if(g_headers_total.load() == 0)
    {
        fprintf(stderr, "[counter-window] no counter records were delivered at all\n");
        return 2;
    }

    if(_deficient > 0)
    {
        fprintf(stderr,
                "[counter-window] FAILED: %lu of %lu windows were short; first was window %lu "
                "with %lu of %lu records\n",
                static_cast<unsigned long>(_deficient),
                static_cast<unsigned long>(kWindows),
                static_cast<unsigned long>(_first_bad),
                static_cast<unsigned long>(_first_bad_count),
                static_cast<unsigned long>(kDispatchesPerWindow));
        return 1;
    }

    fprintf(stdout,
            "[counter-window] PASS: %lu windows, %lu counter records\n",
            static_cast<unsigned long>(kWindows),
            static_cast<unsigned long>(g_headers_total.load()));
    return 0;
}
