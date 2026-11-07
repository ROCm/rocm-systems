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

/**
 * @file tests/range-replay-concurrency/client.cpp
 *
 * @brief LD_PRELOAD tool that replays every range main.cpp opens, with dispatch counters enabled
 * so every dispatch carries a completion handler, and checks that each range closed exactly once
 * with an outcome its concurrent writer allows.
 *
 * Environment:
 *   RR_CONC_MODE    packet (default) or host: whether CONFIG sets host_replay
 *   RR_CONC_RANGES  ranges main.cpp opens (default 16)
 */

#include "range.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define RC(call)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        rocprofiler_status_t _s = (call);                                                          \
        if(_s != ROCPROFILER_STATUS_SUCCESS)                                                       \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[rr-conc] error '%s' @%d: %s\n",                                              \
                    #call,                                                                         \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

namespace
{
constexpr uint64_t kPasses = 4;

rocprofiler_context_id_t g_replay_ctx{0};
rocprofiler_context_id_t g_counters_ctx{0};

bool     g_host   = false;
uint64_t g_ranges = 16;

std::mutex                 g_mtx{};
std::map<uint64_t, int>    g_closes_per_range{};
std::map<std::string, int> g_status_counts{};
std::atomic<int>           g_unexpected{0};

uint64_t pass_count(uint64_t, rocprofiler_user_data_t) { return kPasses; }

const char*
status_name(rocprofiler_range_replay_status_t status)
{
    switch(status)
    {
        case ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED: return "REPLAYED";
        case ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH: return "CONCURRENT_DISPATCH";
        case ROCPROFILER_RANGE_REPLAY_STATUS_MEMORY_COPY_IN_RANGE: return "MEMORY_COPY_IN_RANGE";
        default: return nullptr;
    }
}

void
range_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) return;
    auto* p = static_cast<rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);
    if(p == nullptr) return;

    if(record.operation == ROCPROFILER_RANGE_REPLAY_CONFIG)
    {
        p->pass_count_cb = pass_count;
        p->host_replay   = g_host ? 1 : 0;
    }
    else if(record.operation == ROCPROFILER_RANGE_REPLAY_CLOSE)
    {
        // Another thread writing device memory during the range declines it; nothing else may.
        const char* name = status_name(p->status);
        auto        lk   = std::lock_guard<std::mutex>{g_mtx};
        ++g_closes_per_range[p->range_id];
        if(name == nullptr)
        {
            fprintf(stderr,
                    "[rr-conc] range %lu closed with unexpected status %d\n",
                    static_cast<unsigned long>(p->range_id),
                    static_cast<int>(p->status));
            g_unexpected.fetch_add(1);
            return;
        }
        ++g_status_counts[name];
    }
}

void
counter_record_cb(rocprofiler_dispatch_counting_service_data_t,
                  rocprofiler_counter_record_t*,
                  size_t,
                  rocprofiler_user_data_t,
                  void*)
{}

void
counter_dispatch_cb(rocprofiler_dispatch_counting_service_data_t d,
                    rocprofiler_counter_config_id_t*             config,
                    rocprofiler_user_data_t*,
                    void*)
{
    static std::mutex                                                    m{};
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> cache{};
    const auto agent = d.dispatch_info.agent_id;

    auto lk = std::lock_guard<std::mutex>{m};
    if(auto it = cache.find(agent.handle); it != cache.end())
    {
        *config = it->second;
        return;
    }

    std::vector<rocprofiler_counter_id_t> all{};
    RC(rocprofiler_iterate_agent_supported_counters(
        agent,
        [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* cs, size_t n, void* ud) {
            auto* v = static_cast<std::vector<rocprofiler_counter_id_t>*>(ud);
            for(size_t i = 0; i < n; ++i)
                v->push_back(cs[i]);
            return ROCPROFILER_STATUS_SUCCESS;
        },
        &all));
    std::vector<rocprofiler_counter_id_t> want{};
    for(auto cc : all)
    {
        rocprofiler_counter_info_v0_t info{};
        RC(rocprofiler_query_counter_info(cc, ROCPROFILER_COUNTER_INFO_VERSION_0, &info));
        if(info.name && std::string{info.name} == "SQ_WAVES") want.push_back(cc);
    }
    if(want.empty())
    {
        fprintf(stderr, "[rr-conc] error: SQ_WAVES not found\n");
        std::abort();
    }
    rocprofiler_counter_config_id_t cfg{.handle = 0};
    RC(rocprofiler_create_counter_config(agent, want.data(), want.size(), &cfg));
    cache.emplace(agent.handle, cfg);
    *config = cfg;
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    const char* mode = std::getenv("RR_CONC_MODE");
    g_host           = mode != nullptr && std::string{mode} == "host";
    if(const char* v = std::getenv("RR_CONC_RANGES"); v != nullptr && *v != '\0')
        g_ranges = std::strtoull(v, nullptr, 10);

    RC(rocprofiler_create_context(&g_replay_ctx));
    RC(rocprofiler_configure_callback_tracing_service(g_replay_ctx,
                                                      ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                                                      nullptr,
                                                      0,
                                                      range_replay_cb,
                                                      nullptr));
    RC(rocprofiler_create_context(&g_counters_ctx));
    RC(rocprofiler_configure_callback_dispatch_counting_service(
        g_counters_ctx, counter_dispatch_cb, nullptr, counter_record_cb, nullptr));

    RC(rocprofiler_start_context(g_counters_ctx));
    RC(rocprofiler_start_context(g_replay_ctx));
    return 0;
}

void
tool_fini(void*)
{
    auto lk = std::lock_guard<std::mutex>{g_mtx};
    bool ok = g_unexpected.load() == 0;

    fprintf(stderr, "[rr-conc] mode=%s closes:", g_host ? "host" : "packet");
    for(const auto& [name, count] : g_status_counts)
        fprintf(stderr, " %s=%d", name.c_str(), count);
    fprintf(stderr, "\n");

    for(uint64_t r = 0; r < g_ranges; ++r)
    {
        const auto it = g_closes_per_range.find(kRangeIdBase + r);
        const int  n  = (it == g_closes_per_range.end()) ? 0 : it->second;
        if(n != 1)
        {
            fprintf(stderr,
                    "[rr-conc] range %lu closed %d times, expected once\n",
                    static_cast<unsigned long>(kRangeIdBase + r),
                    n);
            ok = false;
        }
    }
    if(g_closes_per_range.size() != g_ranges) ok = false;

    fprintf(stderr, ok ? "[rr-conc] PASS\n" : "[rr-conc] FAIL\n");
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "rr-conc";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
