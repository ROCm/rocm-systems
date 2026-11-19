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
 * @file tests/range-replay-host/client.cpp
 *
 * @brief LD_PRELOAD tool that host-replays main.cpp's time-stepping range with dispatch counters
 * enabled, and checks what the range reported and what the counter service collected.
 *
 * Environment:
 *   RR_HOST_TOOL_MODE  host (default): set host_replay at CONFIG and expect REPLAYED.
 *                      host-unarmed: set host_replay and expect HOST_JMP_NOT_SET (the application
 *                      opens the range without the jump buffer).
 *                      packet: leave host_replay unset and expect a decline -- the same range is
 *                      not something packet replay can re-execute.
 *   RR_HOST_PASSES     total passes including the application's own (default 4)
 *   RR_HOST_CONTINUE   1: open-ended pass count, stopped by replay_continue_cb at RR_HOST_PASSES
 *   RR_HOST_STOP_PASS  replayed pass at whose PASS PHASE_ENTER the counter context is locally
 *                      stopped; < 1 = never
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
#include <mutex>
#include <set>
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
                    "[rr-host] error '%s' @%d: %s\n",                                              \
                    #call,                                                                         \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

namespace
{
enum class tool_mode
{
    host,
    host_unarmed,
    packet,
};

rocprofiler_context_id_t g_replay_ctx{0};
rocprofiler_context_id_t g_counters_ctx{0};

tool_mode g_mode      = tool_mode::host;
int64_t   g_passes    = 4;
bool      g_continue  = false;
int64_t   g_stop_pass = -1;

std::mutex         g_kernels_mtx{};
std::set<uint64_t> g_target_kernels{};

std::atomic<bool> g_in_range{false};
std::atomic<int>  g_counter_records{0};
std::atomic<int>  g_local_stops{0};
std::atomic<int>  g_pass_enters{0};
std::atomic<int>  g_pass_exits{0};
std::atomic<int>  g_pass_order_errors{0};
std::atomic<int>  g_closes{0};
std::atomic<int>  g_close_status{-1};

std::atomic<uint64_t> g_close_dispatches{0};
std::atomic<uint64_t> g_close_divergence{0};
std::atomic<uint64_t> g_last_pass{0};

int64_t
env_i64(const char* name, int64_t fallback)
{
    const char* v = std::getenv(name);
    if(!v || *v == '\0') return fallback;
    return std::strtoll(v, nullptr, 10);
}

bool
is_target_kernel(uint64_t kernel_id)
{
    auto lk = std::lock_guard<std::mutex>{g_kernels_mtx};
    return g_target_kernels.count(kernel_id) != 0;
}

uint64_t pass_count(uint64_t, rocprofiler_user_data_t)
{
    return g_continue ? 0 : static_cast<uint64_t>(g_passes);
}

int
replay_continue(uint64_t, uint64_t current_pass, uint64_t, rocprofiler_user_data_t)
{
    return static_cast<int64_t>(current_pass) + 1 < g_passes ? 1 : 0;
}

void
range_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY) return;
    auto* p = static_cast<rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);
    if(p == nullptr || p->range_id != kRangeId) return;

    if(record.operation == ROCPROFILER_RANGE_REPLAY_PASS)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            // Passes arrive in order, one at a time: each enter follows the previous pass's exit.
            if(p->current_pass != g_last_pass.load() + 1 ||
               g_pass_enters.load() != g_pass_exits.load())
                g_pass_order_errors.fetch_add(1);
            g_last_pass.store(p->current_pass);
            g_pass_enters.fetch_add(1);

            if(g_stop_pass >= 1 && static_cast<int64_t>(p->current_pass) == g_stop_pass &&
               p->replay_local_stop_context_cb != nullptr)
            {
                auto st = p->replay_local_stop_context_cb(g_counters_ctx);
                if(st == ROCPROFILER_STATUS_SUCCESS)
                    g_local_stops.fetch_add(1);
                else
                    fprintf(stderr,
                            "[rr-host] local_stop failed: %s\n",
                            rocprofiler_get_status_string(st));
            }
        }
        else
        {
            g_pass_exits.fetch_add(1);
        }
        return;
    }

    if(record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) return;

    switch(record.operation)
    {
        case ROCPROFILER_RANGE_REPLAY_CONFIG:
            g_in_range.store(true);
            p->pass_count_cb = pass_count;
            if(g_continue) p->replay_continue_cb = replay_continue;
            p->host_replay = (g_mode == tool_mode::packet) ? 0 : 1;
            break;
        case ROCPROFILER_RANGE_REPLAY_CLOSE:
            g_in_range.store(false);
            g_closes.fetch_add(1);
            g_close_status.store(static_cast<int>(p->status));
            g_close_dispatches.store(p->dispatch_count);
            g_close_divergence.store(p->divergence_count);
            break;
        default: break;
    }
}

void
code_object_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.operation != ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    const auto* data =
        static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(
            record.payload);
    if(data == nullptr || data->kernel_name == nullptr) return;
    if(std::strstr(data->kernel_name, kKernelPrefix) == nullptr) return;

    auto lk = std::lock_guard<std::mutex>{g_kernels_mtx};
    g_target_kernels.insert(data->kernel_id);
}

// Records are counted by what was decided at dispatch: the warm-up step outside the range
// dispatches the same kernels, and its records may complete after the range opens.
void
counter_record_cb(rocprofiler_dispatch_counting_service_data_t,
                  rocprofiler_counter_record_t*,
                  size_t,
                  rocprofiler_user_data_t user_data,
                  void*)
{
    if(user_data.value == 1) g_counter_records.fetch_add(1);
}

void
counter_dispatch_cb(rocprofiler_dispatch_counting_service_data_t d,
                    rocprofiler_counter_config_id_t*             config,
                    rocprofiler_user_data_t*                     user_data,
                    void*)
{
    if(!g_in_range.load() || !is_target_kernel(d.dispatch_info.kernel_id)) return;
    user_data->value = 1;

    static std::mutex                                                    m{};
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> cache{};
    const auto agent = d.dispatch_info.agent_id;
    {
        std::lock_guard<std::mutex> lk{m};
        if(auto it = cache.find(agent.handle); it != cache.end())
        {
            *config = it->second;
            return;
        }
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
        fprintf(stderr, "[rr-host] SQ_WAVES not found\n");
        std::abort();
    }
    rocprofiler_counter_config_id_t cfg{.handle = 0};
    RC(rocprofiler_create_counter_config(agent, want.data(), want.size(), &cfg));
    {
        std::lock_guard<std::mutex> lk{m};
        cache.emplace(agent.handle, cfg);
    }
    *config = cfg;
}

// Passes, out of RR_HOST_PASSES, that collect counters. Pass 0 always does; a local stop is sticky.
int
expected_collecting_passes()
{
    if(g_mode != tool_mode::host) return 1;
    if(g_stop_pass < 1 || g_stop_pass >= g_passes) return static_cast<int>(g_passes);
    return static_cast<int>(g_stop_pass);
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    const char* mode = std::getenv("RR_HOST_TOOL_MODE");
    if(mode != nullptr && std::string{mode} == "packet")
        g_mode = tool_mode::packet;
    else if(mode != nullptr && std::string{mode} == "host-unarmed")
        g_mode = tool_mode::host_unarmed;

    g_passes    = env_i64("RR_HOST_PASSES", 4);
    g_continue  = env_i64("RR_HOST_CONTINUE", 0) != 0;
    g_stop_pass = env_i64("RR_HOST_STOP_PASS", -1);

    RC(rocprofiler_create_context(&g_replay_ctx));
    RC(rocprofiler_configure_callback_tracing_service(g_replay_ctx,
                                                      ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                                                      nullptr,
                                                      0,
                                                      range_replay_cb,
                                                      nullptr));
    RC(rocprofiler_configure_callback_tracing_service(g_replay_ctx,
                                                      ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                      nullptr,
                                                      0,
                                                      code_object_cb,
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
    bool ok = true;

    const auto status = g_close_status.load();
    fprintf(stderr,
            "[rr-host] mode=%d closes=%d status=%d dispatches=%lu divergence=%lu pass_enters=%d "
            "pass_exits=%d counter_records=%d\n",
            static_cast<int>(g_mode),
            g_closes.load(),
            status,
            static_cast<unsigned long>(g_close_dispatches.load()),
            static_cast<unsigned long>(g_close_divergence.load()),
            g_pass_enters.load(),
            g_pass_exits.load(),
            g_counter_records.load());

    if(g_closes.load() != 1)
    {
        fprintf(stderr, "[rr-host] FAIL: expected exactly one CLOSE for the range\n");
        ok = false;
    }

    switch(g_mode)
    {
        case tool_mode::host:
            if(status != static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED))
            {
                fprintf(stderr, "[rr-host] FAIL: the range was not replayed (status %d)\n", status);
                ok = false;
            }
            break;
        case tool_mode::host_unarmed:
            if(status != static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_HOST_JMP_NOT_SET))
            {
                fprintf(stderr, "[rr-host] FAIL: expected HOST_JMP_NOT_SET (status %d)\n", status);
                ok = false;
            }
            break;
        case tool_mode::packet:
            if(status == static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED) ||
               status == static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_HOST_JMP_NOT_SET))
            {
                fprintf(stderr,
                        "[rr-host] FAIL: packet replay should decline this range (status %d)\n",
                        status);
                ok = false;
            }
            break;
    }

    const int replayed = (g_mode == tool_mode::host) ? static_cast<int>(g_passes - 1) : 0;
    if(g_pass_enters.load() != replayed || g_pass_exits.load() != replayed)
    {
        fprintf(stderr, "[rr-host] FAIL: expected %d replayed passes\n", replayed);
        ok = false;
    }
    if(g_pass_order_errors.load() != 0)
    {
        fprintf(stderr, "[rr-host] FAIL: PASS callbacks out of order\n");
        ok = false;
    }
    // Only a host range counts every dispatch: a declined range stops counting at the decline.
    if(g_mode == tool_mode::host &&
       g_close_dispatches.load() < static_cast<uint64_t>(kKernelsPerPass))
    {
        fprintf(stderr,
                "[rr-host] FAIL: CLOSE reported %lu dispatches, the range launches at least %d\n",
                static_cast<unsigned long>(g_close_dispatches.load()),
                kKernelsPerPass);
        ok = false;
    }
    if(std::getenv("ROCPROF_RANGE_REPLAY_VERIFY") != nullptr && g_close_divergence.load() != 0)
    {
        fprintf(stderr, "[rr-host] FAIL: replayed passes diverged from the application's run\n");
        ok = false;
    }

    const int want_records = kKernelsPerPass * expected_collecting_passes();
    fprintf(stderr,
            "[rr-host] counters records=%d expected=%d local_stops=%d\n",
            g_counter_records.load(),
            want_records,
            g_local_stops.load());
    if(g_counter_records.load() != want_records)
    {
        fprintf(stderr, "[rr-host] FAIL: counter record count\n");
        ok = false;
    }

    fprintf(stderr, ok ? "[rr-host] PASS\n" : "[rr-host] FAIL\n");
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "rr-host";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
