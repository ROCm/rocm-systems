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
 * @file tests/range-replay-host-api/client.cpp
 *
 * @brief LD_PRELOAD tool for main.cpp's host_run range: chooses the replay mode and pass count at
 * CONFIG, and checks the PASS sequence and what CLOSE reported.
 *
 * Environment:
 *   RR_API_TOOL_MODE   host (default): set host_replay at CONFIG and expect REPLAYED.
 *                      packet: leave host_replay unset and expect a decline -- the range spans two
 *                      queues and copies into device memory, so packet replay cannot re-execute it.
 *   RR_API_PASSES      total passes including the application's own (default 4)
 *   RR_API_THROW_PASS  pass on which main.cpp's body throws (default -1, never): expect
 *                      BODY_EXCEPTION after that many replayed passes, each entered and exited
 */

#include "range.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define RC(call)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        rocprofiler_status_t _s = (call);                                                          \
        if(_s != ROCPROFILER_STATUS_SUCCESS)                                                       \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[rr-api] error '%s' @%d: %s\n",                                               \
                    #call,                                                                         \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

namespace
{
rocprofiler_context_id_t g_ctx{0};

bool    g_packet     = false;
int64_t g_passes     = 4;
int64_t g_throw_pass = -1;

std::atomic<int>      g_configs{0};
std::atomic<int>      g_pass_enters{0};
std::atomic<int>      g_pass_exits{0};
std::atomic<int>      g_pass_order_errors{0};
std::atomic<uint64_t> g_last_pass{0};
std::atomic<int>      g_closes{0};
std::atomic<int>      g_close_status{-1};
std::atomic<uint64_t> g_close_dispatches{0};
std::atomic<uint64_t> g_close_divergence{0};

int64_t
env_i64(const char* name, int64_t fallback)
{
    const char* v = std::getenv(name);
    if(!v || *v == '\0') return fallback;
    return std::strtoll(v, nullptr, 10);
}

uint64_t pass_count(uint64_t, rocprofiler_user_data_t) { return static_cast<uint64_t>(g_passes); }

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
            if(p->current_pass != g_last_pass.load() + 1 ||
               g_pass_enters.load() != g_pass_exits.load())
                g_pass_order_errors.fetch_add(1);
            g_last_pass.store(p->current_pass);
            g_pass_enters.fetch_add(1);
        }
        else
        {
            if(p->current_pass != g_last_pass.load()) g_pass_order_errors.fetch_add(1);
            g_pass_exits.fetch_add(1);
        }
        return;
    }

    if(record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) return;

    switch(record.operation)
    {
        case ROCPROFILER_RANGE_REPLAY_CONFIG:
            g_configs.fetch_add(1);
            p->pass_count_cb = pass_count;
            p->host_replay   = g_packet ? 0 : 1;
            break;
        case ROCPROFILER_RANGE_REPLAY_CLOSE:
            g_closes.fetch_add(1);
            g_close_status.store(static_cast<int>(p->status));
            g_close_dispatches.store(p->dispatch_count);
            g_close_divergence.store(p->divergence_count);
            break;
        default: break;
    }
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    const char* mode = std::getenv("RR_API_TOOL_MODE");
    g_packet         = (mode != nullptr && std::string{mode} == "packet");
    g_passes         = env_i64("RR_API_PASSES", 4);
    g_throw_pass     = env_i64("RR_API_THROW_PASS", -1);

    RC(rocprofiler_create_context(&g_ctx));
    RC(rocprofiler_configure_callback_tracing_service(
        g_ctx, ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY, nullptr, 0, range_replay_cb, nullptr));
    RC(rocprofiler_start_context(g_ctx));
    return 0;
}

void
tool_fini(void*)
{
    bool ok   = true;
    auto fail = [&ok](const char* what) {
        fprintf(stderr, "[rr-api] FAIL: %s\n", what);
        ok = false;
    };

    const auto status = g_close_status.load();
    fprintf(stderr,
            "[rr-api] packet=%d configs=%d closes=%d status=%d dispatches=%lu divergence=%lu "
            "pass_enters=%d pass_exits=%d\n",
            g_packet ? 1 : 0,
            g_configs.load(),
            g_closes.load(),
            status,
            static_cast<unsigned long>(g_close_dispatches.load()),
            static_cast<unsigned long>(g_close_divergence.load()),
            g_pass_enters.load(),
            g_pass_exits.load());

    if(g_configs.load() != 1) fail("expected exactly one CONFIG for the range");
    if(g_closes.load() != 1) fail("expected exactly one CLOSE for the range");
    if(g_pass_order_errors.load() != 0) fail("PASS callbacks out of order or unpaired");

    auto replayed = 0;
    if(g_packet)
    {
        if(status == static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED) ||
           status == static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_HOST_JMP_NOT_SET) ||
           status == static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_BODY_EXCEPTION))
            fail("packet replay should decline this range for a packet-replay reason");
    }
    else if(g_throw_pass >= 0)
    {
        replayed = static_cast<int>(g_throw_pass);
        if(status != static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_BODY_EXCEPTION))
            fail("expected BODY_EXCEPTION");
    }
    else
    {
        replayed = static_cast<int>(g_passes - 1);
        if(status != static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED))
            fail("the range was not replayed");
        if(g_close_dispatches.load() < static_cast<uint64_t>(kKernelsPerPass))
            fail("CLOSE reported fewer dispatches than the range launches");
        if(std::getenv("ROCPROF_RANGE_REPLAY_VERIFY") != nullptr && g_close_divergence.load() != 0)
            fail("replayed passes diverged from the application's run");
    }

    if(g_pass_enters.load() != replayed || g_pass_exits.load() != replayed)
    {
        fprintf(stderr, "[rr-api] FAIL: expected %d replayed passes\n", replayed);
        ok = false;
    }

    fprintf(stderr, ok ? "[rr-api] PASS\n" : "[rr-api] FAIL\n");
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "rr-api";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
