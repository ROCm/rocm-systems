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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * @file tests/range-replay-verify/client.cpp
 *
 * @brief LD_PRELOAD tool that replays the application's range with verification on and checks that
 * the final replayed pass ended where the application's own run did.
 *
 * With ROCPROF_RANGE_REPLAY_VERIFY on, CLOSE reports how many snapshot regions differ between the
 * end of the application's execution of the range and the end of the final replayed pass. For the
 * kernels this suite runs, that count is zero only if the final pass re-ran every recorded
 * dispatch, in order, from the restored range-entry state: a pass that ran nothing leaves the entry
 * state, one that started from an earlier pass's output applies the in-place steps twice or counts
 * every element twice, and one that reordered the dispatches computes different values.
 *
 * Environment:
 *   RR_VERIFY_EXPECT_DISPATCHES  the dispatch_count CLOSE must report (per mode)
 *   ROCPROF_RANGE_REPLAY_VERIFY  must be 1: with verification off, divergence_count is 0 whatever
 *                                the passes did
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
#include <string_view>

#define RC(call)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        rocprofiler_status_t _s = (call);                                                          \
        if(_s != ROCPROFILER_STATUS_SUCCESS)                                                       \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[rr-verify] error '%s' @%d: %s\n",                                            \
                    #call,                                                                         \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

namespace
{
rocprofiler_context_id_t g_ctx{0};

std::atomic<int>      g_configs{0};
std::atomic<int>      g_passes{0};
std::atomic<int>      g_closes{0};
std::atomic<int>      g_status{ROCPROFILER_RANGE_REPLAY_STATUS_LAST};
std::atomic<uint64_t> g_dispatches{0};
std::atomic<uint64_t> g_divergence{0};

uint64_t pass_count(uint64_t, rocprofiler_user_data_t) { return kPasses; }

void
range_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) return;

    auto* data = static_cast<rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);
    if(data->range_id != kRangeId) return;

    switch(record.operation)
    {
        case ROCPROFILER_RANGE_REPLAY_CONFIG:
            data->pass_count_cb = pass_count;
            g_configs.fetch_add(1);
            break;
        case ROCPROFILER_RANGE_REPLAY_PASS: g_passes.fetch_add(1); break;
        case ROCPROFILER_RANGE_REPLAY_CLOSE:
            g_status.store(data->status);
            g_dispatches.store(data->dispatch_count);
            g_divergence.store(data->divergence_count);
            g_closes.fetch_add(1);
            break;
        default: break;
    }
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    RC(rocprofiler_create_context(&g_ctx));
    RC(rocprofiler_configure_callback_tracing_service(
        g_ctx, ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY, nullptr, 0, range_replay_cb, nullptr));
    RC(rocprofiler_start_context(g_ctx));
    return 0;
}

void
tool_fini(void*)
{
    if(const char* reason = std::getenv(kSkipEnv); reason != nullptr)
    {
        fprintf(stderr, "[rr-verify] SKIP: %s\n", reason);
        return;
    }

    const char* verify            = std::getenv("ROCPROF_RANGE_REPLAY_VERIFY");
    const char* expect_dispatches = std::getenv("RR_VERIFY_EXPECT_DISPATCHES");

    fprintf(stderr,
            "[rr-verify] configs=%d passes=%d closes=%d status=%d dispatches=%lu divergence=%lu "
            "expected_dispatches=%s verify=%s\n",
            g_configs.load(),
            g_passes.load(),
            g_closes.load(),
            g_status.load(),
            static_cast<unsigned long>(g_dispatches.load()),
            static_cast<unsigned long>(g_divergence.load()),
            expect_dispatches != nullptr ? expect_dispatches : "<unset>",
            verify != nullptr ? verify : "<unset>");

    auto ok = true;
    if(verify == nullptr || std::string_view{verify} != "1")
    {
        fprintf(stderr,
                "[rr-verify] FAIL: ROCPROF_RANGE_REPLAY_VERIFY must be 1, or divergence_count is "
                "not computed\n");
        ok = false;
    }
    if(g_configs.load() != 1 || g_closes.load() != 1)
    {
        fprintf(stderr, "[rr-verify] FAIL: expected exactly one CONFIG and one CLOSE\n");
        ok = false;
    }
    if(g_status.load() != static_cast<int>(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED))
    {
        fprintf(stderr,
                "[rr-verify] FAIL: the range was not replayed (rocprofiler_range_replay_status_t "
                "%d)\n",
                g_status.load());
        ok = false;
    }
    if(g_passes.load() != static_cast<int>(kPasses - 1))
    {
        fprintf(stderr,
                "[rr-verify] FAIL: expected %lu replayed passes\n",
                static_cast<unsigned long>(kPasses - 1));
        ok = false;
    }

    if(expect_dispatches == nullptr)
    {
        fprintf(stderr, "[rr-verify] FAIL: RR_VERIFY_EXPECT_DISPATCHES is unset\n");
        ok = false;
    }
    else
    {
        char*      end      = nullptr;
        const auto expected = std::strtoull(expect_dispatches, &end, 10);
        if(end == expect_dispatches || *end != '\0')
        {
            fprintf(stderr,
                    "[rr-verify] FAIL: RR_VERIFY_EXPECT_DISPATCHES '%s' is not an integer\n",
                    expect_dispatches);
            ok = false;
        }
        else if(g_dispatches.load() != expected)
        {
            fprintf(stderr,
                    "[rr-verify] FAIL: CLOSE reported %lu dispatches, expected %lu\n",
                    static_cast<unsigned long>(g_dispatches.load()),
                    static_cast<unsigned long>(expected));
            ok = false;
        }
    }

    if(g_divergence.load() != 0)
    {
        fprintf(stderr,
                "[rr-verify] FAIL: the final replayed pass left %lu snapshot regions different "
                "from the application's run\n",
                static_cast<unsigned long>(g_divergence.load()));
        ok = false;
    }

    fprintf(stderr, ok ? "[rr-verify] PASS\n" : "[rr-verify] FAIL\n");
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "rr-verify";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
