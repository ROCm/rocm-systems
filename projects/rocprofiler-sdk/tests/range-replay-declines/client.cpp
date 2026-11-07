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
 * @file tests/range-replay-declines/client.cpp
 *
 * @brief LD_PRELOAD tool that asks for a range to be replayed and checks that it was declined, for
 * the reason the test names, before any pass ran.
 *
 * It configures the range replay domain and nothing else. A device-writing copy inside a range must
 * decline it whether or not any tool traces memory copies, so the copy cases are only meaningful
 * while no memory-copy tracing is configured in the process.
 *
 * Environment:
 *   RR_DECLINE_EXPECT  the status CLOSE must report, by name (for example MEMORY_COPY_IN_RANGE)
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
                    "[rr-decline] error '%s' @%d: %s\n",                                           \
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

const char*
status_name(int status)
{
    switch(static_cast<rocprofiler_range_replay_status_t>(status))
    {
        case ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED: return "REPLAYED";
        case ROCPROFILER_RANGE_REPLAY_STATUS_NO_DISPATCH: return "NO_DISPATCH";
        case ROCPROFILER_RANGE_REPLAY_STATUS_NO_PASS_COUNT: return "NO_PASS_COUNT";
        case ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_QUEUE: return "MULTI_QUEUE";
        case ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_AGENT: return "MULTI_AGENT";
        case ROCPROFILER_RANGE_REPLAY_STATUS_GRAPH_LAUNCH: return "GRAPH_LAUNCH";
        case ROCPROFILER_RANGE_REPLAY_STATUS_UNKNOWN_KERNARG_SIZE: return "UNKNOWN_KERNARG_SIZE";
        case ROCPROFILER_RANGE_REPLAY_STATUS_MEMORY_COPY_IN_RANGE: return "MEMORY_COPY_IN_RANGE";
        case ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH: return "CONCURRENT_DISPATCH";
        case ROCPROFILER_RANGE_REPLAY_STATUS_PROGRAM_TOO_LARGE: return "PROGRAM_TOO_LARGE";
        case ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED: return "SNAPSHOT_FAILED";
        case ROCPROFILER_RANGE_REPLAY_STATUS_STAGING_FAILED: return "STAGING_FAILED";
        case ROCPROFILER_RANGE_REPLAY_STATUS_ALLOCATION_CHANGED_IN_RANGE:
            return "ALLOCATION_CHANGED_IN_RANGE";
        case ROCPROFILER_RANGE_REPLAY_STATUS_UNSUPPORTED_QUEUE_PATH:
            return "UNSUPPORTED_QUEUE_PATH";
        case ROCPROFILER_RANGE_REPLAY_STATUS_CODE_OBJECT_CHANGED_IN_RANGE:
            return "CODE_OBJECT_CHANGED_IN_RANGE";
        case ROCPROFILER_RANGE_REPLAY_STATUS_LAST: break;
    }
    return "<none>";
}

uint64_t pass_count(uint64_t, rocprofiler_user_data_t) { return kRequestedPasses; }

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
        fprintf(stderr, "[rr-decline] SKIP: %s\n", reason);
        return;
    }

    const char* expect = std::getenv("RR_DECLINE_EXPECT");
    const char* status = status_name(g_status.load());

    fprintf(stderr,
            "[rr-decline] configs=%d passes=%d closes=%d status=%s dispatches=%lu expected=%s\n",
            g_configs.load(),
            g_passes.load(),
            g_closes.load(),
            status,
            static_cast<unsigned long>(g_dispatches.load()),
            expect != nullptr ? expect : "<unset>");

    auto ok = true;
    if(expect == nullptr || std::string_view{status} != std::string_view{expect})
    {
        fprintf(stderr, "[rr-decline] FAIL: CLOSE did not report the expected status\n");
        ok = false;
    }
    if(g_configs.load() != 1 || g_closes.load() != 1)
    {
        fprintf(stderr, "[rr-decline] FAIL: expected exactly one CONFIG and one CLOSE\n");
        ok = false;
    }
    // The decline is known while the range is still recording, so no pass may have run.
    if(g_passes.load() != 0)
    {
        fprintf(stderr, "[rr-decline] FAIL: a declined range ran replayed passes\n");
        ok = false;
    }
    if(g_dispatches.load() != kObservedDispatches)
    {
        fprintf(stderr,
                "[rr-decline] FAIL: CLOSE reported %lu dispatches, expected %lu\n",
                static_cast<unsigned long>(g_dispatches.load()),
                static_cast<unsigned long>(kObservedDispatches));
        ok = false;
    }

    fprintf(stderr, ok ? "[rr-decline] PASS\n" : "[rr-decline] FAIL\n");
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "rr-declines";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
