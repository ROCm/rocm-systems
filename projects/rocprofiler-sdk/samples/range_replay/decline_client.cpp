// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

// A range the SDK refuses to replay, and the two things a tool is entitled to expect of a refusal:
// CLOSE names the reason, and the application's own execution of the range is untouched.
//
// This tool asks for passes exactly as basic_client does. The difference is the application: run
// with RR_APP_MODE=multi-queue, main.cpp dispatches to a second stream inside the range. Replay
// re-submits a recording to one queue, and two queues can interleave their work differently on
// every run, so a range that spans both has no single ordering to re-execute. The range is
// declined rather than replayed under an ordering the application never had.

#include "client.hpp"

#include <rocprofiler-sdk/registration.h>

#include <atomic>
#include <cstdint>

namespace
{
constexpr uint64_t kPasses = 4;

rocprofiler_context_id_t g_ctx{0};

std::atomic<int> g_configs{0};
std::atomic<int> g_passes{0};
std::atomic<int> g_closes{0};
std::atomic<int> g_status{ROCPROFILER_RANGE_REPLAY_STATUS_LAST};

uint64_t pass_count(uint64_t, rocprofiler_user_data_t) { return kPasses; }

void
range_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY) return;

    auto* data = static_cast<rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);
    if(data->range_id != kRangeId) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) return;

    switch(record.operation)
    {
        case ROCPROFILER_RANGE_REPLAY_CONFIG:
            data->pass_count_cb = pass_count;
            g_configs.fetch_add(1);
            break;
        case ROCPROFILER_RANGE_REPLAY_PASS: g_passes.fetch_add(1); break;
        case ROCPROFILER_RANGE_REPLAY_CLOSE:
            fprintf(stderr, "[decline] close status=%s\n", status_name(data->status));
            g_closes.fetch_add(1);
            g_status.store(data->status);
            break;
        default: break;
    }
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    RR_CHECK(rocprofiler_create_context(&g_ctx));
    RR_CHECK(rocprofiler_configure_callback_tracing_service(
        g_ctx, ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY, nullptr, 0, range_replay_cb, nullptr));
    RR_CHECK(rocprofiler_start_context(g_ctx));
    return 0;
}

void
tool_fini(void*)
{
    fprintf(stderr,
            "[decline] configs=%d passes=%d closes=%d status=%s\n",
            g_configs.load(),
            g_passes.load(),
            g_closes.load(),
            status_name(static_cast<rocprofiler_range_replay_status_t>(g_status.load())));

    // The second queue is seen while the range is recording, so the decline is known before any
    // pass would have run: no PASS callback is raised at all.
    const bool ok = g_configs.load() == 1 && g_passes.load() == 0 && g_closes.load() == 1 &&
                    g_status.load() == ROCPROFILER_RANGE_REPLAY_STATUS_MULTI_QUEUE;

    if(!ok)
    {
        fprintf(stderr, "[decline] FAIL\n");
        std::abort();
    }
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "range-replay-decline";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
