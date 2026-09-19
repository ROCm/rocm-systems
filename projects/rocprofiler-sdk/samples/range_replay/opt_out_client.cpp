// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

// A tool that subscribes to range replay but does not want this range re-executed. Leaving
// pass_count_cb NULL is the per-range opt-out: the range is observed, no pass runs, and CLOSE says
// NO_PASS_COUNT rather than reporting a failure the tool would have to interpret.

#include "client.hpp"

#include <rocprofiler-sdk/registration.h>

#include <atomic>
#include <cstdint>

namespace
{
rocprofiler_context_id_t g_ctx{0};

std::atomic<int> g_configs{0};
std::atomic<int> g_passes{0};
std::atomic<int> g_closes{0};
std::atomic<int> g_status{ROCPROFILER_RANGE_REPLAY_STATUS_LAST};

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
            // pass_count_cb deliberately left NULL.
            g_configs.fetch_add(1);
            break;
        case ROCPROFILER_RANGE_REPLAY_PASS: g_passes.fetch_add(1); break;
        case ROCPROFILER_RANGE_REPLAY_CLOSE:
            fprintf(stderr, "[opt-out] close status=%s\n", status_name(data->status));
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
            "[opt-out] configs=%d passes=%d closes=%d status=%s\n",
            g_configs.load(),
            g_passes.load(),
            g_closes.load(),
            status_name(static_cast<rocprofiler_range_replay_status_t>(g_status.load())));

    // The range is still opened, tracked and closed -- only the re-execution is skipped, so the
    // tool that opted out still learns the range happened.
    const bool ok = g_configs.load() == 1 && g_passes.load() == 0 && g_closes.load() == 1 &&
                    g_status.load() == ROCPROFILER_RANGE_REPLAY_STATUS_NO_PASS_COUNT;

    if(!ok)
    {
        fprintf(stderr, "[opt-out] FAIL\n");
        std::abort();
    }
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "range-replay-opt-out";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
