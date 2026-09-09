// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

// The smallest tool that replays a range: ask for four passes and check that the SDK delivered the
// three re-executions and closed the range as replayed.

#include "client.hpp"

#include <rocprofiler-sdk/registration.h>

#include <atomic>
#include <cstdint>

namespace
{
// Counting the application's own execution as pass 0, so the SDK re-executes the recording three
// times and raises three PASS callbacks.
constexpr uint64_t kPasses = 4;

rocprofiler_context_id_t g_ctx{0};

std::atomic<int>      g_configs{0};
std::atomic<int>      g_passes{0};
std::atomic<int>      g_closes{0};
std::atomic<uint64_t> g_last_pass{0};
std::atomic<uint64_t> g_dispatch_count{0};
std::atomic<int>      g_status{ROCPROFILER_RANGE_REPLAY_STATUS_LAST};
std::atomic<uint64_t> g_divergence{0};
std::atomic<bool>     g_agent_bound{false};

uint64_t pass_count(uint64_t, rocprofiler_user_data_t) { return kPasses; }

void
range_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY) return;

    auto* data = static_cast<rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);
    if(data->range_id != kRangeId) return;

    if(record.operation == ROCPROFILER_RANGE_REPLAY_CONFIG &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        data->pass_count_cb = pass_count;
        g_configs.fetch_add(1);
        return;
    }

    if(record.operation == ROCPROFILER_RANGE_REPLAY_PASS &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        fprintf(stderr,
                "[basic] pass %lu / %lu dispatches=%lu\n",
                static_cast<unsigned long>(data->current_pass),
                static_cast<unsigned long>(data->total_passes),
                static_cast<unsigned long>(data->dispatch_count));
        g_passes.fetch_add(1);
        g_last_pass.store(data->current_pass);
        g_dispatch_count.store(data->dispatch_count);
        if(data->agent_id.handle != 0) g_agent_bound.store(true);
        return;
    }

    if(record.operation == ROCPROFILER_RANGE_REPLAY_CLOSE &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        fprintf(stderr,
                "[basic] close status=%s dispatches=%lu divergence=%lu\n",
                status_name(data->status),
                static_cast<unsigned long>(data->dispatch_count),
                static_cast<unsigned long>(data->divergence_count));
        g_closes.fetch_add(1);
        g_status.store(data->status);
        g_divergence.store(data->divergence_count);
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
    const auto expected_passes = static_cast<int>(kPasses - 1);

    fprintf(stderr,
            "[basic] configs=%d passes=%d closes=%d status=%s\n",
            g_configs.load(),
            g_passes.load(),
            g_closes.load(),
            status_name(static_cast<rocprofiler_range_replay_status_t>(g_status.load())));

    bool ok = true;
    // One CONFIG and one CLOSE per range, no matter how many passes ran.
    ok = ok && g_configs.load() == 1;
    ok = ok && g_closes.load() == 1;
    // PASS is raised for the re-executions only: the application's own run is pass 0 and the SDK
    // observes it rather than driving it.
    ok = ok && g_passes.load() == expected_passes;
    ok = ok && g_last_pass.load() == kPasses - 1;
    ok = ok && g_status.load() == ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED;
    ok = ok && g_dispatch_count.load() == kRangeDispatches;
    ok = ok && g_agent_bound.load();
    // Verification is off by default, so nothing should be reported as divergent.
    ok = ok && g_divergence.load() == 0;

    if(!ok)
    {
        fprintf(stderr, "[basic] FAIL\n");
        std::abort();
    }
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "range-replay-basic";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
