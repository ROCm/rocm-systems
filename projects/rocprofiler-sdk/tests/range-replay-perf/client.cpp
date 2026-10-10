// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

// Tool half of the range replay perf regression tests: request RR_PERF_PASSES passes for every
// range the perf application opens, and refuse to let a range that was not actually replayed be
// reported as a timing sample.
//
// That refusal is the point of this file. Every decline path in the SDK is cheap -- the range is
// abandoned during recording and no snapshot, no pass loop and no restore ever run -- so a
// declined range is dramatically *faster* than a replayed one. A perf harness that only timed the
// application would read a newly-introduced decline as a large improvement and go green, which is
// precisely backwards. So the numbers are only trustworthy alongside a check that every range
// reached CLOSE with status REPLAYED and with the dispatch count the application actually issued.

#include "client.hpp"

#include <rocprofiler-sdk/registration.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace
{
uint64_t
pass_count_from_env()
{
    if(const char* env = std::getenv("RR_PERF_PASSES"))
    {
        const long v = std::strtol(env, nullptr, 10);
        // A pass count below 2 means the range is recorded and closed without any re-execution,
        // so it would time the application rather than the replay. Rejected here instead of
        // silently clamped: a harness that asked for 1 has a bug worth surfacing.
        if(v >= 2) return static_cast<uint64_t>(v);
        fprintf(stderr, "[rr-perf-client] RR_PERF_PASSES=%ld is below 2\n", v);
        std::abort();
    }
    return 3;
}

const uint64_t kPasses = pass_count_from_env();

rocprofiler_context_id_t g_ctx{0};

std::atomic<uint64_t> g_configs{0};
std::atomic<uint64_t> g_passes{0};
std::atomic<uint64_t> g_closes{0};
std::atomic<uint64_t> g_replayed{0};
std::atomic<uint64_t> g_dispatches{0};
std::atomic<uint64_t> g_divergence{0};

uint64_t pass_count(uint64_t, rocprofiler_user_data_t) { return kPasses; }

void
range_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) return;

    auto* data = static_cast<rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);
    if(data->range_id != kPerfRangeId) return;

    switch(record.operation)
    {
        case ROCPROFILER_RANGE_REPLAY_CONFIG:
            data->pass_count_cb = pass_count;
            g_configs.fetch_add(1);
            break;
        case ROCPROFILER_RANGE_REPLAY_PASS: g_passes.fetch_add(1); break;
        case ROCPROFILER_RANGE_REPLAY_CLOSE:
            g_closes.fetch_add(1);
            if(data->status == ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED) g_replayed.fetch_add(1);
            g_dispatches.fetch_add(data->dispatch_count);
            g_divergence.fetch_add(data->divergence_count);
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
    const auto closes   = g_closes.load();
    const auto replayed = g_replayed.load();
    const auto passes   = g_passes.load();

    // Parsed by the python drivers, which refuse to record a sample unless every range replayed.
    fprintf(stderr,
            "[rr-perf-client] passes_requested=%lu ranges=%lu replayed=%lu pass_callbacks=%lu "
            "dispatches=%lu divergence=%lu\n",
            static_cast<unsigned long>(kPasses),
            static_cast<unsigned long>(closes),
            static_cast<unsigned long>(replayed),
            static_cast<unsigned long>(passes),
            static_cast<unsigned long>(g_dispatches.load()),
            static_cast<unsigned long>(g_divergence.load()));

    if(closes == 0 || replayed != closes)
    {
        fprintf(stderr,
                "[rr-perf-client] FAIL %lu of %lu range(s) were not replayed; the timing below "
                "would measure the decline path, not the replay\n",
                static_cast<unsigned long>(closes - replayed),
                static_cast<unsigned long>(closes));
        std::abort();
    }

    // Each replayed range raises one PASS per re-execution, so the callback count is a direct
    // check that the pass loop ran the number of times the timing assumes it did.
    if(passes != closes * (kPasses - 1))
    {
        fprintf(stderr,
                "[rr-perf-client] FAIL %lu PASS callbacks across %lu range(s), expected %lu\n",
                static_cast<unsigned long>(passes),
                static_cast<unsigned long>(closes),
                static_cast<unsigned long>(closes * (kPasses - 1)));
        std::abort();
    }
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "range-replay-perf-client";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
