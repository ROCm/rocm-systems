// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "client.hpp"

#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/registration.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace
{
constexpr uint64_t kMaxPasses     = 4;
constexpr uint64_t kStopAfterPass = 1;

// Per-dispatch tool state. A tool reaches this through user_data.ptr instead of keeping a side
// table keyed on dispatch id.
struct replay_user_data
{
    uint64_t          max_passes      = kMaxPasses;
    uint64_t          stop_after_pass = kStopAfterPass;
    std::atomic<int>  replayed{0};
    std::atomic<int>  passes_seen{0};
    std::atomic<bool> continue_passes{true};
};

rocprofiler_context_id_t g_replay_ctx{0};
replay_user_data         g_replay_data{};

// Distinct address parked in user_data.ptr during a PASS to show how far that write travels.
// A real tool has no reason to do this; the sample does it to pin the scoping rules.
replay_user_data g_pass_scratch{};

#define KR_REQUIRE(cond, msg)                                                                      \
    do                                                                                             \
    {                                                                                              \
        if(!(cond))                                                                                \
        {                                                                                          \
            fprintf(stderr, "[basic-user-data] FAILED: %s (%s)\n", msg, #cond);                    \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

replay_user_data*
get_replay_data(rocprofiler_user_data_t user_data)
{
    auto* data = static_cast<replay_user_data*>(user_data.ptr);
    KR_REQUIRE(data != nullptr, "user_data.ptr was not threaded through");
    return data;
}

uint64_t
replay_pass_count(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t user_data)
{
    // Called right after CONFIG PHASE_ENTER, so it sees the value the tool just stored.
    KR_REQUIRE(user_data.ptr == &g_replay_data, "replay_pass_count did not get the CONFIG value");
    return get_replay_data(user_data)->max_passes;
}

int
replay_continue(rocprofiler_kernel_dispatch_info_t,
                uint64_t,
                uint64_t,
                rocprofiler_user_data_t user_data)
{
    // Runs after PASS PHASE_EXIT, yet still receives the CONFIG value: the PASS-phase write to
    // user_data below is not propagated here. State that must influence the continue decision
    // therefore lives behind the pointer, not in the union itself.
    KR_REQUIRE(user_data.ptr == &g_replay_data,
               "replay_continue should observe the CONFIG user_data, not a PASS-phase write");
    return get_replay_data(user_data)->continue_passes.load() ? 1 : 0;
}

void
kernel_replay_cb(rocprofiler_callback_tracing_record_t record,
                 rocprofiler_user_data_t*              user_data,
                 void*                                 callback_args)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY) return;

    auto* payload = static_cast<rocprofiler_callback_tracing_kernel_replay_data_t*>(record.payload);
    if(payload->dispatch_info.workgroup_size.x != kReplayBlockX) return;

    auto* callback_data = static_cast<replay_user_data*>(callback_args);
    KR_REQUIRE(callback_data == &g_replay_data, "unexpected callback_args");

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            // The SDK snapshots this union after CONFIG PHASE_ENTER and hands the same value to
            // replay_pass_count, replay_continue, and every PASS callback for this dispatch.
            user_data->ptr = callback_data;

            callback_data->continue_passes.store(true);
            callback_data->replayed.fetch_add(1);
            payload->replay_pass_count = replay_pass_count;
            payload->replay_continue   = replay_continue;
        }
        else
        {
            // A PASS-phase write to user_data never reaches CONFIG PHASE_EXIT.
            KR_REQUIRE(user_data->ptr == callback_data, "CONFIG EXIT lost the CONFIG user_data");
            KR_REQUIRE(payload->replay_pass_count == replay_pass_count,
                       "CONFIG EXIT should still expose the config callbacks");
            KR_REQUIRE(payload->replay_continue == replay_continue,
                       "CONFIG EXIT should still expose the config callbacks");
        }
        return;
    }

    if(record.operation != ROCPROFILER_KERNEL_REPLAY_PASS) return;

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        // Every pass starts from the CONFIG value, so the previous pass's write is gone.
        KR_REQUIRE(user_data->ptr == callback_data,
                   "PASS ENTER should start from the CONFIG value");

        // Config-only fields read as null during a PASS; the pass-scoped context toggles are
        // live only for the duration of this callback.
        KR_REQUIRE(payload->replay_pass_count == nullptr,
                   "config fields must read as null during a PASS");
        KR_REQUIRE(payload->replay_continue == nullptr,
                   "config fields must read as null during a PASS");
        KR_REQUIRE(
            payload->replay_start_context != nullptr && payload->replay_stop_context != nullptr,
            "PASS ENTER should expose the localized context toggles");

        fprintf(stderr,
                "[basic-user-data] pass %lu / %lu user_data.ptr=%p\n",
                static_cast<unsigned long>(payload->current_pass),
                static_cast<unsigned long>(payload->total_passes),
                user_data->ptr);
        callback_data->passes_seen.fetch_add(1);

        // Scoped to this pass: visible at PASS EXIT below, then discarded.
        user_data->ptr = &g_pass_scratch;
    }
    else
    {
        KR_REQUIRE(user_data->ptr == &g_pass_scratch,
                   "a PASS ENTER write to user_data should reach PASS EXIT of the same pass");
        KR_REQUIRE(
            payload->replay_start_context == nullptr && payload->replay_stop_context == nullptr,
            "context toggles are only valid during PASS ENTER");

        if(payload->current_pass >= callback_data->stop_after_pass)
        {
            // replay_continue reads this through the pointer it was given at CONFIG time, which
            // is why the flag lives in the struct rather than in user_data.value.
            callback_data->continue_passes.store(false);
        }
    }
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    KR_CHECK(rocprofiler_create_context(&g_replay_ctx));
    KR_CHECK(
        rocprofiler_configure_callback_tracing_service(g_replay_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                                       nullptr,
                                                       0,
                                                       kernel_replay_cb,
                                                       &g_replay_data));
    KR_CHECK(rocprofiler_start_context(g_replay_ctx));
    return 0;
}

void
tool_fini(void*)
{
    const int expected_passes = static_cast<int>(kStopAfterPass + 1);
    fprintf(stderr,
            "[basic-user-data] replayed=%d passes_seen=%d expected_passes=%d max_passes=%lu\n",
            g_replay_data.replayed.load(),
            g_replay_data.passes_seen.load(),
            expected_passes,
            static_cast<unsigned long>(g_replay_data.max_passes));
    KR_REQUIRE(g_replay_data.replayed.load() == 1, "expected exactly one replayed dispatch");
    KR_REQUIRE(g_replay_data.passes_seen.load() == expected_passes,
               "early exit did not stop the loop where expected");
    KR_REQUIRE(g_pass_scratch.passes_seen.load() == 0, "the scratch state should never be written");
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "kernel-replay-basic-user-data";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
