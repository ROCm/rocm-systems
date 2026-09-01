// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "client.hpp"

#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/registration.h>

#include <atomic>
#include <cstdint>

namespace
{
constexpr uint64_t kMaxPasses     = 4;
constexpr uint64_t kStopAfterPass = 1;

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

replay_user_data*
get_replay_data(rocprofiler_user_data_t user_data)
{
    auto* data = static_cast<replay_user_data*>(user_data.ptr);
    if(!data) std::abort();
    return data;
}

uint64_t
replay_pass_count(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t user_data)
{
    return get_replay_data(user_data)->max_passes;
}

int
replay_continue(rocprofiler_kernel_dispatch_info_t,
                uint64_t,
                uint64_t,
                rocprofiler_user_data_t user_data)
{
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
    if(!callback_data) std::abort();

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            // The SDK snapshots this union value after CONFIG PHASE_ENTER and passes the same
            // value to replay_pass_count, replay_continue, and every PASS callback.
            user_data->ptr = callback_data;

            callback_data->continue_passes.store(true);
            callback_data->replayed.fetch_add(1);
            payload->replay_pass_count = replay_pass_count;
            payload->replay_continue   = replay_continue;
        }
        else if(user_data->ptr != callback_data)
        {
            std::abort();
        }
        return;
    }

    if(record.operation != ROCPROFILER_KERNEL_REPLAY_PASS) return;
    if(user_data->ptr != callback_data) std::abort();

    auto* replay_data = get_replay_data(*user_data);
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        fprintf(stderr,
                "[basic-user-data] pass %lu / %lu user_data.ptr=%p\n",
                static_cast<unsigned long>(payload->current_pass),
                static_cast<unsigned long>(payload->total_passes),
                user_data->ptr);
        replay_data->passes_seen.fetch_add(1);
    }
    else if(payload->current_pass >= replay_data->stop_after_pass)
    {
        // replay_continue observes this mutation through the pointer stored in user_data.
        replay_data->continue_passes.store(false);
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
    if(g_replay_data.replayed.load() != 1 || g_replay_data.passes_seen.load() != expected_passes)
        std::abort();
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
