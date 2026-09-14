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
constexpr uint64_t kPasses = 4;

rocprofiler_context_id_t g_replay_ctx{0};
std::atomic<int>         g_replayed{0};
std::atomic<int>         g_passes_seen{0};

uint64_t replay_pass_count(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
{
    return kPasses;
}

void
kernel_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY) return;
    auto* p = static_cast<rocprofiler_callback_tracing_kernel_replay_data_t*>(record.payload);
    if(p->dispatch_info.workgroup_size.x != kReplayBlockX) return;

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        p->replay_pass_count = replay_pass_count;
        g_replayed.fetch_add(1);
        return;
    }

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_PASS &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        fprintf(stderr,
                "[basic] pass %lu / %lu dispatch_id=%lu\n",
                static_cast<unsigned long>(p->current_pass),
                static_cast<unsigned long>(p->total_passes),
                static_cast<unsigned long>(p->dispatch_info.dispatch_id));
        g_passes_seen.fetch_add(1);
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
                                                       nullptr));
    KR_CHECK(rocprofiler_start_context(g_replay_ctx));
    return 0;
}

void
tool_fini(void*)
{
    fprintf(stderr,
            "[basic] replayed=%d passes_seen=%d expected_passes=%lu\n",
            g_replayed.load(),
            g_passes_seen.load(),
            static_cast<unsigned long>(kPasses));
    if(g_replayed.load() != 1 || g_passes_seen.load() != static_cast<int>(kPasses)) std::abort();
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "kernel-replay-basic";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
