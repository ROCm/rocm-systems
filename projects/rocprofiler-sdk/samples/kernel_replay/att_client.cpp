// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "client.hpp"

#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/registration.h>

#include <atomic>
#include <cstdint>
#include <vector>

namespace
{
constexpr uint64_t kPasses  = 2;
constexpr uint64_t kAttPass = kPasses - 1;

rocprofiler_context_id_t g_replay_ctx{0};
rocprofiler_context_id_t g_counters_ctx{0};
rocprofiler_context_id_t g_att_ctx{0};
rocprofiler_kernel_id_t  g_target_kernel = UINT64_MAX;

std::atomic<int> g_counter_records{0};
std::atomic<int> g_att_records{0};

uint64_t pass_count_cb(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
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
        p->pass_count_cb = pass_count_cb;
        g_target_kernel  = p->dispatch_info.kernel_id;
        return;
    }

    if(record.operation != ROCPROFILER_KERNEL_REPLAY_PASS ||
       record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER)
        return;

    const bool counters = p->current_pass != kAttPass;
    KR_CHECK((counters ? p->replay_local_enable_context_cb
                       : p->replay_local_disable_context_cb)(g_counters_ctx));
    KR_CHECK((counters ? p->replay_local_disable_context_cb
                       : p->replay_local_enable_context_cb)(g_att_ctx));
}

void
counter_dispatch_cb(rocprofiler_dispatch_counting_service_data_t d,
                    rocprofiler_counter_config_id_t*             config,
                    rocprofiler_user_data_t*,
                    void*)
{
    if(d.dispatch_info.kernel_id != g_target_kernel)
    {
        *config = rocprofiler_counter_config_id_t{.handle = 0};
        return;
    }
    *config = sq_waves_config(d.dispatch_info.agent_id);
}

void
counter_record_cb(rocprofiler_dispatch_counting_service_data_t d,
                  rocprofiler_counter_record_t*,
                  size_t,
                  rocprofiler_user_data_t,
                  void*)
{
    if(d.dispatch_info.kernel_id == g_target_kernel) g_counter_records.fetch_add(1);
}

rocprofiler_thread_trace_control_flags_t
att_dispatch_cb(rocprofiler_agent_id_t,
                rocprofiler_queue_id_t,
                rocprofiler_async_correlation_id_t,
                rocprofiler_kernel_id_t kernel_id,
                rocprofiler_dispatch_id_t,
                void*,
                rocprofiler_user_data_t*)
{
    return (kernel_id == g_target_kernel) ? ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP
                                          : ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
}

void att_shader_cb(rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t)
{
    g_att_records.fetch_add(1);
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

    KR_CHECK(rocprofiler_create_context(&g_counters_ctx));
    KR_CHECK(rocprofiler_configure_callback_dispatch_counting_service(
        g_counters_ctx, counter_dispatch_cb, nullptr, counter_record_cb, nullptr));
    KR_CHECK(rocprofiler_start_context(g_counters_ctx));

    KR_CHECK(rocprofiler_create_context(&g_att_ctx));
    bool any_att = false;
    for(auto agent : gpu_agents())
    {
        auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{
            {ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, 0x1}};
        any_att |= rocprofiler_configure_dispatch_thread_trace_service(g_att_ctx,
                                                                       agent,
                                                                       parameters.data(),
                                                                       parameters.size(),
                                                                       att_dispatch_cb,
                                                                       att_shader_cb,
                                                                       nullptr) ==
                   ROCPROFILER_STATUS_SUCCESS;
    }
    if(!any_att)
    {
        fprintf(stderr, "ATT unavailable\n");
        return -1;
    }
    KR_CHECK(rocprofiler_start_context(g_att_ctx));
    KR_CHECK(rocprofiler_start_context(g_replay_ctx));
    return 0;
}

void
tool_fini(void*)
{
    fprintf(stderr,
            "[att] counter_records=%d att_records=%d\n",
            g_counter_records.load(),
            g_att_records.load());
    if(g_counter_records.load() != 1 || g_att_records.load() == 0) std::abort();
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "kernel-replay-att";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
