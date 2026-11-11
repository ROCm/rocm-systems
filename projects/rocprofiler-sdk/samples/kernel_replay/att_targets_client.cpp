// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

// Traces the same dispatch once per SIMD of one WGP: pass i selects SIMD i through
// replay_set_thread_trace_parameters, so a single application run covers every SIMD with
// identical inputs.

#include "client.hpp"

#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/registration.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace
{
constexpr uint64_t kPasses = 4;  // one pass per SIMD of the traced WGP

rocprofiler_context_id_t g_replay_ctx{0};
rocprofiler_context_id_t g_att_ctx{0};
rocprofiler_kernel_id_t  g_target_kernel = UINT64_MAX;

std::atomic<int64_t>                  g_current_pass{-1};
std::array<std::atomic<int>, kPasses> g_att_records{};
std::atomic<int>                      g_select_failures{0};
std::unordered_map<uint64_t, bool>    g_simd_is_bitmask{};  // agent handle -> gfx9

uint64_t replay_pass_count(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
{
    return kPasses;
}

void
select_simd(rocprofiler_callback_tracing_kernel_replay_data_t* p)
{
    constexpr size_t needed = offsetof(rocprofiler_callback_tracing_kernel_replay_data_t,
                                       replay_set_thread_trace_parameters) +
                              sizeof(rocprofiler_kernel_replay_thread_trace_params_cb_t);
    if(p->size < needed || p->replay_set_thread_trace_parameters == nullptr)
    {
        fprintf(stderr, "[att-targets] this SDK cannot select a thread trace target per pass\n");
        g_select_failures.fetch_add(1);
        return;
    }

    // SIMD_SELECT is a bitmask of SIMDs on gfx9 and a SIMD index on later architectures.
    const auto simd = p->current_pass;
    const auto value =
        g_simd_is_bitmask[p->dispatch_info.agent_id.handle] ? (uint64_t{1} << simd) : simd;
    const auto params = std::vector<rocprofiler_thread_trace_parameter_t>{
        {ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {value}}};

    auto status = p->replay_set_thread_trace_parameters(g_att_ctx, params.data(), params.size());
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[att-targets] selecting SIMD %lu failed: %s\n",
                static_cast<unsigned long>(simd),
                rocprofiler_get_status_string(status));
        g_select_failures.fetch_add(1);
    }
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
        g_target_kernel      = p->dispatch_info.kernel_id;
        return;
    }

    if(record.operation != ROCPROFILER_KERNEL_REPLAY_PASS) return;
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        g_current_pass.store(-1);
        return;
    }

    g_current_pass.store(static_cast<int64_t>(p->current_pass));
    select_simd(p);
}

rocprofiler_thread_trace_control_flags_t
att_dispatch_cb(rocprofiler_agent_id_t,
                rocprofiler_queue_id_t,
                rocprofiler_async_correlation_id_t,
                rocprofiler_kernel_id_t kernel_id,
                rocprofiler_dispatch_id_t,
                void*,
                rocprofiler_user_data_t* userdata_shader)
{
    if(kernel_id != g_target_kernel) return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
    // Shader data may arrive on another thread, so tag the trace with the pass that produced it.
    userdata_shader->value = static_cast<uint64_t>(g_current_pass.load() + 1);
    return ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP;
}

void
att_shader_cb(rocprofiler_thread_trace_shader_data_t, rocprofiler_user_data_t userdata)
{
    if(userdata.value == 0 || userdata.value > kPasses) return;
    g_att_records[userdata.value - 1].fetch_add(1);
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

    KR_CHECK(rocprofiler_query_available_agents(
        ROCPROFILER_AGENT_INFO_VERSION_0,
        [](rocprofiler_agent_version_t, const void** _agents, size_t n, void*) {
            for(size_t i = 0; i < n; ++i)
            {
                auto* agent = static_cast<const rocprofiler_agent_v0_t*>(_agents[i]);
                if(agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                    g_simd_is_bitmask[agent->id.handle] = agent->gfx_target_version / 10000 == 9;
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        sizeof(rocprofiler_agent_v0_t),
        nullptr));

    // Every pass traces WGP 0 of shader engine 0; only the SIMD changes between passes.
    KR_CHECK(rocprofiler_create_context(&g_att_ctx));
    bool any_att = false;
    for(auto agent : gpu_agents())
    {
        auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{
            {ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, {0}},
            {ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {0x1}}};
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
    bool ok = g_select_failures.load() == 0;
    for(uint64_t pass = 0; pass < kPasses; ++pass)
    {
        const int records = g_att_records[pass].load();
        fprintf(stderr,
                "[att-targets] pass %lu (SIMD %lu): att_records=%d\n",
                static_cast<unsigned long>(pass),
                static_cast<unsigned long>(pass),
                records);
        ok = ok && records > 0;
    }
    if(!ok) std::abort();
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "kernel-replay-att-targets";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
