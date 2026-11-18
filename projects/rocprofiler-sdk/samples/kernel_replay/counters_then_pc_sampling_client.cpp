// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "client.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/pc_sampling.h>
#include <rocprofiler-sdk/registration.h>

#include <atomic>
#include <cstdint>
#include <vector>

namespace
{
constexpr uint64_t kPasses  = 4;
constexpr uint64_t kPcsPass = kPasses - 1;

rocprofiler_context_id_t g_replay_ctx{0};
rocprofiler_context_id_t g_counters_ctx{0};
rocprofiler_context_id_t g_pcs_ctx{0};
rocprofiler_buffer_id_t  g_pcs_buffer{0};
std::atomic<int>         g_counter_records{0};
std::atomic<int>         g_pcs_samples{0};
bool                     g_pcs_available = false;

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
        return;
    }

    if(record.operation != ROCPROFILER_KERNEL_REPLAY_PASS ||
       record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER)
        return;

    // Illustrates the intended per-pass toggle pattern, but PC sampling ignores localized
    // overrides today, so this sample must not run under ctest while counters are also enabled.
    // See kernel_replay_callback_api.md (service combination limits).
    if(p->current_pass == kPcsPass)
    {
        if(g_counters_ctx.handle != 0 && p->replay_stop_context)
            KR_CHECK(p->replay_stop_context(g_counters_ctx));
        if(g_pcs_available && p->replay_start_context) KR_CHECK(p->replay_start_context(g_pcs_ctx));
    }
    else if(g_pcs_available && p->replay_stop_context)
    {
        KR_CHECK(p->replay_stop_context(g_pcs_ctx));
    }
}

void
counter_dispatch_cb(rocprofiler_dispatch_counting_service_data_t d,
                    rocprofiler_counter_config_id_t*             config,
                    rocprofiler_user_data_t*,
                    void*)
{
    *config = sq_waves_config(d.dispatch_info.agent_id);
}

void
counter_record_cb(rocprofiler_dispatch_counting_service_data_t d,
                  rocprofiler_counter_record_t*,
                  size_t,
                  rocprofiler_user_data_t,
                  void*)
{
    if(d.dispatch_info.workgroup_size.x == kReplayBlockX) g_counter_records.fetch_add(1);
}

void
pcs_buffer_cb(rocprofiler_context_id_t,
              rocprofiler_buffer_id_t,
              rocprofiler_record_header_t** headers,
              size_t                        num_headers,
              void*,
              uint64_t)
{
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* h = headers[i];
        if(h && h->category == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING) g_pcs_samples.fetch_add(1);
    }
}

bool
configure_pcs()
{
    KR_CHECK(rocprofiler_create_context(&g_pcs_ctx));
    KR_CHECK(rocprofiler_create_buffer(g_pcs_ctx,
                                       8192,
                                       2048,
                                       ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                       pcs_buffer_cb,
                                       nullptr,
                                       &g_pcs_buffer));
    rocprofiler_callback_thread_t thread{};
    KR_CHECK(rocprofiler_create_callback_thread(&thread));
    KR_CHECK(rocprofiler_assign_callback_thread(g_pcs_buffer, thread));

    bool any = false;
    for(auto id : gpu_agents())
    {
        std::vector<rocprofiler_pc_sampling_configuration_t> configs{};
        auto qst = rocprofiler_query_pc_sampling_agent_configurations(
            id,
            [](const rocprofiler_pc_sampling_configuration_t* cfgs, size_t n, void* ud) {
                auto* v = static_cast<std::vector<rocprofiler_pc_sampling_configuration_t>*>(ud);
                v->insert(v->end(), cfgs, cfgs + n);
                return ROCPROFILER_STATUS_SUCCESS;
            },
            &configs);
        if(qst != ROCPROFILER_STATUS_SUCCESS || configs.empty()) continue;
        const auto& cfg = configs.front();
        auto        st  = rocprofiler_configure_pc_sampling_service(
            g_pcs_ctx, id, cfg.method, cfg.unit, cfg.min_interval, g_pcs_buffer, 0);
        if(st == ROCPROFILER_STATUS_SUCCESS) any = true;
    }
    // Started here so the agent sessions and buffer stay live for the whole run; which passes
    // actually sample is decided by the per-pass overrides in kernel_replay_cb.
    if(any) KR_CHECK(rocprofiler_start_context(g_pcs_ctx));
    return any;
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

    g_pcs_available = configure_pcs();
    if(!g_pcs_available) fprintf(stderr, "PC sampling unavailable\n");

    KR_CHECK(rocprofiler_start_context(g_replay_ctx));
    return 0;
}

void
tool_fini(void*)
{
    if(g_pcs_buffer.handle != 0) rocprofiler_flush_buffer(g_pcs_buffer);
    fprintf(stderr,
            "[counters-then-pcs] counter_records=%d expected=%lu pcs_samples=%d\n",
            g_counter_records.load(),
            static_cast<unsigned long>(kPcsPass),
            g_pcs_samples.load());
    if(g_counter_records.load() != static_cast<int>(kPcsPass)) std::abort();
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "kernel-replay-counters-then-pc-sampling";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
