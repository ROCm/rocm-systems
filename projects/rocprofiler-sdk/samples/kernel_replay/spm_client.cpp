// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "client.hpp"

#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/registration.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
constexpr uint64_t kPasses  = 2;
constexpr uint64_t kSpmPass = kPasses - 1;

rocprofiler_context_id_t g_replay_ctx{0};
rocprofiler_context_id_t g_counters_ctx{0};
rocprofiler_context_id_t g_spm_ctx{0};
rocprofiler_kernel_id_t  g_target_kernel = UINT64_MAX;

std::atomic<int> g_counter_records{0};
std::atomic<int> g_spm_records{0};

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

    const bool counters = p->current_pass != kSpmPass;
    KR_CHECK((counters ? p->replay_local_enable_context_cb
                       : p->replay_local_disable_context_cb)(g_counters_ctx));
    KR_CHECK((counters ? p->replay_local_disable_context_cb
                       : p->replay_local_enable_context_cb)(g_spm_ctx));
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

void
spm_dispatch_cb(const rocprofiler_spm_dispatch_counting_service_data_t* data,
                rocprofiler_counter_config_id_t*                        config,
                rocprofiler_user_data_t*,
                void*)
{
    if(!data || data->dispatch_info.kernel_id != g_target_kernel)
    {
        *config = rocprofiler_counter_config_id_t{.handle = 0};
        return;
    }

    static std::mutex                                                    mutex{};
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> cache{};
    const auto agent = data->dispatch_info.agent_id;
    {
        std::lock_guard<std::mutex> lock{mutex};
        if(auto it = cache.find(agent.handle); it != cache.end())
        {
            *config = it->second;
            return;
        }
    }

    std::vector<rocprofiler_counter_id_t> all{};
    KR_CHECK(rocprofiler_spm_iterate_agent_supported_counters(
        agent,
        [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* counters, size_t count, void* ud) {
            auto* out = static_cast<std::vector<rocprofiler_counter_id_t>*>(ud);
            out->insert(out->end(), counters, counters + count);
            return ROCPROFILER_STATUS_SUCCESS;
        },
        &all));
    std::vector<rocprofiler_counter_id_t> want{};
    for(auto counter : all)
    {
        rocprofiler_counter_info_v0_t info{};
        KR_CHECK(
            rocprofiler_query_counter_info(counter, ROCPROFILER_COUNTER_INFO_VERSION_0, &info));
        if(info.name && std::string{info.name} == "SQ_WAVES") want.push_back(counter);
    }
    if(want.empty() && !all.empty()) want.push_back(all.front());
    if(want.empty())
    {
        *config = rocprofiler_counter_config_id_t{.handle = 0};
        return;
    }

    rocprofiler_spm_parameters_t parameter{
        .size  = sizeof(rocprofiler_spm_parameters_t),
        .type  = ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES,
        .value = 1200};
    rocprofiler_spm_parameters_t*   parameters[] = {&parameter};
    rocprofiler_counter_config_id_t created{.handle = 0};
    KR_CHECK(rocprofiler_spm_create_counter_config(
        agent, want.data(), want.size(), parameters, 1, &created));
    {
        std::lock_guard<std::mutex> lock{mutex};
        cache.emplace(agent.handle, created);
    }
    *config = created;
}

void
spm_record_cb(const rocprofiler_spm_dispatch_counting_service_data_t* data,
              const rocprofiler_spm_counter_record_t**,
              size_t,
              rocprofiler_spm_record_flag_t flags,
              rocprofiler_user_data_t,
              void*)
{
    if(data && data->dispatch_info.kernel_id == g_target_kernel &&
       (flags & ROCPROFILER_SPM_RECORD_FLAG_DISPATCH_END) != 0)
        g_spm_records.fetch_add(1);
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

    KR_CHECK(rocprofiler_create_context(&g_spm_ctx));
    if(rocprofiler_spm_configure_callback_dispatch_service(
           g_spm_ctx, spm_dispatch_cb, nullptr, spm_record_cb, nullptr) !=
       ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr, "SPM unavailable\n");
        return -1;
    }
    KR_CHECK(rocprofiler_start_context(g_spm_ctx));
    KR_CHECK(rocprofiler_start_context(g_replay_ctx));
    return 0;
}

void
tool_fini(void*)
{
    fprintf(stderr,
            "[spm] counter_records=%d spm_records=%d\n",
            g_counter_records.load(),
            g_spm_records.load());
    if(g_counter_records.load() != 1 || g_spm_records.load() == 0) std::abort();
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "kernel-replay-spm";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
