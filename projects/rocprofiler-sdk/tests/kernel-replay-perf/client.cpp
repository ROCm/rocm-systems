// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "client.hpp"

#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/registration.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
uint64_t
pass_count_from_env()
{
    if(const char* env = std::getenv("KR_PERF_PASSES"))
    {
        const long v = std::strtol(env, nullptr, 10);
        if(v >= 1) return static_cast<uint64_t>(v);
    }
    return 5;
}

const uint64_t kPasses = pass_count_from_env();

rocprofiler_context_id_t g_replay_ctx{0};
rocprofiler_context_id_t g_counters_ctx{0};
thread_local uint64_t    tl_pass = 0;

const char*
counter_name_for_pass(uint64_t pass)
{
    static constexpr std::array<const char*, 8> names{"SQ_WAVES",
                                                      "GRBM_COUNT",
                                                      "GRBM_GUI_ACTIVE",
                                                      "SQ_INSTS_VALU",
                                                      "SQ_INSTS_SALU",
                                                      "SQ_INSTS_SMEM",
                                                      "SQ_INSTS_LDS",
                                                      "SQ_WAVES"};
    return names[pass % names.size()];
}

rocprofiler_counter_config_id_t
config_for_pass(rocprofiler_agent_id_t agent, uint64_t pass)
{
    static std::mutex                                                    mutex{};
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> cache{};
    const uint64_t key = (agent.handle << 8) | (pass % kPasses);
    {
        std::lock_guard<std::mutex> lock{mutex};
        if(auto it = cache.find(key); it != cache.end()) return it->second;
    }

    const char*                           want_name = counter_name_for_pass(pass);
    std::vector<rocprofiler_counter_id_t> all{};
    KR_CHECK(rocprofiler_iterate_agent_supported_counters(
        agent,
        [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* cs, size_t n, void* ud) {
            auto* v = static_cast<std::vector<rocprofiler_counter_id_t>*>(ud);
            v->insert(v->end(), cs, cs + n);
            return ROCPROFILER_STATUS_SUCCESS;
        },
        &all));

    std::vector<rocprofiler_counter_id_t> want{};
    for(auto id : all)
    {
        rocprofiler_counter_info_v0_t info{};
        KR_CHECK(rocprofiler_query_counter_info(id, ROCPROFILER_COUNTER_INFO_VERSION_0, &info));
        if(info.name && std::string{info.name} == want_name) want.push_back(id);
    }
    if(want.empty()) want.push_back(all.front());

    rocprofiler_counter_config_id_t cfg{.handle = 0};
    KR_CHECK(rocprofiler_create_counter_config(agent, want.data(), want.size(), &cfg));
    {
        std::lock_guard<std::mutex> lock{mutex};
        cache.emplace(key, cfg);
    }
    return cfg;
}

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
        return;
    }
    if(record.operation == ROCPROFILER_KERNEL_REPLAY_PASS &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        tl_pass = p->current_pass;
}

void
counter_dispatch_cb(rocprofiler_dispatch_counting_service_data_t d,
                    rocprofiler_counter_config_id_t*             config,
                    rocprofiler_user_data_t*,
                    void*)
{
    if(d.dispatch_info.workgroup_size.x != kReplayBlockX) return;
    *config = config_for_pass(d.dispatch_info.agent_id, tl_pass);
}

void
counter_record_cb(rocprofiler_dispatch_counting_service_data_t d,
                  rocprofiler_counter_record_t*,
                  size_t,
                  rocprofiler_user_data_t /*user_data*/,
                  void*)
{
    (void) d;
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    KR_CHECK(rocprofiler_create_context(&g_replay_ctx));
    KR_CHECK(rocprofiler_create_context(&g_counters_ctx));

    KR_CHECK(
        rocprofiler_configure_callback_tracing_service(g_replay_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                                       nullptr,
                                                       0,
                                                       kernel_replay_cb,
                                                       nullptr));

    KR_CHECK(rocprofiler_configure_callback_dispatch_counting_service(
        g_counters_ctx, counter_dispatch_cb, nullptr, counter_record_cb, nullptr));

    KR_CHECK(rocprofiler_start_context(g_replay_ctx));
    KR_CHECK(rocprofiler_start_context(g_counters_ctx));
    return 0;
}

void
tool_fini(void*)
{}

rocprofiler_tool_configure_result_t*
configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id)
{
    id->name        = "kernel-replay-perf-client";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    return configure(version, runtime_version, priority, id);
}
