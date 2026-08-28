// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "client.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/kernel_replay.h>
#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/pc_sampling.h>
#include <rocprofiler-sdk/registration.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>

namespace
{
enum class service_kind
{
    counters,
    pc_sampling,
    att,
    spm
};

constexpr uint64_t kPasses = 5;

rocprofiler_context_id_t g_replay_ctx{0};
rocprofiler_context_id_t g_counters_ctx{0};
rocprofiler_context_id_t g_pcs_ctx{0};
rocprofiler_context_id_t g_att_ctx{0};
rocprofiler_context_id_t g_spm_ctx{0};
rocprofiler_buffer_id_t  g_pcs_buffer{0};
rocprofiler_kernel_id_t  g_target_kernel = UINT64_MAX;

std::atomic<int> g_counter_records{0};
std::atomic<int> g_pcs_samples{0};
std::atomic<int> g_att_records{0};
std::atomic<int> g_spm_records{0};

bool g_services_first = true;

const char*
service_name(service_kind kind)
{
    switch(kind)
    {
        case service_kind::counters: return "counters";
        case service_kind::pc_sampling: return "pc-sampling";
        case service_kind::att: return "ATT";
        case service_kind::spm: return "SPM";
    }
    return "unknown";
}

std::array<service_kind, kPasses>
pass_sequence()
{
    if(g_services_first)
        return {service_kind::pc_sampling,
                service_kind::att,
                service_kind::spm,
                service_kind::counters,
                service_kind::counters};

    return {service_kind::counters,
            service_kind::counters,
            service_kind::pc_sampling,
            service_kind::att,
            service_kind::spm};
}

uint64_t pass_count_cb(rocprofiler_kernel_dispatch_info_t, rocprofiler_user_data_t)
{
    return kPasses;
}

void
set_context_for_pass(rocprofiler_callback_tracing_kernel_replay_data_t* payload,
                     rocprofiler_context_id_t                           context,
                     bool                                               enabled)
{
    if(context.handle == 0) return;
    auto callback = enabled ? payload->replay_local_enable_context_cb
                            : payload->replay_local_disable_context_cb;
    KR_CHECK(callback(context));
}

void
kernel_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY) return;
    auto* payload = static_cast<rocprofiler_callback_tracing_kernel_replay_data_t*>(record.payload);
    if(payload->dispatch_info.workgroup_size.x != kReplayBlockX) return;

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        payload->pass_count_cb = pass_count_cb;
        g_target_kernel        = payload->dispatch_info.kernel_id;
        return;
    }

    if(record.operation != ROCPROFILER_KERNEL_REPLAY_PASS ||
       record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER)
        return;

    const auto sequence = pass_sequence();
    const auto selected = sequence.at(payload->current_pass);

    // These services are intentionally exclusive. ATT+SPM both inject AQL instrumentation;
    // SPM+PC sampling share SQ/performance-monitor resources; counters+PC sampling have the
    // MI2xx/MI3xx clock-gating conflict. Give each non-counter service its own pass.
    set_context_for_pass(payload, g_counters_ctx, selected == service_kind::counters);
    set_context_for_pass(payload, g_pcs_ctx, selected == service_kind::pc_sampling);
    set_context_for_pass(payload, g_att_ctx, selected == service_kind::att);
    set_context_for_pass(payload, g_spm_ctx, selected == service_kind::spm);

    fprintf(stderr,
            "[service-sequence] pass=%lu service=%s\n",
            static_cast<unsigned long>(payload->current_pass),
            service_name(selected));
}

void
counter_dispatch_cb(rocprofiler_dispatch_counting_service_data_t data,
                    rocprofiler_counter_config_id_t*             config,
                    rocprofiler_user_data_t*,
                    void*)
{
    if(data.dispatch_info.kernel_id != g_target_kernel)
    {
        *config = rocprofiler_counter_config_id_t{.handle = 0};
        return;
    }
    *config = sq_waves_config(data.dispatch_info.agent_id);
}

void
counter_record_cb(rocprofiler_dispatch_counting_service_data_t data,
                  rocprofiler_counter_record_t*,
                  size_t,
                  rocprofiler_user_data_t,
                  void*)
{
    if(data.dispatch_info.kernel_id == g_target_kernel) g_counter_records.fetch_add(1);
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
        if(headers[i] && headers[i]->category == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
            g_pcs_samples.fetch_add(1);
    }
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
        [](rocprofiler_agent_id_t,
           rocprofiler_counter_id_t* counters,
           size_t                    count,
           void*                     userdata) {
            auto* output = static_cast<std::vector<rocprofiler_counter_id_t>*>(userdata);
            output->insert(output->end(), counters, counters + count);
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
        fprintf(stderr, "SPM unavailable\n");
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

bool
configure_pc_sampling()
{
    KR_CHECK(rocprofiler_create_context(&g_pcs_ctx));
    KR_CHECK(rocprofiler_create_buffer(g_pcs_ctx,
                                       8192,
                                       2048,
                                       ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                       pcs_buffer_cb,
                                       nullptr,
                                       &g_pcs_buffer));
    rocprofiler_callback_thread_t callback_thread{};
    KR_CHECK(rocprofiler_create_callback_thread(&callback_thread));
    KR_CHECK(rocprofiler_assign_callback_thread(g_pcs_buffer, callback_thread));

    bool configured = false;
    for(auto agent : gpu_agents())
    {
        std::vector<rocprofiler_pc_sampling_configuration_t> configurations{};
        auto status = rocprofiler_query_pc_sampling_agent_configurations(
            agent,
            [](const rocprofiler_pc_sampling_configuration_t* values,
               size_t                                         count,
               void*                                          userdata) {
                auto* output =
                    static_cast<std::vector<rocprofiler_pc_sampling_configuration_t>*>(userdata);
                output->insert(output->end(), values, values + count);
                return ROCPROFILER_STATUS_SUCCESS;
            },
            &configurations);
        if(status != ROCPROFILER_STATUS_SUCCESS || configurations.empty()) continue;
        const auto& config = configurations.front();
        status             = rocprofiler_configure_pc_sampling_service(
            g_pcs_ctx, agent, config.method, config.unit, config.min_interval, g_pcs_buffer, 0);
        configured |= status == ROCPROFILER_STATUS_SUCCESS;
    }

    if(configured) KR_CHECK(rocprofiler_start_context(g_pcs_ctx));
    return configured;
}

bool
configure_att()
{
    KR_CHECK(rocprofiler_create_context(&g_att_ctx));
    auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{
        {ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, 0x1}};
    bool configured = false;
    for(auto agent : gpu_agents())
    {
        configured |= rocprofiler_configure_dispatch_thread_trace_service(g_att_ctx,
                                                                          agent,
                                                                          parameters.data(),
                                                                          parameters.size(),
                                                                          att_dispatch_cb,
                                                                          att_shader_cb,
                                                                          nullptr) ==
                      ROCPROFILER_STATUS_SUCCESS;
    }
    if(configured) KR_CHECK(rocprofiler_start_context(g_att_ctx));
    return configured;
}

bool
configure_spm()
{
    KR_CHECK(rocprofiler_create_context(&g_spm_ctx));
    auto status = rocprofiler_spm_configure_callback_dispatch_service(
        g_spm_ctx, spm_dispatch_cb, nullptr, spm_record_cb, nullptr);
    if(status != ROCPROFILER_STATUS_SUCCESS) return false;
    KR_CHECK(rocprofiler_start_context(g_spm_ctx));
    return true;
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    const char* order = std::getenv("KR_SERVICE_ORDER");
    g_services_first  = !order || std::strcmp(order, "services-last") != 0;

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

    if(!configure_pc_sampling())
    {
        fprintf(stderr, "PC sampling unavailable\n");
        return -1;
    }
    if(!configure_att())
    {
        fprintf(stderr, "ATT unavailable\n");
        return -1;
    }
    if(!configure_spm())
    {
        fprintf(stderr, "SPM unavailable\n");
        return -1;
    }

    KR_CHECK(rocprofiler_start_context(g_replay_ctx));
    return 0;
}

void
tool_fini(void*)
{
    if(g_pcs_buffer.handle != 0) rocprofiler_flush_buffer(g_pcs_buffer);
    fprintf(stderr,
            "[service-sequence] order=%s counters=%d pcs=%d att=%d spm=%d\n",
            g_services_first ? "services-first" : "services-last",
            g_counter_records.load(),
            g_pcs_samples.load(),
            g_att_records.load(),
            g_spm_records.load());

    // PC sampling is stochastic, so zero samples is possible even when the pass ran.
    if(g_counter_records.load() != 2 || g_att_records.load() == 0 || g_spm_records.load() == 0)
        std::abort();
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name           = "kernel-replay-service-sequence";
    static auto config = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &config;
}
