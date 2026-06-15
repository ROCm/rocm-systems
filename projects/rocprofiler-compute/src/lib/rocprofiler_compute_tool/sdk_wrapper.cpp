// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "sdk_wrapper.h"

#include <iostream>
#include <sstream>
#include <string>

using namespace rocprofiler_compute_tool;

#define ROCPROFILER_CALL(result, msg)                                                                  \
    {                                                                                                  \
        rocprofiler_status_t CHECKSTATUS = result;                                                     \
        if (CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                                 \
        {                                                                                              \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                       \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg                \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg << std::endl; \
            std::stringstream errmsg{};                                                                \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("      \
                   << status_msg << ")";                                                               \
            throw std::runtime_error(errmsg.str());                                                    \
        }                                                                                              \
    }

void SdkWrapperImpl::create_context(rocprofiler_context_id_t* context_id)
{
    ROCPROFILER_CALL(rocprofiler_create_context(context_id), "context creation");
}

void SdkWrapperImpl::configure_callback_dispatch_counting_service(
    rocprofiler_context_id_t                   context_id,
    rocprofiler_dispatch_counting_service_cb_t dispatch_callback,
    void*                                      dispatch_callback_args,
    rocprofiler_dispatch_counting_record_cb_t  record_callback,
    void*                                      record_callback_args)
{
    ROCPROFILER_CALL(rocprofiler_configure_callback_dispatch_counting_service(context_id,
                                                                              dispatch_callback,
                                                                              dispatch_callback_args,
                                                                              record_callback,
                                                                              record_callback_args),
                     "setup counting service");
}

void SdkWrapperImpl::configure_callback_tracing_service(rocprofiler_context_id_t context_id,
                                                        rocprofiler_callback_tracing_kind_t kind,
                                                        const rocprofiler_tracing_operation_t* operations,
                                                        size_t operations_count,
                                                        rocprofiler_callback_tracing_cb_t callback,
                                                        void* callback_args)
{
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(context_id,
                                                                    kind,
                                                                    operations,
                                                                    operations_count,
                                                                    callback,
                                                                    callback_args),
                     "setup code object tracing service");
}

void SdkWrapperImpl::start_context(rocprofiler_context_id_t context_id)
{
    ROCPROFILER_CALL(rocprofiler_start_context(context_id), "start context");
}

void SdkWrapperImpl::iterate_agent_supported_counters(rocprofiler_agent_id_t              agent_id,
                                                      rocprofiler_available_counters_cb_t cb,
                                                      void*                               user_data)
{
    ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(agent_id, cb, user_data),
                     "iterate agent supported counters");
}

void SdkWrapperImpl::query_counter_info(rocprofiler_counter_id_t              counter_id,
                                        rocprofiler_counter_info_version_id_t version,
                                        void*                                 info)
{
    ROCPROFILER_CALL(rocprofiler_query_counter_info(counter_id, version, info), "query counter info");
}

void SdkWrapperImpl::create_counter_config(rocprofiler_agent_id_t           agent_id,
                                           rocprofiler_counter_id_t*        counters_list,
                                           size_t                           counters_count,
                                           rocprofiler_counter_config_id_t* config_id)
{
    ROCPROFILER_CALL(rocprofiler_create_counter_config(agent_id, counters_list, counters_count, config_id),
                     "create counter config");
}

void SdkWrapperImpl::query_record_counter_id(rocprofiler_counter_instance_id_t id,
                                             rocprofiler_counter_id_t*         counter_id)
{
    ROCPROFILER_CALL(rocprofiler_query_record_counter_id(id, counter_id), "query record counter id");
}

void SdkWrapperImpl::at_intercept_table_registration_hsa(rocprofiler_intercept_library_cb_t callback,
                                                         void* user_data)
{
    ROCPROFILER_CALL(rocprofiler_at_intercept_table_registration(callback, ROCPROFILER_HSA_TABLE, user_data),
                     "register HSA intercept table callback");
}

void SdkWrapperImpl::query_available_gpu_agents(std::vector<rocprofiler_agent_id_t>& out_gpu_agents)
{
    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(
            ROCPROFILER_AGENT_INFO_VERSION_0,
            [](rocprofiler_agent_version_t, const void** agents, size_t num_agents, void* user_data)
            {
                auto* out = static_cast<std::vector<rocprofiler_agent_id_t>*>(user_data);
                for (size_t i = 0; i < num_agents; ++i)
                {
                    const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
                    if (agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                        out->push_back(agent->id);
                }
                return ROCPROFILER_STATUS_SUCCESS;
            },
            sizeof(rocprofiler_agent_v0_t),
            static_cast<void*>(&out_gpu_agents)),
        "query available agents");
}

void SdkWrapperImpl::query_pc_sampling_configs(rocprofiler_agent_id_t agent_id,
                                               rocprofiler_available_pc_sampling_configurations_cb_t cb,
                                               void* user_data)
{
    ROCPROFILER_CALL(rocprofiler_query_pc_sampling_agent_configurations(agent_id, cb, user_data),
                     "query pc sampling agent configurations");
}

void SdkWrapperImpl::create_buffer(rocprofiler_context_id_t        context_id,
                                   size_t                          size,
                                   size_t                          watermark,
                                   rocprofiler_buffer_policy_t     policy,
                                   rocprofiler_buffer_tracing_cb_t callback,
                                   void*                           callback_data,
                                   rocprofiler_buffer_id_t*        buffer_id)
{
    ROCPROFILER_CALL(
        rocprofiler_create_buffer(context_id, size, watermark, policy, callback, callback_data, buffer_id),
        "create buffer");
}

rocprofiler_status_t SdkWrapperImpl::configure_pc_sampling_service(rocprofiler_context_id_t context_id,
                                                                   rocprofiler_agent_id_t agent_id,
                                                                   rocprofiler_pc_sampling_method_t method,
                                                                   rocprofiler_pc_sampling_unit_t unit,
                                                                   uint64_t interval,
                                                                   rocprofiler_buffer_id_t buffer_id,
                                                                   int flags)
{
    return rocprofiler_configure_pc_sampling_service(context_id, agent_id, method, unit, interval, buffer_id, flags);
}

void SdkWrapperImpl::flush_buffer(rocprofiler_buffer_id_t buffer_id)
{
    ROCPROFILER_CALL(rocprofiler_flush_buffer(buffer_id), "flush buffer");
}

void SdkWrapperImpl::configure_buffer_tracing_service(rocprofiler_context_id_t          context_id,
                                                      rocprofiler_buffer_tracing_kind_t kind,
                                                      rocprofiler_buffer_id_t           buffer_id)
{
    ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(context_id, kind, nullptr, 0, buffer_id),
                     "configure buffer tracing service");
}

void SdkWrapperImpl::query_agent_records(std::vector<agent_record_t>& out_agents)
{
    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(
            ROCPROFILER_AGENT_INFO_VERSION_0,
            [](rocprofiler_agent_version_t, const void** agents, size_t num_agents, void* user_data)
            {
                auto* out = static_cast<std::vector<agent_record_t>*>(user_data);
                for (size_t i = 0; i < num_agents; ++i)
                {
                    const auto*    agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
                    agent_record_t rec{};
                    rec.size            = agent->size;
                    rec.id_handle       = agent->id.handle;
                    rec.type            = static_cast<uint32_t>(agent->type);
                    rec.node_id         = agent->node_id;
                    rec.logical_node_id = agent->logical_node_id;
                    rec.cu_count        = agent->cu_count;
                    rec.gpu_id          = agent->gpu_id;
                    rec.wave_front_size = agent->wave_front_size;
                    rec.simd_count      = agent->simd_count;
                    out->push_back(rec);
                }
                return ROCPROFILER_STATUS_SUCCESS;
            },
            sizeof(rocprofiler_agent_v0_t),
            static_cast<void*>(&out_agents)),
        "query available agents");
}
