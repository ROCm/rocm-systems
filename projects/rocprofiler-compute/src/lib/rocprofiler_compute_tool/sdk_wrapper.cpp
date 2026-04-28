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