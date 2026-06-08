// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "agent.hpp"
#include "agent_info.hpp"
#include <cstdint>
#define ROCPROFILER_SDK_CEREAL_NAMESPACE_BEGIN                                           \
    namespace tim                                                                        \
    {                                                                                    \
    namespace cereal                                                                     \
    {
#define ROCPROFILER_SDK_CEREAL_NAMESPACE_END                                             \
    }                                                                                    \
    }  // namespace ::tim::cereal

#include "common/defines.h"
#include "gpu.hpp"

#include <timemory/manager.hpp>

#include <string>

#include "core/agent_manager.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/cxx/serialization.hpp>
#include <rocprofiler-sdk/fwd.h>

#include "logger/debug.hpp"

namespace rocprofsys
{
namespace gpu
{
namespace
{
size_t
query_rocm_agents()
{
    size_t _dev_cnt = 0;
    auto   iterator = []([[maybe_unused]] rocprofiler_agent_version_t version,
                       const void** agents, size_t num_agents,
                       [[maybe_unused]] void* user_data) -> rocprofiler_status_t {
        auto& _agent_manager = get_agent_manager_instance();
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* _agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
            agent       cur_agent;
            cur_agent.type =
                (_agent->type == ROCPROFILER_AGENT_TYPE_GPU ? agent_type::GPU
                                                              : agent_type::CPU);
            cur_agent.handle               = _agent->id.handle;
            cur_agent.device_id            = _agent->device_id;
            cur_agent.node_id              = _agent->node_id;
            cur_agent.logical_node_id      = _agent->logical_node_id;
            cur_agent.logical_node_type_id = _agent->logical_node_type_id;
            cur_agent.name                 = std::string(_agent->name);
            cur_agent.model_name           = std::string(_agent->model_name);
            cur_agent.vendor_name          = std::string(_agent->vendor_name);
            cur_agent.product_name         = std::string(_agent->product_name);

            cur_agent.agent_info = agent_info::to_json_string(*_agent);

            _agent_manager.insert_agent(cur_agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    try
    {
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, iterator,
                                           sizeof(rocprofiler_agent_v0_t), nullptr);
    } catch(std::exception& _e)
    {
        LOG_ERROR("Exception thrown getting the rocm agents: {}. _dev_cnt={}", _e.what(),
                  _dev_cnt);
    }
    _dev_cnt = get_agent_manager_instance().get_gpu_agents_count();
    return _dev_cnt;
}
}  // namespace

int
device_count()
{
    static int _num_devices = query_rocm_agents();
    return _num_devices;
}

template <typename ArchiveT>
void
add_device_metadata(ArchiveT& ar)
{
    namespace cereal = tim::cereal;
    using cereal::make_nvp;

    using agent_vec_t = std::vector<rocprofiler_agent_v0_t>;

    auto iterator_cb = []([[maybe_unused]] rocprofiler_agent_version_t version,
                          const void** agents, size_t num_agents,
                          [[maybe_unused]] void* user_data) -> rocprofiler_status_t {
        auto* agents_vec = static_cast<agent_vec_t*>(user_data);
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* _agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
            if(_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            {
                agents_vec->push_back(*_agent);
            }
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    auto _agents_vec = agent_vec_t{};
    try
    {
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, iterator_cb,
                                           sizeof(rocprofiler_agent_v0_t), &_agents_vec);
    } catch(std::exception& _e)
    {
        LOG_ERROR("Exception thrown getting the rocm agents: {}", _e.what());
    }

    ar(make_nvp("rocm_agents", _agents_vec));
}

void
add_device_metadata()
{
    if(device_count() == 0) return;

    ::tim::manager::add_metadata([](auto& ar) {
        try
        {
            add_device_metadata(ar);
        } catch(std::runtime_error& _e)
        {
            LOG_ERROR("Exception thrown adding device metadata: {}", _e.what());
        }
    });
}

}  // namespace gpu
}  // namespace rocprofsys
