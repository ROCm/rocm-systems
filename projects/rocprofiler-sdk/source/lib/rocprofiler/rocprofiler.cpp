// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler/rocprofiler.h>

#include <algorithm>
#include <vector>

namespace
{
template <typename... Tp>
auto
consume_args(Tp&&...)
{}
}  // namespace

extern "C" {
rocprofiler_status_t
rocprofiler_query_available_agents(rocprofiler_available_agents_cb_t callback,
                                   size_t                            agent_size,
                                   void*                             user_data)
{
    using pc_sampling_config_vec_t = std::vector<rocprofiler_pc_sampling_configuration_t>;

    static const auto _default_pc_config =
        rocprofiler_pc_sampling_configuration_t{ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP,
                                                ROCPROFILER_PC_SAMPLING_UNIT_TIME,
                                                1UL,
                                                1000000000UL,
                                                0};
    static const auto _dummy_pc_configs = pc_sampling_config_vec_t{_default_pc_config};

    static auto _default_cpu_agent = rocprofiler_agent_t{rocprofiler_agent_id_t{0},
                                                         ROCPROFILER_AGENT_TYPE_CPU,
                                                         "cpu",
                                                         rocprofiler_pc_sampling_config_array_t{}};
    static auto _default_gpu_agent = rocprofiler_agent_t{rocprofiler_agent_id_t{1},
                                                         ROCPROFILER_AGENT_TYPE_GPU,
                                                         "gpu",
                                                         rocprofiler_pc_sampling_config_array_t{}};

    // get the agents
    auto _agents = std::vector<rocprofiler_agent_t*>{&_default_cpu_agent, &_default_gpu_agent};
    auto _pc_sampling_config = std::vector<pc_sampling_config_vec_t>{};

    for(auto* itr : _agents)
    {
        auto& _data = _pc_sampling_config.emplace_back();
        if(itr->type == ROCPROFILER_AGENT_TYPE_GPU) _data = {_default_pc_config};
        itr->pc_sampling_configs =
            rocprofiler_pc_sampling_config_array_t{_data.data(), _data.size()};
    }

    assert(agent_size <= sizeof(rocprofiler_agent_t) &&
           "rocprofiler_agent_t used by caller is ABI-incompatible with rocprofiler_agent_t in "
           "rocprofiler");
    return callback(_agents.data(), _agents.size(), user_data);
}

rocprofiler_status_t
rocprofiler_create_context(rocprofiler_context_id_t* context_id)
{
    consume_args(context_id);
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t
rocprofiler_start_context(rocprofiler_context_id_t context_id)
{
    consume_args(context_id);
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t
rocprofiler_stop_context(rocprofiler_context_id_t context_id)
{
    consume_args(context_id);
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t
rocprofiler_flush_buffer(rocprofiler_buffer_id_t buffer_id)
{
    consume_args(buffer_id);
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t
rocprofiler_destroy_buffer(rocprofiler_buffer_id_t buffer_id)
{
    consume_args(buffer_id);
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t
rocprofiler_create_buffer(rocprofiler_context_id_t      context,
                          size_t                        size,
                          size_t                        watermark,
                          rocprofiler_buffer_policy_t   action,
                          rocprofiler_buffer_callback_t callback,
                          void*                         callback_data,
                          rocprofiler_buffer_id_t*      buffer_id)
{
    consume_args(context, size, watermark, action, callback, callback_data, buffer_id);
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t
rocprofiler_configure_pc_sampling_service(rocprofiler_context_id_t         context_id,
                                          rocprofiler_agent_t              agent,
                                          rocprofiler_pc_sampling_method_t method,
                                          rocprofiler_pc_sampling_unit_t   unit,
                                          uint64_t                         interval,
                                          rocprofiler_buffer_id_t          buffer_id)
{
    consume_args(context_id, agent, method, unit, interval, buffer_id);
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t
rocprofiler_query_pc_sampling_agent_configurations(rocprofiler_agent_t                      agent,
                                                   rocprofiler_pc_sampling_configuration_t* config,
                                                   size_t* config_count)
{
    consume_args(agent, config, config_count);
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}
}
