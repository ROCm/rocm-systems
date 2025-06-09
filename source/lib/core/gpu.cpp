// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#define ROCPROFILER_SDK_CEREAL_NAMESPACE_BEGIN                                           \
    namespace tim                                                                        \
    {                                                                                    \
    namespace cereal                                                                     \
    {
#define ROCPROFILER_SDK_CEREAL_NAMESPACE_END                                             \
    }                                                                                    \
    }  // namespace ::tim::cereal

#include "common/defines.h"

#if !defined(ROCPROFSYS_USE_ROCM)
#    define ROCPROFSYS_USE_ROCM 0
#endif

#include "debug.hpp"
#include "defines.hpp"
#include "gpu.hpp"

#include <timemory/manager.hpp>

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#    include <rocprofiler-sdk/agent.h>
#    include <rocprofiler-sdk/cxx/serialization.hpp>
#    include <rocprofiler-sdk/fwd.h>
#endif

namespace rocprofsys
{
namespace gpu
{
namespace
{
#if ROCPROFSYS_USE_ROCM > 0
#    define ROCPROFSYS_AMD_SMI_CALL(ERROR_CODE)                                          \
        ::rocprofsys::gpu::check_amdsmi_error(ERROR_CODE, __FILE__, __LINE__)

void
check_amdsmi_error(amdsmi_status_t _code, const char* _file, int _line)
{
    if(_code == AMDSMI_STATUS_SUCCESS) return;
    const char* _msg = nullptr;
    auto        _err = amdsmi_status_code_to_string(_code, &_msg);
    if(_err != AMDSMI_STATUS_SUCCESS)
        ROCPROFSYS_THROW(
            "amdsmi_status_code_to_string failed. No error message available. "
            "Error code %i originated at %s:%i\n",
            static_cast<int>(_code), _file, _line);
    ROCPROFSYS_THROW("[%s:%i] Error code %i :: %s", _file, _line, static_cast<int>(_code),
                     _msg);
}

bool
amdsmi_init()
{
    auto _amdsmi_init = []() {
        try
        {
            // Currently, only AMDSMI_INIT_AMD_GPUS is supported
            ROCPROFSYS_AMD_SMI_CALL(::amdsmi_init(AMDSMI_INIT_AMD_GPUS));
            get_processor_handles();
        } catch(std::exception& _e)
        {
            ROCPROFSYS_BASIC_VERBOSE(1, "Exception thrown initializing amd-smi: %s\n",
                                     _e.what());
            return false;
        }
        return true;
    }();

    return _amdsmi_init;
}
#endif  // ROCPROFSYS_USE_ROCM > 0

int32_t
query_rocm_gpu_agents()
{
    int32_t _dev_cnt = 0;
#if ROCPROFSYS_USE_ROCM > 0
    auto iterator = [](rocprofiler_agent_version_t /*version*/, const void** agents,
                       size_t num_agents, void* user_data) -> rocprofiler_status_t {
        auto* _cnt = static_cast<int32_t*>(user_data);
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* _agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
            if(_agent && _agent->type == ROCPROFILER_AGENT_TYPE_GPU) *_cnt += 1;
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    try
    {
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, iterator,
                                           sizeof(rocprofiler_agent_v0_t), &_dev_cnt);
    } catch(std::exception& _e)
    {
        ROCPROFSYS_BASIC_VERBOSE(
            1, "Exception thrown getting the rocm agents: %s. _dev_cnt=%d\n", _e.what(),
            _dev_cnt);
    }
    // rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, iterator,
    //                                sizeof(rocprofiler_agent_v0_t), &_dev_cnt);
#endif
    return _dev_cnt;
}
}  // namespace

int
device_count()
{
#if ROCPROFSYS_USE_ROCM > 0
    static int _num_devices = query_rocm_gpu_agents();
    return _num_devices;
#else
    return 0;
#endif
}

bool
initialize_amdsmi()
{
#if ROCPROFSYS_USE_ROCM > 0
    return (amdsmi_init()) ? true : false;
#else
    return false;
#endif
}

template <typename ArchiveT>
void
add_device_metadata(ArchiveT& ar)
{
    namespace cereal = tim::cereal;
    using cereal::make_nvp;

#if ROCPROFSYS_USE_ROCM > 0
    using agent_vec_t = std::vector<rocprofiler_agent_v0_t>;

    auto _agents_vec = agent_vec_t{};
    auto iterator    = [](rocprofiler_agent_version_t /*version*/, const void** agents,
                       size_t num_agents, void* user_data) -> rocprofiler_status_t {
        auto* _agents_vec_v = static_cast<agent_vec_t*>(user_data);
        _agents_vec_v->reserve(num_agents);
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* _agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
            if(_agent) _agents_vec_v->emplace_back(*_agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };
    rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, iterator,
                                       sizeof(rocprofiler_agent_v0_t), &_agents_vec);

    ar(make_nvp("rocm_agents", _agents_vec));
#else
    (void) ar;
#endif
}

void
add_device_metadata()
{
    if(device_count() == 0) return;

    ROCPROFSYS_METADATA([](auto& ar) {
        try
        {
            add_device_metadata(ar);
        } catch(std::runtime_error& _e)
        {
            ROCPROFSYS_VERBOSE(2, "%s\n", _e.what());
        }
    });
}

#if ROCPROFSYS_USE_ROCM > 0
/*
 * Required amdsmi methods to get processors and handles
 */

uint32_t                             processors::total_processor_count   = 0;
std::vector<amdsmi_processor_handle> processors::processors_list         = {};
std::vector<bool>                    processors::vcn_activity_supported  = {};
std::vector<bool>                    processors::jpeg_activity_supported = {};
std::vector<bool>                    processors::vcn_busy_supported      = {};
std::vector<bool>                    processors::jpeg_busy_supported     = {};

void
get_processor_handles()
{
    uint32_t socket_count;
    uint32_t processor_count;

    // Passing nullptr will return us the number of sockets available for read in this
    // system
    auto ret = amdsmi_get_socket_handles(&socket_count, nullptr);
    if(ret != AMDSMI_STATUS_SUCCESS)
    {
        return;
    }
    std::vector<amdsmi_socket_handle> sockets(socket_count);
    ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
    for(auto& socket : sockets)
    {
        // Passing nullptr will return us the number of processors available for read for
        // this socket
        ret = amdsmi_get_processor_handles(socket, &processor_count, nullptr);
        if(ret != AMDSMI_STATUS_SUCCESS)
        {
            return;
        }
        std::vector<amdsmi_processor_handle> all_processors(processor_count);
        ret =
            amdsmi_get_processor_handles(socket, &processor_count, all_processors.data());
        if(ret != AMDSMI_STATUS_SUCCESS)
        {
            return;
        }

        for(auto& processor : all_processors)
        {
            processor_type_t processor_type = {};
            ret = amdsmi_get_processor_type(processor, &processor_type);
            if(processor_type != AMDSMI_PROCESSOR_TYPE_AMD_GPU)
            {
                ROCPROFSYS_THROW("Not AMD_GPU device type!");
                return;
            }
            processors::processors_list.push_back(processor);

            amdsmi_gpu_metrics_t gpu_metrics;
            bool                 vcn_supported    = false;
            bool                 jpeg_supported   = false;
            bool                 v_busy_supported = false;
            bool                 j_busy_supported = false;
            ret = amdsmi_get_gpu_metrics_info(processor, &gpu_metrics);
            if(ret == AMDSMI_STATUS_SUCCESS)
            {
                for(const auto& vcn_activity : gpu_metrics.vcn_activity)
                {
                    if(vcn_activity != UINT16_MAX)
                    {
                        vcn_supported = true;
                        break;
                    }
                }
                for(const auto& jpeg_activity : gpu_metrics.jpeg_activity)
                {
                    if(jpeg_activity != UINT16_MAX)
                    {
                        jpeg_supported = true;
                        break;
                    }
                }
                for(const auto& xcp : gpu_metrics.xcp_stats)
                {
                    if(!v_busy_supported)
                    {
                        v_busy_supported =
                            std::any_of(std::begin(xcp.vcn_busy), std::end(xcp.vcn_busy),
                                        [](uint16_t val) { return val != UINT16_MAX; });
                    }

                    if(!j_busy_supported)
                    {
                        j_busy_supported = std::any_of(
                            std::begin(xcp.jpeg_busy), std::end(xcp.jpeg_busy),
                            [](uint16_t val) { return val != UINT16_MAX; });
                    }

                    if(v_busy_supported && j_busy_supported) break;
                }
            }
            processors::vcn_activity_supported.push_back(vcn_supported);
            processors::jpeg_activity_supported.push_back(jpeg_supported);
            processors::vcn_busy_supported.push_back(v_busy_supported);
            processors::jpeg_busy_supported.push_back(j_busy_supported);
        }
    }
    processors::total_processor_count = processors::processors_list.size();
}

bool
is_vcn_activity_supported(uint32_t dev_id)
{
    if(dev_id >= processors::vcn_activity_supported.size()) return false;
    return processors::vcn_activity_supported[dev_id];
}

bool
is_jpeg_activity_supported(uint32_t dev_id)
{
    if(dev_id >= processors::jpeg_activity_supported.size()) return false;
    return processors::jpeg_activity_supported[dev_id];
}

bool
is_vcn_busy_supported(uint32_t dev_id)
{
    if(dev_id >= processors::vcn_busy_supported.size()) return false;
    return processors::vcn_busy_supported[dev_id];
}

bool
is_jpeg_busy_supported(uint32_t dev_id)
{
    if(dev_id >= processors::jpeg_busy_supported.size()) return false;
    return processors::jpeg_busy_supported[dev_id];
}

uint32_t
get_processor_count()
{
    return processors::total_processor_count;
}

amdsmi_processor_handle
get_handle_from_id(uint32_t dev_id)
{
    return processors::processors_list[dev_id];
}
#endif

}  // namespace gpu
}  // namespace rocprofsys
