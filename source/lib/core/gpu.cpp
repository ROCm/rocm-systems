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

#include "agent.hpp"
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

#include <string>

#include "core/agent_manager.hpp"

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

// Ensures initialization happens only once
std::once_flag amdsmi_once;

// Tracks whether AMD SMI is initialized
bool&
_amdsmi_is_initialized()
{
    static bool initialized = false;
    return initialized;
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
            _amdsmi_is_initialized() = true;  // Mark as initialized
        } catch(std::exception& _e)
        {
            ROCPROFSYS_BASIC_VERBOSE(1, "Exception thrown initializing amd-smi: %s\n",
                                     _e.what());
            _amdsmi_is_initialized() = false;  // Mark as not initialized
            return false;
        }
        return true;
    }();

    return _amdsmi_init;
}
#endif  // ROCPROFSYS_USE_ROCM > 0

size_t
query_rocm_agents()
{
    size_t _dev_cnt = 0;
#if ROCPROFSYS_USE_ROCM > 0
    auto iterator = []([[maybe_unused]] rocprofiler_agent_version_t version,
                       const void** agents, size_t num_agents,
                       [[maybe_unused]] void* user_data) -> rocprofiler_status_t {
        auto& _agent_manager = agent_manager::get_instance();
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* _agent    = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
            auto        cur_agent = agent{
                (_agent->type == ROCPROFILER_AGENT_TYPE_GPU ? agent_type::GPU
                                                                   : agent_type::CPU),
                _agent->device_id,
                _agent->node_id,
                _agent->logical_node_id,
                _agent->logical_node_type_id,
                std::string(_agent->name),
                std::string(_agent->vendor_name),
                std::string(_agent->product_name),
                std::string(_agent->model_name),
            };
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
        ROCPROFSYS_BASIC_VERBOSE(
            1, "Exception thrown getting the rocm agents: %s. _dev_cnt=%ld\n", _e.what(),
            _dev_cnt);
    }
    _dev_cnt = agent_manager::get_instance().get_gpu_agents_count();
#endif
    return _dev_cnt;
}
}  // namespace

int
device_count()
{
#if ROCPROFSYS_USE_ROCM > 0
    static int _num_devices = query_rocm_agents();
    return _num_devices;
#else
    return 0;
#endif
}

bool
initialize_amdsmi()
{
#if ROCPROFSYS_USE_ROCM > 0
    // Ensure initialization happens only once
    std::call_once(amdsmi_once, amdsmi_init);
    return _amdsmi_is_initialized();
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
        ROCPROFSYS_BASIC_VERBOSE(1, "Exception thrown getting the rocm agents: %s.\n",
                                 _e.what());
    }

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
std::vector<bool>                    processors::gpu_category_mi300      = {};

void
get_processor_handles()
{
    uint32_t socket_count;
    uint32_t processor_count;
    processors::processors_list.clear();

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
            amdsmi_asic_info_t   asic_info;
            bool                 vcn_supported = false, jpeg_supported = false;
            bool                 v_busy_supported = false, j_busy_supported = false;
            bool                 gpu_cat_mi300 = false;
            // AMD SMI will not report VCN_activity and JPEG_activity, if VCN_busy or
            // JPEG_busy fields are available.
            if(amdsmi_get_gpu_metrics_info(processor, &gpu_metrics) ==
               AMDSMI_STATUS_SUCCESS)
            {
                // Helper lambda to check if any value in the array is valid
                auto has_valid = [](const auto& arr) {
                    return std::any_of(std::begin(arr), std::end(arr),
                                       [](auto val) { return val != UINT16_MAX; });
                };
                vcn_supported  = has_valid(gpu_metrics.vcn_activity);
                jpeg_supported = has_valid(gpu_metrics.jpeg_activity);
                // Check if VCN and JPEG busy metrics are available
                for(const auto& xcp : gpu_metrics.xcp_stats)
                {
                    if(!v_busy_supported && has_valid(xcp.vcn_busy))
                        v_busy_supported = true;
                    if(!j_busy_supported && has_valid(xcp.jpeg_busy))
                        j_busy_supported = true;
                    if(v_busy_supported && j_busy_supported) break;
                }
            }
            if(amdsmi_get_gpu_asic_info(processor, &asic_info) == AMDSMI_STATUS_SUCCESS)
            {
                uint64_t gfx_version = asic_info.target_graphics_version;

                if(gfx_version >= mi300_gfx_ver && gfx_version < navi10_gfx_ver)
                {
                    gpu_cat_mi300 = true;
                }
            }
            processors::vcn_activity_supported.push_back(vcn_supported);
            processors::jpeg_activity_supported.push_back(jpeg_supported);
            processors::vcn_busy_supported.push_back(v_busy_supported);
            processors::jpeg_busy_supported.push_back(j_busy_supported);
            processors::gpu_category_mi300.push_back(gpu_cat_mi300);
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

bool
is_gpu_category_mi300(uint32_t dev_id)
{
    if(dev_id >= processors::gpu_category_mi300.size()) return false;
    return processors::gpu_category_mi300[dev_id];
}

#endif

}  // namespace gpu
}  // namespace rocprofsys
