// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
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
#    include <rocm_smi/rocm_smi.h>
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
#    define ROCPROFSYS_ROCM_SMI_CALL(ERROR_CODE)                                         \
        ::rocprofsys::gpu::check_rsmi_error(ERROR_CODE, __FILE__, __LINE__)

void
check_rsmi_error(rsmi_status_t _code, const char* _file, int _line)
{
    if(_code == RSMI_STATUS_SUCCESS) return;
    const char* _msg = nullptr;
    auto        _err = rsmi_status_string(_code, &_msg);
    if(_err != RSMI_STATUS_SUCCESS)
        ROCPROFSYS_THROW("rsmi_status_string failed. No error message available. "
                         "Error code %i originated at %s:%i\n",
                         static_cast<int>(_code), _file, _line);
    ROCPROFSYS_THROW("[%s:%i] Error code %i :: %s", _file, _line, static_cast<int>(_code),
                     _msg);
}

bool
rsmi_init()
{
    auto _rsmi_init = []() {
        try
        {
            ROCPROFSYS_ROCM_SMI_CALL(::rsmi_init(0));
        } catch(std::exception& _e)
        {
            ROCPROFSYS_BASIC_VERBOSE(1, "Exception thrown initializing rocm-smi: %s\n",
                                     _e.what());
            return false;
        }
        return true;
    }();

    return _rsmi_init;
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
rocm_device_count()
{
#if ROCPROFSYS_USE_ROCM > 0
    static int _num_devices = query_rocm_gpu_agents();
    return _num_devices;
#else
    return 0;
#endif
}

int
rsmi_device_count()
{
#if ROCPROFSYS_USE_ROCM > 0
    if(!rsmi_init()) return 0;

    static auto _num_devices = []() {
        uint32_t _v = 0;
        try
        {
            ROCPROFSYS_ROCM_SMI_CALL(rsmi_num_monitor_devices(&_v));
        } catch(std::exception& _e)
        {
            ROCPROFSYS_BASIC_VERBOSE(
                1, "Exception thrown getting the rocm-smi devices: %s\n", _e.what());
        }
        return _v;
    }();

    return _num_devices;
#else
    return 0;
#endif
}

int
device_count()
{
#if ROCPROFSYS_USE_ROCM > 0
    return rocm_device_count();
#else
    return 0;
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
}  // namespace gpu
}  // namespace rocprofsys
