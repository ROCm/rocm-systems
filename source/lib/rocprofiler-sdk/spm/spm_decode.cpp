// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/spm/spm_decode.hpp"
#include <rocprofiler-sdk/experimental/spm/decode.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/aql/aql_profile_v2.h"
#include "lib/rocprofiler-sdk/spm/spm_dlsym.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_HSA(fn, message)                                                                     \
    {                                                                                              \
        auto _status = (fn);                                                                       \
        if(_status != HSA_STATUS_SUCCESS)                                                          \
        {                                                                                          \
            ROCP_ERROR << "HSA Err: " << _status << '\n';                                          \
            throw std::runtime_error(message);                                                     \
        }                                                                                          \
    }

namespace SPMDecode
{
struct values_vec_t
{
    std::vector<uint64_t> timestamps;
    std::vector<uint64_t> values;
};

struct instances_t
{
    std::vector<values_vec_t> shaders;
    bool                      is_global = false;
};

typedef std::vector<instances_t> counter_vec;

void
decode_cb(uint64_t timestamp, uint64_t value, uint64_t index, int shader_engine, void* userdata)
{
    auto& counters = *reinterpret_cast<counter_vec*>(userdata);

    // aqlprofile currently reports shadder_engine -1 as global counters
    if(shader_engine < 0)
    {
        counters.at(index).is_global = true;
        shader_engine                = 0;
    }

    if(counters.at(index).shaders.size() <= shader_engine)
        counters.at(index).shaders.resize(shader_engine + 1);

    auto& instance = counters.at(index).shaders.at(shader_engine);
    instance.timestamps.push_back(timestamp);
    instance.values.push_back(value);
    if(shader_engine < 0) counters.at(index).is_global = true;
}
}  // namespace SPMDecode

extern "C" {

rocprofiler_status_t
rocprofiler_spm_decode(rocprofiler_spm_descriptor_t      _desc,
                       rocprofiler_spm_buffer_id_t       buffer_id,
                       rocprofiler_spm_decode_callback_t decode_cb,
                       void*                             data,
                       size_t                            size,
                       void*                             userdata)
{
    static auto*& sym = rocprofiler::common::static_object<rocprofiler::SPM::Dlsym>::construct();
    if(!sym->valid()) return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;

    if(!size || !_desc.data || _desc.size < sizeof(rocprofiler::SPM::spm_desc_v0_t))
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto& desc_v0 = *static_cast<rocprofiler::SPM::spm_desc_v0_t*>(_desc.data);
    if(!desc_v0.valid()) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    SPMDecode::counter_vec       counters{};
    aqlprofile_spm_buffer_desc_t aql_desc{.data = desc_v0.aqlprofile_desc(),
                                          .size = desc_v0.aql_desc_size};

    {
        uint64_t count = 0;

        auto status = sym->spm_query_fn(aql_desc, AQLPROFILE_SPM_DECODE_QUERY_EVENT_COUNT, &count);
        if(status != HSA_STATUS_SUCCESS) return ROCPROFILER_STATUS_ERROR;
        if(count != desc_v0.num_events) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
        counters.resize(count);
    }

    // Prealloc SE_NUM for MI300.
    for(auto& v : counters)
        v.shaders.resize(4);

    auto status = sym->spm_decode_fn(aql_desc, SPMDecode::decode_cb, data, size, &counters);
    if(status != HSA_STATUS_SUCCESS) return ROCPROFILER_STATUS_ERROR;

    std::array<rocprofiler_spm_coord_t, 3> se_coords{};
    se_coords.at(0).name  = "XCC";
    se_coords.at(0).coord = buffer_id;
    se_coords.at(1).name  = "SE";
    se_coords.at(2).name  = "INSTANCE";

    std::array<rocprofiler_spm_coord_t, 2> global_coords{};
    global_coords.at(0).name  = "XCC";
    global_coords.at(0).coord = buffer_id;
    global_coords.at(1).name  = "INSTANCE";

    for(size_t i = 0; i < counters.size(); i++)
    {
        auto& event = desc_v0.events()[i];
        for(size_t se = 0; se < counters.at(i).shaders.size(); se++)
        {
            const auto& times  = counters.at(i).shaders.at(se).timestamps;
            const auto& values = counters.at(i).shaders.at(se).values;

            size_t size = std::min(times.size(), values.size());
            if(!size) continue;

            if(counters.at(i).is_global)
            {
                global_coords.at(1).coord = event.instance;
                decode_cb(event.id,
                          global_coords.data(),
                          global_coords.size(),
                          times.data(),
                          values.data(),
                          size,
                          userdata);
            }
            else
            {
                se_coords.at(1).coord = se;
                se_coords.at(2).coord = event.instance;
                decode_cb(event.id,
                          se_coords.data(),
                          se_coords.size(),
                          times.data(),
                          values.data(),
                          size,
                          userdata);
            }
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}
}
