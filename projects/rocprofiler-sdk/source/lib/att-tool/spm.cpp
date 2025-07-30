// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/att-tool/spm.hpp"
#include "lib/att-tool/att_lib_wrapper.hpp"
#include "lib/att-tool/outputfile.hpp"

#include <rocprofiler-sdk/experimental/spm/decode.h>
#include "rocprofiler-sdk/cxx/operators.hpp"

#include <algorithm>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <vector>

#define SHADER_ENGINE_PER_XCC 4

namespace rocprofiler
{
namespace att_wrapper
{
struct spm_value_t
{
    uint64_t ts{};
    uint64_t val{};
};

typedef std::vector<spm_value_t> instance_t;
typedef std::vector<instance_t>  shader_engine_t;

struct spm_counter_vec_t
{
    std::vector<shader_engine_t> shaders;
    bool                         is_global;
};

typedef std::map<rocprofiler_counter_id_t, spm_counter_vec_t> spm_list_t;

void
spm_decode_callback(rocprofiler_counter_id_t id,
                    rocprofiler_spm_coord_t* dimensions,
                    uint64_t                 num_dimensions,
                    const uint64_t*          timestamps,
                    const uint64_t*          values,
                    uint64_t                 count,
                    void*                    userdata)
{
    if(count == 0) return;

    constexpr size_t SIZET_MAX   = std::numeric_limits<size_t>::max();
    size_t           se_or_xcc   = SIZET_MAX;
    size_t           instance_id = SIZET_MAX;
    size_t           xcc         = SIZET_MAX;

    for(uint64_t i = 0; i < num_dimensions; i++)
    {
        auto name = std::string_view(dimensions[i].name);

        if(name == "INSTANCE")
            instance_id = dimensions[i].coord;
        else if(name == "SE")
            se_or_xcc = dimensions[i].coord;
        else if(name == "XCC")
            xcc = dimensions[i].coord;
    }

    ROCP_FATAL_IF(instance_id == SIZET_MAX) << "Could not find INSTANCE dimension";
    ROCP_FATAL_IF(xcc == SIZET_MAX) << "Could not find XCC dimension";

    auto& counter = (*static_cast<spm_list_t*>(userdata))[id];

    if(se_or_xcc == SIZET_MAX)
    {
        counter.is_global = true;
        se_or_xcc         = xcc;
    }
    else
    {
        se_or_xcc += SHADER_ENGINE_PER_XCC * xcc;
    }

    if(counter.shaders.size() <= se_or_xcc) counter.shaders.resize(se_or_xcc + 1);

    auto& shader = counter.shaders.at(se_or_xcc);
    if(shader.size() <= instance_id) shader.resize(instance_id + 1);

    auto& instance = shader.at(instance_id);
    instance.reserve(instance.size() + count);

    // Remove repeated zero values for counter
    uint64_t last_value = 0;
    // Timestamps are used in deltas
    uint64_t last_ts = 0;
    if(!instance.empty()) last_ts = instance.back().ts;

    for(size_t i = 0; i < count; i++)
    {
        if(last_value != 0 || values[i] != 0 || last_ts == 0)
            instance.push_back({timestamps[i] - last_ts, values[i]});

        last_ts    = timestamps[i];
        last_value = values[i];
    }
};

void
SPMFile::addSpm(std::vector<char>& descriptor, std::vector<std::vector<char>>& spm_data)
{
    rocprofiler_spm_descriptor_t desc{};
    desc.data = descriptor.data();
    desc.size = descriptor.size();

    auto list = spm_list_t{};
    for(size_t xcc = 0; xcc < spm_data.size(); xcc++)
    {
        auto status = rocprofiler_spm_decode(desc,
                                             xcc,
                                             spm_decode_callback,
                                             spm_data.at(xcc).data(),
                                             spm_data.at(xcc).size(),
                                             &list);

        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            ROCP_ERROR << "Unable to decode SPM data!";
            return;
        }
    }

    for(auto& [id, name] : counters)
    {
        if(list.find(id) == list.end()) continue;

        nlohmann::json shader_map;
        for(size_t se = 0; se < list.at(id).shaders.size(); se++)
        {
            auto&          shader = list.at(id).shaders.at(se);
            nlohmann::json instance_map;

            for(size_t i = 0; i < shader.size(); i++)
            {
                auto& instance = shader.at(i);
                if(instance.empty()) continue;

                nlohmann::json instance_vec;
                for(auto& [ts, value] : instance)
                    instance_vec.push_back({ts, value});

                instance_map[std::to_string(i)] = instance_vec;
            }
            shader_map[std::to_string(se)] = instance_map;
        }
        json[name]["values"] = shader_map;

        if(list.at(id).is_global)
            json[name]["dimensions"] = {"XCC", "INSTANCE"};
        else
            json[name]["dimensions"] = {"SE", "INSTANCE"};
    }
}

SPMFile::~SPMFile() { OutputFile(filename) << json; }

}  // namespace att_wrapper
}  // namespace rocprofiler
