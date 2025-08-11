// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/pc_sampling.h>
#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>
#include <rocprofiler-sdk/cxx/serialization.hpp>

#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"

#include <unordered_map>

using pc_sample_config_vec_t = std::vector<rocprofiler_pc_sampling_configuration_t>;
using agent_pc_sample_config_info_map_t =
    std::unordered_map<rocprofiler_agent_id_t, pc_sample_config_vec_t>;

namespace rocprofiler
{
namespace tool
{
struct inst_t
{
    uint64_t code_object_id;
    uint64_t code_object_offset;

    bool operator==(const inst_t& inst) const
    {
        return this->code_object_id == inst.code_object_id &&
               this->code_object_offset == inst.code_object_offset;
    }

    bool operator<(const inst_t& b) const
    {
        if(this->code_object_id == b.code_object_id)
            return this->code_object_offset < b.code_object_offset;
        return this->code_object_id < b.code_object_id;
    };
};

// TODO:: Check if we can template this structure
struct rocprofiler_tool_pc_sampling_host_trap_record_t
{
    rocprofiler_pc_sampling_record_host_trap_v0_t pc_sample_record;
    int64_t                                       inst_index;

    rocprofiler_tool_pc_sampling_host_trap_record_t(
        rocprofiler_pc_sampling_record_host_trap_v0_t record,
        int64_t                                       index)
    : pc_sample_record(record)
    , inst_index(index)
    {}

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("record", pc_sample_record));
        ar(cereal::make_nvp("inst_index", inst_index));
    }
};

// TODO:: Check if we can template this structure
struct rocprofiler_tool_pc_sampling_stochastic_record_t
{
    rocprofiler_pc_sampling_record_stochastic_v0_t pc_sample_record;
    int64_t                                        inst_index;

    rocprofiler_tool_pc_sampling_stochastic_record_t(
        rocprofiler_pc_sampling_record_stochastic_v0_t record,
        int64_t                                        index)
    : pc_sample_record(record)
    , inst_index(index)
    {}

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        ar(cereal::make_nvp("record", pc_sample_record));
        ar(cereal::make_nvp("inst_index", inst_index));
    }
};

struct rocprofiler_tool_pc_sampling_stats
{
    uint64_t valid_samples   = 0;
    uint64_t invalid_samples = 0;
};

}  // namespace tool
}  // namespace rocprofiler
