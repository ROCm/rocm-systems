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

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace tool
{
struct agent_info : rocprofiler_agent_v0_t
{
    using base_type = rocprofiler_agent_v0_t;

    agent_info(base_type _base)
    : base_type{_base}
    {}

    agent_info()                      = default;
    ~agent_info()                     = default;
    agent_info(const agent_info&)     = default;
    agent_info(agent_info&&) noexcept = default;
    agent_info& operator=(const agent_info&) = default;
    agent_info& operator=(agent_info&&) noexcept = default;

    int64_t gpu_index =
        (base_type::type == ROCPROFILER_AGENT_TYPE_GPU) ? base_type::logical_node_type_id : -1;
};

using agent_info_vec_t = std::vector<agent_info>;
using agent_info_map_t = std::unordered_map<rocprofiler_agent_id_t, agent_info>;
}  // namespace tool
}  // namespace rocprofiler

namespace cereal
{
template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::agent_info& data)
{
    cereal::save(ar, static_cast<const rocprofiler_agent_v0_t&>(data));
    ar(cereal::make_nvp("gpu_index", data.gpu_index));
}

template <typename ArchiveT>
void
load(ArchiveT& ar, ::rocprofiler::tool::agent_info& data)
{
    cereal::load(ar, static_cast<rocprofiler_agent_v0_t&>(data));
    ar(cereal::make_nvp("gpu_index", data.gpu_index));
}
}  // namespace cereal
