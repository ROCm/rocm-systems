// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "counter_info.hpp"
#include "lib/common/logging.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace tool
{
struct spm_descriptor_info_t
{
    size_t            num_buffers{};
    std::vector<char> desc_data{};
};

struct spm_dispatch_record_t
{
    rocprofiler_agent_id_t    agent{};
    rocprofiler_dispatch_id_t dispatch{};
};

struct tool_spm_counter_record_t
{
    using container_type = std::vector<tool_counter_value_t>;

    rocprofiler_dispatch_id_t   dispatch_id = {};
    serialized_counter_record_t record      = {};

    template <typename ArchiveT>
    void save(ArchiveT& ar) const
    {
        // should be removed when moving to buffered tracing
        auto tmp = read();

        ar(cereal::make_nvp("dispatch_id", dispatch_id));
        ar(cereal::make_nvp("records", tmp));
    }

    container_type read() const;
    void           write(const container_type& data);
};
}  // namespace tool
}  // namespace rocprofiler
