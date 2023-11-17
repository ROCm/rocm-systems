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

#include <atomic>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "lib/rocprofiler/counters/id_decode.hpp"
#include "lib/rocprofiler/counters/metrics.hpp"

namespace rocprofiler
{
namespace counters
{
class MetricDimension
{
public:
    MetricDimension(std::string                                name,
                    uint64_t                                   dim_size,
                    rocprofiler_profile_counter_instance_types type)
    : name_(std::move(name))
    , size_(dim_size)
    , type_(type){};

    const std::string&                         name() const { return name_; }
    uint64_t                                   size() const { return size_; }
    rocprofiler_profile_counter_instance_types type() const { return type_; }
    bool                                       operator==(const MetricDimension& dim) const
    {
        return std::tie(name_, size_, type_) == std::tie(dim.name_, dim.size_, dim.type_);
    }

private:
    std::string                                name_;
    uint64_t                                   size_;
    rocprofiler_profile_counter_instance_types type_;
};

/*
{
    AgentBlockDimensionsMap = {
        "gfx906":{},
        "gfx908":{
            "TCC": [dim_1, dim_2 ... dim_n]
            "SQ": [dim_1, dim_2 ... dim_n]
            "TCP": [dim_1, dim_2 ... dim_n]
        }
    }
}
*/

// // map block_name -> MetricDimension
// using BlockDimensionMap = std::unordered_map<std::string, std::vector<MetricDimension>>;
// // map agent_name -> BlockDimensionMap
// using AgentBlockDimensionsMap = std::unordered_map<std::string, BlockDimensionMap&>;

// // map dimension_id -> MetricDimension
// using DimensionIdMap = std::unordered_map<uint64_t, MetricDimension>;

// // get the complete AgentBlockDimensionsMap
// const AgentBlockDimensionsMap&
// getAgentBlockDimensionsMap();

// // get specific dimensions for an agent, block_name
// const std::vector<MetricDimension>&
// getBlockDimension(const std::string&                         agent,
//                   std::string                                block_name,
//                   rocprofiler_profile_counter_instance_types dim);

// get all dimensions for an agent, block_name
std::vector<MetricDimension>
getBlockDimensions(std::string_view agent, const counters::Metric&);

// // get a specific dimension by id
// const MetricDimension&
// getDimensionById(uint64_t id);

}  // namespace counters
}  // namespace rocprofiler
