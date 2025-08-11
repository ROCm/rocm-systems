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

#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"

namespace rocprofiler
{
namespace counters
{
class MetricDimension
{
public:
    MetricDimension(std::string_view                           name,
                    uint64_t                                   dim_size,
                    rocprofiler_profile_counter_instance_types type)
    : name_(name)
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

struct metric_dims
{
    const std::unordered_map<uint64_t, std::vector<MetricDimension>> id_to_dim;
};

// get all dimensions for an agent, block_name
std::vector<MetricDimension>
getBlockDimensions(std::string_view agent, const counters::Metric&);

std::shared_ptr<const metric_dims>
get_dimension_cache(bool reload = false);
}  // namespace counters
}  // namespace rocprofiler

namespace fmt
{
// fmt::format support for metric
template <>
struct formatter<rocprofiler::counters::MetricDimension>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(rocprofiler::counters::MetricDimension const& dims, Ctx& ctx) const
    {
        return fmt::format_to(ctx.out(), "[{}, {}]", dims.name(), dims.size());
    }
};
}  // namespace fmt
