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

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocprofiler
{
namespace counters
{
// Base metrics (w/o instance information) defined in gfx_metrics/derived.xml
class Metric
{
public:
    Metric() = default;
    Metric(const std::string&,  // Get rid of this...
           std::string name,
           std::string block,
           std::string event,
           std::string dsc,
           std::string expr,
           std::string constant,
           uint64_t    id);

    const std::string& name() const { return name_; }
    const std::string& block() const { return block_; }
    const std::string& event() const { return event_; }
    const std::string& description() const { return description_; }
    const std::string& expression() const { return expression_; }
    const std::string& constant() const { return constant_; }
    uint64_t           id() const { return id_; }
    uint32_t           flags() const { return flags_; }
    bool               empty() const { return empty_; }

    void setflags(uint32_t flags) { this->flags_ = flags; }

    friend bool operator<(Metric const& lhs, Metric const& rhs);
    friend bool operator==(Metric const& lhs, Metric const& rhs);

private:
    std::string name_        = {};
    std::string block_       = {};
    std::string event_       = {};
    std::string description_ = {};
    std::string expression_  = {};
    std::string constant_    = {};
    int64_t     id_          = -1;
    bool        empty_       = false;
    uint32_t    flags_       = 0;
};

struct CustomCounterDefinition
{
    std::string data   = {};
    bool        append = {false};
    bool        loaded = {false};
};

using MetricMap   = std::unordered_map<std::string, std::vector<Metric>>;
using MetricIdMap = std::unordered_map<uint64_t, Metric>;
using ArchToId    = std::unordered_map<std::string, std::unordered_set<uint64_t>>;
using ArchMetric  = std::pair<std::string, Metric>;

struct counter_metrics_t
{
    const MetricMap   arch_to_metric;
    const MetricIdMap id_to_metric;
    const ArchToId    arch_to_id;
};

std::shared_ptr<const counter_metrics_t>
loadMetrics(bool reload = false, std::optional<ArchMetric> add_metric = std::nullopt);

/**
 * Get the metrics that apply to a specific agent. Supplied parameter
 * is the GFXIP of the agent.
 */
std::vector<Metric>
getMetricsForAgent(const std::string&);

/**
 * Get the metric event ids for perfcounters options in thread trace
 * applicable only for GFX9 agents and SQ block counters
 */
std::unordered_map<uint64_t, int>
getPerfCountersIdMap();

/**
 * Checks if a metric is valid for a given agent
 **/
bool
checkValidMetric(const std::string& agent, const Metric& metric);

/**
 * Set a custom counter definition
 */
rocprofiler_status_t
setCustomCounterDefinition(const CustomCounterDefinition& def);
}  // namespace counters
}  // namespace rocprofiler

namespace fmt
{
// fmt::format support for metric
template <>
struct formatter<rocprofiler::counters::Metric>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(rocprofiler::counters::Metric const& metric, Ctx& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "Metric: {} [Block: {}, Event: {}, Expression: {}, Description: {}, id: {}]",
            metric.name(),
            metric.block(),
            metric.event(),
            metric.expression().empty() ? "<None>" : metric.expression(),
            metric.description(),
            metric.id());
    }
};

// fmt::format support for MetricMap
template <>
struct formatter<rocprofiler::counters::MetricMap>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(rocprofiler::counters::MetricMap const& map, Ctx& ctx) const
    {
        std::string out;
        for(const auto& [gfxName, counters] : map)
        {
            out += fmt::format("Counters for {}\n\t{}\n", gfxName, fmt::join(counters, "\n\t"));
        }
        return fmt::format_to(ctx.out(), "{}", out);
    }
};
}  // namespace fmt
