#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <hsa/hsa_ven_amd_aqlprofile.h>

#include "fmt/core.h"
#include "fmt/ranges.h"

namespace counters
{
// Base metrics (w/o instance information) defined in gfx_metrics/derrived.xml
class Metric
{
public:
    Metric(std::string name,
           std::string block,
           std::string event,
           std::string dsc,
           std::string expr)
    : name_(std::move(name))
    , block_(std::move(block))
    , event_(std::move(event))
    , description_(std::move(dsc))
    , expression_(std::move(expr))
    {}

    const std::string& name() const { return name_; }
    const std::string& block() const { return block_; }
    const std::string& event() const { return event_; }
    const std::string& description() const { return description_; }
    const std::string& expression() const { return expression_; }

private:
    std::string name_;
    std::string block_;
    std::string event_;
    std::string description_;
    std::string expression_;
};

using MetricMap = std::unordered_map<std::string, std::vector<Metric>>;

MetricMap
getBaseHardwareMetrics();

MetricMap
getDerivedHardwareMetrics();

}  // namespace counters

namespace fmt
{
// fmt::format support for metric
template <>
struct formatter<counters::Metric>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(counters::Metric const& metric, Ctx& ctx) const
    {
        return fmt::format_to(ctx.out(),
                              "Metric: {} [Block: {}, Event: {}, Expression: {}, Description: {}]",
                              metric.name(),
                              metric.block(),
                              metric.event(),
                              metric.expression().empty() ? "<None>" : metric.expression(),
                              metric.description());
    }
};

// fmt::format support for MetricMap
template <>
struct formatter<counters::MetricMap>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(counters::MetricMap const& map, Ctx& ctx) const
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
