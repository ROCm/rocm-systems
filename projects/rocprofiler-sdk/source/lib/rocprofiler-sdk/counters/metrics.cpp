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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "metrics.hpp"

#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"

#include <rocprofiler-sdk/fwd.h>

#include "glog/logging.h"
#include "yaml-cpp/exceptions.h"
#include "yaml-cpp/node/convert.h"
#include "yaml-cpp/node/detail/impl.h"
#include "yaml-cpp/node/impl.h"
#include "yaml-cpp/node/iterator.h"
#include "yaml-cpp/node/node.h"
#include "yaml-cpp/node/parse.h"
#include "yaml-cpp/parser.h"

#include <dlfcn.h>  // for dladdr
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace rocprofiler
{
namespace counters
{
namespace
{
common::Synchronized<CustomCounterDefinition>&
getCustomCounterDefinition()
{
    static common::Synchronized<CustomCounterDefinition> def = {};
    return def;
}

/**
 * Constant/speical metrics are treated as psudo-metrics in that they
 * are given their own metric id. MAX_WAVE_SIZE for example is not collected
 * by AQL Profiler but is a constant from the topology. It will still have
 * a counter associated with it. Nearly all metrics contained in
 * rocprofiler_agent_t will have a counter id associated with it and can be
 * used in derived counters (exact support properties that can be used can
 * be viewed in evaluate_ast.cpp:get_agent_property()).
 */
std::vector<Metric>
get_constants(uint64_t starting_id)
{
    std::vector<Metric> constants;
    // Ensure topology is read
    rocprofiler::agent::get_agents();
    for(const auto& prop : rocprofiler::agent::get_agent_available_properties())
    {
        constants.emplace_back("constant",
                               prop,
                               "",
                               "",
                               fmt::format("Constant value {} from agent properties", prop),
                               "",
                               "yes",
                               starting_id);
        starting_id++;
    }
    return constants;
}
/**
 * Expected YAML Format:
 * COUNTER_NAME:
 *  architectures:
 *   gfxXX: // Can be more than one, / deliminated if they share idential data
 *     block: <Optional>
 *     event: <Optional>
 *     expression: <optional>
 *     description: <Optional>
 *   gfxYY:
 *      ...
 *  description: General counter desctiption
 */
counter_metrics_t
loadYAML(const std::string& filename, std::optional<ArchMetric> add_metric)
{
    // Stores metrics that are added via the API
    static MetricMap added_metrics;
    YAML::Node       append_yaml;

    MetricMap ret;
    auto      override = getCustomCounterDefinition().wlock([&](auto& data) {
        data.loaded = true;
        return data;
    });

    std::stringstream counter_data;
    if(override.data.empty() || override.append)
    {
        ROCP_INFO << "Loading Counter Config: " << filename;
        std::ifstream file(filename);
        counter_data << file.rdbuf();
    }
    else
    {
        ROCP_INFO << "Adding Override Config Data: " << override.data;
        counter_data << override.data;
    }

    auto     yaml       = YAML::Load(counter_data.str());
    auto     header     = yaml["rocprofiler-sdk"]["counters"];
    uint64_t current_id = 0;
    if(!override.data.empty() && override.append)
    {
        append_yaml = YAML::Load(override.data);
        if(append_yaml["rocprofiler-sdk"] && append_yaml["rocprofiler-sdk"]["counters"])
        {
            for(const auto& counter : append_yaml["rocprofiler-sdk"]["counters"])
            {
                header.push_back(counter);
            }
        }
    }

    for(const auto& counter : header)
    {
        auto counter_name = counter["name"].as<std::string>();
        auto description  = counter["description"].as<std::string>();
        for(const auto& definition : counter["definitions"])
        {
            for(const auto& arch : definition["architectures"])
            {
                auto& metricVec =
                    ret.emplace(arch.as<std::string>(), std::vector<Metric>()).first->second;
                if(metricVec.empty())
                {
                    const auto constants = get_constants(current_id);
                    metricVec.insert(metricVec.end(), constants.begin(), constants.end());
                    current_id += constants.size();
                }
                metricVec.emplace_back(
                    arch.as<std::string>(),
                    counter_name,
                    (definition["block"] ? definition["block"].as<std::string>() : ""),
                    (definition["event"] ? definition["event"].as<std::string>() : ""),
                    description,
                    (definition["expression"] ? definition["expression"].as<std::string>() : ""),
                    "",
                    current_id);
                current_id++;
            }
        }
    }

    // Add custom counters after adding the above counters, ensures that the mapping is
    // deterministic when generated.
    for(const auto& [arch, metrics] : added_metrics)
    {
        auto& metricVec = ret.emplace(arch, std::vector<Metric>()).first->second;
        metricVec.insert(metricVec.end(), metrics.begin(), metrics.end());
        current_id += metrics.size();
    }

    if(add_metric)
    {
        Metric with_id = Metric(add_metric->first,
                                add_metric->second.name(),
                                add_metric->second.block(),
                                add_metric->second.event(),
                                add_metric->second.description(),
                                add_metric->second.expression(),
                                "",
                                current_id);
        added_metrics.emplace(add_metric->first, std::vector<Metric>{})
            .first->second.push_back(with_id);
        ret.emplace(add_metric->first, std::vector<Metric>{}).first->second.push_back(with_id);
    }

    ROCP_FATAL_IF(current_id > 65536)
        << "Counter count exceeds 16 bits, which may break counter id output";

    return {.arch_to_metric = ret,
            .id_to_metric =
                [&]() {
                    MetricIdMap map;
                    for(const auto& [agent_name, metrics] : ret)
                    {
                        for(const auto& m : metrics)
                        {
                            map.emplace(m.id(), m);
                        }
                    }
                    return map;
                }(),
            .arch_to_id =
                [&]() {
                    ArchToId map;
                    for(const auto& [agent_name, metrics] : ret)
                    {
                        std::unordered_set<uint64_t> ids;
                        for(const auto& m : metrics)
                        {
                            ids.insert(m.id());
                        }
                        map.emplace(agent_name, std::move(ids));
                    }
                    return map;
                }()};
}

std::string
findViaInstallPath(const std::string& filename)
{
    Dl_info dl_info = {};
    ROCP_INFO << filename << " is being looked up via install path";
    if(dladdr(reinterpret_cast<const void*>(rocprofiler_query_available_agents), &dl_info) != 0)
    {
        return common::filesystem::path{dl_info.dli_fname}.parent_path().parent_path() /
               fmt::format("share/rocprofiler-sdk/{}", filename);
    }
    return filename;
}

std::string
findViaEnvironment(const std::string& filename)
{
    if(const char* metrics_path = nullptr; (metrics_path = getenv("ROCPROFILER_METRICS_PATH")))
    {
        ROCP_INFO << filename << " is being looked up via env variable ROCPROFILER_METRICS_PATH";
        return common::filesystem::path{std::string{metrics_path}} / filename;
    }
    // No environment variable, lookup via install path
    return findViaInstallPath(filename);
}

}  // namespace

rocprofiler_status_t
setCustomCounterDefinition(const CustomCounterDefinition& def)
{
    return getCustomCounterDefinition().wlock([&](auto& data) {
        // Counter definition already loaded, cannot override anymore
        if(data.loaded) return ROCPROFILER_STATUS_ERROR;
        data.data   = def.data;
        data.append = def.append;
        return ROCPROFILER_STATUS_SUCCESS;
    });
}

std::shared_ptr<const counter_metrics_t>
loadMetrics(bool reload, const std::optional<ArchMetric> add_metric)
{
    using sync_metric = common::Synchronized<std::shared_ptr<const counter_metrics_t>>;

    if(!reload && add_metric)
    {
        ROCP_FATAL << "Adding a metric without reloading metric list, this should not happen and "
                      "will result in custom metrics not being added";
    }

    auto reload_func = [&]() {
        auto counters_path = findViaEnvironment("counter_defs.yaml");
        ROCP_FATAL_IF(!common::filesystem::exists(counters_path))
            << "metric xml file '" << counters_path << "' does not exist";
        return std::make_shared<counter_metrics_t>(loadYAML(counters_path, add_metric));
    };

    static sync_metric*& id_map =
        common::static_object<sync_metric>::construct([&]() { return reload_func(); }());

    if(!id_map) return nullptr;

    if(!reload)
    {
        return id_map->rlock([](const auto& data) {
            CHECK(data);
            return data;
        });
    }

    return id_map->wlock([&](auto& data) {
        data = reload_func();
        CHECK(data);
        return data;
    });
}

std::unordered_map<uint64_t, int>
getPerfCountersIdMap()
{
    std::unordered_map<uint64_t, int> map;
    auto                              mets = loadMetrics();

    for(const auto& [agent, list] : mets->arch_to_metric)
    {
        if(agent.find("gfx9") == std::string::npos) continue;
        for(const auto& metric : list)
        {
            if(metric.name().find("SQ_") == 0 && !metric.event().empty())
                map.emplace(metric.id(), std::stoi(metric.event()));
        }
    }

    return map;
}

std::vector<Metric>
getMetricsForAgent(const std::string& agent)
{
    auto mets = loadMetrics();
    if(const auto* metric_ptr = rocprofiler::common::get_val(mets->arch_to_metric, agent))
    {
        return *metric_ptr;
    }

    return std::vector<Metric>{};
}

bool
checkValidMetric(const std::string& agent, const Metric& metric)
{
    auto        metrics   = loadMetrics();
    const auto* agent_map = common::get_val(metrics->arch_to_id, agent);
    return agent_map != nullptr && agent_map->count(metric.id()) > 0;
}

bool
operator<(Metric const& lhs, Metric const& rhs)
{
    return std::tie(lhs.id_, lhs.flags_) < std::tie(rhs.id_, rhs.flags_);
}

bool
operator==(Metric const& lhs, Metric const& rhs)
{
    auto get_tie = [](auto& x) {
        return std::tie(x.name_,
                        x.block_,
                        x.event_,
                        x.description_,
                        x.expression_,
                        x.constant_,
                        x.id_,
                        x.empty_,
                        x.flags_);
    };
    return get_tie(lhs) == get_tie(rhs);
}
Metric::Metric(const std::string&,  // Get rid of this...
               std::string name,
               std::string block,
               std::string event,
               std::string dsc,
               std::string expr,
               std::string constant,
               uint64_t    id)
: name_(std::move(name))
, block_(std::move(block))
, event_(std::move(event))
, description_(std::move(dsc))
, expression_(std::move(expr))
, constant_(std::move(constant))
, id_(id)
{
    if(!event_.empty())
    {
        try
        {
            uint64_t event_id  = std::stoul(event_, nullptr);
            uint32_t id_high32 = (event_id >> 32) & 0xFFFFFFFF;
            if(id_high32 != 0u) setflags(id_high32);
        } catch(std::exception& e)
        {
            ROCP_CI_LOG(INFO) << fmt::format(
                "AQL packet construct for '{}' threw an exception: {}", event_, e.what());
        }
    }
}
}  // namespace counters
}  // namespace rocprofiler
