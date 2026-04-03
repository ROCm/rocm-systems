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
#include "id_decode.hpp"

#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

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
#include <system_error>
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

// ============================================================================
// YAML Parsing Helper Functions
// ============================================================================
static void
print_counter_yaml_schema_hint()
{
    ROCP_ERROR << "Expected structure:\n"
               << "rocprofiler-sdk:\n"
               << "  counters-schema-version: 1\n"
               << "  counters:\n"
               << "  - name: COUNTER_NAME\n"
               << "    description: 'Counter description'\n"
               << "    properties: []\n"
               << "    definitions:\n"
               << "    - architectures:\n"
               << "      - gfx942\n"
               << "      block: BLOCK_NAME\n"
               << "      event: EVENT_ID\n";
}

/**
 * @brief Safely load YAML with exception handling
 */
YAML::Node
safe_load_yaml(const std::string& data, const std::string& source_name)
{
    try
    {
        return YAML::Load(data);
    } catch(const YAML::ParserException& e)
    {
        ROCP_ERROR << "Failed to parse " << source_name << ": " << e.what();
    } catch(const YAML::Exception& e)
    {
        ROCP_ERROR << "YAML error while loading " << source_name << ": " << e.what();
    } catch(const std::exception& e)
    {
        ROCP_ERROR << "Unexpected error while loading " << source_name << ": " << e.what();
    }

    return {};
}

/**
 * @brief Extract the counters node from YAML with validation
 */
YAML::Node
get_counters_node(const YAML::Node& root, const std::string& source_name, bool print_schema_hint)
{
    if(!root || !root.IsMap())
    {
        ROCP_ERROR << "Invalid " << source_name << ": root must be a map";
        if(print_schema_hint) print_counter_yaml_schema_hint();
        return {};
    }

    auto sdk = root["rocprofiler-sdk"];
    if(!sdk || !sdk.IsMap())
    {
        ROCP_ERROR << "Invalid " << source_name << ": missing or invalid 'rocprofiler-sdk'";
        if(print_schema_hint) print_counter_yaml_schema_hint();
        return {};
    }

    auto counters = sdk["counters"];
    if(!counters || !counters.IsSequence())
    {
        ROCP_ERROR << "Invalid " << source_name
                   << ": missing or invalid 'rocprofiler-sdk.counters'";
        if(print_schema_hint) print_counter_yaml_schema_hint();
        return {};
    }

    return counters;
}

/**
 * @brief Safely extract a required string field from YAML node
 * Returns std::nullopt if field missing, has wrong type, or is empty
 * Logs appropriate warning on failure
 */
std::optional<std::string>
get_required_scalar(const YAML::Node& parent, const char* key, const std::string& context)
{
    auto node = parent[key];
    if(!node || !node.IsScalar())
    {
        ROCP_WARNING << "Skipping " << context << ": missing or invalid '" << key << "'";
        return std::nullopt;
    }

    try
    {
        return node.as<std::string>();
    } catch(const YAML::Exception& e)
    {
        ROCP_WARNING << "Skipping " << context << ": invalid '" << key << "': " << e.what();
        return std::nullopt;
    }
}

/**
 * @brief Safely extract an optional string field from YAML node
 * Returns empty string if field missing or wrong type
 */
std::string
get_optional_scalar(const YAML::Node& parent, const char* key)
{
    auto node = parent[key];
    if(!node) return {};
    if(!node.IsScalar()) return {};

    try
    {
        return node.as<std::string>();
    } catch(const YAML::Exception&)
    {
        return {};
    }
}

bool
validate_definitions_node(const YAML::Node& counter, const std::string& counter_name)
{
    auto definitions = counter["definitions"];
    if(!definitions || !definitions.IsSequence())
    {
        ROCP_WARNING << "Skipping counter '" << counter_name
                     << "': missing or invalid 'definitions'";
        return false;
    }

    return true;
}
bool
validate_definition_node(const YAML::Node& definition, const std::string& counter_name)
{
    if(!definition || !definition.IsMap())
    {
        ROCP_WARNING << "Skipping invalid definition for counter '" << counter_name
                     << "': definition must be a map";
        return false;
    }

    if(!definition["architectures"] || !definition["architectures"].IsSequence())
    {
        ROCP_WARNING << "Skipping invalid definition for counter '" << counter_name
                     << "': missing or invalid 'architectures'";
        return false;
    }

    if(definition["block"] && !definition["block"].IsScalar())
    {
        ROCP_WARNING << "Skipping invalid definition for counter '" << counter_name
                     << "': 'block' must be a scalar";
        return false;
    }

    if(definition["event"] && !definition["event"].IsScalar())
    {
        ROCP_WARNING << "Skipping invalid definition for counter '" << counter_name
                     << "': 'event' must be a scalar";
        return false;
    }

    if(definition["expression"] && !definition["expression"].IsScalar())
    {
        ROCP_WARNING << "Skipping invalid definition for counter '" << counter_name
                     << "': 'expression' must be a scalar";
        return false;
    }

    return true;
}

bool
should_skip_duplicate(
    std::unordered_map<std::string, std::unordered_set<std::string>>& arch_counter_names,
    std::unordered_set<std::string>&                                  warned_duplicates,
    const std::string&                                                arch_name,
    const std::string&                                                counter_name)
{
    if(arch_counter_names[arch_name].insert(counter_name).second) return false;

    auto dup_key = arch_name + ":" + counter_name;
    if(warned_duplicates.insert(dup_key).second)
    {
        ROCP_WARNING << "Duplicate counter '" << counter_name << "' found in YAML for architecture "
                     << arch_name << ". Using first definition, ignoring duplicate.";
    }

    return true;
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

    auto return_empty = []() {
        return counter_metrics_t{
            .arch_to_metric = MetricMap{}, .id_to_metric = {}, .arch_to_id = {}};
    };

    std::stringstream counter_data;
    if(override.data.empty() || override.append)
    {
        ROCP_INFO << "Loading Counter Config: " << filename;
        std::ifstream file(filename);
        if(!file)
        {
            ROCP_ERROR << "Failed to open counter configuration file: " << filename;
            return return_empty();
        }

        counter_data << file.rdbuf();
    }
    else
    {
        ROCP_INFO << "Adding Override Config Data: " << override.data;
        counter_data << override.data;
    }

    auto yaml = safe_load_yaml(counter_data.str(), "counter config YAML");
    if(!yaml)
    {
        return return_empty();
    }

    // Extract and validate the counters node
    auto header = get_counters_node(yaml, "counter config YAML", false);
    if(!header)
    {
        return return_empty();
    }

    uint64_t current_id = 0;
    if(!override.data.empty() && override.append)
    {
        append_yaml        = safe_load_yaml(override.data, "extra counters YAML");
        auto append_header = get_counters_node(append_yaml, "extra counters YAML", true);

        if(append_header)
        {
            for(const auto& counter : append_header)
            {
                header.push_back(counter);
            }
        }
        else
        {
            ROCP_ERROR << "Ignoring invalid extra counters YAML";
        }
        // Note: Continue with main counters even if append fails
    }

    // Track counter names per architecture to detect duplicates
    std::unordered_map<std::string, std::unordered_set<std::string>> arch_counter_names;
    static std::unordered_set<std::string>                           warned_duplicates;

    for(const auto& counter : header)
    {
        // Extract and validate required counter fields
        if(!counter || !counter.IsMap())
        {
            ROCP_WARNING << "Skipping invalid counter entry: counter must be a map";
            continue;
        }

        auto counter_name_opt = get_required_scalar(counter, "name", "counter entry");
        if(!counter_name_opt) continue;
        const auto& counter_name = *counter_name_opt;

        auto description_opt =
            get_required_scalar(counter, "description", "counter '" + counter_name + "'");
        if(!description_opt) continue;
        const auto& description = *description_opt;

        if(!validate_definitions_node(counter, counter_name)) continue;

        for(const auto& definition : counter["definitions"])
        {
            if(!validate_definition_node(definition, counter_name)) continue;

            auto block      = get_optional_scalar(definition, "block");
            auto event      = get_optional_scalar(definition, "event");
            auto expression = get_optional_scalar(definition, "expression");

            if(event.empty() && expression.empty())
            {
                ROCP_WARNING << "Skipping invalid definition for counter '" << counter_name
                             << "': definition must contain 'event' or 'expression'";
                continue;
            }

            for(const auto& arch : definition["architectures"])
            {
                if(!arch || !arch.IsScalar())
                {
                    ROCP_WARNING << "Skipping invalid architecture entry for counter '"
                                 << counter_name << "': architecture must be a scalar";
                    continue;
                }

                std::string arch_name;
                try
                {
                    arch_name = arch.as<std::string>();
                } catch(const YAML::Exception& e)
                {
                    ROCP_WARNING << "Skipping invalid architecture entry for counter '"
                                 << counter_name << "': " << e.what();
                    continue;
                }

                if(should_skip_duplicate(
                       arch_counter_names, warned_duplicates, arch_name, counter_name))
                    continue;

                auto& metricVec = ret.emplace(arch_name, std::vector<Metric>()).first->second;
                if(metricVec.empty())
                {
                    const auto constants = get_constants(current_id);
                    metricVec.insert(metricVec.end(), constants.begin(), constants.end());
                    current_id += constants.size();
                }

                metricVec.emplace_back(
                    arch_name, counter_name, block, event, description, expression, "", current_id);
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
    namespace fs = common::filesystem;

    Dl_info dl_info = {};
    ROCP_INFO << filename << " is being looked up via install path";
    if(dladdr(reinterpret_cast<const void*>(rocprofiler_query_available_agents), &dl_info) != 0 &&
       dl_info.dli_fname != nullptr)
    {
        // Resolve symlinks to get the absolute physical path of the .so file.
        auto     ec            = std::error_code{};
        auto     lib_path      = fs::path{dl_info.dli_fname};
        fs::path real_lib_path = fs::canonical(lib_path, ec);
        if(!ec)
        {
            lib_path = real_lib_path;
        }

        return lib_path.parent_path().parent_path() /
               fmt::format("share/rocprofiler-sdk/{}", filename);
    }
    return filename;
}

std::string
locateMetricsFile(std::string_view name)
{
    namespace fs = common::filesystem;

    auto metric_env_path = std::string{"not set"};

    // 1) Try env var
    if(const char* env = std::getenv("ROCPROFILER_METRICS_PATH"))
    {
        metric_env_path = env;
        auto env_paths = sdk::parse::tokenize<std::vector<std::string>>(env, std::string_view{":"});
        for(const auto& path : env_paths)
        {
            fs::path candidate = fs::path{path} / std::string{name};
            if(fs::exists(candidate))
            {
                ROCP_INFO << name << " found via ROCPROFILER_METRICS_PATH: " << candidate.string();
                return candidate.string();
            }
        }
        ROCP_INFO << name << " not found at ROCPROFILER_METRICS_PATH (" << env
                  << "). Falling back to install path.";
    }

    // 2) Fall back to install path
    auto install_candidate = findViaInstallPath(std::string{name});
    if(fs::exists(install_candidate))
    {
        ROCP_INFO << name << " found via install path: " << install_candidate;
        return install_candidate;
    }

    ROCP_FATAL << "Metric file '" << name << "' not found.\n"
               << "  Tried: ROCPROFILER_METRICS_PATH (" << metric_env_path << "), and"
               << install_candidate;
    return {};
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
        auto counters_path = locateMetricsFile("config.yaml");
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
getPerfCountersIdMap(const rocprofiler_agent_t* agent)
{
    auto map = std::unordered_map<uint64_t, int>{};
    for(const auto& metric : getMetricsForAgent(agent))
    {
        // Only add basic SQ counters
        if(metric.name().find("SQ_") == 0 && !metric.event().empty())
            map.emplace(metric.id(), std::stoi(metric.event()));
    }

    return map;
}

std::vector<Metric>
getMetricsForAgent(const rocprofiler_agent_t* agent)
{
    auto mets = loadMetrics();
    if(const auto* metric_ptr =
           rocprofiler::common::get_val(mets->arch_to_metric, std::string(agent->name)))
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
