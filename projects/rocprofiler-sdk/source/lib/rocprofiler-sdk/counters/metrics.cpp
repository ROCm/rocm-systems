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

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/spm/interface.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/details/tokenize.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include "yaml-cpp/exceptions.h"
#include "yaml-cpp/node/convert.h"
#include "yaml-cpp/node/detail/impl.h"
#include "yaml-cpp/node/impl.h"
#include "yaml-cpp/node/iterator.h"
#include "yaml-cpp/node/node.h"
#include "yaml-cpp/node/parse.h"
#include "yaml-cpp/parser.h"

#include <dlfcn.h>  // for dladdr
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
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

struct yaml_counter_definition
{
    std::string description = {};
    std::string block       = {};
    std::string event       = {};
    std::string expression  = {};
};

bool
operator==(const yaml_counter_definition& lhs, const yaml_counter_definition& rhs)
{
    return std::tie(lhs.description, lhs.block, lhs.event, lhs.expression) ==
           std::tie(rhs.description, rhs.block, rhs.event, rhs.expression);
}

std::string
format_yaml_counter_definition(const yaml_counter_definition& definition)
{
    return fmt::format("description='{}', block='{}', event='{}', expression='{}'",
                       definition.description,
                       definition.block,
                       definition.event,
                       definition.expression);
}

std::optional<uint64_t>
parse_unsigned_integer(const YAML::Node& node)
{
    if(!node || !node.IsScalar()) return std::nullopt;

    auto value        = node.as<std::string>();
    auto parsed_value = uint64_t{};
    auto result       = std::from_chars(value.data(), value.data() + value.size(), parsed_value);
    if(value.empty() || result.ec != std::errc{} || result.ptr != value.data() + value.size())
        return std::nullopt;
    return parsed_value;
}

/**
 * Constant/special metrics are treated as pseudo-metrics in that they
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

    YAML::Node yaml;
    YAML::Node header;
    uint64_t   current_id = 0;

    try
    {
        yaml = YAML::Load(counter_data.str());
        if(!override.data.empty() && !override.append)
        {
            auto error = validateExtraCounterYAML(yaml);
            if(error)
            {
                ROCP_FATAL << "Invalid extra counters YAML: " << *error << "\n"
                           << "Content:\n"
                           << override.data;
            }
        }
        header = yaml["rocprofiler-sdk"]["counters"];
    } catch(const YAML::Exception& e)
    {
        if(!override.data.empty() && !override.append)
        {
            ROCP_FATAL << "Failed to parse extra counters YAML: " << e.what() << "\n"
                       << "Content:\n"
                       << override.data;
        }
        else
        {
            ROCP_FATAL << "Failed to parse counter file " << filename << ": " << e.what();
        }
    }

    if(!override.data.empty() && override.append)
    {
        try
        {
            append_yaml = YAML::Load(override.data);

            auto error = validateExtraCounterYAML(append_yaml);
            if(error)
            {
                ROCP_FATAL << "Invalid extra counters YAML: " << *error << "\n"
                           << "Content:\n"
                           << override.data;
            }

            for(const auto& counter : append_yaml["rocprofiler-sdk"]["counters"])
                header.push_back(counter);
        } catch(const YAML::Exception& e)
        {
            ROCP_FATAL << "Failed to parse extra counters YAML: " << e.what() << "\n"
                       << "Content:\n"
                       << override.data;
        }
    }

    using definitions_by_name_t     = std::unordered_map<std::string, yaml_counter_definition>;
    auto loaded_counter_definitions = std::unordered_map<std::string, definitions_by_name_t>{};
    for(const auto& counter : header)
    {
        auto counter_name = counter["name"].as<std::string>();
        auto description  = counter["description"].as<std::string>();
        for(const auto& definition : counter["definitions"])
        {
            for(const auto& arch : definition["architectures"])
            {
                auto arch_name = arch.as<std::string>();
                auto block     = definition["block"] ? definition["block"].as<std::string>() : "";
                auto event     = definition["event"] ? definition["event"].as<std::string>() : "";
                auto expression =
                    definition["expression"] ? definition["expression"].as<std::string>() : "";
                auto definition_data =
                    yaml_counter_definition{description, block, event, expression};
                auto [existing_definition, inserted] =
                    loaded_counter_definitions[arch_name].emplace(counter_name, definition_data);
                if(!inserted)
                {
                    if(existing_definition->second == definition_data)
                    {
                        ROCP_WARNING << "Counter '" << counter_name << "' for architecture '"
                                     << arch_name
                                     << "' duplicates an identical definition; ignoring duplicate";
                    }
                    else
                    {
                        ROCP_FATAL
                            << "Conflicting counter definitions for '" << counter_name
                            << "' on architecture '" << arch_name << "'. Existing definition: "
                            << format_yaml_counter_definition(existing_definition->second)
                            << "; new definition: "
                            << format_yaml_counter_definition(definition_data)
                            << ". Counter names must resolve to one definition per architecture"
                            << (override.append
                                    ? "; append mode cannot override an existing counter"
                                    : "");
                    }
                    continue;
                }

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

    if(current_id > 65536)
    {
        ROCP_FATAL << "Counter count exceeds 16 bits, which may break counter id output";
    }

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
    auto env = common::get_env_optional("ROCPROFILER_METRICS_PATH");
    if(env)
    {
        metric_env_path = *env;
        auto env_paths =
            sdk::parse::tokenize<std::vector<std::string>>(*env, std::string_view{":"});
        for(const auto& path : env_paths)
        {
            fs::path candidate = fs::path{path} / std::string{name};
            if(fs::exists(candidate))
            {
                ROCP_INFO << name << " found via ROCPROFILER_METRICS_PATH: " << candidate.string();
                return candidate.string();
            }
        }
        ROCP_INFO << name << " not found at ROCPROFILER_METRICS_PATH (" << *env
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
               << "  Tried: ROCPROFILER_METRICS_PATH (" << metric_env_path << "), and "
               << install_candidate;
}

}  // namespace

std::optional<std::string>
validateExtraCounterYAML(const YAML::Node& root)
{
    if(!root || !root.IsMap()) return "Top-level YAML node must be a map";

    const auto sdk_node = root["rocprofiler-sdk"];
    if(!sdk_node) return "Missing top-level 'rocprofiler-sdk' key";
    if(!sdk_node.IsMap()) return "'rocprofiler-sdk' must be a map";

    const auto schema_version = sdk_node["counters-schema-version"];
    if(schema_version)
    {
        auto parsed_schema_version = parse_unsigned_integer(schema_version);
        if(!parsed_schema_version || *parsed_schema_version != 1)
            return "Unsupported 'counters-schema-version'; expected 1";
    }

    const auto counters_node = sdk_node["counters"];
    if(!counters_node) return "Missing 'counters' array under 'rocprofiler-sdk'";

    if(!counters_node.IsSequence()) return "'counters' must be a sequence";

    for(size_t i = 0; i < counters_node.size(); ++i)
    {
        const auto& counter = counters_node[i];
        auto        ctx     = fmt::format("counters[{}]", i);

        if(!counter.IsMap()) return fmt::format("{}: counter must be a map", ctx);
        if(!counter["name"]) return fmt::format("{}: missing 'name' field", ctx);
        if(!counter["name"].IsScalar()) return fmt::format("{}: 'name' must be a string", ctx);

        auto name = counter["name"].as<std::string>();
        if(name.empty()) return fmt::format("{}: 'name' must not be empty", ctx);

        const auto description = counter["description"];
        if(!description) return fmt::format("Counter '{}': missing 'description'", name);
        if(!description.IsScalar())
            return fmt::format("Counter '{}': 'description' must be a string", name);

        if(!counter["definitions"]) return fmt::format("Counter '{}': missing 'definitions'", name);
        if(!counter["definitions"].IsSequence())
            return fmt::format("Counter '{}': 'definitions' must be a sequence", name);
        if(counter["definitions"].size() == 0)
            return fmt::format("Counter '{}': 'definitions' is empty", name);

        for(size_t j = 0; j < counter["definitions"].size(); ++j)
        {
            const auto& def     = counter["definitions"][j];
            auto        def_ctx = fmt::format("Counter '{}', definition [{}]", name, j);

            if(!def.IsMap()) return fmt::format("{}: definition must be a map", def_ctx);
            if(!def["architectures"]) return fmt::format("{}: missing 'architectures'", def_ctx);
            if(!def["architectures"].IsSequence())
                return fmt::format("{}: 'architectures' must be a sequence", def_ctx);
            if(def["architectures"].size() == 0)
                return fmt::format("{}: 'architectures' is empty", def_ctx);

            for(const auto& arch : def["architectures"])
            {
                if(!arch.IsScalar())
                    return fmt::format("{}: architecture must be a string", def_ctx);
                if(arch.as<std::string>().empty())
                    return fmt::format("{}: architecture must not be empty", def_ctx);
            }

            const auto expression = def["expression"];
            const auto event      = def["event"];
            const auto block      = def["block"];

            if(expression && !expression.IsScalar())
                return fmt::format("{}: 'expression' must be a string", def_ctx);
            if(event && !event.IsScalar())
                return fmt::format("{}: 'event' must be an unsigned integer", def_ctx);
            if(block && !block.IsScalar())
                return fmt::format("{}: 'block' must be a string", def_ctx);

            bool has_expr  = static_cast<bool>(expression);
            bool has_event = static_cast<bool>(event);
            bool has_block = static_cast<bool>(block);

            if(has_expr && expression.as<std::string>().empty())
                return fmt::format("{}: 'expression' must not be empty", def_ctx);
            if(has_block && block.as<std::string>().empty())
                return fmt::format("{}: 'block' must not be empty", def_ctx);
            if(has_expr && (has_event || has_block))
                return fmt::format("{}: 'expression' cannot be combined with 'event' or 'block'",
                                   def_ctx);
            if(has_event && !has_block) return fmt::format("{}: 'event' requires 'block'", def_ctx);
            if(has_block && !has_event) return fmt::format("{}: 'block' requires 'event'", def_ctx);
            if(!has_expr && !has_event)
                return fmt::format("{}: must have 'expression' or 'event'+'block'", def_ctx);

            if(has_event)
            {
                if(!parse_unsigned_integer(event))
                    return fmt::format("{}: 'event' must be an unsigned integer", def_ctx);
            }
        }
    }

    return std::nullopt;
}

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

bool
has_spm_support(const Metric& metric, rocprofiler_agent_id_t agent_id)
{
    using cache_key_t = std::pair<rocprofiler_agent_id_t, uint64_t>;
    using cache_t     = std::map<cache_key_t, bool>;
    using sync_cache  = common::Synchronized<cache_t>;

    static auto*& cache = common::static_object<sync_cache>::construct(cache_t{});

    auto key = cache_key_t{agent_id, metric.id()};

    return cache->wlock([&](auto& data) {
        auto it = data.find(key);
        if(it != data.end()) return it->second;

        bool supported = false;
        if(!metric.event().empty())
        {
            if(const auto* sym = rocprofiler::spm::construct_spm_interface())
            {
                auto aql_agent  = *CHECK_NOTNULL(rocprofiler::agent::get_aql_agent(agent_id));
                auto query_info = rocprofiler::aql::get_query_info(agent_id, metric);
                auto pmc_event  = aqlprofile_pmc_event_t{};
                pmc_event.block_name =
                    static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query_info.id);
                pmc_event.event_id =
                    static_cast<uint32_t>(std::stoul(metric.event().c_str(), nullptr));
                supported = sym->spm_is_event_supported(aql_agent, pmc_event);
            }
        }
        data.emplace(key, supported);
        return supported;
    });
}

}  // namespace counters
}  // namespace rocprofiler
