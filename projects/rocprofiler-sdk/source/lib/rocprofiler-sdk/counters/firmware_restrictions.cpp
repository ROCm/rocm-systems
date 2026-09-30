// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "firmware_restrictions.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <dlfcn.h>
#include <fmt/format.h>
#include <algorithm>
#include <fstream>

#include "yaml-cpp/exceptions.h"
#include "yaml-cpp/node/convert.h"
#include "yaml-cpp/node/detail/impl.h"
#include "yaml-cpp/node/impl.h"
#include "yaml-cpp/node/iterator.h"
#include "yaml-cpp/node/node.h"
#include "yaml-cpp/node/parse.h"
#include "yaml-cpp/parser.h"

namespace rocprofiler
{
namespace counters
{
namespace
{
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
    auto metrics_path = common::get_env_optional("ROCPROFILER_METRICS_PATH");
    if(metrics_path)
    {
        ROCP_INFO << filename << " is being looked up via env variable ROCPROFILER_METRICS_PATH";
        return common::filesystem::path{*metrics_path} / filename;
    }
    // No environment variable, lookup via install path
    return findViaInstallPath(filename);
}

std::optional<std::vector<FirmwareRestriction>>
load_installed_firmware_restrictions()
{
    auto fw_restrictions_path = findViaEnvironment("config.yaml");

    if(!common::filesystem::exists(fw_restrictions_path))
    {
        ROCP_WARNING << "Counter definitions file '" << fw_restrictions_path
                     << "' does not exist, skipping firmware validation";
        return std::nullopt;
    }

    std::ifstream file(fw_restrictions_path);
    if(!file.is_open())
    {
        ROCP_WARNING << "Failed to open counter definitions file '" << fw_restrictions_path << "'";
        return std::nullopt;
    }

    std::string yaml_content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());

    if(yaml_content.empty())
    {
        ROCP_INFO << "Counter definitions file '" << fw_restrictions_path << "' is empty";
        return std::nullopt;
    }

    ROCP_INFO << "Checking firmware restrictions from '" << fw_restrictions_path << "'";

    auto restrictions = parse_firmware_restrictions(yaml_content);
    if(!restrictions)
        ROCP_WARNING << "Failed to parse firmware restrictions from '" << fw_restrictions_path
                     << "'";
    return restrictions;
}

// Parsed once; shared by the startup gate and feature queries.
const std::optional<std::vector<FirmwareRestriction>>&
installed_firmware_restrictions()
{
    static const auto restrictions = load_installed_firmware_restrictions();
    return restrictions;
}

// A restriction applies to an architecture when its affected list is empty
// (applies to all) or contains that architecture.
bool
affects(const FirmwareRestriction& restriction, std::string_view arch)
{
    return restriction.affected_architectures.empty() ||
           std::any_of(restriction.affected_architectures.begin(),
                       restriction.affected_architectures.end(),
                       [&](const auto& a) { return arch == a; });
}

// Select the agent firmware version relevant to a restriction's firmware_type,
// or nullopt for an unrecognized type.
std::optional<uint32_t>
firmware_version(std::string_view type, uint32_t cp, uint32_t sdma)
{
    if(type == "CP" || type == "MEC") return cp;
    if(type == "SDMA") return sdma;
    return std::nullopt;
}

bool
check_agents_against_restrictions(const std::vector<FirmwareRestriction>& restrictions)
{
    bool result = true;

    for(const auto* agent : rocprofiler::agent::get_agents())
    {
        if(!agent || agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;

        const std::string agent_arch = agent->name ? agent->name : "";
        if(agent_arch.empty())
        {
            ROCP_WARNING << "Agent " << agent->node_id << " has no architecture name";
            continue;
        }

        for(const auto& restriction : restrictions)
        {
            // Feature-scoped floors are queried via agent_supports_feature().
            if(!restriction.feature.empty()) continue;
            if(!affects(restriction, agent_arch)) continue;

            auto agent_fw_version = firmware_version(
                restriction.firmware_type, agent->fw_version.Value, agent->sdma_fw_version.Value);
            if(!agent_fw_version)
            {
                ROCP_WARNING << "Unknown firmware type '" << restriction.firmware_type
                             << "' in restriction for agent " << agent->node_id;
                continue;
            }

            if(*agent_fw_version < restriction.min_version)
            {
                ROCP_WARNING << "Agent " << agent->node_id << " (" << agent_arch << ") has "
                             << restriction.firmware_type << " firmware version "
                             << *agent_fw_version << " which is below minimum required version "
                             << restriction.min_version << ". Reason: " << restriction.reason;
                result = false;
            }
        }
    }

    return result;
}

}  // namespace

std::optional<std::vector<FirmwareRestriction>>
parse_firmware_restrictions(const std::string& yaml_content)
{
    std::vector<FirmwareRestriction> result;

    try
    {
        YAML::Node root = YAML::Load(yaml_content);

        if(!root["rocprofiler-sdk"])
        {
            return std::nullopt;
        }

        YAML::Node sdk_node = root["rocprofiler-sdk"];

        if(!sdk_node["fw-restriction-schema-version"])
        {
            return std::nullopt;
        }

        // Schema 2 added the optional per-feature `feature` field.
        const auto schema_version = sdk_node["fw-restriction-schema-version"].as<int>();
        if(schema_version < 1 || schema_version > 2)
        {
            return std::nullopt;
        }

        if(sdk_node["firmware_restrictions"])
        {
            YAML::Node restrictions = sdk_node["firmware_restrictions"];

            for(const auto& fw_entry : restrictions)
            {
                auto& restriction = result.emplace_back(FirmwareRestriction{});

                if(!fw_entry["firmware_type"] || !fw_entry["min_version"])
                {
                    return std::nullopt;
                }

                restriction.firmware_type = fw_entry["firmware_type"].as<std::string>();
                restriction.min_version   = fw_entry["min_version"].as<uint32_t>();
                restriction.reason =
                    (fw_entry["reason"] ? fw_entry["reason"].as<std::string>() : std::string{});
                restriction.feature =
                    (fw_entry["feature"] ? fw_entry["feature"].as<std::string>() : std::string{});

                if(fw_entry["affected_architectures"])
                {
                    for(const auto& arch : fw_entry["affected_architectures"])
                    {
                        restriction.affected_architectures.push_back(arch.as<std::string>());
                    }
                }
            }
        }
    } catch(const YAML::Exception& e)
    {
        return std::nullopt;
    }

    return result;
}

bool
check_agent_firmware_restrictions(const std::string& yaml_content)
{
    auto restrictions_opt = parse_firmware_restrictions(yaml_content);
    if(!restrictions_opt)
    {
        ROCP_WARNING << "Failed to parse firmware restrictions from YAML content";
        return true;
    }

    return check_agents_against_restrictions(*restrictions_opt);
}

bool
check_installed_firmware_restrictions()
{
    static const bool result = []() {
        const auto& restrictions = installed_firmware_restrictions();
        return restrictions ? check_agents_against_restrictions(*restrictions) : true;
    }();

    return result;
}

std::optional<bool>
evaluate_feature_support(const std::vector<FirmwareRestriction>& restrictions,
                         std::string_view                        agent_arch,
                         uint32_t                                cp_fw_version,
                         uint32_t                                sdma_fw_version,
                         std::string_view                        feature)
{
    if(feature.empty()) return std::nullopt;

    bool any_matched = false;
    bool all_met     = true;

    for(const auto& restriction : restrictions)
    {
        if(restriction.feature != feature || !affects(restriction, agent_arch)) continue;

        auto agent_fw_version =
            firmware_version(restriction.firmware_type, cp_fw_version, sdma_fw_version);
        if(!agent_fw_version) return std::nullopt;  // malformed applicable entry

        any_matched = true;
        if(*agent_fw_version < restriction.min_version) all_met = false;
    }

    if(!any_matched) return std::nullopt;
    return all_met;
}

std::optional<bool>
agent_supports_feature(const rocprofiler_agent_t* agent, std::string_view feature)
{
    if(agent == nullptr || agent->type != ROCPROFILER_AGENT_TYPE_GPU || agent->name == nullptr ||
       agent->name[0] == '\0')
        return std::nullopt;

    const auto& restrictions = installed_firmware_restrictions();
    if(!restrictions) return std::nullopt;

    return evaluate_feature_support(
        *restrictions, agent->name, agent->fw_version.Value, agent->sdma_fw_version.Value, feature);
}

}  // namespace counters
}  // namespace rocprofiler
