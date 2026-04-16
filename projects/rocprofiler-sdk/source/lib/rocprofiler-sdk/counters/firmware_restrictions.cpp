// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <dlfcn.h>
#include <fmt/format.h>
#include <fstream>
#include <mutex>

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
    if(const char* metrics_path = nullptr; (metrics_path = getenv("ROCPROFILER_METRICS_PATH")))
    {
        ROCP_INFO << filename << " is being looked up via env variable ROCPROFILER_METRICS_PATH";
        return common::filesystem::path{std::string{metrics_path}} / filename;
    }
    // No environment variable, lookup via install path
    return findViaInstallPath(filename);
}

}  // namespace

std::optional<std::vector<FirmwareRestriction>>
parse_firmware_restrictions(const std::string& yaml_content)
{
    std::vector<FirmwareRestriction> result;

    try
    {
        // Parse the YAML content
        YAML::Node root = YAML::Load(yaml_content);

        // Check for required top-level key
        if(!root["rocprofiler-sdk"])
        {
            return std::nullopt;
        }

        YAML::Node sdk_node = root["rocprofiler-sdk"];

        // Check for schema version
        if(!sdk_node["fw-restriction-schema-version"])
        {
            return std::nullopt;
        }

        // For future reference on how to read schema versions:
        // int schema_version = sdk_node["fw-restriction-schema-version"].as<int>();
        // if(schema_version != 1)
        // {
        //     return std::nullopt;
        // }

        // Parse firmware restrictions if present
        if(sdk_node["firmware_restrictions"])
        {
            YAML::Node restrictions = sdk_node["firmware_restrictions"];

            // Iterate through each firmware restriction entry
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

                // Parse affected architectures
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
    bool result = true;  // Assume valid until proven otherwise

    // Parse firmware restrictions from YAML
    auto restrictions_opt = parse_firmware_restrictions(yaml_content);
    if(!restrictions_opt)
    {
        ROCP_WARNING << "Failed to parse firmware restrictions from YAML content";
        return true;  // If parsing fails, assume no restrictions
    }

    const auto& restrictions = *restrictions_opt;
    // Get all agents
    auto agents = rocprofiler::agent::get_agents();
    // Check each agent against applicable firmware restrictions
    for(const auto* agent : agents)
    {
        if(!agent || agent->type != ROCPROFILER_AGENT_TYPE_GPU)
        {
            continue;  // Skip non-GPU agents
        }

        // Get agent architecture (name field contains gfx architecture)
        const std::string agent_arch = agent->name ? agent->name : "";
        if(agent_arch.empty())
        {
            ROCP_WARNING << "Agent " << agent->node_id << " has no architecture name";
            continue;
        }

        // Check each firmware restriction
        for(const auto& restriction : restrictions)
        {
            // Check if this architecture is affected by this restriction
            bool is_affected = restriction.affected_architectures.empty();
            for(const auto& affected_arch : restriction.affected_architectures)
            {
                if(agent_arch == affected_arch)
                {
                    is_affected = true;
                    break;
                }
            }

            if(!is_affected)
            {
                continue;  // This restriction doesn't apply to this architecture
            }

            // Get firmware version based on type
            uint32_t agent_fw_version = 0;
            if(restriction.firmware_type == "CP" || restriction.firmware_type == "MEC")
            {
                agent_fw_version = agent->fw_version.Value;
            }
            else if(restriction.firmware_type == "SDMA")
            {
                agent_fw_version = agent->sdma_fw_version.Value;
            }
            else
            {
                ROCP_WARNING << "Unknown firmware type '" << restriction.firmware_type
                             << "' in restriction for agent " << agent->node_id;
                continue;
            }

            // Check if firmware version meets minimum requirement
            if(agent_fw_version < restriction.min_version)
            {
                ROCP_WARNING << "Agent " << agent->node_id << " (" << agent_arch << ") has "
                             << restriction.firmware_type << " firmware version "
                             << agent_fw_version << " which is below minimum required version "
                             << restriction.min_version << ". Reason: " << restriction.reason;
                result = false;
            }
        }
    }

    return result;
}

bool
check_installed_firmware_restrictions()
{
    static std::once_flag check_flag;
    static bool           result = true;

    std::call_once(check_flag, []() {
        result = true;  // Assume valid until proven otherwise

        // Find the counter definitions YAML file that contains firmware restrictions
        auto fw_restrictions_path = findViaEnvironment("config.yaml");

        // Check if file exists
        if(!common::filesystem::exists(fw_restrictions_path))
        {
            ROCP_WARNING << "Counter definitions file '" << fw_restrictions_path
                         << "' does not exist, skipping firmware validation";
            return;  // No restrictions file means no restrictions
        }

        // Read the file content
        std::ifstream file(fw_restrictions_path);
        if(!file.is_open())
        {
            ROCP_WARNING << "Failed to open counter definitions file '" << fw_restrictions_path
                         << "'";
            return;  // If we can't read the file, assume no restrictions
        }

        std::string yaml_content((std::istreambuf_iterator<char>(file)),
                                 std::istreambuf_iterator<char>());
        file.close();

        if(yaml_content.empty())
        {
            ROCP_INFO << "Counter definitions file '" << fw_restrictions_path
                      << "' is empty, no restrictions to apply";
            return;
        }

        ROCP_INFO << "Checking firmware restrictions from '" << fw_restrictions_path << "'";

        // Use the existing function to perform the actual check
        result = check_agent_firmware_restrictions(yaml_content);
    });

    return result;
}

}  // namespace counters
}  // namespace rocprofiler
