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

#pragma once

#include <rocprofiler-sdk/agent.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofiler
{
namespace counters
{
struct FirmwareRestriction
{
    std::string              firmware_type          = {};  // Firmware type (e.g., "CP", "SDMA")
    uint32_t                 min_version            = 0;   // Minimum required version
    uint32_t                 current_version        = 0;   // Current firmware version on agent
    std::string              reason                 = {};  // Reason for the restriction
    std::vector<std::string> affected_architectures = {};  // Architectures requiring this minimum
    std::string              feature                = {};  // Optional feature name; empty = hard
                                                           // global floor
};

// Parse YAML string and return a vector of FirmwareRestriction structs
// Returns nullopt on parsing errors or invalid schema
std::optional<std::vector<FirmwareRestriction>>
parse_firmware_restrictions(const std::string& yaml_content);

// Check all agents against firmware restrictions from YAML content
// Returns false if any agent has firmware below minimum requirements
bool
check_agent_firmware_restrictions(const std::string& yaml_content);

// Check all agents against firmware restrictions from installed YAML file
// Returns false if any agent has firmware below minimum requirements
// The result is computed once and cached; feature-scoped entries are excluded.
bool
check_installed_firmware_restrictions();

// Query whether an agent meets the firmware floor(s) for a named feature.
// Feature entries are capability metadata, excluded from the startup gate.
//   std::nullopt -> no matching entry (undeclared, arch mismatch, invalid agent)
//   false        -> a matching feature floor exists and the agent is below it
//   true         -> matching feature floor(s) exist and all are met
// Callers must not treat std::nullopt as "feature unsupported": it means the
// query could not be evaluated. Only false means an evaluated floor was not met.
std::optional<bool>
agent_supports_feature(const rocprofiler_agent_t* agent, std::string_view feature);

// Pure evaluator behind agent_supports_feature(), exposed for testing.
std::optional<bool>
evaluate_feature_support(const std::vector<FirmwareRestriction>& restrictions,
                         std::string_view                        agent_arch,
                         uint32_t                                cp_fw_version,
                         uint32_t                                sdma_fw_version,
                         std::string_view                        feature);

}  // namespace counters
}  // namespace rocprofiler
