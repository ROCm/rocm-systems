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

#pragma once

#include <cstdint>
#include <optional>
#include <string>
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
// Uses std::call_once to ensure this check is only performed once
// Looks for config.yaml in the same way as metrics.cpp
bool
check_installed_firmware_restrictions();

}  // namespace counters
}  // namespace rocprofiler
