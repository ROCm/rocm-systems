// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "common/json_config.hpp"
#include "common/path.hpp"

#include <nlohmann/json.hpp>

#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace rocprofsys
{
namespace preset_loader
{

/**
 * Information about a loaded preset.
 */
struct preset_info
{
    std::string                        name;
    std::string                        cli_flag;
    std::string                        description;
    std::string                        use_case;
    std::string                        category;
    std::map<std::string, std::string> settings;  // Resolved ROCPROFSYS_* env vars
};

/**
 * Locates the preset directory using the following search order:
 *   1. $ROCPROFSYS_PRESET_DIR
 *   2. <exe_dir>/../share/rocprofiler-systems/presets
 *   3. $ROCM_PATH/share/rocprofiler-systems/presets
 * @return Path to preset directory, or empty string if not found.
 */
[[nodiscard]] std::string
find_preset_directory();

/**
 * Loads a preset from a specific file path.
 * @param filepath The path to the preset JSON file.
 * @return The loaded preset info, or nullopt if loading failed.
 */
[[nodiscard]] std::optional<preset_info>
load_preset_file(const std::string& filepath);

/**
 * Loads all presets from the preset directory.
 * @return Map of preset names to preset info.
 */
[[nodiscard]] std::map<std::string, preset_info>
load_all_presets();

/**
 * Loads a preset by name from the preset directory.
 * @param name The name of the preset (without .json extension).
 * @return The loaded preset info, or nullopt if not found.
 */
[[nodiscard]] std::optional<preset_info>
load_preset(std::string_view name);

/**
 * Loads a preset by name or from a file path.
 * If name contains '/' or ends with '.json', treats it as a file path.
 * Otherwise, looks up the preset by name in the preset directory.
 */
[[nodiscard]] std::optional<preset_info>
load_preset_or_file(const std::string& name_or_path);

/**
 * Applies preset settings to the environment.
 * Only sets variables that are not already set in the environment.
 */
void
apply_preset_to_environment(const preset_info& info, bool override_existing = false);

/**
 * Prints a summary of the preset settings for user feedback.
 */
void
print_preset_summary(const preset_info& info, std::ostream& os = std::cout);

}  // namespace preset_loader
}  // namespace rocprofsys
