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

/**
 * Registry for preset configurations.
 *
 * Caches the preset directory (found once at construction) and all loaded
 * presets to avoid redundant I/O.  Provides higher-level operations such as
 * apply, list.
 */
using env_settings = std::map<std::string, std::string>;

class preset_registry
{
public:
    struct preset_info
    {
        std::string  name;
        std::string  cli_flag;
        std::string  description;
        std::string  use_case;
        std::string  category;
        env_settings settings;
    };

    preset_registry();

    /**
     * Check if a top-level section (e.g., "tracing", "profiling") is enabled in a preset.
     * Returns the default_value if the preset or section is not found.
     */
    [[nodiscard]] bool is_section_enabled(std::string_view preset_name,
                                          std::string_view section,
                                          bool             default_value = true) const;

    /**
     * Translate a legacy preset flag (e.g., "--balanced") to new syntax
     * ("--preset=balanced"). Returns empty string if the argument is not
     * a recognized legacy preset flag. Emits a deprecation warning to stderr.
     */
    [[nodiscard]] std::string translate_legacy_flag(std::string_view arg) const;

    /**
     * Load a preset and return its resolved environment settings.
     * @return The settings map if the preset was found, std::nullopt otherwise.
     */
    [[nodiscard]] std::optional<env_settings> get_settings(
        const std::string& name_or_path);

    /**
     * Print a list of all available presets grouped by category.
     */
    void list(std::string_view tool_name, std::ostream& os = std::cout);

    /**
     * Print detailed information about a specific preset.
     * @return true if preset was found and printed.
     */
    bool explain(std::string_view preset_name, std::string_view tool_name,
                 std::ostream& os = std::cout);

    /**
     * Generate a tree-formatted description of a preset from its cached JSON.
     */
    [[nodiscard]] std::string describe(std::string_view preset_name);

private:
    std::optional<preset_info>                find(const std::string& name_or_path);
    const std::map<std::string, preset_info>& all();
    std::optional<preset_info>                load_file(const std::string& filepath);
    std::string resolve_filepath(const std::string& name_or_path);
    void        ensure_all_loaded();
    void        load_embedded();

    std::string                           m_directory;
    std::map<std::string, preset_info>    m_presets;
    std::map<std::string, nlohmann::json> m_json_cache;
    bool                                  m_all_loaded = false;
};

}  // namespace rocprofsys
