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
class preset_registry
{
public:
    struct preset_info
    {
        std::string                        name;
        std::string                        cli_flag;
        std::string                        description;
        std::string                        use_case;
        std::string                        category;
        std::map<std::string, std::string> settings;  // Resolved ROCPROFSYS_* env vars
    };

    preset_registry();

    /**
     * The discovered preset directory (empty if not found).
     */
    [[nodiscard]] const std::string& directory() const noexcept { return m_directory; }

    /**
     * Find a preset by name or file path.  Results are cached.
     * @return Pointer to cached preset_info, or nullptr if not found.
     */
    [[nodiscard]] const preset_info* find(const std::string& name_or_path);

    /**
     * Load and return all available presets.
     * Cached after first call.
     */
    [[nodiscard]] const std::map<std::string, preset_info>& all();

    /**
     * Get the raw parsed JSON for a preset by name.  Cached.
     * @return Pointer to cached JSON, or nullptr if not found.
     */
    [[nodiscard]] const nlohmann::json* raw_json(const std::string& name);

    /**
     * Apply a preset's settings using a caller-provided callback.
     * @return true if the preset was found and applied.
     */
    template <typename EnvUpdaterFn>
    bool apply(const std::string& name_or_path, EnvUpdaterFn&& update_fn)
    {
        const auto* info = find(name_or_path);
        if(!info) return false;
        for(const auto& [key, val] : info->settings)
            update_fn(key, val);
        return true;
    }

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
     * Generate a tree-formatted description of a preset from its JSON.
     * Uses the raw_json() cache to avoid re-reading files.
     */
    [[nodiscard]] std::string describe(std::string_view preset_name);

private:
    std::optional<preset_info> load_file(const std::string& filepath);
    std::string                resolve_filepath(const std::string& name_or_path);
    void                       ensure_all_loaded();

    std::string                           m_directory;
    std::map<std::string, preset_info>    m_presets;
    std::map<std::string, nlohmann::json> m_json_cache;
    bool                                  m_all_loaded = false;
};

}  // namespace rocprofsys
