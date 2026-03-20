// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "common/preset_loader.hpp"

#include <nlohmann/json.hpp>

#include <initializer_list>
#include <map>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

    namespace rocprofsys
{
    namespace common_utils
    {

    // ============================================================================
    // Output directory utilities
    // ============================================================================

    /**
     * Gets the output directory from environment or returns default.
     */
    std::string get_output_directory(const char* env_var = "ROCPROFSYS_OUTPUT_PATH");

    /**
     * Checks if a directory is writable.
     */
    bool check_directory_writable(const std::string& dir);

    // ============================================================================
    // Preset description and display
    // ============================================================================

    /**
     * Generates a tree-formatted description of a preset by reading its JSON data.
     * This avoids hardcoding descriptions that can diverge from the actual preset config.
     */
    std::string generate_preset_description(std::string_view preset_mode);

    /**
     * Prints pre-execution information including preset details and output location.
     */
    void print_pre_execution_info(std::string_view tool_name,
                                  std::string_view preset_mode = "");

    // ============================================================================
    // Validation functions
    // ============================================================================

    /**
     * Collects active preset flags from an argument parser.
     * Template function to work with any parser type.
     */
    template <typename ParserT>
    std::vector<std::string> collect_active_presets(
        ParserT& parser, std::initializer_list<const char*> preset_names)
    {
        std::vector<std::string> active_presets;
        for(const auto* name : preset_names)
        {
            if(parser.exists(name) && parser.template get<bool>(name))
            {
                active_presets.emplace_back(std::string("--") + name);
            }
        }
        return active_presets;
    }

    /**
     * Validates that only one preset mode is specified.
     */
    bool validate_preset_modes(const std::vector<std::string>& active_presets);

    /**
     * Checks if ROCm is available on the system.
     */
    bool check_rocm_available();

    /**
     * Warns if ROCm is not available.
     */
    void warn_if_rocm_unavailable();

    /**
     * Warns if a GPU preset is used without ROCm available.
     */
    void warn_if_gpu_preset_without_rocm(const std::vector<std::string>& active_presets);

    /**
     * Warns if the output directory is not writable.
     */
    void warn_if_output_not_writable(std::string_view tool_name);

    /**
     * Validates the configuration for common issues.
     */
    void validate_configuration(std::string_view tool_name);

    // ============================================================================
    // Preset listing and explanation
    // ============================================================================

    /**
     * Print a list of all available presets grouped by category.
     * @param tool_name The name of the tool (e.g., "run", "sample") for usage message
     */
    void list_presets(std::string_view tool_name);

    /**
     * Print detailed information about a specific preset.
     * @param preset_name The name of the preset to explain
     * @param tool_name The name of the tool (e.g., "run", "sample") for usage message
     * @return true if preset was found and printed, false otherwise
     */
    bool explain_preset(std::string_view preset_name, std::string_view tool_name);

    /**
     * Validate domain flag combinations and print warnings for potential conflicts.
     * @param gpu_enabled Whether --gpu flag was used
     * @param rocm_enabled Whether --rocm flag was used
     * @param cpu_enabled Whether --cpu flag was used
     * @param parallel_enabled Whether --parallel flag was used
     * @param preset_name The active preset name (empty if none)
     */
    void validate_domain_flags(bool gpu_enabled, bool rocm_enabled, bool cpu_enabled,
                               bool parallel_enabled, std::string_view preset_name);

    // ============================================================================
    // Shared functions used by both rocprof-sys-run and rocprof-sys-sample
    // ============================================================================

    /**
     * Collect resolved ROCPROFSYS_* settings by comparing current env against initial
     * env. Returns only variables that were added or changed.
     */
    std::map<std::string, std::string> collect_resolved_settings(
        const std::vector<char*>&              current_env,
        const std::unordered_set<std::string>& initial_envs);

    /**
     * Export configuration to JSON file or stdout.
     */
    void export_config(const std::vector<char*>&              current_env,
                       const std::unordered_set<std::string>& initial_envs,
                       const std::string&                     preset_name,
                       const std::string&                     output_file = "");

    /**
     * Apply a preset's settings using a caller-provided env update function.
     * The callback signature is: void(const std::string& key, const std::string& val)
     */
    template <typename EnvUpdaterFn>
    bool apply_preset_from_json(std::string_view preset_name, EnvUpdaterFn&& update_fn)
    {
        auto info =
            rocprofsys::preset_loader::load_preset_or_file(std::string{ preset_name });
        if(!info) return false;

        for(const auto& [key, val] : info->settings)
        {
            update_fn(key, val);
        }
        return true;
    }

    /**
     * Run the shared post-parse validation sequence.
     * Called by both run and sample after argument parsing.
     */
    void run_post_parse_validation(std::string_view tool_name,
                                   std::string_view preset_name, bool gpu_enabled,
                                   bool rocm_enabled, bool cpu_enabled,
                                   bool parallel_enabled, int verbose_level);

    }  // namespace common_utils
}  // namespace rocprofsys
