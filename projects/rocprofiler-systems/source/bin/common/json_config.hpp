// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "common/env_vars.hpp"

#include <nlohmann/json.hpp>

#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace rocprofsys
{
namespace json_config
{

// ============================================================================
// JSON value conversion
// ============================================================================

/**
 * Converts a JSON value to a string suitable for environment variable assignment.
 */
[[nodiscard]] std::string
json_value_to_string(const nlohmann::json& val);

/**
 * Gets the "value" field from a JSON object, or returns "enabled" bool if present.
 * Handles schema format: { "value": X } or { "enabled": true/false }
 */
[[nodiscard]] std::optional<std::string>
extract_setting_value(const nlohmann::json& obj);

// ============================================================================
// Resolve helpers (JSON -> env var map)
// ============================================================================

/**
 * Set result[env_var] from a JSON "enabled" bool field.
 */
void
resolve_enabled(std::map<std::string, std::string>& result, const nlohmann::json& section,
                std::string_view json_key, std::string_view env_var);

/**
 * Set result[env_var] from a JSON value/enabled field (uses extract_setting_value).
 */
void
resolve_value(std::map<std::string, std::string>& result, const nlohmann::json& section,
              std::string_view json_key, std::string_view env_var);

/**
 * Resolves the new schema-based JSON format into a flat map of ROCPROFSYS_* env vars.
 *
 * Handles the hierarchical schema format with sections:
 *   - domains.gpu, domains.rocm, domains.cpu, domains.parallel
 *   - sampling, tracing, profiling, output
 *   - causal, hardware_counters, advanced
 *
 * @return Resolved map of env var name to string value.
 */
[[nodiscard]] std::map<std::string, std::string>
resolve_schema_config(const nlohmann::json& j);

/**
 * Resolves a JSON config into a flat map of ROCPROFSYS_* env vars.
 * Uses the schema-based format with sections like metadata, domains, tracing, etc.
 */
[[nodiscard]] std::map<std::string, std::string>
resolve_config(const nlohmann::json& j);

/**
 * Loads a JSON file and resolves it to environment variables.
 * Auto-detects format (schema-based vs legacy).
 */
[[nodiscard]] std::optional<std::map<std::string, std::string>>
load_and_resolve(const std::string& filepath);

// ============================================================================
// Config metadata
// ============================================================================

/**
 * Returns the metadata from a preset/config JSON file.
 */
struct config_metadata
{
    std::string name;
    std::string description;
    std::string use_case;
    std::string category;
    std::string cli_flag;
};

[[nodiscard]] std::optional<config_metadata>
get_config_metadata(const nlohmann::json& j);

[[nodiscard]] std::optional<config_metadata>
load_config_metadata(const std::string& filepath);

// ============================================================================
// Domain Flag Shorthand Expansion
// ============================================================================

/**
 * Expands ROCm domain shorthand names to full domain names.
 * E.g., "hip" -> "hip_runtime_api", "kernel" -> "kernel_dispatch"
 */
[[nodiscard]] std::string
expand_rocm_domain_shorthand(const std::string& shorthand);

/**
 * Expands a comma-separated list of ROCm domain shorthand names.
 */
[[nodiscard]] std::string
expand_rocm_domains(const std::string& domains_str);

/**
 * Expands parallel runtime shorthand names to env var suffixes.
 * Returns a map of ROCPROFSYS_USE_* env vars to enable.
 */
[[nodiscard]] std::map<std::string, std::string>
expand_parallel_runtimes(const std::string& runtimes_str);

/**
 * Expands GPU metrics shorthand. Empty or "all" means default metrics.
 */
[[nodiscard]] std::string
expand_gpu_metrics(const std::string& metrics_str);

// ============================================================================
// Safe numeric conversion
// ============================================================================

/**
 * Safely converts a string to int, returning std::nullopt on failure.
 */
[[nodiscard]] std::optional<int>
safe_stoi(const std::string& s);

/**
 * Safely converts a string to double, returning std::nullopt on failure.
 */
[[nodiscard]] std::optional<double>
safe_stod(const std::string& s);

/**
 * Sets a JSON field to a numeric int value if conversion succeeds,
 * otherwise stores as a string to avoid data loss.
 */
void
set_json_int(nlohmann::json& target, const std::string& value);

/**
 * Sets a JSON field to a numeric double value if conversion succeeds,
 * otherwise stores as a string to avoid data loss.
 */
void
set_json_double(nlohmann::json& target, const std::string& value);

// ============================================================================
// Export helpers (env var map -> JSON schema)
// ============================================================================

/**
 * Checks if a string represents a truthy boolean value.
 */
[[nodiscard]] bool
is_truthy(const std::string& v);

/**
 * Export an "enabled" bool field from env vars to JSON.
 */
void
export_enabled(nlohmann::json& j, const std::map<std::string, std::string>& env_map,
               std::string_view env_var, const std::string& json_path_section,
               const std::string& json_path_key);

/**
 * Export a top-level "enabled" field (directly under a section, not nested in a subkey).
 */
void
export_section_enabled(nlohmann::json&                           j,
                       const std::map<std::string, std::string>& env_map,
                       std::string_view env_var, const std::string& json_path_section);

/**
 * Export a string value field from env vars to JSON.
 */
void
export_string_value(nlohmann::json& j, const std::map<std::string, std::string>& env_map,
                    std::string_view env_var, const std::string& json_path_section,
                    const std::string& json_path_key);

/**
 * Export an int value field from env vars to JSON.
 */
void
export_int_value(nlohmann::json& j, const std::map<std::string, std::string>& env_map,
                 std::string_view env_var, const std::string& json_path_section,
                 const std::string& json_path_key);

/**
 * Export a double value field from env vars to JSON.
 */
void
export_double_value(nlohmann::json& j, const std::map<std::string, std::string>& env_map,
                    std::string_view env_var, const std::string& json_path_section,
                    const std::string& json_path_key);

/**
 * Converts a map of ROCPROFSYS_* env vars back to JSON schema format.
 * This allows exporting the resolved configuration for reuse.
 */
[[nodiscard]] nlohmann::json
env_vars_to_json_schema(const std::map<std::string, std::string>& env_map);

/**
 * Exports the configuration as a formatted JSON string.
 */
[[nodiscard]] std::string
export_config_as_json(const std::map<std::string, std::string>& env_vars,
                      const std::string& preset_name = "", int indent = 4);

}  // namespace json_config
}  // namespace rocprofsys
