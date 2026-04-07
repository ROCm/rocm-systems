// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <nlohmann/json.hpp>

#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace rocprofsys
{
namespace json_config
{

[[nodiscard]] std::string
json_value_to_string(const nlohmann::json& val);

/**
 * Resolves a JSON config into a flat map of ROCPROFSYS_* env vars.
 */
[[nodiscard]] std::map<std::string, std::string>
resolve_config(const nlohmann::json& config);

struct config_metadata
{
    std::string name;
    std::string description;
    std::string use_case;
    std::string category;
    std::string cli_flag;
};

[[nodiscard]] std::optional<config_metadata>
get_config_metadata(const nlohmann::json& config);

/**
 * Expands a comma-separated list of ROCm domain shorthand names.
 * E.g., "hip,kernel" -> "hip_runtime_api,kernel_dispatch"
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

/**
 * Converts a map of ROCPROFSYS_* env vars back to JSON schema format.
 */
[[nodiscard]] nlohmann::json
env_vars_to_json_schema(const std::map<std::string, std::string>& env_map);

/**
 * Exports the configuration as a formatted JSON string.
 */
[[nodiscard]] std::string
export_config_as_json(const std::map<std::string, std::string>& env_vars,
                      const std::string&                        preset_name = "",
                      std::string_view tool_name = "", int indent = 4);

}  // namespace json_config
}  // namespace rocprofsys
