// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <nlohmann/json.hpp>

#include <charconv>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{
namespace json_config
{

/**
 * Converts a JSON value to a string suitable for environment variable assignment.
 */
[[nodiscard]] inline std::string
json_value_to_string(const nlohmann::json& val)
{
    if(val.is_string())
        return val.get<std::string>();
    else if(val.is_boolean())
        return val.get<bool>() ? "true" : "false";
    else if(val.is_number_integer())
        return std::to_string(val.get<int64_t>());
    else if(val.is_number_float())
        return std::to_string(val.get<double>());
    else if(val.is_array())
    {
        std::string result;
        for(const auto& item : val)
        {
            if(!result.empty()) result += ',';
            result += json_value_to_string(item);
        }
        return result;
    }
    return val.dump();
}

/**
 * Gets the "value" field from a JSON object, or returns "enabled" bool if present.
 * Handles schema format: { "value": X } or { "enabled": true/false }
 */
[[nodiscard]] inline std::optional<std::string>
extract_setting_value(const nlohmann::json& obj)
{
    if(obj.is_object())
    {
        if(obj.contains("value")) return json_value_to_string(obj["value"]);
        if(obj.contains("enabled")) return obj["enabled"].get<bool>() ? "true" : "false";
    }
    else if(obj.is_boolean())
    {
        return obj.get<bool>() ? "true" : "false";
    }
    else if(!obj.is_null())
    {
        return json_value_to_string(obj);
    }
    return std::nullopt;
}

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
[[nodiscard]] inline std::map<std::string, std::string>
resolve_schema_config(const nlohmann::json& j)
{
    std::map<std::string, std::string> result;

    // --- Tracing section ---
    if(j.contains("tracing"))
    {
        const auto& tracing = j["tracing"];
        if(tracing.contains("enabled"))
            result["ROCPROFSYS_TRACE"] =
                tracing["enabled"].get<bool>() ? "true" : "false";
        if(tracing.contains("buffer_size_kb"))
        {
            if(auto val = extract_setting_value(tracing["buffer_size_kb"]))
                result["ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB"] = *val;
        }
        if(tracing.contains("fill_policy"))
        {
            if(auto val = extract_setting_value(tracing["fill_policy"]))
                result["ROCPROFSYS_PERFETTO_FILL_POLICY"] = *val;
        }
    }

    // --- Profiling section ---
    if(j.contains("profiling"))
    {
        const auto& profiling = j["profiling"];
        if(profiling.contains("enabled"))
            result["ROCPROFSYS_PROFILE"] =
                profiling["enabled"].get<bool>() ? "true" : "false";
        if(profiling.contains("flat_profile"))
        {
            const auto& flat = profiling["flat_profile"];
            if(flat.contains("enabled"))
                result["ROCPROFSYS_FLAT_PROFILE"] =
                    flat["enabled"].get<bool>() ? "true" : "false";
        }
    }

    // --- Sampling section ---
    if(j.contains("sampling"))
    {
        const auto& sampling = j["sampling"];
        if(sampling.contains("enabled"))
            result["ROCPROFSYS_USE_SAMPLING"] =
                sampling["enabled"].get<bool>() ? "true" : "false";
        if(sampling.contains("timer"))
        {
            if(auto val = extract_setting_value(sampling["timer"]))
                result["ROCPROFSYS_SAMPLING_TIMER"] = *val;
        }
        if(sampling.contains("frequency_hz"))
        {
            if(auto val = extract_setting_value(sampling["frequency_hz"]))
                result["ROCPROFSYS_SAMPLING_FREQ"] = *val;
        }
        if(sampling.contains("delay_sec"))
        {
            if(auto val = extract_setting_value(sampling["delay_sec"]))
                result["ROCPROFSYS_SAMPLING_DELAY"] = *val;
        }
        if(sampling.contains("duration_sec"))
        {
            if(auto val = extract_setting_value(sampling["duration_sec"]))
                result["ROCPROFSYS_SAMPLING_DURATION"] = *val;
        }
        if(sampling.contains("cpus"))
        {
            if(auto val = extract_setting_value(sampling["cpus"]))
                result["ROCPROFSYS_SAMPLING_CPUS"] = *val;
        }
        if(sampling.contains("gpus"))
        {
            if(auto val = extract_setting_value(sampling["gpus"]))
                result["ROCPROFSYS_SAMPLING_GPUS"] = *val;
        }
    }

    // --- Domains section ---
    if(j.contains("domains"))
    {
        const auto& domains = j["domains"];

        // GPU domain (AMD SMI metrics)
        if(domains.contains("gpu"))
        {
            const auto& gpu = domains["gpu"];
            if(gpu.contains("enabled") && gpu["enabled"].get<bool>())
            {
                result["ROCPROFSYS_USE_AMD_SMI"]          = "true";
                result["ROCPROFSYS_USE_PROCESS_SAMPLING"] = "true";

                // Collect enabled metrics
                if(gpu.contains("metrics"))
                {
                    std::vector<std::string> enabled_metrics;
                    const auto&              metrics = gpu["metrics"];
                    for(const auto& [name, metric] : metrics.items())
                    {
                        if(metric.is_object() && metric.contains("enabled") &&
                           metric["enabled"].get<bool>())
                        {
                            enabled_metrics.push_back(name);
                        }
                    }
                    if(!enabled_metrics.empty())
                    {
                        std::string metrics_str;
                        for(const auto& m : enabled_metrics)
                        {
                            if(!metrics_str.empty()) metrics_str += ',';
                            metrics_str += m;
                        }
                        result["ROCPROFSYS_AMD_SMI_METRICS"] = metrics_str;
                    }
                }

                if(gpu.contains("sampling_rate_hz"))
                {
                    if(auto val = extract_setting_value(gpu["sampling_rate_hz"]))
                        result["ROCPROFSYS_AMD_SMI_FREQ"] = *val;
                }
            }
        }

        // ROCm domain (API tracing)
        if(domains.contains("rocm"))
        {
            const auto& rocm = domains["rocm"];
            if(rocm.contains("enabled") && rocm["enabled"].get<bool>())
            {
                // Top-level rocm.enabled ensures tracing is on and default domains set
                if(result.find("ROCPROFSYS_TRACE") == result.end())
                    result["ROCPROFSYS_TRACE"] = "true";
            }
            if(rocm.contains("api_domains"))
            {
                std::vector<std::string> enabled_apis;
                const auto&              api_domains = rocm["api_domains"];
                for(const auto& [name, api] : api_domains.items())
                {
                    if(api.is_object() && api.contains("enabled") &&
                       api["enabled"].get<bool>())
                    {
                        enabled_apis.push_back(name);
                    }
                }
                if(!enabled_apis.empty())
                {
                    std::string apis_str;
                    for(const auto& a : enabled_apis)
                    {
                        if(!apis_str.empty()) apis_str += ',';
                        apis_str += a;
                    }
                    result["ROCPROFSYS_ROCM_DOMAINS"] = apis_str;
                }
            }
            if(rocm.contains("group_by_queue"))
            {
                const auto& gbq = rocm["group_by_queue"];
                if(gbq.contains("enabled"))
                    result["ROCPROFSYS_ROCM_GROUP_BY_QUEUE"] =
                        gbq["enabled"].get<bool>() ? "true" : "false";
            }
        }

        // CPU domain
        if(domains.contains("cpu"))
        {
            const auto& cpu = domains["cpu"];
            if(cpu.contains("enabled") && cpu["enabled"].get<bool>())
            {
                result["ROCPROFSYS_USE_PROCESS_SAMPLING"] = "true";
                if(cpu.contains("metrics"))
                {
                    const auto& metrics = cpu["metrics"];
                    if(metrics.contains("freq") && metrics["freq"].contains("enabled") &&
                       metrics["freq"]["enabled"].get<bool>())
                    {
                        result["ROCPROFSYS_CPU_FREQ"] = "true";
                    }
                }
            }
        }

        // Parallel runtimes domain
        if(domains.contains("parallel"))
        {
            const auto& parallel = domains["parallel"];
            if(parallel.contains("runtimes"))
            {
                const auto& runtimes = parallel["runtimes"];
                if(runtimes.contains("mpi") && runtimes["mpi"].contains("enabled") &&
                   runtimes["mpi"]["enabled"].get<bool>())
                {
                    result["ROCPROFSYS_USE_MPIP"] = "true";
                }
                if(runtimes.contains("openmp") &&
                   runtimes["openmp"].contains("enabled") &&
                   runtimes["openmp"]["enabled"].get<bool>())
                {
                    result["ROCPROFSYS_USE_OMPT"] = "true";
                }
                if(runtimes.contains("kokkos") &&
                   runtimes["kokkos"].contains("enabled") &&
                   runtimes["kokkos"]["enabled"].get<bool>())
                {
                    result["ROCPROFSYS_USE_KOKKOSP"] = "true";
                }
                if(runtimes.contains("rccl") && runtimes["rccl"].contains("enabled") &&
                   runtimes["rccl"]["enabled"].get<bool>())
                {
                    result["ROCPROFSYS_USE_RCCLP"] = "true";
                }
            }
        }
    }

    // --- Output section ---
    if(j.contains("output"))
    {
        const auto& output = j["output"];
        if(output.contains("path"))
        {
            if(auto val = extract_setting_value(output["path"]))
                result["ROCPROFSYS_OUTPUT_PATH"] = *val;
        }
        if(output.contains("time_output"))
        {
            const auto& to = output["time_output"];
            if(to.contains("enabled"))
                result["ROCPROFSYS_TIME_OUTPUT"] =
                    to["enabled"].get<bool>() ? "true" : "false";
        }
        if(output.contains("file_output"))
        {
            const auto& fo = output["file_output"];
            if(fo.contains("enabled"))
                result["ROCPROFSYS_FILE_OUTPUT"] =
                    fo["enabled"].get<bool>() ? "true" : "false";
        }
        if(output.contains("rocpd_output"))
        {
            const auto& rpd = output["rocpd_output"];
            if(rpd.contains("enabled"))
                result["ROCPROFSYS_USE_ROCPD"] =
                    rpd["enabled"].get<bool>() ? "true" : "false";
        }
    }

    // --- Causal profiling section ---
    if(j.contains("causal"))
    {
        const auto& causal = j["causal"];
        if(causal.contains("enabled"))
            result["ROCPROFSYS_USE_CAUSAL"] =
                causal["enabled"].get<bool>() ? "true" : "false";
        if(causal.contains("mode"))
        {
            if(auto val = extract_setting_value(causal["mode"]))
                result["ROCPROFSYS_CAUSAL_MODE"] = *val;
        }
        if(causal.contains("backend"))
        {
            if(auto val = extract_setting_value(causal["backend"]))
                result["ROCPROFSYS_CAUSAL_BACKEND"] = *val;
        }
        if(causal.contains("binary_scope"))
        {
            if(auto val = extract_setting_value(causal["binary_scope"]))
                result["ROCPROFSYS_CAUSAL_BINARY_SCOPE"] = *val;
        }
        if(causal.contains("function_scope"))
        {
            if(auto val = extract_setting_value(causal["function_scope"]))
                result["ROCPROFSYS_CAUSAL_FUNCTION_SCOPE"] = *val;
        }
        if(causal.contains("source_scope"))
        {
            if(auto val = extract_setting_value(causal["source_scope"]))
                result["ROCPROFSYS_CAUSAL_SOURCE_SCOPE"] = *val;
        }
        if(causal.contains("end_to_end"))
        {
            const auto& e2e = causal["end_to_end"];
            if(e2e.contains("enabled"))
                result["ROCPROFSYS_CAUSAL_END_TO_END"] =
                    e2e["enabled"].get<bool>() ? "true" : "false";
        }
    }

    // --- Hardware counters section ---
    if(j.contains("hardware_counters"))
    {
        const auto& hw = j["hardware_counters"];
        if(hw.contains("enabled") && hw["enabled"].get<bool>())
        {
            if(hw.contains("rocm_events"))
            {
                if(auto val = extract_setting_value(hw["rocm_events"]))
                    result["ROCPROFSYS_ROCM_EVENTS"] = *val;
            }
            if(hw.contains("papi_events"))
            {
                if(auto val = extract_setting_value(hw["papi_events"]))
                    result["ROCPROFSYS_PAPI_EVENTS"] = *val;
            }
        }
        if(hw.contains("papi_multiplexing"))
        {
            const auto& pm = hw["papi_multiplexing"];
            if(pm.contains("enabled"))
                result["ROCPROFSYS_PAPI_MULTIPLEXING"] =
                    pm["enabled"].get<bool>() ? "true" : "false";
        }
    }

    // --- Advanced section ---
    if(j.contains("advanced"))
    {
        const auto& adv = j["advanced"];
        if(adv.contains("cpu_affinity"))
        {
            const auto& ca = adv["cpu_affinity"];
            if(ca.contains("enabled"))
                result["ROCPROFSYS_CPU_AFFINITY"] =
                    ca["enabled"].get<bool>() ? "true" : "false";
        }
        if(adv.contains("collapse_threads"))
        {
            const auto& ct = adv["collapse_threads"];
            if(ct.contains("enabled"))
                result["ROCPROFSYS_COLLAPSE_THREADS"] =
                    ct["enabled"].get<bool>() ? "true" : "false";
        }
        if(adv.contains("max_depth"))
        {
            if(auto val = extract_setting_value(adv["max_depth"]))
                result["ROCPROFSYS_MAX_DEPTH"] = *val;
        }
        if(adv.contains("trace_delay_sec"))
        {
            if(auto val = extract_setting_value(adv["trace_delay_sec"]))
                result["ROCPROFSYS_TRACE_DELAY"] = *val;
        }
        if(adv.contains("trace_duration_sec"))
        {
            if(auto val = extract_setting_value(adv["trace_duration_sec"]))
                result["ROCPROFSYS_TRACE_DURATION"] = *val;
        }
        if(adv.contains("verbose"))
        {
            if(auto val = extract_setting_value(adv["verbose"]))
                result["ROCPROFSYS_VERBOSE"] = *val;
        }
        if(adv.contains("debug"))
        {
            const auto& dbg = adv["debug"];
            if(dbg.contains("enabled"))
                result["ROCPROFSYS_DEBUG"] =
                    dbg["enabled"].get<bool>() ? "true" : "false";
        }
        if(adv.contains("timemory_components"))
        {
            if(auto val = extract_setting_value(adv["timemory_components"]))
                result["ROCPROFSYS_TIMEMORY_COMPONENTS"] = *val;
        }
        if(adv.contains("network_interface"))
        {
            if(auto val = extract_setting_value(adv["network_interface"]))
                result["ROCPROFSYS_NETWORK_INTERFACE"] = *val;
        }
        if(adv.contains("trace_periods"))
        {
            if(auto val = extract_setting_value(adv["trace_periods"]))
                result["ROCPROFSYS_TRACE_PERIODS"] = *val;
        }
    }

    return result;
}

/**
 * Resolves a JSON config into a flat map of ROCPROFSYS_* env vars.
 * Uses the schema-based format with sections like metadata, domains, tracing, etc.
 */
[[nodiscard]] inline std::map<std::string, std::string>
resolve_config(const nlohmann::json& j)
{
    return resolve_schema_config(j);
}

/**
 * Loads a JSON file and resolves it to environment variables.
 * Auto-detects format (schema-based vs legacy).
 */
[[nodiscard]] inline std::optional<std::map<std::string, std::string>>
load_and_resolve(const std::string& filepath)
{
    std::ifstream ifs{ filepath };
    if(!ifs.is_open()) return std::nullopt;

    try
    {
        auto j = nlohmann::json::parse(ifs);
        return resolve_config(j);
    } catch(const nlohmann::json::exception& e)
    {
        std::cerr << "[rocprof-sys] WARNING: Failed to parse config '" << filepath
                  << "': " << e.what() << '\n';
        return std::nullopt;
    }
}

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

[[nodiscard]] inline std::optional<config_metadata>
get_config_metadata(const nlohmann::json& j)
{
    if(!j.contains("metadata")) return std::nullopt;

    const auto&     meta = j["metadata"];
    config_metadata result;
    if(meta.contains("name")) result.name = meta["name"].get<std::string>();
    if(meta.contains("description"))
        result.description = meta["description"].get<std::string>();
    if(meta.contains("use_case")) result.use_case = meta["use_case"].get<std::string>();
    if(meta.contains("category")) result.category = meta["category"].get<std::string>();
    if(meta.contains("cli_flag")) result.cli_flag = meta["cli_flag"].get<std::string>();
    return result;
}

[[nodiscard]] inline std::optional<config_metadata>
load_config_metadata(const std::string& filepath)
{
    std::ifstream ifs{ filepath };
    if(!ifs.is_open()) return std::nullopt;

    try
    {
        auto j = nlohmann::json::parse(ifs);
        return get_config_metadata(j);
    } catch(const nlohmann::json::exception&)
    {
        return std::nullopt;
    }
}

// ============================================================================
// Domain Flag Shorthand Expansion
// ============================================================================

/**
 * Expands ROCm domain shorthand names to full domain names.
 * E.g., "hip" -> "hip_runtime_api", "kernel" -> "kernel_dispatch"
 */
[[nodiscard]] inline std::string
expand_rocm_domain_shorthand(const std::string& shorthand)
{
    static const std::map<std::string, std::string> shortcuts = {
        { "hip", "hip_runtime_api" },
        { "hip_runtime", "hip_runtime_api" },
        { "hip_compiler", "hip_compiler_api" },
        { "hsa", "hsa_api" },
        { "kernel", "kernel_dispatch" },
        { "kernels", "kernel_dispatch" },
        { "memory", "memory_copy" },
        { "mem", "memory_copy" },
        { "scratch", "scratch_memory" },
        { "marker", "marker_api" },
        { "roctx", "marker_api" },
        { "rccl", "rccl_api" },
    };

    auto it = shortcuts.find(shorthand);
    if(it != shortcuts.end()) return it->second;
    return shorthand;  // Return as-is if no mapping
}

/**
 * Expands a comma-separated list of ROCm domain shorthand names.
 */
[[nodiscard]] inline std::string
expand_rocm_domains(const std::string& domains_str)
{
    std::string              result;
    std::string              token;
    std::istringstream       ss(domains_str);
    std::vector<std::string> expanded;

    while(std::getline(ss, token, ','))
    {
        // Trim whitespace
        auto start = token.find_first_not_of(" \t");
        auto end   = token.find_last_not_of(" \t");
        if(start != std::string::npos)
        {
            token = token.substr(start, end - start + 1);
            expanded.push_back(expand_rocm_domain_shorthand(token));
        }
    }

    for(const auto& d : expanded)
    {
        if(!result.empty()) result += ',';
        result += d;
    }
    return result;
}

/**
 * Expands parallel runtime shorthand names to env var suffixes.
 * Returns a map of ROCPROFSYS_USE_* env vars to enable.
 */
[[nodiscard]] inline std::map<std::string, std::string>
expand_parallel_runtimes(const std::string& runtimes_str)
{
    std::map<std::string, std::string> result;

    // If empty or "all", enable all runtimes
    if(runtimes_str.empty() || runtimes_str == "all")
    {
        result["ROCPROFSYS_USE_MPIP"]    = "true";
        result["ROCPROFSYS_USE_OMPT"]    = "true";
        result["ROCPROFSYS_USE_KOKKOSP"] = "true";
        result["ROCPROFSYS_USE_RCCLP"]   = "true";
        return result;
    }

    static const std::map<std::string, std::string> shortcuts = {
        { "mpi", "ROCPROFSYS_USE_MPIP" },        { "mpip", "ROCPROFSYS_USE_MPIP" },
        { "openmp", "ROCPROFSYS_USE_OMPT" },     { "ompt", "ROCPROFSYS_USE_OMPT" },
        { "omp", "ROCPROFSYS_USE_OMPT" },        { "kokkos", "ROCPROFSYS_USE_KOKKOSP" },
        { "kokkosp", "ROCPROFSYS_USE_KOKKOSP" }, { "rccl", "ROCPROFSYS_USE_RCCLP" },
        { "rcclp", "ROCPROFSYS_USE_RCCLP" },
    };

    std::string        token;
    std::istringstream ss(runtimes_str);
    while(std::getline(ss, token, ','))
    {
        // Trim and lowercase
        auto start = token.find_first_not_of(" \t");
        auto end   = token.find_last_not_of(" \t");
        if(start != std::string::npos)
        {
            token = token.substr(start, end - start + 1);
            for(auto& c : token)
                c = std::tolower(c);

            auto it = shortcuts.find(token);
            if(it != shortcuts.end()) result[it->second] = "true";
        }
    }
    return result;
}

/**
 * Expands GPU metrics shorthand. Empty or "all" means default metrics.
 */
[[nodiscard]] inline std::string
expand_gpu_metrics(const std::string& metrics_str)
{
    if(metrics_str.empty()) return "";  // Use default

    static const std::map<std::string, std::string> shortcuts = {
        { "temperature", "temp" },
        { "usage", "busy" },
        { "utilization", "busy" },
        { "memory", "mem_usage" },
    };

    std::string              result;
    std::string              token;
    std::istringstream       ss(metrics_str);
    std::vector<std::string> expanded;

    while(std::getline(ss, token, ','))
    {
        auto start = token.find_first_not_of(" \t");
        auto end   = token.find_last_not_of(" \t");
        if(start != std::string::npos)
        {
            token = token.substr(start, end - start + 1);
            for(auto& c : token)
                c = std::tolower(c);

            auto it = shortcuts.find(token);
            expanded.push_back(it != shortcuts.end() ? it->second : token);
        }
    }

    for(const auto& m : expanded)
    {
        if(!result.empty()) result += ',';
        result += m;
    }
    return result;
}

// ============================================================================
// Config Export (env vars -> JSON schema)
// ============================================================================

/**
 * Safely converts a string to int, returning std::nullopt on failure.
 */
[[nodiscard]] inline std::optional<int>
safe_stoi(const std::string& s)
{
    if(s.empty()) return std::nullopt;
    int        value  = 0;
    const auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    if(result.ec != std::errc{}) return std::nullopt;
    return value;
}

/**
 * Safely converts a string to double, returning std::nullopt on failure.
 */
[[nodiscard]] inline std::optional<double>
safe_stod(const std::string& s)
{
    if(s.empty()) return std::nullopt;
    double     value  = 0.0;
    const auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    if(result.ec != std::errc{}) return std::nullopt;
    return value;
}

/**
 * Sets a JSON field to a numeric int value if conversion succeeds,
 * otherwise stores as a string to avoid data loss.
 */
inline void
set_json_int(nlohmann::json& target, const std::string& value)
{
    if(auto n = safe_stoi(value))
        target = *n;
    else
        target = value;
}

/**
 * Sets a JSON field to a numeric double value if conversion succeeds,
 * otherwise stores as a string to avoid data loss.
 */
inline void
set_json_double(nlohmann::json& target, const std::string& value)
{
    if(auto n = safe_stod(value))
        target = *n;
    else
        target = value;
}

/**
 * Converts a map of ROCPROFSYS_* env vars back to JSON schema format.
 * This allows exporting the resolved configuration for reuse.
 */
[[nodiscard]] inline nlohmann::json
env_vars_to_json_schema(const std::map<std::string, std::string>& env_vars)
{
    nlohmann::json j;

    // Helper to check if env var exists and get value
    auto get_val = [&](const std::string& key) -> std::optional<std::string> {
        auto it = env_vars.find(key);
        if(it != env_vars.end()) return it->second;
        return std::nullopt;
    };

    auto is_true = [](const std::string& v) {
        return v == "true" || v == "TRUE" || v == "1" || v == "ON" || v == "on" ||
               v == "yes" || v == "YES";
    };

    // --- Tracing ---
    if(auto v = get_val("ROCPROFSYS_TRACE")) j["tracing"]["enabled"] = is_true(*v);
    if(auto v = get_val("ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB"))
        set_json_int(j["tracing"]["buffer_size_kb"]["value"], *v);
    if(auto v = get_val("ROCPROFSYS_PERFETTO_FILL_POLICY"))
        j["tracing"]["fill_policy"]["value"] = *v;

    // --- Profiling ---
    if(auto v = get_val("ROCPROFSYS_PROFILE")) j["profiling"]["enabled"] = is_true(*v);
    if(auto v = get_val("ROCPROFSYS_FLAT_PROFILE"))
        j["profiling"]["flat_profile"]["enabled"] = is_true(*v);

    // --- Sampling ---
    if(auto v = get_val("ROCPROFSYS_USE_SAMPLING"))
        j["sampling"]["enabled"] = is_true(*v);
    if(auto v = get_val("ROCPROFSYS_SAMPLING_FREQ"))
        set_json_int(j["sampling"]["frequency_hz"]["value"], *v);
    if(auto v = get_val("ROCPROFSYS_SAMPLING_TIMER"))
        j["sampling"]["timer"]["value"] = *v;
    if(auto v = get_val("ROCPROFSYS_SAMPLING_DELAY"))
        set_json_double(j["sampling"]["delay_sec"]["value"], *v);
    if(auto v = get_val("ROCPROFSYS_SAMPLING_DURATION"))
        set_json_double(j["sampling"]["duration_sec"]["value"], *v);
    if(auto v = get_val("ROCPROFSYS_SAMPLING_CPUS")) j["sampling"]["cpus"]["value"] = *v;
    if(auto v = get_val("ROCPROFSYS_SAMPLING_GPUS")) j["sampling"]["gpus"]["value"] = *v;

    // --- Domains: GPU ---
    if(auto v = get_val("ROCPROFSYS_USE_AMD_SMI"))
    {
        j["domains"]["gpu"]["enabled"] = is_true(*v);
        if(auto metrics = get_val("ROCPROFSYS_AMD_SMI_METRICS"))
        {
            std::istringstream ss(*metrics);
            std::string        token;
            while(std::getline(ss, token, ','))
            {
                auto start = token.find_first_not_of(" \t");
                auto end   = token.find_last_not_of(" \t");
                if(start != std::string::npos)
                {
                    token = token.substr(start, end - start + 1);
                    j["domains"]["gpu"]["metrics"][token]["enabled"] = true;
                }
            }
        }
        if(auto freq = get_val("ROCPROFSYS_AMD_SMI_FREQ"))
            set_json_int(j["domains"]["gpu"]["sampling_rate_hz"]["value"], *freq);
    }

    // --- Domains: ROCm ---
    if(auto v = get_val("ROCPROFSYS_ROCM_DOMAINS"))
    {
        std::istringstream ss(*v);
        std::string        token;
        while(std::getline(ss, token, ','))
        {
            auto start = token.find_first_not_of(" \t");
            auto end   = token.find_last_not_of(" \t");
            if(start != std::string::npos)
            {
                token = token.substr(start, end - start + 1);
                j["domains"]["rocm"]["api_domains"][token]["enabled"] = true;
            }
        }
    }

    // --- Domains: CPU ---
    if(auto v = get_val("ROCPROFSYS_USE_PROCESS_SAMPLING"))
    {
        if(is_true(*v))
        {
            if(auto freq = get_val("ROCPROFSYS_CPU_FREQ"))
            {
                if(is_true(*freq))
                {
                    j["domains"]["cpu"]["enabled"]                    = true;
                    j["domains"]["cpu"]["metrics"]["freq"]["enabled"] = true;
                }
            }
        }
    }

    // --- Domains: Parallel ---
    if(auto v = get_val("ROCPROFSYS_USE_MPIP"))
        j["domains"]["parallel"]["runtimes"]["mpi"]["enabled"] = is_true(*v);
    if(auto v = get_val("ROCPROFSYS_USE_OMPT"))
        j["domains"]["parallel"]["runtimes"]["openmp"]["enabled"] = is_true(*v);
    if(auto v = get_val("ROCPROFSYS_USE_KOKKOSP"))
        j["domains"]["parallel"]["runtimes"]["kokkos"]["enabled"] = is_true(*v);
    if(auto v = get_val("ROCPROFSYS_USE_RCCLP"))
        j["domains"]["parallel"]["runtimes"]["rccl"]["enabled"] = is_true(*v);

    // --- Output ---
    if(auto v = get_val("ROCPROFSYS_OUTPUT_PATH")) j["output"]["path"]["value"] = *v;
    if(auto v = get_val("ROCPROFSYS_TIME_OUTPUT"))
        j["output"]["time_output"]["enabled"] = is_true(*v);
    if(auto v = get_val("ROCPROFSYS_FILE_OUTPUT"))
        j["output"]["file_output"]["enabled"] = is_true(*v);
    if(auto v = get_val("ROCPROFSYS_USE_ROCPD"))
        j["output"]["rocpd_output"]["enabled"] = is_true(*v);

    // --- Hardware counters ---
    if(auto v = get_val("ROCPROFSYS_ROCM_EVENTS"))
    {
        j["hardware_counters"]["enabled"]              = true;
        j["hardware_counters"]["rocm_events"]["value"] = *v;
    }
    if(auto v = get_val("ROCPROFSYS_PAPI_EVENTS"))
    {
        j["hardware_counters"]["enabled"]              = true;
        j["hardware_counters"]["papi_events"]["value"] = *v;
    }

    // --- Advanced ---
    if(auto v = get_val("ROCPROFSYS_VERBOSE"))
        set_json_int(j["advanced"]["verbose"]["value"], *v);
    if(auto v = get_val("ROCPROFSYS_DEBUG"))
        j["advanced"]["debug"]["enabled"] = is_true(*v);
    if(auto v = get_val("ROCPROFSYS_MAX_DEPTH"))
        set_json_int(j["advanced"]["max_depth"]["value"], *v);
    if(auto v = get_val("ROCPROFSYS_TRACE_DELAY"))
        set_json_double(j["advanced"]["trace_delay_sec"]["value"], *v);
    if(auto v = get_val("ROCPROFSYS_TRACE_DURATION"))
        set_json_double(j["advanced"]["trace_duration_sec"]["value"], *v);
    if(auto v = get_val("ROCPROFSYS_CPU_AFFINITY"))
        j["advanced"]["cpu_affinity"]["enabled"] = is_true(*v);
    if(auto v = get_val("ROCPROFSYS_COLLAPSE_THREADS"))
        j["advanced"]["collapse_threads"]["enabled"] = is_true(*v);
    if(auto v = get_val("ROCPROFSYS_TIMEMORY_COMPONENTS"))
        j["advanced"]["timemory_components"]["value"] = *v;
    if(auto v = get_val("ROCPROFSYS_NETWORK_INTERFACE"))
        j["advanced"]["network_interface"]["value"] = *v;
    if(auto v = get_val("ROCPROFSYS_TRACE_PERIODS"))
        j["advanced"]["trace_periods"]["value"] = *v;

    // --- Hardware counters: papi_multiplexing ---
    if(auto v = get_val("ROCPROFSYS_PAPI_MULTIPLEXING"))
        j["hardware_counters"]["papi_multiplexing"]["enabled"] = is_true(*v);

    return j;
}

/**
 * Exports the configuration as a formatted JSON string.
 */
[[nodiscard]] inline std::string
export_config_as_json(const std::map<std::string, std::string>& env_vars,
                      const std::string& preset_name = "", int indent = 4)
{
    auto j = env_vars_to_json_schema(env_vars);

    // Add metadata if preset name provided
    if(!preset_name.empty())
    {
        j["metadata"]["name"]        = preset_name;
        j["metadata"]["description"] = "Exported configuration from rocprof-sys-run";
    }

    return j.dump(indent);
}

}  // namespace json_config
}  // namespace rocprofsys
