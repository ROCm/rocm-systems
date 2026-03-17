// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <nlohmann/json.hpp>

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

}  // namespace json_config
}  // namespace rocprofsys
