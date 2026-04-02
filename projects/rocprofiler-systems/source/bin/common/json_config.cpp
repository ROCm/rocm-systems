// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "common/json_config.hpp"

#include "common/env_vars.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <iostream>
#include <sstream>

namespace rocprofsys
{
namespace json_config
{

std::string
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

std::optional<std::string>
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

void
resolve_enabled(std::map<std::string, std::string>& result, const nlohmann::json& section,
                std::string_view json_key, std::string_view env_var)
{
    if(section.contains(json_key))
        result[std::string{ env_var }] = section[json_key].get<bool>() ? "true" : "false";
}

void
resolve_value(std::map<std::string, std::string>& result, const nlohmann::json& section,
              std::string_view json_key, std::string_view env_var)
{
    if(section.contains(json_key))
    {
        if(auto val = extract_setting_value(section[json_key]))
            result[std::string{ env_var }] = *val;
    }
}

std::map<std::string, std::string>
resolve_schema_config(const nlohmann::json& j)
{
    std::map<std::string, std::string> result;

    // --- Tracing section ---
    if(j.contains("tracing"))
    {
        const auto& tracing = j["tracing"];
        resolve_enabled(result, tracing, "enabled", env_vars::TRACE);
        if(tracing.contains("legacy"))
            resolve_enabled(result, tracing["legacy"], "enabled", env_vars::TRACE_LEGACY);
        resolve_value(result, tracing, "buffer_size_kb",
                      env_vars::PERFETTO_BUFFER_SIZE_KB);
        resolve_value(result, tracing, "fill_policy", env_vars::PERFETTO_FILL_POLICY);
    }

    // --- Profiling section ---
    if(j.contains("profiling"))
    {
        const auto& profiling = j["profiling"];
        resolve_enabled(result, profiling, "enabled", env_vars::PROFILE);
        if(profiling.contains("flat_profile"))
        {
            resolve_enabled(result, profiling["flat_profile"], "enabled",
                            env_vars::FLAT_PROFILE);
        }
    }

    // --- Sampling section ---
    if(j.contains("sampling"))
    {
        const auto& sampling = j["sampling"];
        resolve_enabled(result, sampling, "enabled", env_vars::USE_SAMPLING);
        resolve_value(result, sampling, "timer", env_vars::SAMPLING_TIMER);
        resolve_value(result, sampling, "frequency_hz", env_vars::SAMPLING_FREQ);
        resolve_value(result, sampling, "delay_sec", env_vars::SAMPLING_DELAY);
        resolve_value(result, sampling, "duration_sec", env_vars::SAMPLING_DURATION);
        resolve_value(result, sampling, "cpus", env_vars::SAMPLING_CPUS);
        resolve_value(result, sampling, "gpus", env_vars::SAMPLING_GPUS);
        resolve_value(result, sampling, "ainics", env_vars::SAMPLING_AINICS);
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
                result[std::string{ env_vars::USE_AMD_SMI }]          = "true";
                result[std::string{ env_vars::USE_PROCESS_SAMPLING }] = "true";

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
                        result[std::string{ env_vars::AMD_SMI_METRICS }] = metrics_str;
                    }
                }

                resolve_value(result, gpu, "sampling_rate_hz", env_vars::AMD_SMI_FREQ);
                resolve_value(result, gpu, "process_sampling_freq",
                              env_vars::PROCESS_SAMPLING_FREQ);
                if(gpu.contains("ainic"))
                    resolve_enabled(result, gpu["ainic"], "enabled", env_vars::USE_AINIC);
            }
        }

        // ROCm domain (API tracing)
        if(domains.contains("rocm"))
        {
            const auto& rocm = domains["rocm"];
            if(rocm.contains("enabled") && rocm["enabled"].get<bool>())
            {
                // Top-level rocm.enabled ensures tracing is on and default domains set
                if(result.find(std::string{ env_vars::TRACE }) == result.end())
                    result[std::string{ env_vars::TRACE }] = "true";
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
                    result[std::string{ env_vars::ROCM_DOMAINS }] = apis_str;
                }
            }
            if(rocm.contains("group_by_queue"))
            {
                resolve_enabled(result, rocm["group_by_queue"], "enabled",
                                env_vars::ROCM_GROUP_BY_QUEUE);
            }
        }

        // CPU domain
        if(domains.contains("cpu"))
        {
            const auto& cpu = domains["cpu"];
            if(cpu.contains("enabled") && cpu["enabled"].get<bool>())
            {
                result[std::string{ env_vars::USE_PROCESS_SAMPLING }] = "true";
                if(cpu.contains("metrics"))
                {
                    const auto& metrics = cpu["metrics"];
                    if(metrics.contains("freq") && metrics["freq"].contains("enabled") &&
                       metrics["freq"]["enabled"].get<bool>())
                    {
                        result[std::string{ env_vars::CPU_FREQ }] = "true";
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
                    result[std::string{ env_vars::USE_MPIP }] = "true";
                }
                if(runtimes.contains("openmp") &&
                   runtimes["openmp"].contains("enabled") &&
                   runtimes["openmp"]["enabled"].get<bool>())
                {
                    result[std::string{ env_vars::USE_OMPT }] = "true";
                }
                if(runtimes.contains("kokkos") &&
                   runtimes["kokkos"].contains("enabled") &&
                   runtimes["kokkos"]["enabled"].get<bool>())
                {
                    result[std::string{ env_vars::USE_KOKKOSP }] = "true";
                }
                if(runtimes.contains("rccl") && runtimes["rccl"].contains("enabled") &&
                   runtimes["rccl"]["enabled"].get<bool>())
                {
                    result[std::string{ env_vars::USE_RCCLP }] = "true";
                }
                if(runtimes.contains("shmem") && runtimes["shmem"].contains("enabled") &&
                   runtimes["shmem"]["enabled"].get<bool>())
                {
                    result[std::string{ env_vars::USE_SHMEM }] = "true";
                }
                if(runtimes.contains("ucx") && runtimes["ucx"].contains("enabled") &&
                   runtimes["ucx"]["enabled"].get<bool>())
                {
                    result[std::string{ env_vars::USE_UCX }] = "true";
                }
            }
        }
    }

    // --- Output section ---
    if(j.contains("output"))
    {
        const auto& output = j["output"];
        resolve_value(result, output, "path", env_vars::OUTPUT_PATH);
        if(output.contains("time_output"))
            resolve_enabled(result, output["time_output"], "enabled",
                            env_vars::TIME_OUTPUT);
        if(output.contains("file_output"))
            resolve_enabled(result, output["file_output"], "enabled",
                            env_vars::FILE_OUTPUT);
        if(output.contains("rocpd_output"))
            resolve_enabled(result, output["rocpd_output"], "enabled",
                            env_vars::USE_ROCPD);
    }

    // --- Causal profiling section ---
    if(j.contains("causal"))
    {
        const auto& causal = j["causal"];
        resolve_enabled(result, causal, "enabled", env_vars::USE_CAUSAL);
        resolve_value(result, causal, "mode", env_vars::CAUSAL_MODE);
        resolve_value(result, causal, "backend", env_vars::CAUSAL_BACKEND);
        resolve_value(result, causal, "binary_scope", env_vars::CAUSAL_BINARY_SCOPE);
        resolve_value(result, causal, "binary_exclude", env_vars::CAUSAL_BINARY_EXCLUDE);
        resolve_value(result, causal, "function_scope", env_vars::CAUSAL_FUNCTION_SCOPE);
        resolve_value(result, causal, "function_exclude",
                      env_vars::CAUSAL_FUNCTION_EXCLUDE);
        resolve_value(result, causal, "source_scope", env_vars::CAUSAL_SOURCE_SCOPE);
        resolve_value(result, causal, "source_exclude", env_vars::CAUSAL_SOURCE_EXCLUDE);
        if(causal.contains("end_to_end"))
            resolve_enabled(result, causal["end_to_end"], "enabled",
                            env_vars::CAUSAL_END_TO_END);
        resolve_value(result, causal, "delay_sec", env_vars::CAUSAL_DELAY);
        resolve_value(result, causal, "duration_sec", env_vars::CAUSAL_DURATION);
        resolve_value(result, causal, "random_seed", env_vars::CAUSAL_RANDOM_SEED);
    }

    // --- Hardware counters section ---
    if(j.contains("hardware_counters"))
    {
        const auto& hw = j["hardware_counters"];
        if(hw.contains("enabled") && hw["enabled"].get<bool>())
        {
            resolve_value(result, hw, "rocm_events", env_vars::ROCM_EVENTS);
            resolve_value(result, hw, "papi_events", env_vars::PAPI_EVENTS);
        }
        if(hw.contains("papi_multiplexing"))
            resolve_enabled(result, hw["papi_multiplexing"], "enabled",
                            env_vars::PAPI_MULTIPLEXING);
    }

    // --- Advanced section ---
    if(j.contains("advanced"))
    {
        const auto& adv = j["advanced"];
        if(adv.contains("cpu_affinity"))
            resolve_enabled(result, adv["cpu_affinity"], "enabled",
                            env_vars::CPU_AFFINITY);
        if(adv.contains("collapse_threads"))
            resolve_enabled(result, adv["collapse_threads"], "enabled",
                            env_vars::COLLAPSE_THREADS);
        resolve_value(result, adv, "max_depth", env_vars::MAX_DEPTH);
        resolve_value(result, adv, "trace_delay_sec", env_vars::TRACE_DELAY);
        resolve_value(result, adv, "trace_duration_sec", env_vars::TRACE_DURATION);
        resolve_value(result, adv, "verbose", env_vars::VERBOSE);
        if(adv.contains("debug"))
            resolve_enabled(result, adv["debug"], "enabled", env_vars::DEBUG);
        resolve_value(result, adv, "timemory_components", env_vars::TIMEMORY_COMPONENTS);
        resolve_value(result, adv, "network_interface", env_vars::NETWORK_INTERFACE);
        resolve_value(result, adv, "trace_periods", env_vars::TRACE_PERIODS);
        resolve_value(result, adv, "trace_period_clock_id",
                      env_vars::TRACE_PERIOD_CLOCK_ID);
    }

    return result;
}

std::map<std::string, std::string>
resolve_config(const nlohmann::json& j)
{
    return resolve_schema_config(j);
}

std::optional<std::map<std::string, std::string>>
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

std::optional<config_metadata>
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

std::optional<config_metadata>
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

std::string
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

std::string
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

std::map<std::string, std::string>
expand_parallel_runtimes(const std::string& runtimes_str)
{
    std::map<std::string, std::string> result;

    // If empty or "all", enable all runtimes
    if(runtimes_str.empty() || runtimes_str == "all")
    {
        result[std::string{ env_vars::USE_MPIP }]    = "true";
        result[std::string{ env_vars::USE_OMPT }]    = "true";
        result[std::string{ env_vars::USE_KOKKOSP }] = "true";
        result[std::string{ env_vars::USE_RCCLP }]   = "true";
        return result;
    }

    static const std::map<std::string, std::string> shortcuts = {
        { "mpi", std::string{ env_vars::USE_MPIP } },
        { "mpip", std::string{ env_vars::USE_MPIP } },
        { "openmp", std::string{ env_vars::USE_OMPT } },
        { "ompt", std::string{ env_vars::USE_OMPT } },
        { "omp", std::string{ env_vars::USE_OMPT } },
        { "kokkos", std::string{ env_vars::USE_KOKKOSP } },
        { "kokkosp", std::string{ env_vars::USE_KOKKOSP } },
        { "rccl", std::string{ env_vars::USE_RCCLP } },
        { "rcclp", std::string{ env_vars::USE_RCCLP } },
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

std::string
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

std::optional<int>
safe_stoi(const std::string& s)
{
    if(s.empty()) return std::nullopt;
    int        value  = 0;
    const auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    // Reject partial parses (e.g. "12.5" -> 12) by checking entire string was consumed
    if(result.ec != std::errc{} || result.ptr != s.data() + s.size()) return std::nullopt;
    return value;
}

std::optional<double>
safe_stod(const std::string& s)
{
    if(s.empty()) return std::nullopt;
    double value = 0.0;
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    // Prefer from_chars: locale-independent, zero-allocation
    const auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    if(result.ec != std::errc{} || result.ptr != s.data() + s.size()) return std::nullopt;
#else
    // Fallback for older compilers (GCC <11, Clang <16).
    // Note: std::stod is locale-sensitive -assumes C/POSIX locale.
    try
    {
        std::size_t pos = 0;
        value           = std::stod(s, &pos);
        if(pos != s.size()) return std::nullopt;
    } catch(const std::exception&)
    {
        return std::nullopt;
    }
#endif
    return value;
}

void
set_json_int(nlohmann::json& target, const std::string& value)
{
    if(auto n = safe_stoi(value))
        target = *n;
    else
        target = value;
}

void
set_json_double(nlohmann::json& target, const std::string& value)
{
    if(auto n = safe_stod(value))
        target = *n;
    else
        target = value;
}

bool
is_truthy(const std::string& v)
{
    if(v == "1") return true;
    if(v.size() < 2 || v.size() > 4) return false;
    std::string lower;
    lower.reserve(v.size());
    for(auto c : v)
        lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return lower == "true" || lower == "on" || lower == "yes";
}

void
export_enabled(nlohmann::json& j, const std::map<std::string, std::string>& env_map,
               std::string_view env_var, const std::string& json_path_section,
               const std::string& json_path_key)
{
    auto it = env_map.find(std::string{ env_var });
    if(it != env_map.end())
        j[json_path_section][json_path_key]["enabled"] = is_truthy(it->second);
}

void
export_section_enabled(nlohmann::json&                           j,
                       const std::map<std::string, std::string>& env_map,
                       std::string_view env_var, const std::string& json_path_section)
{
    auto it = env_map.find(std::string{ env_var });
    if(it != env_map.end()) j[json_path_section]["enabled"] = is_truthy(it->second);
}

void
export_string_value(nlohmann::json& j, const std::map<std::string, std::string>& env_map,
                    std::string_view env_var, const std::string& json_path_section,
                    const std::string& json_path_key)
{
    auto it = env_map.find(std::string{ env_var });
    if(it != env_map.end()) j[json_path_section][json_path_key]["value"] = it->second;
}

void
export_int_value(nlohmann::json& j, const std::map<std::string, std::string>& env_map,
                 std::string_view env_var, const std::string& json_path_section,
                 const std::string& json_path_key)
{
    auto it = env_map.find(std::string{ env_var });
    if(it != env_map.end())
        set_json_int(j[json_path_section][json_path_key]["value"], it->second);
}

void
export_double_value(nlohmann::json& j, const std::map<std::string, std::string>& env_map,
                    std::string_view env_var, const std::string& json_path_section,
                    const std::string& json_path_key)
{
    auto it = env_map.find(std::string{ env_var });
    if(it != env_map.end())
        set_json_double(j[json_path_section][json_path_key]["value"], it->second);
}

nlohmann::json
env_vars_to_json_schema(const std::map<std::string, std::string>& env_map)
{
    nlohmann::json j;

    // Helper to check if env var exists and get value
    auto get_val = [&](std::string_view key) -> std::optional<std::string> {
        auto it = env_map.find(std::string{ key });
        if(it != env_map.end()) return it->second;
        return std::nullopt;
    };

    // --- Tracing ---
    export_section_enabled(j, env_map, env_vars::TRACE, "tracing");
    export_enabled(j, env_map, env_vars::TRACE_LEGACY, "tracing", "legacy");
    export_int_value(j, env_map, env_vars::PERFETTO_BUFFER_SIZE_KB, "tracing",
                     "buffer_size_kb");
    export_string_value(j, env_map, env_vars::PERFETTO_FILL_POLICY, "tracing",
                        "fill_policy");

    // --- Profiling ---
    export_section_enabled(j, env_map, env_vars::PROFILE, "profiling");
    export_enabled(j, env_map, env_vars::FLAT_PROFILE, "profiling", "flat_profile");

    // --- Sampling ---
    export_section_enabled(j, env_map, env_vars::USE_SAMPLING, "sampling");
    export_int_value(j, env_map, env_vars::SAMPLING_FREQ, "sampling", "frequency_hz");
    export_string_value(j, env_map, env_vars::SAMPLING_TIMER, "sampling", "timer");
    export_double_value(j, env_map, env_vars::SAMPLING_DELAY, "sampling", "delay_sec");
    export_double_value(j, env_map, env_vars::SAMPLING_DURATION, "sampling",
                        "duration_sec");
    export_string_value(j, env_map, env_vars::SAMPLING_CPUS, "sampling", "cpus");
    export_string_value(j, env_map, env_vars::SAMPLING_GPUS, "sampling", "gpus");
    export_string_value(j, env_map, env_vars::SAMPLING_AINICS, "sampling", "ainics");

    // --- Domains: GPU ---
    if(auto v = get_val(env_vars::USE_AMD_SMI))
    {
        j["domains"]["gpu"]["enabled"] = is_truthy(*v);
        if(auto metrics = get_val(env_vars::AMD_SMI_METRICS))
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
        if(auto freq = get_val(env_vars::AMD_SMI_FREQ))
            set_json_int(j["domains"]["gpu"]["sampling_rate_hz"]["value"], *freq);
        if(auto freq = get_val(env_vars::PROCESS_SAMPLING_FREQ))
            set_json_double(j["domains"]["gpu"]["process_sampling_freq"]["value"], *freq);
        if(auto v = get_val(env_vars::USE_AINIC))
            j["domains"]["gpu"]["ainic"]["enabled"] = is_truthy(*v);
    }

    // --- Domains: ROCm ---
    if(auto v = get_val(env_vars::ROCM_DOMAINS))
    {
        j["domains"]["rocm"]["enabled"] = true;
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

    if(auto v = get_val(env_vars::ROCM_GROUP_BY_QUEUE))
        j["domains"]["rocm"]["group_by_queue"]["enabled"] = is_truthy(*v);

    // --- Domains: CPU ---
    if(auto v = get_val(env_vars::USE_PROCESS_SAMPLING))
    {
        if(is_truthy(*v))
        {
            if(auto freq = get_val(env_vars::CPU_FREQ))
            {
                if(is_truthy(*freq))
                {
                    j["domains"]["cpu"]["enabled"]                    = true;
                    j["domains"]["cpu"]["metrics"]["freq"]["enabled"] = true;
                }
            }
        }
    }

    // --- Domains: Parallel ---
    if(auto v = get_val(env_vars::USE_MPIP))
        j["domains"]["parallel"]["runtimes"]["mpi"]["enabled"] = is_truthy(*v);
    if(auto v = get_val(env_vars::USE_OMPT))
        j["domains"]["parallel"]["runtimes"]["openmp"]["enabled"] = is_truthy(*v);
    if(auto v = get_val(env_vars::USE_KOKKOSP))
        j["domains"]["parallel"]["runtimes"]["kokkos"]["enabled"] = is_truthy(*v);
    if(auto v = get_val(env_vars::USE_RCCLP))
        j["domains"]["parallel"]["runtimes"]["rccl"]["enabled"] = is_truthy(*v);
    if(auto v = get_val(env_vars::USE_SHMEM))
        j["domains"]["parallel"]["runtimes"]["shmem"]["enabled"] = is_truthy(*v);
    if(auto v = get_val(env_vars::USE_UCX))
        j["domains"]["parallel"]["runtimes"]["ucx"]["enabled"] = is_truthy(*v);

    // --- Output ---
    export_string_value(j, env_map, env_vars::OUTPUT_PATH, "output", "path");
    export_enabled(j, env_map, env_vars::TIME_OUTPUT, "output", "time_output");
    export_enabled(j, env_map, env_vars::FILE_OUTPUT, "output", "file_output");
    export_enabled(j, env_map, env_vars::USE_ROCPD, "output", "rocpd_output");

    // --- Hardware counters ---
    if(auto v = get_val(env_vars::ROCM_EVENTS))
    {
        j["hardware_counters"]["enabled"]              = true;
        j["hardware_counters"]["rocm_events"]["value"] = *v;
    }
    if(auto v = get_val(env_vars::PAPI_EVENTS))
    {
        j["hardware_counters"]["enabled"]              = true;
        j["hardware_counters"]["papi_events"]["value"] = *v;
    }
    export_enabled(j, env_map, env_vars::PAPI_MULTIPLEXING, "hardware_counters",
                   "papi_multiplexing");

    // --- Causal profiling ---
    export_section_enabled(j, env_map, env_vars::USE_CAUSAL, "causal");
    export_string_value(j, env_map, env_vars::CAUSAL_MODE, "causal", "mode");
    export_string_value(j, env_map, env_vars::CAUSAL_BACKEND, "causal", "backend");
    export_string_value(j, env_map, env_vars::CAUSAL_BINARY_SCOPE, "causal",
                        "binary_scope");
    export_string_value(j, env_map, env_vars::CAUSAL_BINARY_EXCLUDE, "causal",
                        "binary_exclude");
    export_string_value(j, env_map, env_vars::CAUSAL_FUNCTION_SCOPE, "causal",
                        "function_scope");
    export_string_value(j, env_map, env_vars::CAUSAL_FUNCTION_EXCLUDE, "causal",
                        "function_exclude");
    export_string_value(j, env_map, env_vars::CAUSAL_SOURCE_SCOPE, "causal",
                        "source_scope");
    export_string_value(j, env_map, env_vars::CAUSAL_SOURCE_EXCLUDE, "causal",
                        "source_exclude");
    export_enabled(j, env_map, env_vars::CAUSAL_END_TO_END, "causal", "end_to_end");
    export_double_value(j, env_map, env_vars::CAUSAL_DELAY, "causal", "delay_sec");
    export_double_value(j, env_map, env_vars::CAUSAL_DURATION, "causal", "duration_sec");
    export_int_value(j, env_map, env_vars::CAUSAL_RANDOM_SEED, "causal", "random_seed");

    // --- Advanced ---
    export_int_value(j, env_map, env_vars::VERBOSE, "advanced", "verbose");
    export_enabled(j, env_map, env_vars::DEBUG, "advanced", "debug");
    export_int_value(j, env_map, env_vars::MAX_DEPTH, "advanced", "max_depth");
    export_double_value(j, env_map, env_vars::TRACE_DELAY, "advanced", "trace_delay_sec");
    export_double_value(j, env_map, env_vars::TRACE_DURATION, "advanced",
                        "trace_duration_sec");
    export_enabled(j, env_map, env_vars::CPU_AFFINITY, "advanced", "cpu_affinity");
    export_enabled(j, env_map, env_vars::COLLAPSE_THREADS, "advanced",
                   "collapse_threads");
    export_string_value(j, env_map, env_vars::TIMEMORY_COMPONENTS, "advanced",
                        "timemory_components");
    export_string_value(j, env_map, env_vars::NETWORK_INTERFACE, "advanced",
                        "network_interface");
    export_string_value(j, env_map, env_vars::TRACE_PERIODS, "advanced", "trace_periods");
    export_string_value(j, env_map, env_vars::TRACE_PERIOD_CLOCK_ID, "advanced",
                        "trace_period_clock_id");

    // ========================================================================
    // Intentionally excluded environment variables
    // ========================================================================
    //
    // The following ROCPROFSYS_* env vars are NOT included in the JSON preset
    // schema. These are internal runtime settings whose values depend on the
    // specific invocation context, tooling infrastructure, or low-level
    // implementation details. Including them in presets would be harmful:
    // a preset should describe *what* to profile, not *how* the profiler
    // manages its own internals.
    //
    // If any of these are later deemed useful as preset configuration, they
    // should be moved into the appropriate schema section (e.g., tracing,
    // advanced) on a case-by-case basis — NOT bulk-exposed via an "internal"
    // domain.
    //
    // Session-specific (depend on the invocation, not the profiling intent):
    //   ROCPROFSYS_CONFIG_FILE     - Path to the user's config file; set at
    //                                invocation time, not a profiling choice.
    //   ROCPROFSYS_OUTPUT_PREFIX   - Per-run output prefix (e.g., test name);
    //                                set by test harness or user per-run.
    //   ROCPROFSYS_TRACE_REGION    - Region filter for selective tracing;
    //                                depends on the specific application.
    //
    // Internal plumbing (implementation details users should not configure):
    //   ROCPROFSYS_ENABLED                 - Master profiler enable flag.
    //                                        Always true when running via CLI
    //                                        tools; setting to false in a
    //                                        preset would silently disable
    //                                        all profiling.
    //   ROCPROFSYS_SUPPRESS_CONFIG         - Suppress config file loading.
    //   ROCPROFSYS_SUPPRESS_PARSING        - Suppress config parsing.
    //                                        These are used internally by
    //                                        rocprof-sys-avail; setting them
    //                                        in a preset would break config
    //                                        file handling.
    //   ROCPROFSYS_USE_PID                 - Include PID in output paths.
    //                                        Managed automatically by the
    //                                        output subsystem.
    //   ROCPROFSYS_PERFETTO_BACKEND        - Perfetto backend (inprocess/
    //                                        system). Low-level transport
    //                                        choice, not a profiling concern.
    //   ROCPROFSYS_PERFETTO_FLUSH_PERIOD_MS - Perfetto flush interval.
    //                                        Performance tuning for the trace
    //                                        writer, not user-facing.
    //   ROCPROFSYS_PROCESS_SAMPLING_DURATION - Duration of process sampling.
    //                                        Controlled via SAMPLING_DURATION
    //                                        in the sampling section instead.
    //   ROCPROFSYS_SAMPLING_OVERFLOW_EVENT  - Hardware overflow event name
    //                                        (e.g., PERF_COUNT_HW_CACHE_REFERENCES).
    //                                        Highly platform-specific; not
    //                                        portable across machines.
    //   ROCPROFSYS_CPU_FREQ_ENABLED        - CPU frequency monitoring flag.
    //                                        Controlled indirectly via the
    //                                        domains.cpu section instead.

    return j;
}

std::string
export_config_as_json(const std::map<std::string, std::string>& env_vars,
                      const std::string& preset_name, std::string_view tool_name,
                      int indent)
{
    auto j = env_vars_to_json_schema(env_vars);

    if(!preset_name.empty())
    {
        j["metadata"]["name"] = preset_name;
        std::string desc      = "Exported configuration from rocprof-sys";
        if(!tool_name.empty())
        {
            desc += '-';
            desc += tool_name;
        }
        j["metadata"]["description"] = desc;
    }

    return j.dump(indent);
}

}  // namespace json_config
}  // namespace rocprofsys
