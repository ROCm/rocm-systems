// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "common/preset_loader.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofsys
{
namespace common_utils
{
inline std::string
get_output_directory(const char* env_var = "ROCPROFSYS_OUTPUT_PATH")
{
    const char* output_path = std::getenv(env_var);
    if(output_path && strlen(output_path) > 0) return std::string(output_path);

    return "rocprof-sys-output";
}

inline std::string
generate_preset_description(std::string_view preset_mode)
{
    // Normalize the preset_mode by stripping leading "--" if present
    std::string normalized{ preset_mode };
    if(normalized.size() > 2 && normalized.substr(0, 2) == "--")
        normalized = normalized.substr(2);

    auto info = rocprofsys::preset_loader::load_preset_or_file(normalized);
    if(!info) return "";

    // Load the raw JSON to access hierarchical structure
    auto preset_dir = rocprofsys::preset_loader::find_preset_directory();
    if(preset_dir.empty()) return info->description;

    auto          filepath = preset_dir + "/" + normalized + ".json";
    std::ifstream ifs{ filepath };
    if(!ifs.is_open()) return info->description;

    nlohmann::json j;
    try
    {
        j = nlohmann::json::parse(ifs);
    } catch(const nlohmann::json::exception&)
    {
        return info->description;
    }

    // Build tree lines from JSON sections
    std::vector<std::string> lines;

    // Tracing
    if(j.contains("tracing"))
    {
        const auto& t     = j["tracing"];
        bool        on    = t.value("enabled", false);
        std::string entry = std::string("Tracing:         ") + (on ? "ON" : "OFF");
        if(on && t.contains("buffer_size_kb"))
        {
            auto kb = t["buffer_size_kb"].value("value", 0);
            if(kb >= 1024000)
                entry += " (buffer: " + std::to_string(kb / 1024000) + " GB)";
            else if(kb > 0)
                entry += " (buffer: " + std::to_string(kb) + " KB)";
        }
        lines.push_back(entry);
    }

    // Profiling
    if(j.contains("profiling"))
    {
        const auto& p     = j["profiling"];
        bool        on    = p.value("enabled", false);
        std::string entry = std::string("Profiling:       ") + (on ? "ON" : "OFF");
        if(on && p.contains("flat_profile") && p["flat_profile"].value("enabled", false))
            entry += " (flat profile)";
        lines.push_back(entry);
    }

    // Sampling
    if(j.contains("sampling"))
    {
        const auto& s     = j["sampling"];
        bool        on    = s.value("enabled", false);
        std::string entry = std::string("CPU Sampling:    ") + (on ? "ON" : "OFF");
        if(on && s.contains("frequency_hz"))
        {
            auto freq = s["frequency_hz"].value("value", 0);
            if(freq > 0) entry += " @ " + std::to_string(freq) + " Hz";
        }
        if(s.contains("cpus") && s["cpus"].value("value", "") == "none")
        {
            entry = "CPU Sampling:    Disabled (none)";
        }
        lines.push_back(entry);
    }

    // Domains: GPU
    if(j.contains("domains") && j["domains"].contains("gpu"))
    {
        const auto& gpu = j["domains"]["gpu"];
        if(gpu.value("enabled", false))
        {
            std::string entry = "GPU Metrics:     ON";
            if(gpu.contains("metrics"))
            {
                std::vector<std::string> names;
                for(const auto& [name, m] : gpu["metrics"].items())
                {
                    if(m.value("enabled", false)) names.push_back(name);
                }
                if(!names.empty())
                {
                    entry += " (";
                    for(size_t i = 0; i < names.size(); ++i)
                    {
                        if(i > 0) entry += ", ";
                        entry += names[i];
                    }
                    entry += ")";
                }
            }
            lines.push_back(entry);
        }
    }

    // Domains: ROCm
    if(j.contains("domains") && j["domains"].contains("rocm"))
    {
        const auto& rocm = j["domains"]["rocm"];
        if(rocm.value("enabled", false) && rocm.contains("api_domains"))
        {
            std::vector<std::string> apis;
            for(const auto& [name, api] : rocm["api_domains"].items())
            {
                if(api.value("enabled", false)) apis.push_back(name);
            }
            if(!apis.empty())
            {
                std::string entry = "ROCm Domains:    ";
                for(size_t i = 0; i < apis.size(); ++i)
                {
                    if(i > 0) entry += ", ";
                    entry += apis[i];
                }
                lines.push_back(entry);
            }
        }
    }

    // Domains: Parallel runtimes
    if(j.contains("domains") && j["domains"].contains("parallel"))
    {
        const auto& par = j["domains"]["parallel"];
        if(par.contains("runtimes"))
        {
            std::vector<std::string> runtimes;
            for(const auto& [name, rt] : par["runtimes"].items())
            {
                if(rt.value("enabled", false)) runtimes.push_back(name);
            }
            if(!runtimes.empty())
            {
                std::string entry = "Parallel:        ";
                for(size_t i = 0; i < runtimes.size(); ++i)
                {
                    if(i > 0) entry += ", ";
                    entry += runtimes[i];
                }
                lines.push_back(entry);
            }
        }
    }

    // Hardware counters
    if(j.contains("hardware_counters") && j["hardware_counters"].value("enabled", false))
    {
        const auto& hw = j["hardware_counters"];
        if(hw.contains("papi_events"))
        {
            auto val =
                rocprofsys::json_config::json_value_to_string(hw["papi_events"]["value"]);
            lines.push_back("PAPI Events:     " + val);
        }
        if(hw.contains("rocm_events"))
        {
            auto val =
                rocprofsys::json_config::json_value_to_string(hw["rocm_events"]["value"]);
            lines.push_back("ROCm Events:     " + val);
        }
    }

    // Output: rocPD
    if(j.contains("output") && j["output"].contains("rocpd_output") &&
       j["output"]["rocpd_output"].value("enabled", false))
    {
        lines.push_back("rocPD Output:    ON");
    }

    if(lines.empty()) return info->description;

    // Format with tree characters
    std::ostringstream oss;
    oss << info->description << "\n";
    for(size_t i = 0; i < lines.size(); ++i)
    {
        bool is_last = (i + 1 == lines.size());
        oss << "  " << (is_last ? "└─ " : "├─ ") << lines[i];
        if(!is_last) oss << "\n";
    }
    return oss.str();
}

inline void
print_pre_execution_info(std::string_view tool_name, std::string_view preset_mode = "")
{
    auto output_dir = get_output_directory();

    if(!preset_mode.empty() && !tool_name.empty())
    {
        constexpr size_t           box_width       = 60;
        constexpr size_t           box_inner_width = box_width - 2;
        constexpr std::string_view box_line =
            "════════════════════════════════════════════════════════════";
        constexpr std::string_view prefix       = "ROCm Systems Profiler - ";
        const size_t               content_size = prefix.size() + tool_name.size();
        const size_t               padding =
            content_size < box_inner_width ? box_inner_width - content_size : 0;

        std::cout << "\n"
                  << "╔" << box_line << "╗\n"
                  << "║ " << prefix << tool_name << std::string(padding, ' ') << " ║\n"
                  << "╚" << box_line << "╝\n"
                  << "\n";

        std::cout << "Preset:        " << preset_mode << "\n";

        auto description = generate_preset_description(preset_mode);
        if(!description.empty())
        {
            std::cout << "\n" << description << "\n";
        }
    }

    std::cout << "\nOutput:        " << output_dir << "\n";

    std::cout << "\nResults will be available in:\n"
              << "  • Text profile:  " << output_dir << "/wall_clock.txt\n"
              << "  • Trace (visual): " << output_dir << "/perfetto-trace.proto\n"
              << "  • JSON data:      " << output_dir << "/wall_clock.json\n"
              << "\nTo visualize trace:\n"
              << "  Open " << output_dir
              << "/perfetto-trace.proto in https://ui.perfetto.dev\n"
              << "\n";
}

template <typename ParserT>
std::vector<std::string>
collect_active_presets(ParserT& parser, std::initializer_list<const char*> preset_names)
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

inline bool
validate_preset_modes(const std::vector<std::string>& active_presets)
{
    if(active_presets.size() > 1)
    {
        std::cerr << "\nERROR: Multiple preset modes specified: ";
        for(const auto& active_preset : active_presets)
        {
            std::cerr << active_preset;
            if(active_preset != active_presets.back()) std::cerr << ", ";
        }
        std::cerr << "\n\n";

        std::cerr << "Only ONE preset mode can be used at a time.\n\n";
        std::cerr
            << "Available presets (use with --preset=<name>):\n"
            << "  General Purpose:\n"
            << "    balanced           Balanced profiling with moderate overhead\n"
            << "    profile-only       Profiling without tracing, minimal overhead\n"
            << "    detailed           Full trace + hardware counters\n"
            << "  Workload-Specific:\n"
            << "    trace-hpc          MPI/OpenMP/HPC applications\n"
            << "    workload-trace     General compute workloads (AI/ML, HPC, etc.)\n"
            << "    trace-gpu          GPU workload analysis\n"
            << "    trace-openmp       OpenMP offload workloads\n"
            << "    profile-mpi        MPI communication latency profiling\n"
            << "    trace-hw-counters  Hardware counter collection\n"
            << "  API Tracing:\n"
            << "    sys-trace          Comprehensive system API tracing\n"
            << "    runtime-trace      Runtime API tracing (no compiler/HSA)\n\n";

        std::cerr
            << "Choose one preset or use manual options for custom configuration.\n";
        std::cerr << "See --help for all options.\n\n";

        return false;
    }
    return true;
}

}  // namespace common_utils
}  // namespace rocprofsys
