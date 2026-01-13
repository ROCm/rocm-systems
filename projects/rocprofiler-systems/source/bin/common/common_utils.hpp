// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
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

inline bool
check_directory_writable(const std::string& dir)
{
    struct stat st;
    if(stat(dir.c_str(), &st) == 0)
    {
        return (access(dir.c_str(), W_OK) == 0);
    }

    std::string parent = dir;
    size_t      pos    = parent.find_last_of('/');
    if(pos != std::string::npos)
    {
        parent = parent.substr(0, pos);
        if(parent.empty()) parent = ".";
    }
    else
    {
        parent = ".";
    }

    return (access(parent.c_str(), W_OK) == 0);
}

inline std::string
get_preset_description(std::string_view preset_mode)
{
    static const std::unordered_map<std::string_view, std::string> descriptions = {
        { "--quick", "Fast profiling with sensible defaults\n"
                     "  ├─ Tracing:         ON (Perfetto timeline)\n"
                     "  ├─ Profiling:       ON (call-stack based)\n"
                     "  ├─ CPU Sampling:    ON @ 50 Hz\n"
                     "  └─ Process Metrics: ON (CPU freq, memory)" },
        { "--simple", "Minimal overhead flat profiling\n"
                      "  ├─ Tracing:         OFF\n"
                      "  ├─ Profiling:       ON (flat profile)\n"
                      "  ├─ CPU Sampling:    ON @ 100 Hz\n"
                      "  └─ Process Metrics: OFF" },
        { "--detailed", "Comprehensive profiling with full system metrics\n"
                        "  ├─ Tracing:         ON (Perfetto timeline)\n"
                        "  ├─ Profiling:       ON (call-stack based)\n"
                        "  ├─ CPU Sampling:    ON @ 100 Hz (all CPUs)\n"
                        "  └─ Process Metrics: ON (CPU freq, memory)" },
        { "--trace-hpc", "Optimized for HPC/MPI/OpenMP applications\n"
                         "  ├─ Tracing:         ON (Perfetto timeline)\n"
                         "  ├─ Profiling:       ON (call-stack based)\n"
                         "  ├─ CPU Sampling:    OFF (reduced overhead)\n"
                         "  ├─ Process Metrics: ON\n"
                         "  ├─ OpenMP (OMPT):   ON\n"
                         "  ├─ MPI (MPIP):      ON\n"
                         "  ├─ Kokkos:          ON\n"
                         "  ├─ RCCL:            ON\n"
                         "  ├─ PAPI Events:     PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_L3_TCM\n"
                         "  ├─ ROCm Domains:    HIP API, kernels, memory, scratch\n"
                         "  └─ GPU Metrics:     busy, temp, power, mem_usage" },
        { "--trace-ai", "Optimized for AI/ML workloads (PyTorch, TensorFlow, JAX)\n"
                        "  ├─ Tracing:         ON (Perfetto timeline)\n"
                        "  ├─ Profiling:       ON (call-stack based)\n"
                        "  ├─ CPU Sampling:    OFF (reduced overhead)\n"
                        "  ├─ Process Metrics: ON\n"
                        "  ├─ ROCtracer:       ON\n"
                        "  ├─ HIP API Trace:   ON\n"
                        "  ├─ HIP Activity:    ON (kernel timing)\n"
                        "  ├─ RCCL:            ON (collective comms)\n"
                        "  ├─ rocPD:           ON (PyTorch dispatcher)\n"
                        "  ├─ MPI (MPIP):      ON\n"
                        "  ├─ ROCm Domains:    HIP API, kernels, memory, scratch\n"
                        "  ├─ GPU Metrics:     busy, temp, power, mem_usage\n"
                        "  └─ Buffer Size:     2 GB (for long traces)" }
    };

    auto it = descriptions.find(preset_mode);
    if(it != descriptions.end())
    {
        return it->second;
    }
    return "";
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
        constexpr std::string_view prefix = "ROCm Systems Profiler - ";
        const size_t padding = box_inner_width - prefix.size() - tool_name.size();

        std::cout << "\n"
                  << "╔" << box_line << "╗\n"
                  << "║ " << prefix << tool_name << std::string(padding, ' ') << " ║\n"
                  << "╚" << box_line << "╝\n"
                  << "\n";

        std::cout << "Preset:        " << preset_mode << "\n";

        auto description = get_preset_description(preset_mode);
        if(!description.empty())
        {
            std::cout << "\n" << description << "\n";
        }
    }

    std::cout << "\nOutput:        " << output_dir << "\n";

    if(!check_directory_writable(output_dir))
    {
        std::cerr << "\nWARNING: Output directory may not be writable!\n";
        std::cerr << "   Try: rocprof-sys-" << tool_name
                  << " -o /tmp/profile -- <command>\n\n";
    }

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
        for(size_t i = 0; i < active_presets.size(); ++i)
        {
            std::cerr << active_presets[i];
            if(i < active_presets.size() - 1) std::cerr << ", ";
        }
        std::cerr << "\n\n";

        std::cerr << "Only ONE preset mode can be used at a time.\n\n";
        std::cerr << "Available presets:\n"
                  << "  --quick         Fast profiling with sensible defaults\n"
                  << "  --simple        Flat profile, minimal overhead\n"
                  << "  --detailed      Full trace + hardware counters\n"
                  << "  --trace-hpc     MPI/OpenMP/HPC applications\n"
                  << "  --trace-ai      PyTorch/TensorFlow/JAX\n\n";

        std::cerr
            << "Choose one preset or use manual options for custom configuration.\n";
        std::cerr << "See --help for all options.\n\n";

        return false;
    }
    return true;
}

inline bool
check_rocm_available()
{
#if !defined(ROCPROFSYS_USE_ROCM) || ROCPROFSYS_USE_ROCM == 0
    return false;
#else
    return (system("which hipconfig > /dev/null 2>&1") == 0 ||
            access("/opt/rocm/bin/hipconfig", X_OK) == 0);
#endif
}

inline void
warn_if_rocm_unavailable()
{
    if(!check_rocm_available())
    {
        std::cerr << "\nWARNING: GPU tracing requested but ROCm is not available\n\n";
        std::cerr << "GPU features will be disabled.\n\n";
    }
}

inline void
warn_if_gpu_preset_without_rocm(const std::vector<std::string>& active_presets)
{
    for(const auto& preset : active_presets)
    {
        if(preset == "--trace-ai" || preset == "--trace-hpc")
        {
            warn_if_rocm_unavailable();
            return;
        }
    }
}

}  // namespace common_utils
}  // namespace rocprofsys
