// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

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
        { "--balanced",
          "Balanced profiling with moderate overhead and comprehensive data\n"
          "  ├─ Tracing:         ON (Perfetto timeline)\n"
          "  ├─ Profiling:       ON (call-stack based)\n"
          "  ├─ CPU Sampling:    ON @ 50 Hz\n"
          "  └─ Process Metrics: ON (CPU freq, memory)" },
        { "--profile-only", "Profiling-only mode without tracing (flat profile)\n"
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
        { "--workload-trace",
          "Optimized for general compute workloads (AI/ML, HPC, etc.)\n"
          "  ├─ Tracing:         ON (Perfetto timeline)\n"
          "  ├─ Profiling:       ON (call-stack based)\n"
          "  ├─ CPU Sampling:    OFF (reduced overhead)\n"
          "  ├─ Process Metrics: ON\n"
          "  ├─ ROCtracer:       ON\n"
          "  ├─ HIP API Trace:   ON\n"
          "  ├─ HIP Activity:    ON (kernel timing)\n"
          "  ├─ RCCL:            ON (collective comms)\n"
          "  ├─ rocPD:           ON (SQLite Database Output Format)\n"
          "  ├─ MPI (MPIP):      ON\n"
          "  ├─ ROCm Domains:    HIP API, kernels, memory, scratch\n"
          "  ├─ GPU Metrics:     busy, temp, power, mem_usage\n"
          "  └─ Buffer Size:     2 GB (for long traces)" },
        { "--sys-trace", "Comprehensive system API tracing\n"
                         "  ├─ Tracing:         ON (Perfetto timeline)\n"
                         "  ├─ Profiling:       ON (call-stack based)\n"
                         "  ├─ ROCm APIs:       HIP API, HSA API\n"
                         "  ├─ Marker API:      ROCTx\n"
                         "  ├─ RCCL:            ON (collective communications)\n"
                         "  ├─ Decode/JPEG:     rocDecode, rocJPEG\n"
                         "  ├─ Memory Ops:      copies, scratch, allocations\n"
                         "  └─ Kernel Dispatch: ON" },
        { "--runtime-trace", "Runtime API tracing (excludes compiler and low-level HSA)\n"
                             "  ├─ Tracing:         ON (Perfetto timeline)\n"
                             "  ├─ Profiling:       ON (call-stack based)\n"
                             "  ├─ HIP Runtime:     ON (excludes compiler API)\n"
                             "  ├─ Marker API:      ROCTx\n"
                             "  ├─ RCCL:            ON (collective communications)\n"
                             "  ├─ Decode/JPEG:     rocDecode, rocJPEG\n"
                             "  ├─ Memory Ops:      copies, scratch, allocations\n"
                             "  └─ Kernel Dispatch: ON" },
        { "--trace-gpu",
          "GPU workload analysis with host functions, MPI, and device activity\n"
          "  ├─ Tracing:         ON (Perfetto timeline)\n"
          "  ├─ Profiling:       OFF (reduced overhead)\n"
          "  ├─ ROCm:            ON\n"
          "  ├─ AMD SMI:         ON (GPU metrics)\n"
          "  ├─ CPU Sampling:    Disabled (none)\n"
          "  └─ ROCm Domains:    HIP runtime, ROCTx, kernels, memory, scratch" },
        { "--trace-openmp",
          "OpenMP offload workloads with HSA domains\n"
          "  ├─ Tracing:         ON (Perfetto timeline)\n"
          "  ├─ Profiling:       OFF (reduced overhead)\n"
          "  ├─ ROCm:            ON\n"
          "  ├─ OMPT:            ON (OpenMP tools interface)\n"
          "  └─ ROCm Domains:    HIP runtime, ROCTx, kernels, memory, HSA API" },
        { "--profile-mpi", "MPI communication latency profiling\n"
                           "  ├─ Tracing:         OFF\n"
                           "  ├─ Profiling:       ON (flat profile)\n"
                           "  ├─ AMD SMI:         OFF\n"
                           "  ├─ ROCm:            OFF\n"
                           "  └─ Focus:           Wall-clock files per rank" },
        { "--trace-hw-counters", "Hardware counter collection during execution\n"
                                 "  ├─ Profiling:       ON\n"
                                 "  ├─ CPU Sampling:    Disabled (none)\n"
                                 "  ├─ ROCm Events:     VALUUtilization, Occupancy\n"
                                 "  └─ Focus:           GPU performance counters" }
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
        for(const auto& active_preset : active_presets)
        {
            std::cerr << active_preset;
            if(active_preset != active_presets.back()) std::cerr << ", ";
        }
        std::cerr << "\n\n";

        std::cerr << "Only ONE preset mode can be used at a time.\n\n";
        std::cerr
            << "Available presets:\n"
            << "  General Purpose:\n"
            << "    --balanced           Balanced profiling with moderate overhead\n"
            << "    --profile-only       Profiling without tracing, minimal overhead\n"
            << "    --detailed           Full trace + hardware counters\n"
            << "  Workload-Specific:\n"
            << "    --trace-hpc          MPI/OpenMP/HPC applications\n"
            << "    --workload-trace     General compute workloads (AI/ML, HPC, etc.)\n"
            << "    --trace-gpu          GPU workload analysis\n"
            << "    --trace-openmp       OpenMP offload workloads\n"
            << "    --profile-mpi        MPI communication latency profiling\n"
            << "    --trace-hw-counters  Hardware counter collection\n"
            << "  API Tracing:\n"
            << "    --sys-trace          Comprehensive system API tracing\n"
            << "    --runtime-trace      Runtime API tracing (no compiler/HSA)\n\n";

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
    return (access("/opt/rocm/bin/hipconfig", X_OK) == 0);
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
        if(preset == "--workload-trace" || preset == "--trace-hpc" ||
           preset == "--sys-trace" || preset == "--runtime-trace" ||
           preset == "--trace-gpu" || preset == "--trace-openmp" ||
           preset == "--trace-hw-counters")
        {
            warn_if_rocm_unavailable();
            return;
        }
    }
}

}  // namespace common_utils
}  // namespace rocprofsys
