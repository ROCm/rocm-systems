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
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace rocprofsys
{
namespace common_utils
{
inline std::string
get_output_directory(const char* env_var = "ROCPROFSYS_OUTPUT_PATH")
{
    const char* output_path = getenv(env_var);
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
get_preset_description(const std::string& preset_mode)
{
    if(preset_mode == "--quick")
    {
        return "Fast profiling with sensible defaults\n"
               "  ├─ Tracing:         ON (Perfetto timeline)\n"
               "  ├─ Profiling:       ON (call-stack based)\n"
               "  ├─ CPU Sampling:    ON @ 50 Hz\n"
               "  └─ Process Metrics: ON (CPU freq, memory)";
    }
    else if(preset_mode == "--simple")
    {
        return "Minimal overhead flat profiling\n"
               "  ├─ Tracing:         OFF\n"
               "  ├─ Profiling:       ON (flat profile)\n"
               "  ├─ CPU Sampling:    ON @ 100 Hz\n"
               "  └─ Process Metrics: OFF";
    }
    else if(preset_mode == "--detailed")
    {
        return "Comprehensive profiling with full system metrics\n"
               "  ├─ Tracing:         ON (Perfetto timeline)\n"
               "  ├─ Profiling:       ON (call-stack based)\n"
               "  ├─ CPU Sampling:    ON @ 100 Hz (all CPUs)\n"
               "  └─ Process Metrics: ON (CPU freq, memory)";
    }
    else if(preset_mode == "--trace-hpc")
    {
        return "Optimized for HPC/MPI/OpenMP applications\n"
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
               "  └─ GPU Metrics:     busy, temp, power, mem_usage";
    }
    else if(preset_mode == "--trace-ai")
    {
        return "Optimized for AI/ML workloads (PyTorch, TensorFlow, JAX)\n"
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
               "  └─ Buffer Size:     2 GB (for long traces)";
    }
    return "";
}

inline void
print_pre_execution_info(const std::string& tool_name,
                         const std::string& preset_mode = "")
{
    auto output_dir = get_output_directory();

    if(!preset_mode.empty())
    {
        constexpr size_t  box_width       = 60;
        constexpr size_t  box_inner_width = box_width - 2;
        const std::string box_line =
            "════════════════════════════════════════════════════════════";
        const std::string prefix  = "ROCm Systems Profiler - ";
        const size_t      padding = box_inner_width - prefix.size() - tool_name.size();

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

inline void
print_error_with_guidance(const std::string& error_msg, const std::string& tool_name)
{
    std::cerr << "\nERROR: " << error_msg << "\n\n";

    if(error_msg.find("output") != std::string::npos ||
       error_msg.find("write") != std::string::npos)
    {
        std::cerr << "Possible solutions:\n"
                  << "  1. Specify writable output: rocprof-sys-" << tool_name
                  << " -o /tmp/profile -- ./app\n"
                  << "  2. Check permissions: ls -ld ./\n"
                  << "  3. Set environment: export ROCPROFSYS_OUTPUT_PATH=/tmp/profile\n";
    }
    else if(error_msg.find("HIP") != std::string::npos ||
            error_msg.find("GPU") != std::string::npos ||
            error_msg.find("ROCm") != std::string::npos)
    {
        std::cerr << "GPU/ROCm troubleshooting:\n"
                  << "  1. Verify ROCm installation: hipconfig\n"
                  << "  2. Check devices: rocminfo\n"
                  << "  3. Ensure ROCm in PATH: which hipcc\n"
                  << "  4. Source environment: source /opt/rocm/setup.sh\n";
    }
    else if(error_msg.find("command") != std::string::npos ||
            error_msg.find("executable") != std::string::npos)
    {
        std::cerr << "Command troubleshooting:\n"
                  << "  1. Check executable exists: ls -l ./app\n"
                  << "  2. Verify it's executable: chmod +x ./app\n"
                  << "  3. Try absolute path: rocprof-sys-" << tool_name
                  << " -- $(pwd)/app\n";
    }
    else
    {
        std::cerr << "General troubleshooting:\n"
                  << "  1. Check help: rocprof-sys-" << tool_name << " --help\n"
                  << "  2. Enable verbose mode: rocprof-sys-" << tool_name
                  << " -v -- ./app\n"
                  << "  3. Try quick preset: rocprof-sys-" << tool_name
                  << " --quick -- ./app\n";
    }

    std::cerr << "\nDocumentation: /opt/rocprofiler-systems/share/docs/\n"
              << "Online help: https://rocm.docs.amd.com/projects/rocprofiler-systems/\n"
              << "\n";
}

inline int
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

        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

inline bool
check_rocm_available()
{
    return (system("which hipconfig > /dev/null 2>&1") == 0 ||
            access("/opt/rocm/bin/hipconfig", X_OK) == 0);
}

inline void
warn_if_hip_trace_without_rocm(bool hip_trace_requested, const std::string& /*tool_name*/)
{
    if(hip_trace_requested && !check_rocm_available())
    {
        std::cerr << "\nWARNING: HIP tracing requested but ROCm may not be available\n\n";
        std::cerr << "Verify ROCm installation:\n";
        std::cerr << "  $ hipconfig\n";
        std::cerr << "  $ rocminfo\n\n";
        std::cerr << "If ROCm is installed, ensure it's in your PATH:\n";
        std::cerr << "  $ export PATH=/opt/rocm/bin:$PATH\n";
        std::cerr << "  $ export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH\n\n";
        std::cerr << "Continuing without GPU tracing...\n\n";
    }
}

}  // namespace common_utils
}  // namespace rocprofsys
