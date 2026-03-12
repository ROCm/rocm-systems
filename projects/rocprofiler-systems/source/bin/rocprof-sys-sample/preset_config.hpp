// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string_view>
#include <variant>
#include <vector>

namespace rocprofsys
{
namespace sample
{

namespace common_values
{
constexpr auto ROCM_DOMAINS_STANDARD =
    "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,scratch_memory";

constexpr auto ROCM_DOMAINS_FULL =
    "hip_api,hsa_api,marker_api,rccl_api,memory_copy,scratch_memory,kernel_dispatch";

constexpr auto ROCM_DOMAINS_RUNTIME =
    "hip_runtime_api,marker_api,rccl_api,memory_copy,scratch_memory,kernel_dispatch";

constexpr auto ROCM_DOMAINS_WITH_HSA =
    "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,hsa_api";

constexpr auto AMD_SMI_METRICS_STANDARD = "busy,temp,power,mem_usage";

constexpr auto PAPI_EVENTS_STANDARD = "PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_L3_TCM";

constexpr auto ROCM_EVENTS_STANDARD = "VALUUtilization,Occupancy";
}  // namespace common_values

using EnvValue = std::variant<bool, int, std::string_view>;

struct PresetConfig
{
    std::string_view                                   name;
    std::string_view                                   description;
    std::vector<std::pair<std::string_view, EnvValue>> env_settings;
    bool                                               has_gpu_devices_check = false;
};

inline const std::vector<PresetConfig> PRESETS = {
    { "balanced",
      "Balanced profiling mode: moderate overhead with comprehensive data "
      "(tracing, call-stack profiling, and sampling at 50Hz)",
      { { "ROCPROFSYS_TRACE", true },
        { "ROCPROFSYS_PROFILE", true },
        { "ROCPROFSYS_USE_SAMPLING", true },
        { "ROCPROFSYS_SAMPLING_FREQ", 50 },
        { "ROCPROFSYS_USE_PROCESS_SAMPLING", true } } },

    { "profile-only",
      "Profiling-only mode: lightweight profiling without tracing "
      "(flat profile, minimal overhead)",
      { { "ROCPROFSYS_TRACE", false },
        { "ROCPROFSYS_PROFILE", true },
        { "ROCPROFSYS_USE_SAMPLING", true },
        { "ROCPROFSYS_SAMPLING_FREQ", 100 },
        { "ROCPROFSYS_USE_PROCESS_SAMPLING", false },
        { "ROCPROFSYS_FLAT_PROFILE", true } } },

    { "detailed",
      "Detailed profiling mode: full trace, profile, hardware counters, and "
      "process sampling",
      { { "ROCPROFSYS_TRACE", true },
        { "ROCPROFSYS_PROFILE", true },
        { "ROCPROFSYS_USE_SAMPLING", true },
        { "ROCPROFSYS_SAMPLING_FREQ", 100 },
        { "ROCPROFSYS_USE_PROCESS_SAMPLING", true },
        { "ROCPROFSYS_SAMPLING_CPUS", "all" } },
      true },

    { "trace-hpc",
      "HPC workload preset: optimized for MPI, OpenMP, and compute-intensive "
      "applications with hardware counter collection",
      { { "ROCPROFSYS_TRACE", true },
        { "ROCPROFSYS_PROFILE", true },
        { "ROCPROFSYS_USE_SAMPLING", false },
        { "ROCPROFSYS_SAMPLING_FREQ", 100 },
        { "ROCPROFSYS_USE_PROCESS_SAMPLING", true },
        { "ROCPROFSYS_USE_OMPT", true },
        { "ROCPROFSYS_USE_RCCL", true },
        { "ROCPROFSYS_USE_KOKKOSP", true },
        { "ROCPROFSYS_USE_MPIP", true },
        { "ROCPROFSYS_SAMPLING_CPUS", "none" },
        { "ROCPROFSYS_ROCM_DOMAINS", common_values::ROCM_DOMAINS_STANDARD },
        { "ROCPROFSYS_AMD_SMI_METRICS", common_values::AMD_SMI_METRICS_STANDARD },
        { "ROCPROFSYS_PAPI_EVENTS", common_values::PAPI_EVENTS_STANDARD } } },

    { "workload-trace",
      "General compute workload preset: optimized for AI/ML, HPC, and "
      "GPU workloads with comprehensive tracing and Python profiling",
      { { "ROCPROFSYS_TRACE", true },
        { "ROCPROFSYS_PROFILE", true },
        { "ROCPROFSYS_USE_SAMPLING", false },
        { "ROCPROFSYS_SAMPLING_FREQ", 50 },
        { "ROCPROFSYS_USE_PROCESS_SAMPLING", true },
        { "ROCPROFSYS_USE_MPIP", true },
        { "ROCPROFSYS_SAMPLING_CPUS", "none" },
        { "ROCPROFSYS_ROCM_DOMAINS", common_values::ROCM_DOMAINS_STANDARD },
        { "ROCPROFSYS_AMD_SMI_METRICS", common_values::AMD_SMI_METRICS_STANDARD },
        { "ROCPROFSYS_USE_ROCTRACER", true },
        { "ROCPROFSYS_TRACE_HIP_API", true },
        { "ROCPROFSYS_TRACE_HIP_ACTIVITY", true },
        { "ROCPROFSYS_USE_RCCL", true },
        { "ROCPROFSYS_USE_ROCPD", true },
        { "ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB", 2048000 } },
      true },

    { "sys-trace",
      "Comprehensive system API tracing: HIP API, HSA API, ROCTx, RCCL, "
      "rocDecode, rocJPEG, memory operations, and kernel dispatches",
      { { "ROCPROFSYS_TRACE", true },
        { "ROCPROFSYS_PROFILE", true },
        { "ROCPROFSYS_USE_ROCM", true },
        { "ROCPROFSYS_ROCM_DOMAINS", common_values::ROCM_DOMAINS_FULL } } },

    { "runtime-trace",
      "Runtime API tracing: HIP runtime API, ROCTx, RCCL, rocDecode, rocJPEG, "
      "memory operations, and kernel dispatches (excludes HIP compiler and HSA APIs)",
      { { "ROCPROFSYS_TRACE", true },
        { "ROCPROFSYS_PROFILE", true },
        { "ROCPROFSYS_USE_ROCM", true },
        { "ROCPROFSYS_ROCM_DOMAINS", common_values::ROCM_DOMAINS_RUNTIME } } },

    { "trace-gpu",
      "GPU workload analysis: trace with host functions, MPI, and device activity",
      { { "ROCPROFSYS_TRACE", true },
        { "ROCPROFSYS_PROFILE", false },
        { "ROCPROFSYS_USE_ROCM", true },
        { "ROCPROFSYS_USE_AMD_SMI", true },
        { "ROCPROFSYS_SAMPLING_CPUS", "none" },
        { "ROCPROFSYS_ROCM_DOMAINS", common_values::ROCM_DOMAINS_STANDARD } } },

    { "trace-openmp",
      "OpenMP offload workloads: tracing with HSA domains enabled",
      { { "ROCPROFSYS_TRACE", true },
        { "ROCPROFSYS_PROFILE", false },
        { "ROCPROFSYS_USE_ROCM", true },
        { "ROCPROFSYS_ROCM_DOMAINS", common_values::ROCM_DOMAINS_WITH_HSA },
        { "ROCPROFSYS_USE_OMPT", true } } },

    { "profile-mpi",
      "MPI communication latency profiling: flat profiling with wall-clock per rank",
      { { "ROCPROFSYS_TRACE", false },
        { "ROCPROFSYS_PROFILE", true },
        { "ROCPROFSYS_FLAT_PROFILE", true },
        { "ROCPROFSYS_USE_AMD_SMI", false },
        { "ROCPROFSYS_USE_ROCM", false } } },

    { "trace-hw-counters",
      "Hardware counter collection: GPU performance counters during execution",
      { { "ROCPROFSYS_PROFILE", true },
        { "ROCPROFSYS_SAMPLING_CPUS", "none" },
        { "ROCPROFSYS_ROCM_EVENTS", common_values::ROCM_EVENTS_STANDARD } } }
};

namespace descriptions
{
constexpr auto main = R"(
Call-stack sampling profiler for applications without binary instrumentation.
QUICK REFERENCE:
  Presets:  --balanced (default), --profile-only (minimal), --trace-hpc (HPC/MPI), --workload-trace (GPU/ML)
  Output:   Results saved to rocprof-sys-output/ directory
  Visualize: Open perfetto-trace.proto in https://ui.perfetto.dev
EXAMPLES:
  Quick Start:
    rocprof-sys-sample --balanced -- ./myapp
  Workload-Specific Presets:
    rocprof-sys-sample --trace-hpc -- ./hpc_app              # HPC/MPI/OpenMP
    rocprof-sys-sample --workload-trace -- python train.py   # AI/ML/GPU workloads
    rocprof-sys-sample --profile-only -- ./myapp             # Minimal overhead
  Custom Configuration:
    rocprof-sys-sample -f 100 --trace --hip-trace -- ./myapp
    rocprof-sys-sample -o ./results myrun -- ./myapp
    mpirun -n 4 rocprof-sys-sample --trace-hpc -- ./mpi_app
PROFILING WORKFLOW:
  1. Profile:   rocprof-sys-sample --balanced -- ./app
  2. Analyze:   cat rocprof-sys-output/wall_clock.txt
  3. Visualize: Open rocprof-sys-output/perfetto-trace.proto in ui.perfetto.dev
)";

}

}  
}  
