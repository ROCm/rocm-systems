// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string_view>
#include <vector>

namespace rocprofsys
{
namespace run
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

constexpr auto ON  = "ON";
constexpr auto OFF = "OFF";
}  // namespace common_values

struct PresetConfig
{
    std::string_view                                           name;
    std::string_view                                           description;
    std::vector<std::string_view>                              env_vars;
    std::vector<std::pair<std::string_view, std::string_view>> env_settings;
};

inline const std::vector<PresetConfig> PRESETS = {
    { "balanced",
      "Balanced profiling mode: moderate overhead with comprehensive data "
      "(tracing, call-stack profiling, and sampling)",
      { "ROCPROFSYS_TRACE", "ROCPROFSYS_PROFILE", "ROCPROFSYS_USE_SAMPLING",
        "ROCPROFSYS_USE_PROCESS_SAMPLING" },
      { { "ROCPROFSYS_TRACE", common_values::ON },
        { "ROCPROFSYS_PROFILE", common_values::ON },
        { "ROCPROFSYS_USE_SAMPLING", common_values::ON },
        { "ROCPROFSYS_USE_PROCESS_SAMPLING", common_values::ON },
        { "ROCPROFSYS_SAMPLING_FREQ", "50" } } },

    { "profile-only",
      "Profiling-only mode: lightweight profiling without tracing "
      "(flat profile, minimal overhead)",
      { "ROCPROFSYS_TRACE", "ROCPROFSYS_PROFILE", "ROCPROFSYS_FLAT_PROFILE" },
      { { "ROCPROFSYS_TRACE", common_values::OFF },
        { "ROCPROFSYS_PROFILE", common_values::ON },
        { "ROCPROFSYS_FLAT_PROFILE", common_values::ON } } },

    { "detailed",
      "Detailed profiling mode: full trace, profile, and system metrics",
      { "ROCPROFSYS_TRACE", "ROCPROFSYS_PROFILE", "ROCPROFSYS_USE_SAMPLING",
        "ROCPROFSYS_USE_PROCESS_SAMPLING", "ROCPROFSYS_SAMPLING_GPUS" },
      { { "ROCPROFSYS_TRACE", common_values::ON },
        { "ROCPROFSYS_PROFILE", common_values::ON },
        { "ROCPROFSYS_USE_SAMPLING", common_values::ON },
        { "ROCPROFSYS_USE_PROCESS_SAMPLING", common_values::ON },
        { "ROCPROFSYS_SAMPLING_CPUS", "all" } } },

    { "trace-hpc",
      "HPC workload preset: optimized for MPI, OpenMP, and compute-intensive "
      "applications with hardware counter collection",
      { "ROCPROFSYS_TRACE", "ROCPROFSYS_PROFILE", "ROCPROFSYS_USE_SAMPLING",
        "ROCPROFSYS_USE_PROCESS_SAMPLING", "ROCPROFSYS_USE_OMPT",
        "ROCPROFSYS_USE_KOKKOSP", "ROCPROFSYS_USE_RCCL", "ROCPROFSYS_USE_MPIP",
        "ROCPROFSYS_SAMPLING_CPUS", "ROCPROFSYS_ROCM_DOMAINS",
        "ROCPROFSYS_AMD_SMI_METRICS", "ROCPROFSYS_PAPI_EVENTS" },
      { { "ROCPROFSYS_TRACE", common_values::ON },
        { "ROCPROFSYS_PROFILE", common_values::ON },
        { "ROCPROFSYS_USE_SAMPLING", common_values::OFF },
        { "ROCPROFSYS_SAMPLING_FREQ", "100" },
        { "ROCPROFSYS_USE_PROCESS_SAMPLING", common_values::ON },
        { "ROCPROFSYS_USE_OMPT", common_values::ON },
        { "ROCPROFSYS_USE_RCCL", common_values::ON },
        { "ROCPROFSYS_USE_KOKKOSP", common_values::ON },
        { "ROCPROFSYS_USE_MPIP", common_values::ON },
        { "ROCPROFSYS_SAMPLING_CPUS", "none" },
        { "ROCPROFSYS_ROCM_DOMAINS", common_values::ROCM_DOMAINS_STANDARD },
        { "ROCPROFSYS_AMD_SMI_METRICS", common_values::AMD_SMI_METRICS_STANDARD },
        { "ROCPROFSYS_PAPI_EVENTS", common_values::PAPI_EVENTS_STANDARD } } },

    { "workload-trace",
      "General compute workload preset: optimized for AI/ML, HPC, and GPU workloads with "
      "comprehensive tracing and increased buffer size",
      { "ROCPROFSYS_TRACE", "ROCPROFSYS_PROFILE", "ROCPROFSYS_USE_SAMPLING",
        "ROCPROFSYS_USE_PROCESS_SAMPLING", "ROCPROFSYS_USE_MPIP",
        "ROCPROFSYS_SAMPLING_CPUS", "ROCPROFSYS_ROCM_DOMAINS",
        "ROCPROFSYS_AMD_SMI_METRICS", "ROCPROFSYS_SAMPLING_GPUS",
        "ROCPROFSYS_USE_ROCTRACER", "ROCPROFSYS_TRACE_HIP_API",
        "ROCPROFSYS_TRACE_HIP_ACTIVITY", "ROCPROFSYS_USE_RCCL", "ROCPROFSYS_USE_ROCPD",
        "ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB" },
      { { "ROCPROFSYS_TRACE", common_values::ON },
        { "ROCPROFSYS_PROFILE", common_values::ON },
        { "ROCPROFSYS_USE_SAMPLING", common_values::OFF },
        { "ROCPROFSYS_SAMPLING_FREQ", "50" },
        { "ROCPROFSYS_USE_PROCESS_SAMPLING", common_values::ON },
        { "ROCPROFSYS_USE_MPIP", common_values::ON },
        { "ROCPROFSYS_SAMPLING_CPUS", "none" },
        { "ROCPROFSYS_ROCM_DOMAINS", common_values::ROCM_DOMAINS_STANDARD },
        { "ROCPROFSYS_AMD_SMI_METRICS", common_values::AMD_SMI_METRICS_STANDARD },
        { "ROCPROFSYS_USE_ROCTRACER", common_values::ON },
        { "ROCPROFSYS_TRACE_HIP_API", common_values::ON },
        { "ROCPROFSYS_TRACE_HIP_ACTIVITY", common_values::ON },
        { "ROCPROFSYS_USE_RCCL", common_values::ON },
        { "ROCPROFSYS_USE_ROCPD", common_values::ON },
        { "ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB", "2048000" } } },

    { "sys-trace",
      "Comprehensive system API tracing: HIP API, HSA API, ROCTx, RCCL, "
      "rocDecode, rocJPEG, memory operations, and kernel dispatches",
      { "ROCPROFSYS_TRACE", "ROCPROFSYS_PROFILE", "ROCPROFSYS_USE_ROCM",
        "ROCPROFSYS_ROCM_DOMAINS" },
      { { "ROCPROFSYS_TRACE", common_values::ON },
        { "ROCPROFSYS_PROFILE", common_values::ON },
        { "ROCPROFSYS_USE_ROCM", common_values::ON },
        { "ROCPROFSYS_ROCM_DOMAINS", common_values::ROCM_DOMAINS_FULL } } },

    { "runtime-trace",
      "Runtime API tracing: HIP runtime API, ROCTx, RCCL, rocDecode, rocJPEG, "
      "memory operations, and kernel dispatches (excludes HIP compiler and HSA APIs)",
      { "ROCPROFSYS_TRACE", "ROCPROFSYS_PROFILE", "ROCPROFSYS_USE_ROCM",
        "ROCPROFSYS_ROCM_DOMAINS" },
      { { "ROCPROFSYS_TRACE", common_values::ON },
        { "ROCPROFSYS_PROFILE", common_values::ON },
        { "ROCPROFSYS_USE_ROCM", common_values::ON },
        { "ROCPROFSYS_ROCM_DOMAINS", common_values::ROCM_DOMAINS_RUNTIME } } },

    { "trace-gpu",
      "GPU workload analysis: trace with host functions, MPI, and device activity",
      { "ROCPROFSYS_TRACE", "ROCPROFSYS_PROFILE", "ROCPROFSYS_USE_ROCM",
        "ROCPROFSYS_USE_AMD_SMI", "ROCPROFSYS_SAMPLING_CPUS", "ROCPROFSYS_ROCM_DOMAINS" },
      { { "ROCPROFSYS_TRACE", common_values::ON },
        { "ROCPROFSYS_PROFILE", common_values::OFF },
        { "ROCPROFSYS_USE_ROCM", common_values::ON },
        { "ROCPROFSYS_USE_AMD_SMI", common_values::ON },
        { "ROCPROFSYS_SAMPLING_CPUS", "none" },
        { "ROCPROFSYS_ROCM_DOMAINS", common_values::ROCM_DOMAINS_STANDARD } } },

    { "trace-openmp",
      "OpenMP offload workloads: tracing with HSA domains enabled",
      { "ROCPROFSYS_TRACE", "ROCPROFSYS_PROFILE", "ROCPROFSYS_USE_ROCM",
        "ROCPROFSYS_ROCM_DOMAINS", "ROCPROFSYS_USE_OMPT" },
      { { "ROCPROFSYS_TRACE", common_values::ON },
        { "ROCPROFSYS_PROFILE", common_values::OFF },
        { "ROCPROFSYS_USE_ROCM", common_values::ON },
        { "ROCPROFSYS_ROCM_DOMAINS", common_values::ROCM_DOMAINS_WITH_HSA },
        { "ROCPROFSYS_USE_OMPT", common_values::ON } } },

    { "profile-mpi",
      "MPI communication latency profiling: flat profiling with wall-clock per rank",
      { "ROCPROFSYS_TRACE", "ROCPROFSYS_PROFILE", "ROCPROFSYS_FLAT_PROFILE",
        "ROCPROFSYS_USE_AMD_SMI", "ROCPROFSYS_USE_ROCM" },
      { { "ROCPROFSYS_TRACE", common_values::OFF },
        { "ROCPROFSYS_PROFILE", common_values::ON },
        { "ROCPROFSYS_FLAT_PROFILE", common_values::ON },
        { "ROCPROFSYS_USE_AMD_SMI", common_values::OFF },
        { "ROCPROFSYS_USE_ROCM", common_values::OFF } } },

    { "trace-hw-counters",
      "Hardware counter collection: GPU performance counters during execution",
      { "ROCPROFSYS_PROFILE", "ROCPROFSYS_SAMPLING_CPUS", "ROCPROFSYS_ROCM_EVENTS" },
      { { "ROCPROFSYS_PROFILE", common_values::ON },
        { "ROCPROFSYS_SAMPLING_CPUS", "none" },
        { "ROCPROFSYS_ROCM_EVENTS", common_values::ROCM_EVENTS_STANDARD } } }
};

namespace descriptions
{
constexpr auto main = R"desc(
Execute instrumented binaries with ROCm Systems Profiler configuration.
QUICK REFERENCE:
  Presets:  --balanced (default), --profile-only (minimal), --trace-hpc (HPC/MPI), --workload-trace (GPU/ML)
  Output:   Results saved to rocprof-sys-output/ directory
  Visualize: Open perfetto-trace.proto in https://ui.perfetto.dev
EXAMPLES:
  Quick Start:
    rocprof-sys-run --balanced -- ./myapp.inst
  Workload-Specific Presets:
    rocprof-sys-run --trace-hpc -- ./hpc_app.inst         # HPC/MPI/OpenMP
    rocprof-sys-run --workload-trace -- ./gpu_app.inst    # AI/ML/GPU workloads
    rocprof-sys-run --profile-only -- ./myapp.inst        # Minimal overhead
  Custom Configuration:
    rocprof-sys-run --trace-buffer-size=500000 -- ./myapp.inst
    rocprof-sys-run -o ./results -- ./myapp.inst
    mpirun -n 4 rocprof-sys-run --trace-hpc -- ./mpi_app.inst
INSTRUMENTATION WORKFLOW:
  1. Instrument: rocprof-sys-instrument -o app.inst -- ./app
  2. Run:        rocprof-sys-run --balanced -- ./app.inst
  3. Analyze:    cat rocprof-sys-output/wall_clock.txt
    )desc";

}

}  // namespace run
}  // namespace rocprofsys
