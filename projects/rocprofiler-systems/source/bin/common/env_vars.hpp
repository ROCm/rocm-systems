// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cctype>
#include <string_view>

namespace rocprofsys
{
namespace env_vars
{

// --- General ---
constexpr std::string_view ROOT               = "ROCPROFSYS_ROOT";
constexpr std::string_view MODE               = "ROCPROFSYS_MODE";
constexpr std::string_view SCRIPT_PATH        = "ROCPROFSYS_SCRIPT_PATH";
constexpr std::string_view CONFIG_FILE        = "ROCPROFSYS_CONFIG_FILE";
constexpr std::string_view PRESET_DIR         = "ROCPROFSYS_PRESET_DIR";
constexpr std::string_view MONOCHROME         = "ROCPROFSYS_MONOCHROME";
constexpr std::string_view LOG_LEVEL          = "ROCPROFSYS_LOG_LEVEL";
constexpr std::string_view TMPDIR             = "ROCPROFSYS_TMPDIR";
constexpr std::string_view ENABLE_CATEGORIES  = "ROCPROFSYS_ENABLE_CATEGORIES";
constexpr std::string_view DISABLE_CATEGORIES = "ROCPROFSYS_DISABLE_CATEGORIES";

// --- Tracing ---
constexpr std::string_view TRACE                   = "ROCPROFSYS_TRACE";
constexpr std::string_view TRACE_LEGACY            = "ROCPROFSYS_TRACE_LEGACY";
constexpr std::string_view PERFETTO_FILE           = "ROCPROFSYS_PERFETTO_FILE";
constexpr std::string_view PERFETTO_BUFFER_SIZE_KB = "ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB";
constexpr std::string_view PERFETTO_FILL_POLICY    = "ROCPROFSYS_PERFETTO_FILL_POLICY";
constexpr std::string_view PERFETTO_BACKEND        = "ROCPROFSYS_PERFETTO_BACKEND";
constexpr std::string_view PERFETTO_FLUSH_PERIOD = "ROCPROFSYS_PERFETTO_FLUSH_PERIOD_MS";
constexpr std::string_view TRACE_REGION          = "ROCPROFSYS_TRACE_REGION";
constexpr std::string_view TRACE_THREAD_LOCKS    = "ROCPROFSYS_TRACE_THREAD_LOCKS";
constexpr std::string_view TRACE_THREAD_RW_LOCKS = "ROCPROFSYS_TRACE_THREAD_RW_LOCKS";
constexpr std::string_view TRACE_THREAD_SPIN_LOCKS = "ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS";

// --- Profiling ---
constexpr std::string_view PROFILE      = "ROCPROFSYS_PROFILE";
constexpr std::string_view FLAT_PROFILE = "ROCPROFSYS_FLAT_PROFILE";

// --- Sampling ---
constexpr std::string_view USE_SAMPLING      = "ROCPROFSYS_USE_SAMPLING";
constexpr std::string_view SAMPLING_TIMER    = "ROCPROFSYS_SAMPLING_TIMER";
constexpr std::string_view SAMPLING_FREQ     = "ROCPROFSYS_SAMPLING_FREQ";
constexpr std::string_view SAMPLING_DELAY    = "ROCPROFSYS_SAMPLING_DELAY";
constexpr std::string_view SAMPLING_DURATION = "ROCPROFSYS_SAMPLING_DURATION";
constexpr std::string_view SAMPLING_CPUS     = "ROCPROFSYS_SAMPLING_CPUS";
constexpr std::string_view SAMPLING_GPUS     = "ROCPROFSYS_SAMPLING_GPUS";
constexpr std::string_view SAMPLING_AINICS   = "ROCPROFSYS_SAMPLING_AINICS";
constexpr std::string_view SAMPLING_TIDS     = "ROCPROFSYS_SAMPLING_TIDS";
constexpr std::string_view SAMPLING_INCLUDE_INLINES =
    "ROCPROFSYS_SAMPLING_INCLUDE_INLINES";
constexpr std::string_view SAMPLING_OVERFLOW_EVENT = "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT";

// --- Sampling: cputime timer ---
constexpr std::string_view SAMPLING_CPUTIME       = "ROCPROFSYS_SAMPLING_CPUTIME";
constexpr std::string_view SAMPLING_CPUTIME_FREQ  = "ROCPROFSYS_SAMPLING_CPUTIME_FREQ";
constexpr std::string_view SAMPLING_CPUTIME_DELAY = "ROCPROFSYS_SAMPLING_CPUTIME_DELAY";
constexpr std::string_view SAMPLING_CPUTIME_TIDS  = "ROCPROFSYS_SAMPLING_CPUTIME_TIDS";

// --- Sampling: realtime timer ---
constexpr std::string_view SAMPLING_REALTIME       = "ROCPROFSYS_SAMPLING_REALTIME";
constexpr std::string_view SAMPLING_REALTIME_FREQ  = "ROCPROFSYS_SAMPLING_REALTIME_FREQ";
constexpr std::string_view SAMPLING_REALTIME_DELAY = "ROCPROFSYS_SAMPLING_REALTIME_DELAY";
constexpr std::string_view SAMPLING_REALTIME_TIDS  = "ROCPROFSYS_SAMPLING_REALTIME_TIDS";

// --- Domains: GPU (AMD SMI) ---
constexpr std::string_view USE_AMD_SMI          = "ROCPROFSYS_USE_AMD_SMI";
constexpr std::string_view USE_PROCESS_SAMPLING = "ROCPROFSYS_USE_PROCESS_SAMPLING";
constexpr std::string_view AMD_SMI_METRICS      = "ROCPROFSYS_AMD_SMI_METRICS";
constexpr std::string_view AMD_SMI_FREQ         = "ROCPROFSYS_AMD_SMI_FREQ";
constexpr std::string_view CPU_FREQ_ENABLED     = "ROCPROFSYS_CPU_FREQ_ENABLED";

// --- Domains: ROCm ---
constexpr std::string_view ROCM_DOMAINS        = "ROCPROFSYS_ROCM_DOMAINS";
constexpr std::string_view ROCM_GROUP_BY_QUEUE = "ROCPROFSYS_ROCM_GROUP_BY_QUEUE";

// --- Domains: CPU ---
constexpr std::string_view CPU_FREQ = "ROCPROFSYS_CPU_FREQ";

// --- Domains: Parallel runtimes ---
constexpr std::string_view USE_MPIP    = "ROCPROFSYS_USE_MPIP";
constexpr std::string_view USE_OMPT    = "ROCPROFSYS_USE_OMPT";
constexpr std::string_view USE_KOKKOSP = "ROCPROFSYS_USE_KOKKOSP";
constexpr std::string_view USE_RCCLP   = "ROCPROFSYS_USE_RCCLP";
constexpr std::string_view USE_AINIC   = "ROCPROFSYS_USE_AINIC";
constexpr std::string_view USE_SHMEM   = "ROCPROFSYS_USE_SHMEM";
constexpr std::string_view USE_UCX     = "ROCPROFSYS_USE_UCX";

// --- Output ---
constexpr std::string_view OUTPUT_PATH   = "ROCPROFSYS_OUTPUT_PATH";
constexpr std::string_view OUTPUT_PREFIX = "ROCPROFSYS_OUTPUT_PREFIX";
constexpr std::string_view USE_PID       = "ROCPROFSYS_USE_PID";
constexpr std::string_view TIME_OUTPUT   = "ROCPROFSYS_TIME_OUTPUT";
constexpr std::string_view FILE_OUTPUT   = "ROCPROFSYS_FILE_OUTPUT";
constexpr std::string_view TEXT_OUTPUT   = "ROCPROFSYS_TEXT_OUTPUT";
constexpr std::string_view JSON_OUTPUT   = "ROCPROFSYS_JSON_OUTPUT";
constexpr std::string_view COUT_OUTPUT   = "ROCPROFSYS_COUT_OUTPUT";
constexpr std::string_view DIFF_OUTPUT   = "ROCPROFSYS_DIFF_OUTPUT";
constexpr std::string_view INPUT_PATH    = "ROCPROFSYS_INPUT_PATH";
constexpr std::string_view INPUT_PREFIX  = "ROCPROFSYS_INPUT_PREFIX";
constexpr std::string_view USE_ROCPD     = "ROCPROFSYS_USE_ROCPD";

// --- Process sampling ---
constexpr std::string_view PROCESS_SAMPLING_FREQ  = "ROCPROFSYS_PROCESS_SAMPLING_FREQ";
constexpr std::string_view PROCESS_SAMPLING_DELAY = "ROCPROFSYS_PROCESS_SAMPLING_DELAY";
constexpr std::string_view PROCESS_SAMPLING_DURATION =
    "ROCPROFSYS_PROCESS_SAMPLING_DURATION";
constexpr std::string_view SAMPLING_PROCESS_DURATION =
    "ROCPROFSYS_SAMPLING_PROCESS_DURATION";

// --- Causal profiling ---
constexpr std::string_view USE_CAUSAL              = "ROCPROFSYS_USE_CAUSAL";
constexpr std::string_view CAUSAL_MODE             = "ROCPROFSYS_CAUSAL_MODE";
constexpr std::string_view CAUSAL_BACKEND          = "ROCPROFSYS_CAUSAL_BACKEND";
constexpr std::string_view CAUSAL_VERBOSE          = "ROCPROFSYS_CAUSAL_VERBOSE";
constexpr std::string_view CAUSAL_DEBUG            = "ROCPROFSYS_CAUSAL_DEBUG";
constexpr std::string_view CAUSAL_BINARY_SCOPE     = "ROCPROFSYS_CAUSAL_BINARY_SCOPE";
constexpr std::string_view CAUSAL_BINARY_EXCLUDE   = "ROCPROFSYS_CAUSAL_BINARY_EXCLUDE";
constexpr std::string_view CAUSAL_FUNCTION_SCOPE   = "ROCPROFSYS_CAUSAL_FUNCTION_SCOPE";
constexpr std::string_view CAUSAL_FUNCTION_EXCLUDE = "ROCPROFSYS_CAUSAL_FUNCTION_EXCLUDE";
constexpr std::string_view CAUSAL_SOURCE_SCOPE     = "ROCPROFSYS_CAUSAL_SOURCE_SCOPE";
constexpr std::string_view CAUSAL_SOURCE_EXCLUDE   = "ROCPROFSYS_CAUSAL_SOURCE_EXCLUDE";
constexpr std::string_view CAUSAL_END_TO_END       = "ROCPROFSYS_CAUSAL_END_TO_END";
constexpr std::string_view CAUSAL_DELAY            = "ROCPROFSYS_CAUSAL_DELAY";
constexpr std::string_view CAUSAL_DURATION         = "ROCPROFSYS_CAUSAL_DURATION";
constexpr std::string_view CAUSAL_RANDOM_SEED      = "ROCPROFSYS_CAUSAL_RANDOM_SEED";

// --- Hardware counters ---
constexpr std::string_view ROCM_EVENTS       = "ROCPROFSYS_ROCM_EVENTS";
constexpr std::string_view PAPI_EVENTS       = "ROCPROFSYS_PAPI_EVENTS";
constexpr std::string_view PAPI_MULTIPLEXING = "ROCPROFSYS_PAPI_MULTIPLEXING";

// --- Advanced ---
constexpr std::string_view CPU_AFFINITY          = "ROCPROFSYS_CPU_AFFINITY";
constexpr std::string_view COLLAPSE_THREADS      = "ROCPROFSYS_COLLAPSE_THREADS";
constexpr std::string_view MAX_DEPTH             = "ROCPROFSYS_MAX_DEPTH";
constexpr std::string_view TRACE_DELAY           = "ROCPROFSYS_TRACE_DELAY";
constexpr std::string_view TRACE_DURATION        = "ROCPROFSYS_TRACE_DURATION";
constexpr std::string_view TRACE_PERIODS         = "ROCPROFSYS_TRACE_PERIODS";
constexpr std::string_view TRACE_PERIOD_CLOCK_ID = "ROCPROFSYS_TRACE_PERIOD_CLOCK_ID";
constexpr std::string_view VERBOSE               = "ROCPROFSYS_VERBOSE";
constexpr std::string_view DEBUG                 = "ROCPROFSYS_DEBUG";
// well above the highest verbose threshold (3) so debug mode enables all verbose output
constexpr int              DEBUG_VERBOSE_BOOST = 8;
constexpr std::string_view TIMEMORY_COMPONENTS = "ROCPROFSYS_TIMEMORY_COMPONENTS";
constexpr std::string_view NETWORK_INTERFACE   = "ROCPROFSYS_NETWORK_INTERFACE";

[[nodiscard]] inline int
log_level_to_verbose(std::string_view level) noexcept
{
    auto eq = [](std::string_view a, std::string_view b) noexcept {
        if(a.size() != b.size()) return false;
        for(std::size_t i = 0; i < a.size(); ++i)
            if(std::tolower(static_cast<unsigned char>(a[i])) !=
               std::tolower(static_cast<unsigned char>(b[i])))
                return false;
        return true;
    };
    if(eq(level, "trace")) return 2;
    if(eq(level, "debug")) return 1;
    if(eq(level, "info")) return 0;
    return -1;
}

}  // namespace env_vars
}  // namespace rocprofsys
