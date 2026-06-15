// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cctype>
#include <string_view>

namespace rocprofsys
{
namespace env_vars
{

// --- General ---
inline constexpr const char* ROOT               = "ROCPROFSYS_ROOT";
inline constexpr const char* MODE               = "ROCPROFSYS_MODE";
inline constexpr const char* SCRIPT_PATH        = "ROCPROFSYS_SCRIPT_PATH";
inline constexpr const char* CONFIG_FILE        = "ROCPROFSYS_CONFIG_FILE";
inline constexpr const char* PRESET_DIR         = "ROCPROFSYS_PRESET_DIR";
inline constexpr const char* MONOCHROME         = "ROCPROFSYS_MONOCHROME";
inline constexpr const char* LOG_LEVEL          = "ROCPROFSYS_LOG_LEVEL";
inline constexpr const char* LOG_FILE           = "ROCPROFSYS_LOG_FILE";
inline constexpr const char* LOG_COUNT          = "ROCPROFSYS_LOG_COUNT";
inline constexpr const char* TMPDIR             = "ROCPROFSYS_TMPDIR";
inline constexpr const char* ENABLE_CATEGORIES  = "ROCPROFSYS_ENABLE_CATEGORIES";
inline constexpr const char* DISABLE_CATEGORIES = "ROCPROFSYS_DISABLE_CATEGORIES";
inline constexpr const char* ENABLED            = "ROCPROFSYS_ENABLED";
inline constexpr const char* INIT_ENABLED       = "ROCPROFSYS_INIT_ENABLED";
inline constexpr const char* INIT_TOOLING       = "ROCPROFSYS_INIT_TOOLING";
inline constexpr const char* STRICT_CONFIG      = "ROCPROFSYS_STRICT_CONFIG";
inline constexpr const char* SUPPRESS_CONFIG    = "ROCPROFSYS_SUPPRESS_CONFIG";
inline constexpr const char* SUPPRESS_PARSING   = "ROCPROFSYS_SUPPRESS_PARSING";
inline constexpr const char* PRINT_ENV          = "ROCPROFSYS_PRINT_ENV";

// --- Tracing ---
inline constexpr const char* TRACE         = "ROCPROFSYS_TRACE";
inline constexpr const char* USE_TRACE     = "ROCPROFSYS_USE_TRACE";
inline constexpr const char* TRACE_LEGACY  = "ROCPROFSYS_TRACE_LEGACY";
inline constexpr const char* PERFETTO_FILE = "ROCPROFSYS_PERFETTO_FILE";
inline constexpr const char* PERFETTO_BUFFER_SIZE_KB =
    "ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB";
inline constexpr const char* PERFETTO_FILL_POLICY = "ROCPROFSYS_PERFETTO_FILL_POLICY";
inline constexpr const char* PERFETTO_BACKEND     = "ROCPROFSYS_PERFETTO_BACKEND";
inline constexpr const char* PERFETTO_BACKEND_SYSTEM =
    "ROCPROFSYS_PERFETTO_BACKEND_SYSTEM";
inline constexpr const char* PERFETTO_FLUSH_PERIOD =
    "ROCPROFSYS_PERFETTO_FLUSH_PERIOD_MS";
inline constexpr const char* PERFETTO_ANNOTATIONS = "ROCPROFSYS_PERFETTO_ANNOTATIONS";
inline constexpr const char* PERFETTO_COMBINE_TRACES =
    "ROCPROFSYS_PERFETTO_COMBINE_TRACES";
inline constexpr const char* PERFETTO_SHMEM_SIZE_HINT_KB =
    "ROCPROFSYS_PERFETTO_SHMEM_SIZE_HINT_KB";
inline constexpr const char* MERGE_PERFETTO_FILES  = "ROCPROFSYS_MERGE_PERFETTO_FILES";
inline constexpr const char* SELECTED_REGIONS      = "ROCPROFSYS_SELECTED_REGIONS";
inline constexpr const char* TRACE_THREAD_LOCKS    = "ROCPROFSYS_TRACE_THREAD_LOCKS";
inline constexpr const char* TRACE_THREAD_RW_LOCKS = "ROCPROFSYS_TRACE_THREAD_RW_LOCKS";
inline constexpr const char* TRACE_THREAD_SPIN_LOCKS =
    "ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS";
inline constexpr const char* TRACE_THREAD_BARRIERS = "ROCPROFSYS_TRACE_THREAD_BARRIERS";
inline constexpr const char* TRACE_THREAD_JOIN     = "ROCPROFSYS_TRACE_THREAD_JOIN";

// --- Profiling ---
inline constexpr const char* PROFILE          = "ROCPROFSYS_PROFILE";
inline constexpr const char* FLAT_PROFILE     = "ROCPROFSYS_FLAT_PROFILE";
inline constexpr const char* TIMELINE_PROFILE = "ROCPROFSYS_TIMELINE_PROFILE";

// --- Sampling ---
inline constexpr const char* USE_SAMPLING        = "ROCPROFSYS_USE_SAMPLING";
inline constexpr const char* USE_THREAD_SAMPLING = "ROCPROFSYS_USE_THREAD_SAMPLING";
inline constexpr const char* SAMPLING_TIMER      = "ROCPROFSYS_SAMPLING_TIMER";
inline constexpr const char* SAMPLING_FREQ       = "ROCPROFSYS_SAMPLING_FREQ";
inline constexpr const char* SAMPLING_DELAY      = "ROCPROFSYS_SAMPLING_DELAY";
inline constexpr const char* SAMPLING_DURATION   = "ROCPROFSYS_SAMPLING_DURATION";
inline constexpr const char* SAMPLING_CPUS       = "ROCPROFSYS_SAMPLING_CPUS";
inline constexpr const char* SAMPLING_GPUS       = "ROCPROFSYS_SAMPLING_GPUS";
inline constexpr const char* SAMPLING_AINICS     = "ROCPROFSYS_SAMPLING_AINICS";
inline constexpr const char* SAMPLING_TIDS       = "ROCPROFSYS_SAMPLING_TIDS";
inline constexpr const char* SAMPLING_INCLUDE_INLINES =
    "ROCPROFSYS_SAMPLING_INCLUDE_INLINES";
inline constexpr const char* SAMPLING_OVERFLOW_EVENT =
    "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT";
inline constexpr const char* SAMPLING_ALLOCATOR_SIZE =
    "ROCPROFSYS_SAMPLING_ALLOCATOR_SIZE";
inline constexpr const char* SAMPLING_KEEP_DYNINST_SUFFIX =
    "ROCPROFSYS_SAMPLING_KEEP_DYNINST_SUFFIX";
inline constexpr const char* SAMPLING_KEEP_INTERNAL = "ROCPROFSYS_SAMPLING_KEEP_INTERNAL";

// --- Sampling: cputime timer ---
inline constexpr const char* SAMPLING_CPUTIME       = "ROCPROFSYS_SAMPLING_CPUTIME";
inline constexpr const char* SAMPLING_CPUTIME_FREQ  = "ROCPROFSYS_SAMPLING_CPUTIME_FREQ";
inline constexpr const char* SAMPLING_CPUTIME_DELAY = "ROCPROFSYS_SAMPLING_CPUTIME_DELAY";
inline constexpr const char* SAMPLING_CPUTIME_TIDS  = "ROCPROFSYS_SAMPLING_CPUTIME_TIDS";
inline constexpr const char* SAMPLING_CPUTIME_SIGNAL =
    "ROCPROFSYS_SAMPLING_CPUTIME_SIGNAL";

// --- Sampling: realtime timer ---
inline constexpr const char* SAMPLING_REALTIME      = "ROCPROFSYS_SAMPLING_REALTIME";
inline constexpr const char* SAMPLING_REALTIME_FREQ = "ROCPROFSYS_SAMPLING_REALTIME_FREQ";
inline constexpr const char* SAMPLING_REALTIME_DELAY =
    "ROCPROFSYS_SAMPLING_REALTIME_DELAY";
inline constexpr const char* SAMPLING_REALTIME_TIDS = "ROCPROFSYS_SAMPLING_REALTIME_TIDS";
inline constexpr const char* SAMPLING_REALTIME_SIGNAL =
    "ROCPROFSYS_SAMPLING_REALTIME_SIGNAL";

// --- Sampling: overflow (PAPI event-based) ---
inline constexpr const char* SAMPLING_OVERFLOW      = "ROCPROFSYS_SAMPLING_OVERFLOW";
inline constexpr const char* SAMPLING_OVERFLOW_FREQ = "ROCPROFSYS_SAMPLING_OVERFLOW_FREQ";
inline constexpr const char* SAMPLING_OVERFLOW_TIDS = "ROCPROFSYS_SAMPLING_OVERFLOW_TIDS";
inline constexpr const char* SAMPLING_OVERFLOW_SIGNAL =
    "ROCPROFSYS_SAMPLING_OVERFLOW_SIGNAL";

// --- Domains: GPU (AMD SMI) ---
inline constexpr const char* USE_AMD_SMI          = "ROCPROFSYS_USE_AMD_SMI";
inline constexpr const char* USE_PROCESS_SAMPLING = "ROCPROFSYS_USE_PROCESS_SAMPLING";
inline constexpr const char* AMD_SMI_METRICS      = "ROCPROFSYS_AMD_SMI_METRICS";
inline constexpr const char* AMD_SMI_FREQ         = "ROCPROFSYS_AMD_SMI_FREQ";
inline constexpr const char* AMD_SMI_DEVICES      = "ROCPROFSYS_AMD_SMI_DEVICES";

// --- Domains: ROCm ---
inline constexpr const char* ROCM_DOMAINS        = "ROCPROFSYS_ROCM_DOMAINS";
inline constexpr const char* ROCM_GROUP_BY_QUEUE = "ROCPROFSYS_ROCM_GROUP_BY_QUEUE";
inline constexpr const char* GPU_PERF_COUNTERS   = "ROCPROFSYS_GPU_PERF_COUNTERS";

// --- Domains: CPU ---
inline constexpr const char* CPU_FREQ         = "ROCPROFSYS_CPU_FREQ";
inline constexpr const char* CPU_FREQ_ENABLED = "ROCPROFSYS_CPU_FREQ_ENABLED";
inline constexpr const char* CPU_METRICS      = "ROCPROFSYS_CPU_METRICS";

// --- Domains: Parallel runtimes ---
inline constexpr const char* USE_MPI     = "ROCPROFSYS_USE_MPI";
inline constexpr const char* USE_MPIP    = "ROCPROFSYS_USE_MPIP";
inline constexpr const char* USE_OMPT    = "ROCPROFSYS_USE_OMPT";
inline constexpr const char* USE_KOKKOSP = "ROCPROFSYS_USE_KOKKOSP";
inline constexpr const char* USE_RCCLP   = "ROCPROFSYS_USE_RCCLP";
inline constexpr const char* USE_AINIC   = "ROCPROFSYS_USE_AINIC";
inline constexpr const char* USE_SHMEM   = "ROCPROFSYS_USE_SHMEM";
inline constexpr const char* USE_UCX     = "ROCPROFSYS_USE_UCX";

// --- Domains: Shmem ---
inline constexpr const char* SHMEM_PERMIT_LIST = "ROCPROFSYS_SHMEM_PERMIT_LIST";
inline constexpr const char* SHMEM_REJECT_LIST = "ROCPROFSYS_SHMEM_REJECT_LIST";

// --- Output ---
inline constexpr const char* OUTPUT_PATH   = "ROCPROFSYS_OUTPUT_PATH";
inline constexpr const char* OUTPUT_PREFIX = "ROCPROFSYS_OUTPUT_PREFIX";
inline constexpr const char* OUTPUT        = "ROCPROFSYS_OUTPUT";
inline constexpr const char* OUTPUT_FILE   = "ROCPROFSYS_OUTPUT_FILE";
inline constexpr const char* OUTPUT_USE_CURRENT_TIME =
    "ROCPROFSYS_OUTPUT_USE_CURRENT_TIME";
inline constexpr const char* USE_PID             = "ROCPROFSYS_USE_PID";
inline constexpr const char* TIME_OUTPUT         = "ROCPROFSYS_TIME_OUTPUT";
inline constexpr const char* FILE_OUTPUT         = "ROCPROFSYS_FILE_OUTPUT";
inline constexpr const char* TEXT_OUTPUT         = "ROCPROFSYS_TEXT_OUTPUT";
inline constexpr const char* JSON_OUTPUT         = "ROCPROFSYS_JSON_OUTPUT";
inline constexpr const char* COUT_OUTPUT         = "ROCPROFSYS_COUT_OUTPUT";
inline constexpr const char* DIFF_OUTPUT         = "ROCPROFSYS_DIFF_OUTPUT";
inline constexpr const char* TREE_OUTPUT         = "ROCPROFSYS_TREE_OUTPUT";
inline constexpr const char* INPUT_PATH          = "ROCPROFSYS_INPUT_PATH";
inline constexpr const char* INPUT_PREFIX        = "ROCPROFSYS_INPUT_PREFIX";
inline constexpr const char* INPUT_EXTENSIONS    = "ROCPROFSYS_INPUT_EXTENSIONS";
inline constexpr const char* USE_ROCPD           = "ROCPROFSYS_USE_ROCPD";
inline constexpr const char* USE_TEMPORARY_FILES = "ROCPROFSYS_USE_TEMPORARY_FILES";
inline constexpr const char* USE_UNIFIED_MEMORY_PROFILING =
    "ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING";
inline constexpr const char* TIME_FORMAT = "ROCPROFSYS_TIME_FORMAT";

// --- Output: number formatting ---
inline constexpr const char* PRECISION         = "ROCPROFSYS_PRECISION";
inline constexpr const char* SCIENTIFIC        = "ROCPROFSYS_SCIENTIFIC";
inline constexpr const char* WIDTH             = "ROCPROFSYS_WIDTH";
inline constexpr const char* MAX_WIDTH         = "ROCPROFSYS_MAX_WIDTH";
inline constexpr const char* TIMING_PRECISION  = "ROCPROFSYS_TIMING_PRECISION";
inline constexpr const char* TIMING_SCIENTIFIC = "ROCPROFSYS_TIMING_SCIENTIFIC";
inline constexpr const char* TIMING_UNITS      = "ROCPROFSYS_TIMING_UNITS";
inline constexpr const char* TIMING_WIDTH      = "ROCPROFSYS_TIMING_WIDTH";
inline constexpr const char* MEMORY_PRECISION  = "ROCPROFSYS_MEMORY_PRECISION";
inline constexpr const char* MEMORY_SCIENTIFIC = "ROCPROFSYS_MEMORY_SCIENTIFIC";
inline constexpr const char* MEMORY_UNITS      = "ROCPROFSYS_MEMORY_UNITS";
inline constexpr const char* MEMORY_WIDTH      = "ROCPROFSYS_MEMORY_WIDTH";

// --- MPI output filtering ---
inline constexpr const char* RANK_FILTER_ID     = "ROCPROFSYS_RANK_FILTER_ID";
inline constexpr const char* RANK_FILTER_OUTPUT = "ROCPROFSYS_RANK_FILTER_OUTPUT";
inline constexpr const char* RANK_FILTER_LOGS   = "ROCPROFSYS_RANK_FILTER_LOGS";

// --- Process sampling ---
inline constexpr const char* PROCESS_SAMPLING_FREQ  = "ROCPROFSYS_PROCESS_SAMPLING_FREQ";
inline constexpr const char* PROCESS_SAMPLING_DELAY = "ROCPROFSYS_PROCESS_SAMPLING_DELAY";
inline constexpr const char* PROCESS_SAMPLING_DURATION =
    "ROCPROFSYS_PROCESS_SAMPLING_DURATION";
inline constexpr const char* SAMPLING_PROCESS_DURATION =
    "ROCPROFSYS_SAMPLING_PROCESS_DURATION";

// --- Causal profiling ---
inline constexpr const char* USE_CAUSAL            = "ROCPROFSYS_USE_CAUSAL";
inline constexpr const char* CAUSAL_MODE           = "ROCPROFSYS_CAUSAL_MODE";
inline constexpr const char* CAUSAL_BACKEND        = "ROCPROFSYS_CAUSAL_BACKEND";
inline constexpr const char* CAUSAL_VERBOSE        = "ROCPROFSYS_CAUSAL_VERBOSE";
inline constexpr const char* CAUSAL_DEBUG          = "ROCPROFSYS_CAUSAL_DEBUG";
inline constexpr const char* CAUSAL_BINARY_SCOPE   = "ROCPROFSYS_CAUSAL_BINARY_SCOPE";
inline constexpr const char* CAUSAL_BINARY_EXCLUDE = "ROCPROFSYS_CAUSAL_BINARY_EXCLUDE";
inline constexpr const char* CAUSAL_FUNCTION_SCOPE = "ROCPROFSYS_CAUSAL_FUNCTION_SCOPE";
inline constexpr const char* CAUSAL_FUNCTION_EXCLUDE =
    "ROCPROFSYS_CAUSAL_FUNCTION_EXCLUDE";
inline constexpr const char* CAUSAL_FUNCTION_EXCLUDE_DEFAULTS =
    "ROCPROFSYS_CAUSAL_FUNCTION_EXCLUDE_DEFAULTS";
inline constexpr const char* CAUSAL_SOURCE_SCOPE   = "ROCPROFSYS_CAUSAL_SOURCE_SCOPE";
inline constexpr const char* CAUSAL_SOURCE_EXCLUDE = "ROCPROFSYS_CAUSAL_SOURCE_EXCLUDE";
inline constexpr const char* CAUSAL_END_TO_END     = "ROCPROFSYS_CAUSAL_END_TO_END";
inline constexpr const char* CAUSAL_DELAY          = "ROCPROFSYS_CAUSAL_DELAY";
inline constexpr const char* CAUSAL_DURATION       = "ROCPROFSYS_CAUSAL_DURATION";
inline constexpr const char* CAUSAL_RANDOM_SEED    = "ROCPROFSYS_CAUSAL_RANDOM_SEED";
inline constexpr const char* CAUSAL_FIXED_SPEEDUP  = "ROCPROFSYS_CAUSAL_FIXED_SPEEDUP";
inline constexpr const char* CAUSAL_SPEEDUP_DIVISIONS =
    "ROCPROFSYS_CAUSAL_SPEEDUP_DIVISIONS";
inline constexpr const char* CAUSAL_SCALE_EXPERIMENT_TIME_BY_SPEEDUP =
    "ROCPROFSYS_CAUSAL_SCALE_EXPERIMENT_TIME_BY_SPEEDUP";
inline constexpr const char* CAUSAL_FILE       = "ROCPROFSYS_CAUSAL_FILE";
inline constexpr const char* CAUSAL_FILE_RESET = "ROCPROFSYS_CAUSAL_FILE_RESET";

// --- Hardware counters ---
// Note: PAPI_MULTIPLEXING and PAPI_QUIET would collide with integer macros defined in
// PAPI's C header (papi.h). The identifiers carry a trailing suffix to avoid
// preprocessor substitution; the env-var strings retain the original names.
inline constexpr const char* ROCM_EVENTS               = "ROCPROFSYS_ROCM_EVENTS";
inline constexpr const char* PAPI_EVENTS               = "ROCPROFSYS_PAPI_EVENTS";
inline constexpr const char* PAPI_MULTIPLEXING_ENABLED = "ROCPROFSYS_PAPI_MULTIPLEXING";
inline constexpr const char* PAPI_FAIL_ON_ERROR        = "ROCPROFSYS_PAPI_FAIL_ON_ERROR";
inline constexpr const char* PAPI_OVERFLOW             = "ROCPROFSYS_PAPI_OVERFLOW";
inline constexpr const char* PAPI_QUIET_MODE           = "ROCPROFSYS_PAPI_QUIET";
inline constexpr const char* PAPI_THREADING            = "ROCPROFSYS_PAPI_THREADING";
inline constexpr const char* USE_CODE_COVERAGE         = "ROCPROFSYS_USE_CODE_COVERAGE";

// --- MPI ---
inline constexpr const char* MPI_INIT             = "ROCPROFSYS_MPI_INIT";
inline constexpr const char* MPI_FINALIZE         = "ROCPROFSYS_MPI_FINALIZE";
inline constexpr const char* MPI_FAIL_ON_ERROR    = "ROCPROFSYS_MPI_FAIL_ON_ERROR";
inline constexpr const char* MPI_QUIET            = "ROCPROFSYS_MPI_QUIET";
inline constexpr const char* MPI_THREAD           = "ROCPROFSYS_MPI_THREAD";
inline constexpr const char* MPI_THREAD_TYPE      = "ROCPROFSYS_MPI_THREAD_TYPE";
inline constexpr const char* MPI_MAX_COMM_UPDATES = "ROCPROFSYS_MPI_MAX_COMM_UPDATES";

// --- Kokkos profiling ---
inline constexpr const char* KOKKOSP_PREFIX        = "ROCPROFSYS_KOKKOSP_PREFIX";
inline constexpr const char* KOKKOSP_KERNEL_LOGGER = "ROCPROFSYS_KOKKOSP_KERNEL_LOGGER";
inline constexpr const char* KOKKOSP_NAME_LENGTH_MAX =
    "ROCPROFSYS_KOKKOSP_NAME_LENGTH_MAX";
inline constexpr const char* KOKKOSP_DEEP_COPY = "ROCPROFSYS_KOKKOSP_DEEP_COPY";

// --- DL (dynamic loader / preload) ---
inline constexpr const char* DL_VERBOSE = "ROCPROFSYS_DL_VERBOSE";
inline constexpr const char* DL_DEBUG   = "ROCPROFSYS_DL_DEBUG";
inline constexpr const char* DL_LIBRARY = "ROCPROFSYS_DL_LIBRARY";

// --- Regex include/exclude filters ---
inline constexpr const char* REGEX_INCLUDE          = "ROCPROFSYS_REGEX_INCLUDE";
inline constexpr const char* REGEX_EXCLUDE          = "ROCPROFSYS_REGEX_EXCLUDE";
inline constexpr const char* REGEX_RESTRICT         = "ROCPROFSYS_REGEX_RESTRICT";
inline constexpr const char* REGEX_CALLER_INCLUDE   = "ROCPROFSYS_REGEX_CALLER_INCLUDE";
inline constexpr const char* REGEX_INTERNAL_INCLUDE = "ROCPROFSYS_REGEX_INTERNAL_INCLUDE";
inline constexpr const char* REGEX_INSTRUCTION_EXCLUDE =
    "ROCPROFSYS_REGEX_INSTRUCTION_EXCLUDE";
inline constexpr const char* REGEX_MODULE_INCLUDE  = "ROCPROFSYS_REGEX_MODULE_INCLUDE";
inline constexpr const char* REGEX_MODULE_EXCLUDE  = "ROCPROFSYS_REGEX_MODULE_EXCLUDE";
inline constexpr const char* REGEX_MODULE_RESTRICT = "ROCPROFSYS_REGEX_MODULE_RESTRICT";
inline constexpr const char* REGEX_MODULE_INTERNAL_INCLUDE =
    "ROCPROFSYS_REGEX_MODULE_INTERNAL_INCLUDE";

// --- CI / continuous integration ---
inline constexpr const char* CI                  = "ROCPROFSYS_CI";
inline constexpr const char* CI_TIMEOUT          = "ROCPROFSYS_CI_TIMEOUT";
inline constexpr const char* CI_TIMEOUT_COUNT    = "ROCPROFSYS_CI_TIMEOUT_COUNT";
inline constexpr const char* CI_TIMEOUT_OVERRIDE = "ROCPROFSYS_CI_TIMEOUT_OVERRIDE";

// --- Process / threading / behavior ---
inline constexpr const char* NUM_THREADS        = "ROCPROFSYS_NUM_THREADS";
inline constexpr const char* NUM_THREADS_HINT   = "ROCPROFSYS_NUM_THREADS_HINT";
inline constexpr const char* THREAD_POOL_SIZE   = "ROCPROFSYS_THREAD_POOL_SIZE";
inline constexpr const char* RECYCLE_TIDS       = "ROCPROFSYS_RECYCLE_TIDS";
inline constexpr const char* KILL_DELAY         = "ROCPROFSYS_KILL_DELAY";
inline constexpr const char* COLLAPSE_PROCESSES = "ROCPROFSYS_COLLAPSE_PROCESSES";
inline constexpr const char* NODE_COUNT         = "ROCPROFSYS_NODE_COUNT";
inline constexpr const char* ROOT_PROCESS       = "ROCPROFSYS_ROOT_PROCESS";
inline constexpr const char* REATTACH_ADD_SESSION_ID =
    "ROCPROFSYS_REATTACH_ADD_SESSION_ID";

// --- Instrumentation ---
inline constexpr const char* INSTRUMENT_MODE = "ROCPROFSYS_INSTRUMENT_MODE";
inline constexpr const char* IGNORE_DYNINST_TRAMPOLINE =
    "ROCPROFSYS_IGNORE_DYNINST_TRAMPOLINE";
inline constexpr const char* DEFAULT_MIN_INSTRUCTIONS =
    "ROCPROFSYS_DEFAULT_MIN_INSTRUCTIONS";

// --- Runtime / launcher (set by instrumenter / read by the loaded library) ---
inline constexpr const char* PATH                   = "ROCPROFSYS_PATH";
inline constexpr const char* PRELOAD                = "ROCPROFSYS_PRELOAD";
inline constexpr const char* LIBRARY                = "ROCPROFSYS_LIBRARY";
inline constexpr const char* USER_LIBRARY           = "ROCPROFSYS_USER_LIBRARY";
inline constexpr const char* COMMAND_LINE           = "ROCPROFSYS_COMMAND_LINE";
inline constexpr const char* LAUNCHER               = "ROCPROFSYS_LAUNCHER";
inline constexpr const char* SCRIPT_DIR             = "ROCPROFSYS_SCRIPT_DIR";
inline constexpr const char* ROCM_PATH              = "ROCPROFSYS_ROCM_PATH";
inline constexpr const char* CONFIG                 = "ROCPROFSYS_CONFIG";
inline constexpr const char* ENVIRONMENT            = "ROCPROFSYS_ENVIRONMENT";
inline constexpr const char* SETTINGS_DESC          = "ROCPROFSYS_SETTINGS_DESC";
inline constexpr const char* SETTINGS_DESC_MARKDOWN = "ROCPROFSYS_SETTINGS_DESC_MARKDOWN";
inline constexpr const char* CRAYPAT                = "ROCPROFSYS_CRAYPAT";

// --- Advanced ---
inline constexpr const char* CPU_AFFINITY          = "ROCPROFSYS_CPU_AFFINITY";
inline constexpr const char* COLLAPSE_THREADS      = "ROCPROFSYS_COLLAPSE_THREADS";
inline constexpr const char* MAX_DEPTH             = "ROCPROFSYS_MAX_DEPTH";
inline constexpr const char* TRACE_DELAY           = "ROCPROFSYS_TRACE_DELAY";
inline constexpr const char* TRACE_DURATION        = "ROCPROFSYS_TRACE_DURATION";
inline constexpr const char* TRACE_PERIODS         = "ROCPROFSYS_TRACE_PERIODS";
inline constexpr const char* TRACE_PERIOD_CLOCK_ID = "ROCPROFSYS_TRACE_PERIOD_CLOCK_ID";
inline constexpr const char* VERBOSE               = "ROCPROFSYS_VERBOSE";
inline constexpr const char* VERBOSE_AVAIL         = "ROCPROFSYS_VERBOSE_AVAIL";
inline constexpr const char* VERBOSE_INSTRUMENT    = "ROCPROFSYS_VERBOSE_INSTRUMENT";
// Note: identifier is DEBUG_MODE to avoid collision with `#define DEBUG 1` injected by
// the project's generated common/defines.h on CI builds (ROCPROFSYS_CI > 0). The
// env-var string itself remains "ROCPROFSYS_DEBUG".
inline constexpr const char* DEBUG_MODE = "ROCPROFSYS_DEBUG";
// well above the highest verbose threshold (3) so debug mode enables all verbose output
constexpr int                DEBUG_VERBOSE_BOOST   = 8;
inline constexpr const char* DEBUG_INIT            = "ROCPROFSYS_DEBUG_INIT";
inline constexpr const char* DEBUG_FINALIZE        = "ROCPROFSYS_DEBUG_FINALIZE";
inline constexpr const char* DEBUG_AVAIL           = "ROCPROFSYS_DEBUG_AVAIL";
inline constexpr const char* DEBUG_MARK            = "ROCPROFSYS_DEBUG_MARK";
inline constexpr const char* DEBUG_PIDS            = "ROCPROFSYS_DEBUG_PIDS";
inline constexpr const char* DEBUG_TIDS            = "ROCPROFSYS_DEBUG_TIDS";
inline constexpr const char* DEBUG_PUSH            = "ROCPROFSYS_DEBUG_PUSH";
inline constexpr const char* DEBUG_POP             = "ROCPROFSYS_DEBUG_POP";
inline constexpr const char* DEBUG_SAMPLING        = "ROCPROFSYS_DEBUG_SAMPLING";
inline constexpr const char* DEBUG_USER_REGIONS    = "ROCPROFSYS_DEBUG_USER_REGIONS";
inline constexpr const char* ENABLE_SIGNAL_HANDLER = "ROCPROFSYS_ENABLE_SIGNAL_HANDLER";
inline constexpr const char* TIMEMORY_COMPONENTS   = "ROCPROFSYS_TIMEMORY_COMPONENTS";
inline constexpr const char* NETWORK_INTERFACE     = "ROCPROFSYS_NETWORK_INTERFACE";

// --- Deprecated aliases ---
// Names retained only for legacy/migration paths — the codebase emits deprecation
// warnings when these are encountered. Do not introduce new references; prefer the
// replacement (TRACE for USE_PERFETTO, PROFILE for USE_TIMEMORY).
inline constexpr const char* USE_PERFETTO = "ROCPROFSYS_USE_PERFETTO";
inline constexpr const char* USE_TIMEMORY = "ROCPROFSYS_USE_TIMEMORY";

[[nodiscard]] inline int
log_level_to_verbose(std::string_view level) noexcept
{
    auto iequal = [](std::string_view lhs, std::string_view rhs) noexcept {
        if(lhs.size() != rhs.size()) return false;
        for(std::size_t idx = 0; idx < lhs.size(); ++idx)
            if(std::tolower(static_cast<unsigned char>(lhs[idx])) !=
               std::tolower(static_cast<unsigned char>(rhs[idx])))
                return false;
        return true;
    };
    if(iequal(level, "trace")) return 2;
    if(iequal(level, "debug")) return 1;
    if(iequal(level, "info")) return 0;
    return -1;
}

}  // namespace env_vars
}  // namespace rocprofsys
