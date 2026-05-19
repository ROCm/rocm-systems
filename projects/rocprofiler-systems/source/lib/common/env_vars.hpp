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
constexpr std::string_view ROOT               = "ROCPROFSYS_ROOT";
constexpr std::string_view MODE               = "ROCPROFSYS_MODE";
constexpr std::string_view SCRIPT_PATH        = "ROCPROFSYS_SCRIPT_PATH";
constexpr std::string_view CONFIG_FILE        = "ROCPROFSYS_CONFIG_FILE";
constexpr std::string_view PRESET_DIR         = "ROCPROFSYS_PRESET_DIR";
constexpr std::string_view MONOCHROME         = "ROCPROFSYS_MONOCHROME";
constexpr std::string_view LOG_LEVEL          = "ROCPROFSYS_LOG_LEVEL";
constexpr std::string_view LOG_FILE           = "ROCPROFSYS_LOG_FILE";
constexpr std::string_view LOG_COUNT          = "ROCPROFSYS_LOG_COUNT";
constexpr std::string_view TMPDIR             = "ROCPROFSYS_TMPDIR";
constexpr std::string_view ENABLE_CATEGORIES  = "ROCPROFSYS_ENABLE_CATEGORIES";
constexpr std::string_view DISABLE_CATEGORIES = "ROCPROFSYS_DISABLE_CATEGORIES";
constexpr std::string_view ENABLED            = "ROCPROFSYS_ENABLED";
constexpr std::string_view INIT_ENABLED       = "ROCPROFSYS_INIT_ENABLED";
constexpr std::string_view INIT_TOOLING       = "ROCPROFSYS_INIT_TOOLING";
constexpr std::string_view STRICT_CONFIG      = "ROCPROFSYS_STRICT_CONFIG";
constexpr std::string_view SUPPRESS_CONFIG    = "ROCPROFSYS_SUPPRESS_CONFIG";
constexpr std::string_view SUPPRESS_PARSING   = "ROCPROFSYS_SUPPRESS_PARSING";
constexpr std::string_view PRINT_ENV          = "ROCPROFSYS_PRINT_ENV";

// --- Tracing ---
constexpr std::string_view TRACE                   = "ROCPROFSYS_TRACE";
constexpr std::string_view USE_TRACE               = "ROCPROFSYS_USE_TRACE";
constexpr std::string_view TRACE_LEGACY            = "ROCPROFSYS_TRACE_LEGACY";
constexpr std::string_view PERFETTO_FILE           = "ROCPROFSYS_PERFETTO_FILE";
constexpr std::string_view PERFETTO_BUFFER_SIZE_KB = "ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB";
constexpr std::string_view PERFETTO_FILL_POLICY    = "ROCPROFSYS_PERFETTO_FILL_POLICY";
constexpr std::string_view PERFETTO_BACKEND        = "ROCPROFSYS_PERFETTO_BACKEND";
constexpr std::string_view PERFETTO_BACKEND_SYSTEM = "ROCPROFSYS_PERFETTO_BACKEND_SYSTEM";
constexpr std::string_view PERFETTO_FLUSH_PERIOD = "ROCPROFSYS_PERFETTO_FLUSH_PERIOD_MS";
constexpr std::string_view PERFETTO_ANNOTATIONS  = "ROCPROFSYS_PERFETTO_ANNOTATIONS";
constexpr std::string_view PERFETTO_COMBINE_TRACES = "ROCPROFSYS_PERFETTO_COMBINE_TRACES";
constexpr std::string_view PERFETTO_SHMEM_SIZE_HINT_KB =
    "ROCPROFSYS_PERFETTO_SHMEM_SIZE_HINT_KB";
constexpr std::string_view MERGE_PERFETTO_FILES    = "ROCPROFSYS_MERGE_PERFETTO_FILES";
constexpr std::string_view SELECTED_REGIONS        = "ROCPROFSYS_SELECTED_REGIONS";
constexpr std::string_view TRACE_THREAD_LOCKS      = "ROCPROFSYS_TRACE_THREAD_LOCKS";
constexpr std::string_view TRACE_THREAD_RW_LOCKS   = "ROCPROFSYS_TRACE_THREAD_RW_LOCKS";
constexpr std::string_view TRACE_THREAD_SPIN_LOCKS = "ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS";
constexpr std::string_view TRACE_THREAD_BARRIERS   = "ROCPROFSYS_TRACE_THREAD_BARRIERS";
constexpr std::string_view TRACE_THREAD_JOIN       = "ROCPROFSYS_TRACE_THREAD_JOIN";

// --- Profiling ---
constexpr std::string_view PROFILE          = "ROCPROFSYS_PROFILE";
constexpr std::string_view FLAT_PROFILE     = "ROCPROFSYS_FLAT_PROFILE";
constexpr std::string_view TIMELINE_PROFILE = "ROCPROFSYS_TIMELINE_PROFILE";

// --- Sampling ---
constexpr std::string_view USE_SAMPLING        = "ROCPROFSYS_USE_SAMPLING";
constexpr std::string_view USE_THREAD_SAMPLING = "ROCPROFSYS_USE_THREAD_SAMPLING";
constexpr std::string_view SAMPLING_TIMER      = "ROCPROFSYS_SAMPLING_TIMER";
constexpr std::string_view SAMPLING_FREQ       = "ROCPROFSYS_SAMPLING_FREQ";
constexpr std::string_view SAMPLING_DELAY      = "ROCPROFSYS_SAMPLING_DELAY";
constexpr std::string_view SAMPLING_DURATION   = "ROCPROFSYS_SAMPLING_DURATION";
constexpr std::string_view SAMPLING_CPUS       = "ROCPROFSYS_SAMPLING_CPUS";
constexpr std::string_view SAMPLING_GPUS       = "ROCPROFSYS_SAMPLING_GPUS";
constexpr std::string_view SAMPLING_AINICS     = "ROCPROFSYS_SAMPLING_AINICS";
constexpr std::string_view SAMPLING_TIDS       = "ROCPROFSYS_SAMPLING_TIDS";
constexpr std::string_view SAMPLING_INCLUDE_INLINES =
    "ROCPROFSYS_SAMPLING_INCLUDE_INLINES";
constexpr std::string_view SAMPLING_OVERFLOW_EVENT = "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT";
constexpr std::string_view SAMPLING_ALLOCATOR_SIZE = "ROCPROFSYS_SAMPLING_ALLOCATOR_SIZE";
constexpr std::string_view SAMPLING_KEEP_DYNINST_SUFFIX =
    "ROCPROFSYS_SAMPLING_KEEP_DYNINST_SUFFIX";
constexpr std::string_view SAMPLING_KEEP_INTERNAL = "ROCPROFSYS_SAMPLING_KEEP_INTERNAL";

// --- Sampling: cputime timer ---
constexpr std::string_view SAMPLING_CPUTIME        = "ROCPROFSYS_SAMPLING_CPUTIME";
constexpr std::string_view SAMPLING_CPUTIME_FREQ   = "ROCPROFSYS_SAMPLING_CPUTIME_FREQ";
constexpr std::string_view SAMPLING_CPUTIME_DELAY  = "ROCPROFSYS_SAMPLING_CPUTIME_DELAY";
constexpr std::string_view SAMPLING_CPUTIME_TIDS   = "ROCPROFSYS_SAMPLING_CPUTIME_TIDS";
constexpr std::string_view SAMPLING_CPUTIME_SIGNAL = "ROCPROFSYS_SAMPLING_CPUTIME_SIGNAL";

// --- Sampling: realtime timer ---
constexpr std::string_view SAMPLING_REALTIME       = "ROCPROFSYS_SAMPLING_REALTIME";
constexpr std::string_view SAMPLING_REALTIME_FREQ  = "ROCPROFSYS_SAMPLING_REALTIME_FREQ";
constexpr std::string_view SAMPLING_REALTIME_DELAY = "ROCPROFSYS_SAMPLING_REALTIME_DELAY";
constexpr std::string_view SAMPLING_REALTIME_TIDS  = "ROCPROFSYS_SAMPLING_REALTIME_TIDS";
constexpr std::string_view SAMPLING_REALTIME_SIGNAL =
    "ROCPROFSYS_SAMPLING_REALTIME_SIGNAL";

// --- Sampling: overflow (PAPI event-based) ---
constexpr std::string_view SAMPLING_OVERFLOW      = "ROCPROFSYS_SAMPLING_OVERFLOW";
constexpr std::string_view SAMPLING_OVERFLOW_FREQ = "ROCPROFSYS_SAMPLING_OVERFLOW_FREQ";
constexpr std::string_view SAMPLING_OVERFLOW_TIDS = "ROCPROFSYS_SAMPLING_OVERFLOW_TIDS";
constexpr std::string_view SAMPLING_OVERFLOW_SIGNAL =
    "ROCPROFSYS_SAMPLING_OVERFLOW_SIGNAL";

// --- Domains: GPU (AMD SMI) ---
constexpr std::string_view USE_AMD_SMI          = "ROCPROFSYS_USE_AMD_SMI";
constexpr std::string_view USE_PROCESS_SAMPLING = "ROCPROFSYS_USE_PROCESS_SAMPLING";
constexpr std::string_view AMD_SMI_METRICS      = "ROCPROFSYS_AMD_SMI_METRICS";
constexpr std::string_view AMD_SMI_FREQ         = "ROCPROFSYS_AMD_SMI_FREQ";
constexpr std::string_view AMD_SMI_DEVICES      = "ROCPROFSYS_AMD_SMI_DEVICES";

// --- Domains: ROCm ---
constexpr std::string_view ROCM_DOMAINS        = "ROCPROFSYS_ROCM_DOMAINS";
constexpr std::string_view ROCM_GROUP_BY_QUEUE = "ROCPROFSYS_ROCM_GROUP_BY_QUEUE";
constexpr std::string_view GPU_PERF_COUNTERS   = "ROCPROFSYS_GPU_PERF_COUNTERS";

// --- Domains: CPU ---
constexpr std::string_view CPU_FREQ         = "ROCPROFSYS_CPU_FREQ";
constexpr std::string_view CPU_FREQ_ENABLED = "ROCPROFSYS_CPU_FREQ_ENABLED";
constexpr std::string_view CPU_METRICS      = "ROCPROFSYS_CPU_METRICS";

// --- Domains: Parallel runtimes ---
constexpr std::string_view USE_MPI     = "ROCPROFSYS_USE_MPI";
constexpr std::string_view USE_MPIP    = "ROCPROFSYS_USE_MPIP";
constexpr std::string_view USE_OMPT    = "ROCPROFSYS_USE_OMPT";
constexpr std::string_view USE_KOKKOSP = "ROCPROFSYS_USE_KOKKOSP";
constexpr std::string_view USE_RCCLP   = "ROCPROFSYS_USE_RCCLP";
constexpr std::string_view USE_AINIC   = "ROCPROFSYS_USE_AINIC";
constexpr std::string_view USE_SHMEM   = "ROCPROFSYS_USE_SHMEM";
constexpr std::string_view USE_UCX     = "ROCPROFSYS_USE_UCX";

// --- Domains: Shmem ---
constexpr std::string_view SHMEM_PERMIT_LIST = "ROCPROFSYS_SHMEM_PERMIT_LIST";
constexpr std::string_view SHMEM_REJECT_LIST = "ROCPROFSYS_SHMEM_REJECT_LIST";

// --- Output ---
constexpr std::string_view OUTPUT_PATH             = "ROCPROFSYS_OUTPUT_PATH";
constexpr std::string_view OUTPUT_PREFIX           = "ROCPROFSYS_OUTPUT_PREFIX";
constexpr std::string_view OUTPUT                  = "ROCPROFSYS_OUTPUT";
constexpr std::string_view OUTPUT_FILE             = "ROCPROFSYS_OUTPUT_FILE";
constexpr std::string_view OUTPUT_USE_CURRENT_TIME = "ROCPROFSYS_OUTPUT_USE_CURRENT_TIME";
constexpr std::string_view USE_PID                 = "ROCPROFSYS_USE_PID";
constexpr std::string_view TIME_OUTPUT             = "ROCPROFSYS_TIME_OUTPUT";
constexpr std::string_view FILE_OUTPUT             = "ROCPROFSYS_FILE_OUTPUT";
constexpr std::string_view TEXT_OUTPUT             = "ROCPROFSYS_TEXT_OUTPUT";
constexpr std::string_view JSON_OUTPUT             = "ROCPROFSYS_JSON_OUTPUT";
constexpr std::string_view COUT_OUTPUT             = "ROCPROFSYS_COUT_OUTPUT";
constexpr std::string_view DIFF_OUTPUT             = "ROCPROFSYS_DIFF_OUTPUT";
constexpr std::string_view TREE_OUTPUT             = "ROCPROFSYS_TREE_OUTPUT";
constexpr std::string_view INPUT_PATH              = "ROCPROFSYS_INPUT_PATH";
constexpr std::string_view INPUT_PREFIX            = "ROCPROFSYS_INPUT_PREFIX";
constexpr std::string_view INPUT_EXTENSIONS        = "ROCPROFSYS_INPUT_EXTENSIONS";
constexpr std::string_view USE_ROCPD               = "ROCPROFSYS_USE_ROCPD";
constexpr std::string_view USE_TEMPORARY_FILES     = "ROCPROFSYS_USE_TEMPORARY_FILES";
constexpr std::string_view USE_UNIFIED_MEMORY_PROFILING =
    "ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING";
constexpr std::string_view TIME_FORMAT = "ROCPROFSYS_TIME_FORMAT";

// --- Output: number formatting ---
constexpr std::string_view PRECISION         = "ROCPROFSYS_PRECISION";
constexpr std::string_view SCIENTIFIC        = "ROCPROFSYS_SCIENTIFIC";
constexpr std::string_view WIDTH             = "ROCPROFSYS_WIDTH";
constexpr std::string_view MAX_WIDTH         = "ROCPROFSYS_MAX_WIDTH";
constexpr std::string_view TIMING_PRECISION  = "ROCPROFSYS_TIMING_PRECISION";
constexpr std::string_view TIMING_SCIENTIFIC = "ROCPROFSYS_TIMING_SCIENTIFIC";
constexpr std::string_view TIMING_UNITS      = "ROCPROFSYS_TIMING_UNITS";
constexpr std::string_view TIMING_WIDTH      = "ROCPROFSYS_TIMING_WIDTH";
constexpr std::string_view MEMORY_PRECISION  = "ROCPROFSYS_MEMORY_PRECISION";
constexpr std::string_view MEMORY_SCIENTIFIC = "ROCPROFSYS_MEMORY_SCIENTIFIC";
constexpr std::string_view MEMORY_UNITS      = "ROCPROFSYS_MEMORY_UNITS";
constexpr std::string_view MEMORY_WIDTH      = "ROCPROFSYS_MEMORY_WIDTH";

// --- MPI output filtering ---
constexpr std::string_view RANK_FILTER_ID     = "ROCPROFSYS_RANK_FILTER_ID";
constexpr std::string_view RANK_FILTER_OUTPUT = "ROCPROFSYS_RANK_FILTER_OUTPUT";
constexpr std::string_view RANK_FILTER_LOGS   = "ROCPROFSYS_RANK_FILTER_LOGS";

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
constexpr std::string_view CAUSAL_FUNCTION_EXCLUDE_DEFAULTS =
    "ROCPROFSYS_CAUSAL_FUNCTION_EXCLUDE_DEFAULTS";
constexpr std::string_view CAUSAL_SOURCE_SCOPE   = "ROCPROFSYS_CAUSAL_SOURCE_SCOPE";
constexpr std::string_view CAUSAL_SOURCE_EXCLUDE = "ROCPROFSYS_CAUSAL_SOURCE_EXCLUDE";
constexpr std::string_view CAUSAL_END_TO_END     = "ROCPROFSYS_CAUSAL_END_TO_END";
constexpr std::string_view CAUSAL_DELAY          = "ROCPROFSYS_CAUSAL_DELAY";
constexpr std::string_view CAUSAL_DURATION       = "ROCPROFSYS_CAUSAL_DURATION";
constexpr std::string_view CAUSAL_RANDOM_SEED    = "ROCPROFSYS_CAUSAL_RANDOM_SEED";
constexpr std::string_view CAUSAL_FIXED_SPEEDUP  = "ROCPROFSYS_CAUSAL_FIXED_SPEEDUP";
constexpr std::string_view CAUSAL_SPEEDUP_DIVISIONS =
    "ROCPROFSYS_CAUSAL_SPEEDUP_DIVISIONS";
constexpr std::string_view CAUSAL_SCALE_EXPERIMENT_TIME_BY_SPEEDUP =
    "ROCPROFSYS_CAUSAL_SCALE_EXPERIMENT_TIME_BY_SPEEDUP";
constexpr std::string_view CAUSAL_FILE       = "ROCPROFSYS_CAUSAL_FILE";
constexpr std::string_view CAUSAL_FILE_RESET = "ROCPROFSYS_CAUSAL_FILE_RESET";

// --- Hardware counters ---
// Note: PAPI_MULTIPLEXING and PAPI_QUIET would collide with integer macros defined in
// PAPI's C header (papi.h). The identifiers carry a trailing suffix to avoid
// preprocessor substitution; the env-var strings retain the original names.
constexpr std::string_view ROCM_EVENTS               = "ROCPROFSYS_ROCM_EVENTS";
constexpr std::string_view PAPI_EVENTS               = "ROCPROFSYS_PAPI_EVENTS";
constexpr std::string_view PAPI_MULTIPLEXING_ENABLED = "ROCPROFSYS_PAPI_MULTIPLEXING";
constexpr std::string_view PAPI_FAIL_ON_ERROR        = "ROCPROFSYS_PAPI_FAIL_ON_ERROR";
constexpr std::string_view PAPI_OVERFLOW             = "ROCPROFSYS_PAPI_OVERFLOW";
constexpr std::string_view PAPI_QUIET_MODE           = "ROCPROFSYS_PAPI_QUIET";
constexpr std::string_view PAPI_THREADING            = "ROCPROFSYS_PAPI_THREADING";
constexpr std::string_view USE_CODE_COVERAGE         = "ROCPROFSYS_USE_CODE_COVERAGE";

// --- MPI ---
constexpr std::string_view MPI_INIT             = "ROCPROFSYS_MPI_INIT";
constexpr std::string_view MPI_FINALIZE         = "ROCPROFSYS_MPI_FINALIZE";
constexpr std::string_view MPI_FAIL_ON_ERROR    = "ROCPROFSYS_MPI_FAIL_ON_ERROR";
constexpr std::string_view MPI_QUIET            = "ROCPROFSYS_MPI_QUIET";
constexpr std::string_view MPI_THREAD           = "ROCPROFSYS_MPI_THREAD";
constexpr std::string_view MPI_THREAD_TYPE      = "ROCPROFSYS_MPI_THREAD_TYPE";
constexpr std::string_view MPI_MAX_COMM_UPDATES = "ROCPROFSYS_MPI_MAX_COMM_UPDATES";

// --- Kokkos profiling ---
constexpr std::string_view KOKKOSP_PREFIX          = "ROCPROFSYS_KOKKOSP_PREFIX";
constexpr std::string_view KOKKOSP_KERNEL_LOGGER   = "ROCPROFSYS_KOKKOSP_KERNEL_LOGGER";
constexpr std::string_view KOKKOSP_NAME_LENGTH_MAX = "ROCPROFSYS_KOKKOSP_NAME_LENGTH_MAX";
constexpr std::string_view KOKKOSP_DEEP_COPY       = "ROCPROFSYS_KOKKOSP_DEEP_COPY";

// --- DL (dynamic loader / preload) ---
constexpr std::string_view DL_VERBOSE = "ROCPROFSYS_DL_VERBOSE";
constexpr std::string_view DL_DEBUG   = "ROCPROFSYS_DL_DEBUG";
constexpr std::string_view DL_LIBRARY = "ROCPROFSYS_DL_LIBRARY";

// --- Regex include/exclude filters ---
constexpr std::string_view REGEX_INCLUDE          = "ROCPROFSYS_REGEX_INCLUDE";
constexpr std::string_view REGEX_EXCLUDE          = "ROCPROFSYS_REGEX_EXCLUDE";
constexpr std::string_view REGEX_RESTRICT         = "ROCPROFSYS_REGEX_RESTRICT";
constexpr std::string_view REGEX_CALLER_INCLUDE   = "ROCPROFSYS_REGEX_CALLER_INCLUDE";
constexpr std::string_view REGEX_INTERNAL_INCLUDE = "ROCPROFSYS_REGEX_INTERNAL_INCLUDE";
constexpr std::string_view REGEX_INSTRUCTION_EXCLUDE =
    "ROCPROFSYS_REGEX_INSTRUCTION_EXCLUDE";
constexpr std::string_view REGEX_MODULE_INCLUDE  = "ROCPROFSYS_REGEX_MODULE_INCLUDE";
constexpr std::string_view REGEX_MODULE_EXCLUDE  = "ROCPROFSYS_REGEX_MODULE_EXCLUDE";
constexpr std::string_view REGEX_MODULE_RESTRICT = "ROCPROFSYS_REGEX_MODULE_RESTRICT";
constexpr std::string_view REGEX_MODULE_INTERNAL_INCLUDE =
    "ROCPROFSYS_REGEX_MODULE_INTERNAL_INCLUDE";

// --- CI / continuous integration ---
constexpr std::string_view CI                     = "ROCPROFSYS_CI";
constexpr std::string_view CI_SKIP_PUSH_POP_CHECK = "ROCPROFSYS_CI_SKIP_PUSH_POP_CHECK";
constexpr std::string_view CI_TIMEOUT             = "ROCPROFSYS_CI_TIMEOUT";
constexpr std::string_view CI_TIMEOUT_COUNT       = "ROCPROFSYS_CI_TIMEOUT_COUNT";
constexpr std::string_view CI_TIMEOUT_OVERRIDE    = "ROCPROFSYS_CI_TIMEOUT_OVERRIDE";

// --- Process / threading / behavior ---
constexpr std::string_view NUM_THREADS             = "ROCPROFSYS_NUM_THREADS";
constexpr std::string_view NUM_THREADS_HINT        = "ROCPROFSYS_NUM_THREADS_HINT";
constexpr std::string_view THREAD_POOL_SIZE        = "ROCPROFSYS_THREAD_POOL_SIZE";
constexpr std::string_view RECYCLE_TIDS            = "ROCPROFSYS_RECYCLE_TIDS";
constexpr std::string_view KILL_DELAY              = "ROCPROFSYS_KILL_DELAY";
constexpr std::string_view COLLAPSE_PROCESSES      = "ROCPROFSYS_COLLAPSE_PROCESSES";
constexpr std::string_view NODE_COUNT              = "ROCPROFSYS_NODE_COUNT";
constexpr std::string_view ROOT_PROCESS            = "ROCPROFSYS_ROOT_PROCESS";
constexpr std::string_view RANK_FILTER_ID          = "ROCPROFSYS_RANK_FILTER_ID";
constexpr std::string_view RANK_FILTER_OUTPUT      = "ROCPROFSYS_RANK_FILTER_OUTPUT";
constexpr std::string_view REATTACH_ADD_SESSION_ID = "ROCPROFSYS_REATTACH_ADD_SESSION_ID";

// --- Instrumentation ---
constexpr std::string_view INSTRUMENT_MODE = "ROCPROFSYS_INSTRUMENT_MODE";
constexpr std::string_view IGNORE_DYNINST_TRAMPOLINE =
    "ROCPROFSYS_IGNORE_DYNINST_TRAMPOLINE";
constexpr std::string_view DEFAULT_MIN_INSTRUCTIONS =
    "ROCPROFSYS_DEFAULT_MIN_INSTRUCTIONS";

// --- Runtime / launcher (set by instrumenter / read by the loaded library) ---
constexpr std::string_view PATH                   = "ROCPROFSYS_PATH";
constexpr std::string_view PRELOAD                = "ROCPROFSYS_PRELOAD";
constexpr std::string_view LIBRARY                = "ROCPROFSYS_LIBRARY";
constexpr std::string_view USER_LIBRARY           = "ROCPROFSYS_USER_LIBRARY";
constexpr std::string_view COMMAND_LINE           = "ROCPROFSYS_COMMAND_LINE";
constexpr std::string_view LAUNCHER               = "ROCPROFSYS_LAUNCHER";
constexpr std::string_view SCRIPT_DIR             = "ROCPROFSYS_SCRIPT_DIR";
constexpr std::string_view ROCM_PATH              = "ROCPROFSYS_ROCM_PATH";
constexpr std::string_view CONFIG                 = "ROCPROFSYS_CONFIG";
constexpr std::string_view ENVIRONMENT            = "ROCPROFSYS_ENVIRONMENT";
constexpr std::string_view SETTINGS_DESC          = "ROCPROFSYS_SETTINGS_DESC";
constexpr std::string_view SETTINGS_DESC_MARKDOWN = "ROCPROFSYS_SETTINGS_DESC_MARKDOWN";
constexpr std::string_view CRAYPAT                = "ROCPROFSYS_CRAYPAT";

// --- Advanced ---
constexpr std::string_view CPU_AFFINITY          = "ROCPROFSYS_CPU_AFFINITY";
constexpr std::string_view COLLAPSE_THREADS      = "ROCPROFSYS_COLLAPSE_THREADS";
constexpr std::string_view MAX_DEPTH             = "ROCPROFSYS_MAX_DEPTH";
constexpr std::string_view TRACE_DELAY           = "ROCPROFSYS_TRACE_DELAY";
constexpr std::string_view TRACE_DURATION        = "ROCPROFSYS_TRACE_DURATION";
constexpr std::string_view TRACE_PERIODS         = "ROCPROFSYS_TRACE_PERIODS";
constexpr std::string_view TRACE_PERIOD_CLOCK_ID = "ROCPROFSYS_TRACE_PERIOD_CLOCK_ID";
constexpr std::string_view VERBOSE               = "ROCPROFSYS_VERBOSE";
constexpr std::string_view VERBOSE_AVAIL         = "ROCPROFSYS_VERBOSE_AVAIL";
constexpr std::string_view VERBOSE_INSTRUMENT    = "ROCPROFSYS_VERBOSE_INSTRUMENT";
// Note: identifier is DEBUG_MODE to avoid collision with `#define DEBUG 1` injected by
// the project's generated common/defines.h on CI builds (ROCPROFSYS_CI > 0). The
// env-var string itself remains "ROCPROFSYS_DEBUG".
constexpr std::string_view DEBUG_MODE = "ROCPROFSYS_DEBUG";
// well above the highest verbose threshold (3) so debug mode enables all verbose output
constexpr int              DEBUG_VERBOSE_BOOST   = 8;
constexpr std::string_view DEBUG_INIT            = "ROCPROFSYS_DEBUG_INIT";
constexpr std::string_view DEBUG_FINALIZE        = "ROCPROFSYS_DEBUG_FINALIZE";
constexpr std::string_view DEBUG_AVAIL           = "ROCPROFSYS_DEBUG_AVAIL";
constexpr std::string_view DEBUG_MARK            = "ROCPROFSYS_DEBUG_MARK";
constexpr std::string_view DEBUG_PIDS            = "ROCPROFSYS_DEBUG_PIDS";
constexpr std::string_view DEBUG_TIDS            = "ROCPROFSYS_DEBUG_TIDS";
constexpr std::string_view DEBUG_PUSH            = "ROCPROFSYS_DEBUG_PUSH";
constexpr std::string_view DEBUG_POP             = "ROCPROFSYS_DEBUG_POP";
constexpr std::string_view DEBUG_SAMPLING        = "ROCPROFSYS_DEBUG_SAMPLING";
constexpr std::string_view DEBUG_USER_REGIONS    = "ROCPROFSYS_DEBUG_USER_REGIONS";
constexpr std::string_view ENABLE_SIGNAL_HANDLER = "ROCPROFSYS_ENABLE_SIGNAL_HANDLER";
constexpr std::string_view TIMEMORY_COMPONENTS   = "ROCPROFSYS_TIMEMORY_COMPONENTS";
constexpr std::string_view NETWORK_INTERFACE     = "ROCPROFSYS_NETWORK_INTERFACE";

// --- Deprecated aliases ---
// Names retained only for legacy/migration paths — the codebase emits deprecation
// warnings when these are encountered. Do not introduce new references; prefer the
// replacement (TRACE for USE_PERFETTO, PROFILE for USE_TIMEMORY).
constexpr std::string_view USE_PERFETTO = "ROCPROFSYS_USE_PERFETTO";
constexpr std::string_view USE_TIMEMORY = "ROCPROFSYS_USE_TIMEMORY";

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
