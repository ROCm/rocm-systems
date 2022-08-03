// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"
#include "library/gpu.hpp"
#include "library/mproc.hpp"
#include "library/perfetto.hpp"
#include "library/runtime.hpp"

#include <timemory/backends/dmp.hpp>
#include <timemory/backends/mpi.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/environment.hpp>
#include <timemory/environment/types.hpp>
#include <timemory/manager.hpp>
#include <timemory/sampling/allocator.hpp>
#include <timemory/settings.hpp>
#include <timemory/settings/types.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/declaration.hpp>
#include <timemory/utility/signals.hpp>

#include <algorithm>
#include <array>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <unistd.h>

namespace omnitrace
{
using settings = tim::settings;

namespace
{
auto
get_config()
{
    static auto _once = (configure_settings(), true);
    return settings::shared_instance();
    (void) _once;
}

std::string
get_setting_name(std::string _v)
{
    static const auto _prefix = tim::string_view_t{ "omnitrace_" };
    for(auto& itr : _v)
        itr = tolower(itr);
    auto _pos = _v.find(_prefix);
    if(_pos == 0) return _v.substr(_prefix.length());
    return _v;
}

template <typename Tp>
Tp
get_available_perfetto_categories()
{
    auto _v = Tp{};
    for(auto itr : { OMNITRACE_PERFETTO_CATEGORIES })
        tim::utility::emplace(_v, itr.name);
    return _v;
}

#define OMNITRACE_CONFIG_SETTING(TYPE, ENV_NAME, DESCRIPTION, INITIAL_VALUE, ...)        \
    [&]() {                                                                              \
        auto _ret = _config->insert<TYPE, TYPE>(                                         \
            ENV_NAME, get_setting_name(ENV_NAME), DESCRIPTION, TYPE{ INITIAL_VALUE },    \
            std::set<std::string>{ "custom", "omnitrace", "libomnitrace",                \
                                   __VA_ARGS__ });                                       \
        if(!_ret.second)                                                                 \
        {                                                                                \
            OMNITRACE_PRINT("Warning! Duplicate setting: %s / %s\n",                     \
                            get_setting_name(ENV_NAME).c_str(), ENV_NAME);               \
        }                                                                                \
        return _config->find(ENV_NAME)->second;                                          \
    }()

// below does not include "libomnitrace"
#define OMNITRACE_CONFIG_EXT_SETTING(TYPE, ENV_NAME, DESCRIPTION, INITIAL_VALUE, ...)    \
    [&]() {                                                                              \
        auto _ret = _config->insert<TYPE, TYPE>(                                         \
            ENV_NAME, get_setting_name(ENV_NAME), DESCRIPTION, TYPE{ INITIAL_VALUE },    \
            std::set<std::string>{ "custom", "omnitrace", __VA_ARGS__ });                \
        if(!_ret.second)                                                                 \
        {                                                                                \
            OMNITRACE_PRINT("Warning! Duplicate setting: %s / %s\n",                     \
                            get_setting_name(ENV_NAME).c_str(), ENV_NAME);               \
        }                                                                                \
        return _config->find(ENV_NAME)->second;                                          \
    }()

// setting + command line option
#define OMNITRACE_CONFIG_CL_SETTING(TYPE, ENV_NAME, DESCRIPTION, INITIAL_VALUE,          \
                                    CMD_LINE, ...)                                       \
    [&]() {                                                                              \
        auto _ret = _config->insert<TYPE, TYPE>(                                         \
            ENV_NAME, get_setting_name(ENV_NAME), DESCRIPTION, TYPE{ INITIAL_VALUE },    \
            std::set<std::string>{ "custom", "omnitrace", "libomnitrace", __VA_ARGS__ }, \
            std::vector<std::string>{ CMD_LINE });                                       \
        if(!_ret.second)                                                                 \
        {                                                                                \
            OMNITRACE_PRINT("Warning! Duplicate setting: %s / %s\n",                     \
                            get_setting_name(ENV_NAME).c_str(), ENV_NAME);               \
        }                                                                                \
        return _config->find(ENV_NAME)->second;                                          \
    }()
}  // namespace

inline namespace config
{
namespace
{
TIMEMORY_NOINLINE bool&
_settings_are_configured()
{
    static bool _v = false;
    return _v;
}
}  // namespace

void
finalize()
{
    OMNITRACE_DEBUG("[omnitrace_finalize] Disabling signal handling...\n");
    tim::disable_signal_detection();
    _settings_are_configured() = false;
}

bool
settings_are_configured()
{
    volatile bool _v = _settings_are_configured();
    return _v;
}

void
configure_settings(bool _init)
{
    volatile bool _v = _settings_are_configured();
    if(_v) return;

    static bool _once = false;
    if(_once) return;
    _once = true;

    if(get_state() < State::Init)
    {
        ::tim::print_demangled_backtrace<64>();
        OMNITRACE_THROW("config::configure_settings() called before "
                        "omnitrace_init_library. state = %s",
                        std::to_string(get_state()).c_str());
    }

    tim::manager::add_metadata("OMNITRACE_VERSION", OMNITRACE_VERSION_STRING);
    tim::manager::add_metadata("OMNITRACE_VERSION_MAJOR", OMNITRACE_VERSION_MAJOR);
    tim::manager::add_metadata("OMNITRACE_VERSION_MINOR", OMNITRACE_VERSION_MINOR);
    tim::manager::add_metadata("OMNITRACE_VERSION_PATCH", OMNITRACE_VERSION_PATCH);
    tim::manager::add_metadata("OMNITRACE_GIT_DESCRIBE", OMNITRACE_GIT_DESCRIBE);
    tim::manager::add_metadata("OMNITRACE_GIT_REVISION", OMNITRACE_GIT_REVISION);

#if OMNITRACE_HIP_VERSION > 0
    tim::manager::add_metadata("OMNITRACE_HIP_VERSION", OMNITRACE_HIP_VERSION_STRING);
    tim::manager::add_metadata("OMNITRACE_HIP_VERSION_MAJOR",
                               OMNITRACE_HIP_VERSION_MAJOR);
    tim::manager::add_metadata("OMNITRACE_HIP_VERSION_MINOR",
                               OMNITRACE_HIP_VERSION_MINOR);
    tim::manager::add_metadata("OMNITRACE_HIP_VERSION_PATCH",
                               OMNITRACE_HIP_VERSION_PATCH);
#endif

    static auto _config = settings::shared_instance();

    // if using timemory, default to perfetto being off
    auto _default_perfetto_v =
        !tim::get_env<bool>("OMNITRACE_USE_TIMEMORY", false, false);

    auto _system_backend =
        tim::get_env("OMNITRACE_PERFETTO_BACKEND_SYSTEM", false, false);

    auto _omnitrace_debug = _config->get<bool>("OMNITRACE_DEBUG");
    if(_omnitrace_debug) tim::set_env("TIMEMORY_DEBUG_SETTINGS", "1", 0);

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_MODE",
        "Data collection mode. Used to set default values for OMNITRACE_USE_* options. "
        "Typically set by omnitrace binary instrumenter.",
        std::string{ "trace" }, "backend", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_CI",
                             "Enable some runtime validation checks (typically enabled "
                             "for continuous integration)",
                             false, "debugging", "advanced");

    OMNITRACE_CONFIG_EXT_SETTING(int, "OMNITRACE_DL_VERBOSE",
                                 "Verbosity within the omnitrace-dl library", 0,
                                 "debugging", "libomnitrace-dl", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_PERFETTO", "Enable perfetto backend",
                             _default_perfetto_v, "backend", "perfetto");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_TIMEMORY", "Enable timemory backend",
                             !_config->get<bool>("OMNITRACE_USE_PERFETTO"), "backend",
                             "timemory");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_ROCTRACER",
                             "Enable ROCm API and kernel tracing", true, "backend",
                             "roctracer", "rocm");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_ROCPROFILER",
                             "Enable ROCm hardware counters", true, "backend",
                             "rocprofiler", "rocm");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_USE_ROCM_SMI",
        "Enable sampling GPU power, temp, utilization, and memory usage", true, "backend",
        "rocm_smi", "rocm");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_USE_ROCTX",
        "Enable ROCtx API. Warning! Out-of-order ranges may corrupt perfetto flamegraph",
        false, "backend", "roctracer", "rocm", "roctx");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_SAMPLING",
                             "Enable statistical sampling of call-stack", false,
                             "backend", "sampling");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_THREAD_SAMPLING",
                             "[DEPRECATED] Renamed to OMNITRACE_USE_PROCESS_SAMPLING",
                             true, "backend", "sampling", "process_sampling",
                             "deprecated", "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_USE_PROCESS_SAMPLING",
        "Enable a background thread which samples process-level and system metrics "
        "such as the CPU/GPU freq, power, memory usage, etc.",
        true, "backend", "sampling", "process_sampling");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_USE_PID",
        "Enable tagging filenames with process identifier (either MPI rank or pid)", true,
        "io", "filename");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_KOKKOSP",
                             "Enable support for Kokkos Tools", false, "kokkos",
                             "backend");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_USE_RCCLP",
        "Enable support for ROCm Communication Collectives Library (RCCL) Performance",
        false, "rocm", "rccl", "backend");

    OMNITRACE_CONFIG_CL_SETTING(
        bool, "OMNITRACE_KOKKOS_KERNEL_LOGGER", "Enables kernel logging", false,
        "--omnitrace-kokkos-kernel-logger", "kokkos", "debugging", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_OMPT",
                             "Enable support for OpenMP-Tools", false, "openmp", "ompt",
                             "backend");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_CODE_COVERAGE",
                             "Enable support for code coverage", false, "coverage",
                             "backend", "advanced");

    OMNITRACE_CONFIG_SETTING(size_t, "OMNITRACE_INSTRUMENTATION_INTERVAL",
                             "Instrumentation only takes measurements once every N "
                             "function calls (not statistical)",
                             size_t{ 1 }, "instrumentation", "data_sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(
        double, "OMNITRACE_SAMPLING_FREQ",
        "Number of software interrupts per second when OMNITTRACE_USE_SAMPLING=ON", 10.0,
        "sampling", "process_sampling");

    OMNITRACE_CONFIG_SETTING(
        double, "OMNITRACE_SAMPLING_DELAY",
        "Time (in seconds) to wait before the first sampling signal is delivered, "
        "increasing this value can fix deadlocks during init",
        0.5, "sampling", "process_sampling");

    OMNITRACE_CONFIG_SETTING(
        double, "OMNITRACE_PROCESS_SAMPLING_FREQ",
        "Number of measurements per second when OMNITTRACE_USE_PROCESS_SAMPLING=ON. If "
        "set to zero, uses OMNITRACE_SAMPLING_FREQ value",
        0.0, "process_sampling");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_SAMPLING_CPUS",
        "CPUs to collect frequency information for. Values should be separated by commas "
        "and can be explicit or ranges, e.g. 0,1,5-8. An empty value implies 'all' and "
        "'none' suppresses all CPU frequency sampling",
        std::string{}, "process_sampling");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_ROCM_SMI_DEVICES",
                             "[DEPRECATED] Renamed to OMNITRACE_SAMPLING_GPUS",
                             std::string{ "all" }, "rocm_smi", "rocm", "process_sampling",
                             "deprecated", "advanced");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_SAMPLING_GPUS",
        "Devices to query when OMNITRACE_USE_ROCM_SMI=ON. Values should be separated by "
        "commas and can be explicit or ranges, e.g. 0,1,5-8. An empty value implies "
        "'all' and 'none' suppresses all GPU sampling",
        std::string{ "all" }, "rocm_smi", "rocm", "process_sampling");

    auto _backend = tim::get_env_choice<std::string>(
        "OMNITRACE_PERFETTO_BACKEND",
        (_system_backend) ? "system"      // if OMNITRACE_PERFETTO_BACKEND_SYSTEM is true,
                                          // default to system.
                          : "inprocess",  // Otherwise, default to inprocess
        { "inprocess", "system", "all" }, false);

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_PERFETTO_BACKEND",
                             "Specify the perfetto backend to activate. Options are: "
                             "'inprocess', 'system', or 'all'",
                             _backend, "perfetto")
        ->set_choices({ "inprocess", "system", "all" });

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_CRITICAL_TRACE",
                             "Enable generation of the critical trace", false, "backend",
                             "critical_trace");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_TRACE_THREAD_LOCKS",
                             "Enable tracing calls to pthread_mutex_lock, "
                             "pthread_mutex_unlock, pthread_mutex_trylock",
                             false, "backend", "parallelism", "gotcha", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_TRACE_THREAD_RW_LOCKS",
                             "Enable tracing calls to pthread_rwlock_* functions. May "
                             "cause deadlocks with ROCm-enabled OpenMPI.",
                             false, "backend", "parallelism", "gotcha", "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_SAMPLING_KEEP_INTERNAL",
        "Configure whether the statistical samples should include call-stack entries "
        "from internal routines in omnitrace. E.g. when ON, the call-stack will show "
        "functions like omnitrace_push_trace. If disabled, omnitrace will attempt to "
        "filter out internal routines from the sampling call-stacks",
        true, "sampling", "data", "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_SAMPLING_REALTIME",
        "Enable sampling frequency via a wall-clock timer on child threads. This may "
        "result in typically idle child threads consuming an unnecessary large amount of "
        "CPU time. The main thread always has this enabled.",
        false, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_SAMPLING_CPUTIME",
                             "Enable sampling frequency via a timer that measures both "
                             "CPU time used by the current process, "
                             "and CPU time expended on behalf of the process by the "
                             "system. This is recommended.",
                             true, "timemory", "sampling", "advanced");

    auto _sigrt_range = SIGRTMAX - SIGRTMIN;

    OMNITRACE_CONFIG_SETTING(
        int, "OMNITRACE_SAMPLING_REALTIME_OFFSET",
        std::string{
            "Modify this value only if the target process is also using SIGRTMIN. E.g. "
            "the signal used is SIGRTMIN + <THIS_VALUE>. Value must be <= " } +
            std::to_string(_sigrt_range),
        0, "sampling", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_FLAT_SAMPLING",
                             "Ignore hierarchy in all statistical sampling entries",
                             _config->get_flat_profile(), "timemory", "sampling",
                             "data_layout", "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_TIMELINE_SAMPLING",
        "Create unique entries for every sample when statistical sampling is enabled",
        _config->get_timeline_profile(), "timemory", "sampling", "data_layout",
        "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_ROCTRACER_FLAT_PROFILE",
        "Ignore hierarchy in all kernels entries with timemory backend",
        _config->get_flat_profile(), "timemory", "roctracer", "data_layout", "rocm",
        "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_ROCTRACER_TIMELINE_PROFILE",
        "Create unique entries for every kernel with timemory backend",
        _config->get_timeline_profile(), "timemory", "roctracer", "data_layout", "rocm",
        "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_ROCTRACER_HSA_ACTIVITY",
                             "Enable HSA activity tracing support", true, "roctracer",
                             "rocm", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_ROCTRACER_HSA_API",
                             "Enable HSA API tracing support", true, "roctracer", "rocm",
                             "advanced");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_ROCTRACER_HSA_API_TYPES",
                             "HSA API type to collect", "", "roctracer", "rocm",
                             "advanced");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_ROCM_EVENTS",
        "ROCm hardware counters. Use ':device=N' syntax to specify collection on device "
        "number N, e.g. ':device=0'. If no device specification is provided, the event "
        "is collected on every available device",
        "", "rocprofiler", "rocm", "hardware_counters");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_CRITICAL_TRACE_DEBUG",
                             "Enable debugging for critical trace", _omnitrace_debug,
                             "debugging", "critical_trace", "advanced");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_CRITICAL_TRACE_SERIALIZE_NAMES",
        "Include names in serialization of critical trace (mainly for debugging)",
        _omnitrace_debug, "debugging", "critical_trace", "advanced");

    OMNITRACE_CONFIG_SETTING(size_t, "OMNITRACE_PERFETTO_SHMEM_SIZE_HINT_KB",
                             "Hint for shared-memory buffer size in perfetto (in KB)",
                             size_t{ 4096 }, "perfetto", "data", "advanced");

    OMNITRACE_CONFIG_SETTING(size_t, "OMNITRACE_PERFETTO_BUFFER_SIZE_KB",
                             "Size of perfetto buffer (in KB)", size_t{ 1024000 },
                             "perfetto", "data", "advanced");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_PERFETTO_COMBINE_TRACES",
                             "Combine Perfetto traces. If not explicitly set, it will "
                             "default to the value of OMNITRACE_COLLAPSE_PROCESSES",
                             _config->get<bool>("collapse_processes"), "perfetto", "data",
                             "advanced");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_PERFETTO_FILL_POLICY",
        "Behavior when perfetto buffer is full. 'discard' will ignore new entries, "
        "'ring_buffer' will overwrite old entries",
        "discard", "perfetto", "data")
        ->set_choices({ "fill", "discard" });

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_PERFETTO_CATEGORIES",
                             "Categories to collect within perfetto", "", "perfetto",
                             "data", "advanced")
        ->set_choices(get_available_perfetto_categories<std::vector<std::string>>());

    OMNITRACE_CONFIG_EXT_SETTING(int64_t, "OMNITRACE_CRITICAL_TRACE_COUNT",
                                 "Number of critical trace to export (0 == all)",
                                 int64_t{ 0 }, "data", "critical_trace",
                                 "omnitrace-critical-trace", "perfetto", "advanced");

    OMNITRACE_CONFIG_SETTING(uint64_t, "OMNITRACE_CRITICAL_TRACE_BUFFER_COUNT",
                             "Number of critical trace records to store in thread-local "
                             "memory before submitting to shared buffer",
                             uint64_t{ 2000 }, "data", "critical_trace", "advanced");

    OMNITRACE_CONFIG_SETTING(
        uint64_t, "OMNITRACE_CRITICAL_TRACE_NUM_THREADS",
        "Number of threads to use when generating the critical trace",
        std::min<uint64_t>(8, std::thread::hardware_concurrency()), "parallelism",
        "critical_trace", "advanced");

    OMNITRACE_CONFIG_EXT_SETTING(
        int64_t, "OMNITRACE_CRITICAL_TRACE_PER_ROW",
        "How many critical traces per row in perfetto (0 == all in one row)",
        int64_t{ 0 }, "io", "critical_trace", "omnitrace-critical-trace", "perfetto",
        "advanced");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_TIMEMORY_COMPONENTS",
        "List of components to collect via timemory (see `omnitrace-avail -C`)",
        "wall_clock", "timemory", "component");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_OUTPUT_FILE",
                             "[DEPRECATED] See OMNITRACE_PERFETTO_FILE", std::string{},
                             "perfetto", "io", "filename", "deprecated", "advanced");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_PERFETTO_FILE", "Perfetto filename",
                             std::string{ "perfetto-trace.proto" }, "perfetto", "io",
                             "filename", "advanced");

    // set the defaults
    _config->get_flamegraph_output()     = false;
    _config->get_ctest_notes()           = false;
    _config->get_cout_output()           = false;
    _config->get_file_output()           = true;
    _config->get_json_output()           = true;
    _config->get_tree_output()           = true;
    _config->get_enable_signal_handler() = true;
    _config->get_collapse_processes()    = false;
    _config->get_collapse_threads()      = false;
    _config->get_stack_clearing()        = false;
    _config->get_time_output()           = true;
    _config->get_timing_precision()      = 6;
    _config->get_max_thread_bookmarks()  = 1;
    _config->get_timing_units()          = "sec";
    _config->get_memory_units()          = "MB";
    _config->get_papi_events()           = "PAPI_TOT_CYC";

    // settings native to timemory but critically and/or extensively used by omnitrace
    auto _add_omnitrace_category = [](auto itr) {
        if(itr != _config->end())
        {
            auto _categories = itr->second->get_categories();
            _categories.emplace("omnitrace");
            _categories.emplace("libomnitrace");
            itr->second->set_categories(_categories);
        }
    };

    _add_omnitrace_category(_config->find("OMNITRACE_CONFIG_FILE"));
    _add_omnitrace_category(_config->find("OMNITRACE_DEBUG"));
    _add_omnitrace_category(_config->find("OMNITRACE_VERBOSE"));
    _add_omnitrace_category(_config->find("OMNITRACE_TIME_OUTPUT"));
    _add_omnitrace_category(_config->find("OMNITRACE_OUTPUT_PREFIX"));
    _add_omnitrace_category(_config->find("OMNITRACE_OUTPUT_PATH"));

    auto _add_advanced_category = [](const std::string& _name) {
        auto itr = _config->find(_name);
        if(itr != _config->end())
        {
            auto _categories = itr->second->get_categories();
            _categories.emplace("advanced");
            itr->second->set_categories(_categories);
        }
        else
        {
            if(_config->get<bool>("OMNITRACE_CI"))
            {
                OMNITRACE_THROW("Error! Setting '%s' not found!", _name.c_str());
            }
        }
    };

    _add_advanced_category("OMNITRACE_CPU_AFFINITY");
    _add_advanced_category("OMNITRACE_COUT_OUTPUT");
    _add_advanced_category("OMNITRACE_FILE_OUTPUT");
    _add_advanced_category("OMNITRACE_JSON_OUTPUT");
    _add_advanced_category("OMNITRACE_TREE_OUTPUT");
    _add_advanced_category("OMNITRACE_TEXT_OUTPUT");
    _add_advanced_category("OMNITRACE_DIFF_OUTPUT");
    _add_advanced_category("OMNITRACE_DEBUG");
    _add_advanced_category("OMNITRACE_ENABLE_SIGNAL_HANDLER");
    _add_advanced_category("OMNITRACE_FLAT_PROFILE");
    _add_advanced_category("OMNITRACE_INPUT_EXTENSIONS");
    _add_advanced_category("OMNITRACE_INPUT_PATH");
    _add_advanced_category("OMNITRACE_INPUT_PREFIX");
    _add_advanced_category("OMNITRACE_MAX_DEPTH");
    _add_advanced_category("OMNITRACE_MAX_WIDTH");
    _add_advanced_category("OMNITRACE_MEMORY_PRECISION");
    _add_advanced_category("OMNITRACE_MEMORY_SCIENTIFIC");
    _add_advanced_category("OMNITRACE_MEMORY_UNITS");
    _add_advanced_category("OMNITRACE_MEMORY_WIDTH");
    _add_advanced_category("OMNITRACE_NETWORK_INTERFACE");
    _add_advanced_category("OMNITRACE_NODE_COUNT");
    _add_advanced_category("OMNITRACE_PAPI_FAIL_ON_ERROR");
    _add_advanced_category("OMNITRACE_PAPI_OVERFLOW");
    _add_advanced_category("OMNITRACE_PAPI_MULTIPLEXING");
    _add_advanced_category("OMNITRACE_PAPI_QUIET");
    _add_advanced_category("OMNITRACE_PAPI_THREADING");
    _add_advanced_category("OMNITRACE_PRECISION");
    _add_advanced_category("OMNITRACE_SCIENTIFIC");
    _add_advanced_category("OMNITRACE_STRICT_CONFIG");
    _add_advanced_category("OMNITRACE_TIMELINE_PROFILE");
    _add_advanced_category("OMNITRACE_SCIENTIFIC");
    _add_advanced_category("OMNITRACE_TIME_FORMAT");
    _add_advanced_category("OMNITRACE_TIMING_PRECISION");
    _add_advanced_category("OMNITRACE_TIMING_SCIENTIFIC");
    _add_advanced_category("OMNITRACE_TIMING_UNITS");
    _add_advanced_category("OMNITRACE_TIMING_WIDTH");
    _add_advanced_category("OMNITRACE_WIDTH");
    _add_advanced_category("OMNITRACE_COLLAPSE_THREADS");
    _add_advanced_category("OMNITRACE_COLLAPSE_PROCESSES");

#if defined(TIMEMORY_USE_PAPI)
    int _paranoid = 2;
    {
        std::ifstream _fparanoid{ "/proc/sys/kernel/perf_event_paranoid" };
        if(_fparanoid) _fparanoid >> _paranoid;
    }

    if(_paranoid > 1)
    {
        OMNITRACE_BASIC_VERBOSE(0,
                                "/proc/sys/kernel/perf_event_paranoid has a value of %i. "
                                "Disabling PAPI (requires a value <= 1)...\n",
                                _paranoid);
        OMNITRACE_BASIC_VERBOSE(0,
                                "In order to enable PAPI support, run 'echo N | sudo tee "
                                "/proc/sys/kernel/perf_event_paranoid' where N is < 2\n");
        tim::trait::runtime_enabled<comp::papi_common>::set(false);
        tim::trait::runtime_enabled<comp::papi_array_t>::set(false);
        tim::trait::runtime_enabled<comp::papi_vector>::set(false);
        tim::trait::runtime_enabled<comp::cpu_roofline_flops>::set(false);
        tim::trait::runtime_enabled<comp::cpu_roofline_dp_flops>::set(false);
        tim::trait::runtime_enabled<comp::cpu_roofline_sp_flops>::set(false);
        _config->get_papi_events() = std::string{};
    }
    else
    {
        auto _papi_events = _config->find("OMNITRACE_PAPI_EVENTS");
        _add_omnitrace_category(_papi_events);
        std::vector<std::string> _papi_choices = {};
        for(auto itr : tim::papi::available_events_info())
        {
            if(itr.available()) _papi_choices.emplace_back(itr.symbol());
        }
        _papi_events->second->set_choices(_papi_choices);
    }
#else
    _config->find("OMNITRACE_PAPI_EVENTS")->second->set_hidden(true);
    _config->get_papi_quiet() = true;
#endif

    // always initialize timemory because gotcha wrappers are always used
    auto _cmd     = tim::read_command_line(process::get_id());
    auto _cmd_env = tim::get_env<std::string>("OMNITRACE_COMMAND_LINE", "");
    if(!_cmd_env.empty()) _cmd = tim::delimit(_cmd_env, " ");
    auto _exe = (_cmd.empty()) ? "exe" : _cmd.front();
    auto _pos = _exe.find_last_of('/');
    if(_pos < _exe.length() - 1) _exe = _exe.substr(_pos + 1);
    get_exe_name() = _exe;
    _config->set_tag(_exe);

    bool _found_sep = false;
    for(const auto& itr : _cmd)
    {
        if(itr == "--") _found_sep = true;
    }
    if(!_found_sep && _cmd.size() > 1) _cmd.insert(_cmd.begin() + 1, "--");

    auto _pid       = getpid();
    auto _ppid      = getppid();
    auto _proc      = mproc::get_concurrent_processes(_ppid);
    bool _main_proc = (_proc.size() < 2 || *_proc.begin() == _pid);

    for(auto&& itr :
        tim::delimit(_config->get<std::string>("OMNITRACE_CONFIG_FILE"), ";:"))
    {
        if(_config->get_suppress_config()) continue;
        OMNITRACE_BASIC_VERBOSE(1, "Reading config file %s\n", itr.c_str());
        _config->read(itr);
        if(_config->get<bool>("OMNITRACE_CI") && _main_proc)
        {
            std::ifstream     _in{ itr };
            std::stringstream _iss{};
            while(_in)
            {
                std::string _s{};
                getline(_in, _s);
                _iss << _s << "\n";
            }
            if(!_iss.str().empty())
            {
                OMNITRACE_BASIC_PRINT("config file '%s':\n%s\n", itr.c_str(),
                                      _iss.str().c_str());
            }
        }
    }

    settings::suppress_config() = true;

    if(_init)
    {
        using argparser_t = tim::argparse::argument_parser;
        argparser_t _parser{ _exe };
        tim::timemory_init(_cmd, _parser, "omnitrace-");
        _settings_are_configured() = true;
    }

    _config->get_global_components() =
        _config->get<std::string>("OMNITRACE_TIMEMORY_COMPONENTS");

    auto _combine_perfetto_traces = _config->find("OMNITRACE_PERFETTO_COMBINE_TRACES");
    if(!_combine_perfetto_traces->second->get_environ_updated() &&
       _combine_perfetto_traces->second->get_config_updated())
    {
        _combine_perfetto_traces->second->set(_config->get<bool>("collapse_processes"));
    }

    handle_deprecated_setting("OMNITRACE_ROCM_SMI_DEVICES", "OMNITRACE_SAMPLING_GPUS");
    handle_deprecated_setting("OMNITRACE_USE_THREAD_SAMPLING",
                              "OMNITRACE_USE_PROCESS_SAMPLING");
    handle_deprecated_setting("OMNITRACE_OUTPUT_FILE", "OMNITRACE_PERFETTO_FILE");

    scope::get_fields()[scope::flat::value]     = _config->get_flat_profile();
    scope::get_fields()[scope::timeline::value] = _config->get_timeline_profile();

    settings::suppress_parsing()  = true;
    settings::use_output_suffix() = _config->get<bool>("OMNITRACE_USE_PID");
    if(settings::use_output_suffix())
        settings::default_process_suffix() = process::get_id();
#if !defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_MPI_HEADERS)
    if(tim::dmp::is_initialized()) settings::default_process_suffix() = tim::dmp::rank();
#endif

    auto _dl_verbose = _config->find("OMNITRACE_DL_VERBOSE");
    if(_dl_verbose->second->get_config_updated())
        tim::set_env(std::string{ _dl_verbose->first }, _dl_verbose->second->as_string(),
                     0);

    configure_mode_settings();
    configure_signal_handler();
    configure_disabled_settings();

    OMNITRACE_VERBOSE(1, "configuration complete\n");
}

void
configure_mode_settings()
{
    auto _set = [](const std::string& _name, bool _v) {
        if(!set_setting_value(_name, _v))
        {
            OMNITRACE_VERBOSE(
                4, "[configure_mode_settings] No configuration setting named '%s'...\n",
                _name.data());
        }
        else
        {
            OMNITRACE_VERBOSE(
                1, "[configure_mode_settings] Overriding %s to %s in %s mode...\n",
                _name.c_str(), JOIN("", std::boolalpha, _v).c_str(),
                std::to_string(get_mode()).c_str());
        }
    };

    if(get_mode() == Mode::Coverage)
    {
        set_default_setting_value("OMNITRACE_USE_CODE_COVERAGE", true);
        _set("OMNITRACE_USE_PERFETTO", false);
        _set("OMNITRACE_USE_TIMEMORY", false);
        _set("OMNITRACE_USE_ROCM_SMI", false);
        _set("OMNITRACE_USE_ROCTRACER", false);
        _set("OMNITRACE_USE_ROCPROFILER", false);
        _set("OMNITRACE_USE_KOKKOSP", false);
        _set("OMNITRACE_USE_RCCLP", false);
        _set("OMNITRACE_USE_OMPT", false);
        _set("OMNITRACE_USE_SAMPLING", false);
        _set("OMNITRACE_USE_PROCESS_SAMPLING", false);
        _set("OMNITRACE_CRITICAL_TRACE", false);
    }
    else if(get_mode() == Mode::Sampling)
    {
        set_default_setting_value("OMNITRACE_USE_SAMPLING", true);
        set_default_setting_value("OMNITRACE_USE_PROCESS_SAMPLING", true);
        _set("OMNITRACE_CRITICAL_TRACE", false);
    }

    if(gpu::device_count() == 0)
    {
        OMNITRACE_VERBOSE(1, "No HIP devices were found: disabling roctracer, "
                             "rocprofiler, and rocm_smi...\n");
        _set("OMNITRACE_USE_ROCPROFILER", false);
        _set("OMNITRACE_USE_ROCTRACER", false);
        _set("OMNITRACE_USE_ROCM_SMI", false);
    }

    get_instrumentation_interval() = std::max<size_t>(get_instrumentation_interval(), 1);

    if(get_use_kokkosp())
    {
        auto _current_kokkosp_lib = tim::get_env<std::string>("KOKKOS_PROFILE_LIBRARY");
        if(_current_kokkosp_lib.find("libomnitrace-dl.so") == std::string::npos &&
           _current_kokkosp_lib.find("libomnitrace.so") == std::string::npos)
        {
            auto        _force   = 0;
            std::string _message = {};
            if(std::regex_search(_current_kokkosp_lib, std::regex{ "libtimemory\\." }))
            {
                _force = 1;
                _message =
                    JOIN("", " (forced. Previous value: '", _current_kokkosp_lib, "')");
            }
            OMNITRACE_VERBOSE_F(1, "Setting KOKKOS_PROFILE_LIBRARY=%s%s\n",
                                "libomnitrace.so", _message.c_str());
            tim::set_env("KOKKOS_PROFILE_LIBRARY", "libomnitrace.so", _force);
        }
    }

    // recycle all subsequent thread ids
    threading::recycle_ids() =
        tim::get_env<bool>("OMNITRACE_RECYCLE_TIDS", !get_use_sampling());

    if(!get_config()->get_enabled())
    {
        _set("OMNITRACE_USE_PERFETTO", false);
        _set("OMNITRACE_USE_TIMEMORY", false);
        _set("OMNITRACE_USE_ROCM_SMI", false);
        _set("OMNITRACE_USE_ROCTRACER", false);
        _set("OMNITRACE_USE_ROCPROFILER", false);
        _set("OMNITRACE_USE_KOKKOSP", false);
        _set("OMNITRACE_USE_RCCLP", false);
        _set("OMNITRACE_USE_OMPT", false);
        _set("OMNITRACE_USE_SAMPLING", false);
        _set("OMNITRACE_USE_PROCESS_SAMPLING", false);
        _set("OMNITRACE_USE_CODE_COVERAGE", false);
        _set("OMNITRACE_CRITICAL_TRACE", false);
        set_setting_value("OMNITRACE_TIMEMORY_COMPONENTS", std::string{});
        set_setting_value("OMNITRACE_PAPI_EVENTS", std::string{});
    }
}

void
configure_signal_handler()
{
    auto _config = settings::shared_instance();
    auto _ignore_dyninst_trampoline =
        tim::get_env("OMNITRACE_IGNORE_DYNINST_TRAMPOLINE", false);
    // this is how dyninst looks up the env variable
    static auto _dyninst_trampoline_signal =
        getenv("DYNINST_SIGNAL_TRAMPOLINE_SIGILL") ? SIGILL : SIGTRAP;

    if(_config->get_enable_signal_handler())
    {
        using signal_settings = tim::signal_settings;
        using sys_signal      = tim::sys_signal;
        tim::disable_signal_detection();
        auto _exit_action = [](int nsig) {
            tim::sampling::block_signals(get_sampling_signals(),
                                         tim::sampling::sigmask_scope::process);
            OMNITRACE_BASIC_PRINT(
                "Finalizing afer signal %i :: %s\n", nsig,
                signal_settings::str(static_cast<sys_signal>(nsig)).c_str());
            if(get_state() == State::Active) omnitrace_finalize();
            kill(process::get_id(), nsig);
        };
        signal_settings::set_exit_action(_exit_action);
        signal_settings::check_environment();
        auto default_signals = signal_settings::get_default();
        for(const auto& itr : default_signals)
            signal_settings::enable(itr);
        if(_ignore_dyninst_trampoline)
            signal_settings::disable(static_cast<sys_signal>(_dyninst_trampoline_signal));
        auto enabled_signals = signal_settings::get_enabled();
        tim::enable_signal_detection(enabled_signals);
    }

    if(_ignore_dyninst_trampoline)
    {
        using signal_handler_t                      = void (*)(int);
        static signal_handler_t _old_handler        = nullptr;
        static auto             _trampoline_handler = [](int _v) {
            if(_v == _dyninst_trampoline_signal)
            {
                auto _info =
                    ::tim::signal_settings::get_info(static_cast<tim::sys_signal>(_v));
                OMNITRACE_VERBOSE(
                    2,
                    "signal %s (%i) ignored (OMNITRACE_IGNORE_DYNINST_TRAMPOLINE=ON)\n",
                    std::get<0>(_info).c_str(), _v);
                if(get_verbose_env() > 1 || get_debug_env())
                    ::tim::print_demangled_backtrace<64>();
                if(_old_handler) _old_handler(_v);
            }
        };
        _old_handler = signal(_dyninst_trampoline_signal,
                              static_cast<signal_handler_t>(_trampoline_handler));
    }
}

void
configure_disabled_settings()
{
    auto _config            = settings::shared_instance();
    auto _handle_use_option = [_config](const std::string& _opt,
                                        const std::string& _category) {
        if(!_config->get<bool>(_opt))
        {
            auto _disabled = _config->disable_category(_category);
            _config->enable(_opt);
            for(auto&& itr : _disabled)
                OMNITRACE_VERBOSE(3, "[%s=OFF]    disabled option :: '%s'\n",
                                  _opt.c_str(), itr.c_str());
            return false;
        }
        auto _enabled = _config->enable_category(_category);
        for(auto&& itr : _enabled)
            OMNITRACE_VERBOSE(3, "[%s=ON]      enabled option :: '%s'\n", _opt.c_str(),
                              itr.c_str());
        return true;
    };

    _handle_use_option("OMNITRACE_USE_SAMPLING", "sampling");
    _handle_use_option("OMNITRACE_USE_PROCESS_SAMPLING", "process_sampling");
    _handle_use_option("OMNITRACE_USE_KOKKOSP", "kokkos");
    _handle_use_option("OMNITRACE_USE_PERFETTO", "perfetto");
    _handle_use_option("OMNITRACE_USE_TIMEMORY", "timemory");
    _handle_use_option("OMNITRACE_USE_OMPT", "ompt");
    _handle_use_option("OMNITRACE_USE_RCCLP", "rcclp");
    _handle_use_option("OMNITRACE_USE_ROCM_SMI", "rocm_smi");
    _handle_use_option("OMNITRACE_USE_ROCTRACER", "roctracer");
    _handle_use_option("OMNITRACE_USE_ROCPROFILER", "rocprofiler");
    _handle_use_option("OMNITRACE_CRITICAL_TRACE", "critical_trace");

#if !defined(OMNITRACE_USE_ROCTRACER) || OMNITRACE_USE_ROCTRACER == 0
    _config->find("OMNITRACE_USE_ROCTRACER")->second->set_hidden(true);
    for(const auto& itr : _config->disable_category("roctracer"))
        _config->find(itr)->second->set_hidden(true);
#endif

#if !defined(OMNITRACE_USE_ROCPROFILER) || OMNITRACE_USE_ROCPROFILER == 0
    _config->find("OMNITRACE_USE_ROCPROFILER")->second->set_hidden(true);
    for(const auto& itr : _config->disable_category("rocprofiler"))
        _config->find(itr)->second->set_hidden(true);
#endif

#if !defined(OMNITRACE_USE_ROCM_SMI) || OMNITRACE_USE_ROCM_SMI == 0
    _config->find("OMNITRACE_USE_ROCM_SMI")->second->set_hidden(true);
    for(const auto& itr : _config->disable_category("rocm_smi"))
        _config->find(itr)->second->set_hidden(true);
#endif

#if defined(OMNITRACE_USE_OMPT) || OMNITRACE_USE_OMPT == 0
    _config->find("OMNITRACE_USE_OMPT")->second->set_hidden(true);
    for(const auto& itr : _config->disable_category("ompt"))
        _config->find(itr)->second->set_hidden(true);
#endif

#if !defined(TIMEMORY_USE_MPI) || TIMEMORY_USE_MPI == 0
    _config->disable("OMNITRACE_PERFETTO_COMBINE_TRACES");
    _config->disable("OMNITRACE_COLLAPSE_PROCESSES");
    _config->find("OMNITRACE_PERFETTO_COMBINE_TRACES")->second->set_hidden(true);
    _config->find("OMNITRACE_COLLAPSE_PROCESSES")->second->set_hidden(true);
#endif

    _config->disable_category("throttle");

    // user bundle components
    _config->disable("components");
    _config->disable("global_components");
    _config->disable("ompt_components");
    _config->disable("kokkos_components");
    _config->disable("trace_components");
    _config->disable("profiler_components");

    // miscellaneous
    _config->disable("destructor_report");
    _config->disable("stack_clearing");
    _config->disable("add_secondary");

    // output fields
    _config->disable("auto_output");
    _config->disable("file_output");
    _config->disable("plot_output");
    _config->disable("dart_output");
    _config->disable("flamegraph_output");
    _config->disable("separator_freq");

    // exclude some timemory settings which are not relevant to omnitrace
    //  exact matches, e.g. OMNITRACE_BANNER
    std::string _hidden_exact_re =
        "^OMNITRACE_(BANNER|DESTRUCTOR_REPORT|COMPONENTS|(GLOBAL|MPIP|NCCLP|OMPT|"
        "PROFILER|TRACE|KOKKOS)_COMPONENTS|PYTHON_EXE|PAPI_ATTACH|PLOT_OUTPUT|SEPARATOR_"
        "FREQ|STACK_CLEARING|TARGET_PID|THROTTLE_(COUNT|VALUE)|(AUTO|FLAMEGRAPH)_OUTPUT|"
        "(ENABLE|DISABLE)_ALL_SIGNALS|ALLOW_SIGNAL_HANDLER|CTEST_NOTES|INSTRUCTION_"
        "ROOFLINE|ADD_SECONDARY|MAX_THREAD_BOOKMARKS)$";

    //  leading matches, e.g. OMNITRACE_MPI_[A-Z_]+
    std::string _hidden_begin_re =
        "^OMNITRACE_(ERT|DART|MPI|UPCXX|ROOFLINE|CUDA|NVTX|CUPTI)_[A-Z_]+$";

    auto _hidden_exact = std::set<std::string>{};

#if !defined(TIMEMORY_USE_CRAYPAT)
    _hidden_exact.emplace("OMNITRACE_CRAYPAT");
#endif

    for(const auto& itr : *_config)
    {
        auto _v = itr.second->get_env_name();
        if(_hidden_exact.count(_v) > 0 ||
           std ::regex_match(_v, std::regex{ _hidden_exact_re }) ||
           std::regex_match(_v, std::regex{ _hidden_begin_re }))
        {
            itr.second->set_enabled(false);
            itr.second->set_hidden(true);
        }
    }
}

void
handle_deprecated_setting(const std::string& _old, const std::string& _new, int _verbose)
{
    auto _config      = settings::shared_instance();
    auto _old_setting = _config->find(_old);
    auto _new_setting = _config->find(_new);

    if(_old_setting == _config->end()) return;

    OMNITRACE_CI_THROW(_new_setting == _config->end(),
                       "New configuration setting not found: '%s'", _new.c_str());

    if(_old_setting->second->get_environ_updated() ||
       _old_setting->second->get_config_updated())
    {
        auto _separator = [_verbose]() {
            std::array<char, 79> _v = {};
            _v.fill('=');
            _v.back() = '\0';
            OMNITRACE_VERBOSE(_verbose, "#%s#\n", _v.data());
        };
        _separator();
        OMNITRACE_VERBOSE(_verbose, "#\n");
        OMNITRACE_VERBOSE(_verbose, "# DEPRECATION NOTICE:\n");
        OMNITRACE_VERBOSE(_verbose, "#   %s is deprecated!\n", _old.c_str());
        OMNITRACE_VERBOSE(_verbose, "#   Use %s instead!\n", _new.c_str());

        if(!_new_setting->second->get_environ_updated() &&
           !_new_setting->second->get_config_updated())
        {
            auto _before = _new_setting->second->as_string();
            _new_setting->second->parse(_old_setting->second->as_string());
            auto _after = _new_setting->second->as_string();

            if(_before != _after)
            {
                std::string _cause =
                    (_old_setting->second->get_environ_updated()) ? "environ" : "config";
                OMNITRACE_VERBOSE(_verbose, "#\n");
                OMNITRACE_VERBOSE(_verbose, "# %s :: '%s' -> '%s'\n", _new.c_str(),
                                  _before.c_str(), _after.c_str());
                OMNITRACE_VERBOSE(_verbose, "#   via %s (%s)\n", _old.c_str(),
                                  _cause.c_str());
            }
        }

        OMNITRACE_VERBOSE(_verbose, "#\n");
        _separator();
    }
}

void
print_banner(std::ostream& _os)
{
    static const char* _banner = R"banner(

      ______   .___  ___. .__   __.  __  .___________..______          ___       ______  _______
     /  __  \  |   \/   | |  \ |  | |  | |           ||   _  \        /   \     /      ||   ____|
    |  |  |  | |  \  /  | |   \|  | |  | `---|  |----`|  |_)  |      /  ^  \   |  ,----'|  |__
    |  |  |  | |  |\/|  | |  . `  | |  |     |  |     |      /      /  /_\  \  |  |     |   __|
    |  `--'  | |  |  |  | |  |\   | |  |     |  |     |  |\  \----./  _____  \ |  `----.|  |____
     \______/  |__|  |__| |__| \__| |__|     |__|     | _| `._____/__/     \__\ \______||_______|

    )banner";
    _os << _banner << std::endl;
}

void
print_settings(
    std::ostream&                                                                _ros,
    std::function<bool(const std::string_view&, const std::set<std::string>&)>&& _filter)
{
    OMNITRACE_CONDITIONAL_BASIC_PRINT(true, "configuration:\n");

    std::stringstream _os{};

    bool _print_desc = get_debug() || tim::get_env("OMNITRACE_SETTINGS_DESC", false);
    bool _md         = tim::get_env<bool>("OMNITRACE_SETTINGS_DESC_MARKDOWN", false);

    constexpr size_t nfields = 3;
    using str_array_t        = std::array<std::string, nfields>;
    std::vector<str_array_t>    _data{};
    std::array<size_t, nfields> _widths{};
    _widths.fill(0);
    for(const auto& itr : *get_config())
    {
        if(itr.second->get_hidden()) continue;
        if(!itr.second->get_enabled()) continue;
        if(_filter(itr.first, itr.second->get_categories()))
        {
            auto _disp = itr.second->get_display(std::ios::boolalpha);
            _data.emplace_back(str_array_t{ _disp.at("env_name"), _disp.at("value"),
                                            _disp.at("description") });
            for(size_t i = 0; i < nfields; ++i)
            {
                size_t _wextra = (_md && i < 2) ? 2 : 0;
                _widths.at(i)  = std::max<size_t>(_widths.at(i),
                                                 _data.back().at(i).length() + _wextra);
            }
        }
    }

    std::sort(_data.begin(), _data.end(), [](const auto& lhs, const auto& rhs) {
        auto _npos = std::string::npos;
        // OMNITRACE_CONFIG_FILE always first
        if(lhs.at(0) == "OMNITRACE_MODE") return true;
        if(rhs.at(0) == "OMNITRACE_MODE") return false;
        // OMNITRACE_CONFIG_FILE always second
        if(lhs.at(0).find("OMNITRACE_CONFIG") != _npos) return true;
        if(rhs.at(0).find("OMNITRACE_CONFIG") != _npos) return false;
        // OMNITRACE_USE_* prioritized
        auto _lhs_use = lhs.at(0).find("OMNITRACE_USE_");
        auto _rhs_use = rhs.at(0).find("OMNITRACE_USE_");
        if(_lhs_use != _rhs_use && _lhs_use < _rhs_use) return true;
        if(_lhs_use != _rhs_use && _lhs_use > _rhs_use) return false;
        // alphabetical sort
        return lhs.at(0) < rhs.at(0);
    });

    auto tot_width = std::accumulate(_widths.begin(), _widths.end(), 0);
    if(!_print_desc) tot_width -= _widths.back() + 4;

    size_t _spacer_extra = 9;
    if(!_md)
        _spacer_extra += 2;
    else if(_md && _print_desc)
        _spacer_extra -= 1;
    std::stringstream _spacer{};
    _spacer.fill('-');
    _spacer << "#" << std::setw(tot_width + _spacer_extra) << ""
            << "#";
    _os << _spacer.str() << "\n";
    // _os << "# api::omnitrace settings:" << std::setw(tot_width - 8) << "#" << "\n";
    for(const auto& itr : _data)
    {
        _os << ((_md) ? "| " : "# ");
        for(size_t i = 0; i < nfields; ++i)
        {
            switch(i)
            {
                case 0: _os << std::left; break;
                case 1: _os << std::left; break;
                case 2: _os << std::left; break;
            }
            if(_md)
            {
                std::stringstream _ss{};
                _ss.setf(_os.flags());
                std::string _extra = (i < 2) ? "`" : "";
                _ss << _extra << itr.at(i) << _extra;
                _os << std::setw(_widths.at(i)) << _ss.str() << " | ";
                if(!_print_desc && i == 1) break;
            }
            else
            {
                _os << std::setw(_widths.at(i)) << itr.at(i) << " ";
                if(!_print_desc && i == 1) break;
                switch(i)
                {
                    case 0: _os << "= "; break;
                    case 1: _os << "[ "; break;
                    case 2: _os << "]"; break;
                }
            }
        }
        _os << ((_md) ? "\n" : "  #\n");
    }
    _os << _spacer.str() << "\n";

    _ros << _os.str() << std::flush;
}

void
print_settings(bool _include_env)
{
    if(dmp::rank() > 0) return;

    // generic filter for filtering relevant options
    auto _is_omnitrace_option = [](const auto& _v, const auto&) {
        return (_v.find("OMNITRACE_") == 0);
    };

    if(_include_env)
    {
        tim::print_env(std::cerr, [_is_omnitrace_option](const std::string& _v) {
            return _is_omnitrace_option(_v, std::set<std::string>{});
        });
    }

    print_settings(std::cerr, _is_omnitrace_option);

    fprintf(stderr, "\n");
}

std::string&
get_exe_name()
{
    static std::string _v = {};
    return _v;
}

std::string
get_config_file()
{
    static auto _v = get_config()->find("OMNITRACE_CONFIG_FILE");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
}

Mode
get_mode()
{
    if(!settings_are_configured())
    {
        auto _mode = tim::get_env_choice<std::string>(
            "OMNITRACE_MODE", "trace", { "trace", "sampling", "coverage" });
        if(_mode == "sampling")
            return Mode::Sampling;
        else if(_mode == "coverage")
            return Mode::Coverage;
        return Mode::Trace;
    }
    static auto _m =
        std::unordered_map<std::string_view, Mode>{ { "trace", Mode::Trace },
                                                    { "sampling", Mode::Sampling },
                                                    { "coverage", Mode::Coverage } };
    static auto _v = get_config()->find("OMNITRACE_MODE");
    try
    {
        return _m.at(static_cast<tim::tsettings<std::string>&>(*_v->second).get());
    } catch(std::runtime_error& _e)
    {
        auto _mode = static_cast<tim::tsettings<std::string>&>(*_v->second).get();
        std::stringstream _ss{};
        for(const auto& itr : _v->second->get_choices())
            _ss << ", " << itr;
        auto _msg = (_ss.str().length() > 2) ? _ss.str().substr(2) : std::string{};
        OMNITRACE_THROW("[%s] invalid mode %s. Choices: %s\n", __FUNCTION__,
                        _mode.c_str(), _msg.c_str());
    }
    return Mode::Trace;
}

bool&
is_attached()
{
    static bool _v = false;
    return _v;
}

bool&
is_binary_rewrite()
{
    static bool _v = false;
    return _v;
}

bool
get_debug_env()
{
    return (settings_are_configured())
               ? get_debug()
               : tim::get_env<bool>("OMNITRACE_DEBUG", false, false);
}

bool
get_is_continuous_integration()
{
    if(!settings_are_configured())
        return tim::get_env<bool>("OMNITRACE_CI", false, false);
    static auto _v = get_config()->find("OMNITRACE_CI");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_debug_init()
{
    return tim::get_env<bool>("OMNITRACE_DEBUG_INIT", get_debug_env());
}

bool
get_debug_finalize()
{
    return tim::get_env<bool>("OMNITRACE_DEBUG_FINALIZE", false);
}

bool
get_debug()
{
    static auto _v = get_config()->find("OMNITRACE_DEBUG");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_debug_sampling()
{
    static bool _v =
        tim::get_env<bool>("OMNITRACE_DEBUG_SAMPLING",
                           (settings_are_configured() ? get_debug() : get_debug_env()));
    return _v;
}

int
get_verbose_env()
{
    return (settings_are_configured()) ? get_verbose()
                                       : tim::get_env<int>("OMNITRACE_VERBOSE", 0, false);
}

int
get_verbose()
{
    static auto _v = get_config()->find("OMNITRACE_VERBOSE");
    return static_cast<tim::tsettings<int>&>(*_v->second).get();
}

bool&
get_use_perfetto()
{
    static auto _v = get_config()->find("OMNITRACE_USE_PERFETTO");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool&
get_use_timemory()
{
    static auto _v = get_config()->find("OMNITRACE_USE_TIMEMORY");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_roctracer()
{
#if defined(OMNITRACE_USE_ROCTRACER) && OMNITRACE_USE_ROCTRACER > 0
    static auto _v = get_config()->find("OMNITRACE_USE_ROCTRACER");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    return false;
#endif
}

bool
get_use_rocprofiler()
{
#if defined(OMNITRACE_USE_ROCPROFILER) && OMNITRACE_USE_ROCPROFILER > 0
    static auto _v = get_config()->find("OMNITRACE_USE_ROCPROFILER");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    return false;
#endif
}

bool
get_use_rocm_smi()
{
#if defined(OMNITRACE_USE_ROCM_SMI) && OMNITRACE_USE_ROCM_SMI > 0
    static auto _v = get_config()->find("OMNITRACE_USE_ROCM_SMI");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    return false;
#endif
}

bool
get_use_roctx()
{
#if defined(OMNITRACE_USE_ROCTRACER) && OMNITRACE_USE_ROCTRACER > 0
    static auto _v = get_config()->find("OMNITRACE_USE_ROCTX");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    return false;
#endif
}

bool&
get_use_sampling()
{
#if defined(TIMEMORY_USE_LIBUNWIND)
    static auto _v = get_config()->find("OMNITRACE_USE_SAMPLING");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    OMNITRACE_THROW(
        "Error! sampling was enabled but omnitrace was not built with libunwind support");
    static bool _v = false;
    return _v;
#endif
}

bool&
get_use_process_sampling()
{
    static auto _v = get_config()->find("OMNITRACE_USE_PROCESS_SAMPLING");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool&
get_use_pid()
{
    static auto _v = get_config()->find("OMNITRACE_USE_PID");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool&
get_use_mpip()
{
    static bool _v = tim::get_env("OMNITRACE_USE_MPIP", false, false);
    return _v;
}

bool&
get_use_critical_trace()
{
    static auto _v = get_config()->find("OMNITRACE_CRITICAL_TRACE");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_kokkosp()
{
    static auto _v = get_config()->find("OMNITRACE_USE_KOKKOSP");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_kokkosp_kernel_logger()
{
    static auto _v = get_config()->find("OMNITRACE_KOKKOS_KERNEL_LOGGER");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_ompt()
{
#if defined(TIMEMORY_USE_OMPT)
    static auto _v = get_config()->find("OMNITRACE_USE_OMPT");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    return false;
#endif
}

bool
get_use_code_coverage()
{
    static auto _v = get_config()->find("OMNITRACE_USE_CODE_COVERAGE");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_rcclp()
{
    static auto _v = get_config()->find("OMNITRACE_USE_RCCLP");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_critical_trace_debug()
{
    static auto _v = get_config()->find("OMNITRACE_CRITICAL_TRACE_DEBUG");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_critical_trace_serialize_names()
{
    static auto _v = get_config()->find("OMNITRACE_CRITICAL_TRACE_SERIALIZE_NAMES");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_sampling_keep_internal()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_KEEP_INTERNAL");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_sampling_realtime()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_REALTIME");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_sampling_cputime()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_CPUTIME");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

int
get_sampling_rtoffset()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_REALTIME_OFFSET");
    return static_cast<tim::tsettings<int>&>(*_v->second).get();
}

bool
get_timeline_sampling()
{
    static auto _v = get_config()->find("OMNITRACE_TIMELINE_SAMPLING");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_flat_sampling()
{
    static auto _v = get_config()->find("OMNITRACE_FLAT_SAMPLING");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_roctracer_timeline_profile()
{
    static auto _v = get_config()->find("OMNITRACE_ROCTRACER_TIMELINE_PROFILE");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_roctracer_flat_profile()
{
    static auto _v = get_config()->find("OMNITRACE_ROCTRACER_FLAT_PROFILE");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_trace_hsa_api()
{
    static auto _v = get_config()->find("OMNITRACE_ROCTRACER_HSA_API");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_trace_hsa_activity()
{
    static auto _v = get_config()->find("OMNITRACE_ROCTRACER_HSA_ACTIVITY");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

int64_t
get_critical_trace_per_row()
{
    static auto _v = get_config()->find("OMNITRACE_CRITICAL_TRACE_PER_ROW");
    return static_cast<tim::tsettings<int64_t>&>(*_v->second).get();
}

size_t
get_perfetto_shmem_size_hint()
{
    static auto _v = get_config()->find("OMNITRACE_PERFETTO_SHMEM_SIZE_HINT_KB");
    return static_cast<tim::tsettings<size_t>&>(*_v->second).get();
}

size_t
get_perfetto_buffer_size()
{
    static auto _v = get_config()->find("OMNITRACE_PERFETTO_BUFFER_SIZE_KB");
    return static_cast<tim::tsettings<size_t>&>(*_v->second).get();
}

bool
get_perfetto_combined_traces()
{
#if defined(TIMEMORY_USE_MPI) && TIMEMORY_USE_MPI > 0
    static auto _v = get_config()->find("OMNITRACE_PERFETTO_COMBINE_TRACES");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    return false;
#endif
}

std::string
get_perfetto_fill_policy()
{
    static auto _v = get_config()->find("OMNITRACE_PERFETTO_FILL_POLICY");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
}

std::set<std::string>
get_perfetto_categories()
{
    static auto _v     = get_config()->find("OMNITRACE_PERFETTO_CATEGORIES");
    static auto _avail = get_available_perfetto_categories<std::set<std::string>>();
    auto        _ret   = std::set<std::string>{};
    for(auto itr : tim::delimit(
            static_cast<tim::tsettings<std::string>&>(*_v->second).get(), " ,;:"))
    {
        if(_avail.count(itr) > 0) _ret.emplace(itr);
    }
    return _ret;
}

uint64_t
get_critical_trace_update_freq()
{
    static uint64_t _v =
        get_config()->get<uint64_t>("OMNITRACE_CRITICAL_TRACE_BUFFER_COUNT");
    return _v;
}

uint64_t
get_critical_trace_num_threads()
{
    static uint64_t _v =
        get_config()->get<uint64_t>("OMNITRACE_CRITICAL_TRACE_NUM_THREADS");
    return _v;
}

std::string
get_trace_hsa_api_types()
{
    static std::string _v =
        get_config()->get<std::string>("OMNITRACE_ROCTRACER_HSA_API_TYPES");
    return _v;
}

std::string&
get_backend()
{
    // select inprocess, system, or both (i.e. all)
    static auto _v = get_config()->find("OMNITRACE_PERFETTO_BACKEND");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
}

std::string
get_perfetto_output_filename()
{
    static auto _v       = get_config()->find("OMNITRACE_PERFETTO_FILE");
    auto        _val     = static_cast<tim::tsettings<std::string>&>(*_v->second).get();
    auto        _pos_dir = _val.find_last_of('/');
    auto        _dir     = std::string{};
    auto        _ext     = std::string{ "proto" };
    if(_pos_dir != std::string::npos)
    {
        _dir = _val.substr(0, _pos_dir + 1);
        _val = _val.substr(_pos_dir + 1);
    }
    auto _pos_ext = _val.find_last_of('.');
    if(_pos_ext + 1 < _val.length())
    {
        _ext = _val.substr(_pos_ext + 1);
        _val = _val.substr(0, _pos_ext);
    }
    _val = settings::compose_output_filename(_val, _ext, settings::use_output_suffix(),
                                             settings::default_process_suffix(), false,
                                             _dir);
    if(!_val.empty() && _val.at(0) != '/')
        return settings::format(JOIN('/', "%env{PWD}%", _val), get_config()->get_tag());
    return _val;
}

size_t&
get_instrumentation_interval()
{
    static auto _v = get_config()->find("OMNITRACE_INSTRUMENTATION_INTERVAL");
    return static_cast<tim::tsettings<size_t>&>(*_v->second).get();
}

double
get_sampling_freq()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_FREQ");
    return static_cast<tim::tsettings<double>&>(*_v->second).get();
}

double&
get_sampling_delay()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_DELAY");
    return static_cast<tim::tsettings<double>&>(*_v->second).get();
}

std::string
get_sampling_cpus()
{
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_CPUS");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
}

int64_t
get_critical_trace_count()
{
    static auto _v = get_config()->find("OMNITRACE_CRITICAL_TRACE_COUNT");
    return static_cast<tim::tsettings<int64_t>&>(*_v->second).get();
}

double
get_process_sampling_freq()
{
    static auto _v = get_config()->find("OMNITRACE_PROCESS_SAMPLING_FREQ");
    auto        _val =
        std::min<double>(static_cast<tim::tsettings<double>&>(*_v->second).get(), 1000.0);
    if(_val < 1.0e-9) return get_sampling_freq();
    return _val;
}

std::string
get_sampling_gpus()
{
#if defined(OMNITRACE_USE_ROCM_SMI) && OMNITRACE_USE_ROCM_SMI > 0
    static auto _v = get_config()->find("OMNITRACE_SAMPLING_GPUS");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
#else
    return std::string{};
#endif
}

bool
get_trace_thread_locks()
{
    static auto _v = get_config()->find("OMNITRACE_TRACE_THREAD_LOCKS");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

std::string
get_rocm_events()
{
    static auto _v = get_config()->find("OMNITRACE_ROCM_EVENTS");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
}

bool
get_trace_thread_rwlocks()
{
    static auto _v = get_config()->find("OMNITRACE_TRACE_THREAD_RW_LOCKS");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_debug_tid()
{
    static auto _vlist = []() {
        std::unordered_set<int64_t> _tids{};
        for(auto itr : tim::delimit<std::vector<int64_t>>(
                tim::get_env<std::string>("OMNITRACE_DEBUG_TIDS", ""),
                ",: ", [](const std::string& _v) { return std::stoll(_v); }))
            _tids.insert(itr);
        return _tids;
    }();
    static thread_local bool _v =
        _vlist.empty() || _vlist.count(tim::threading::get_id()) > 0;
    return _v;
}

bool
get_debug_pid()
{
    static auto _vlist = []() {
        std::unordered_set<int64_t> _pids{};
        for(auto itr : tim::delimit<std::vector<int64_t>>(
                tim::get_env<std::string>("OMNITRACE_DEBUG_PIDS", ""),
                ",: ", [](const std::string& _v) { return std::stoll(_v); }))
            _pids.insert(itr);
        return _pids;
    }();
    static bool _v = _vlist.empty() || _vlist.count(tim::process::get_id()) > 0 ||
                     _vlist.count(dmp::rank()) > 0;
    return _v;
}
}  // namespace config

State&
get_state()
{
    static State _v{ State::PreInit };
    return _v;
}

State
set_state(State _n)
{
    auto _o = get_state();
    OMNITRACE_CONDITIONAL_PRINT_F(get_debug_init(), "Setting state :: %s -> %s\n",
                                  std::to_string(_o).c_str(), std::to_string(_n).c_str());
    // state should always be increased, not decreased
    OMNITRACE_CI_BASIC_THROW(_n < _o,
                             "State is being assigned to a lesser value :: %s -> %s",
                             std::to_string(_o).c_str(), std::to_string(_n).c_str());
    get_state() = _n;
    return _o;
}
}  // namespace omnitrace
