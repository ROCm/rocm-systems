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

#include <timemory/backends/dmp.hpp>
#include <timemory/backends/mpi.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/environment.hpp>
#include <timemory/sampling/allocator.hpp>
#include <timemory/settings.hpp>
#include <timemory/settings/types.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/declaration.hpp>
#include <timemory/utility/signals.hpp>

#include <array>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <ostream>
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

#define OMNITRACE_CONFIG_SETTING(TYPE, ENV_NAME, DESCRIPTION, INITIAL_VALUE, ...)        \
    {                                                                                    \
        auto _ret = _config->insert<TYPE, TYPE>(                                         \
            ENV_NAME, get_setting_name(ENV_NAME), DESCRIPTION, INITIAL_VALUE,            \
            std::set<std::string>{ "custom", "omnitrace", "omnitrace-library",           \
                                   __VA_ARGS__ });                                       \
        if(!_ret.second)                                                                 \
            OMNITRACE_PRINT("Warning! Duplicate setting: %s / %s\n",                     \
                            get_setting_name(ENV_NAME).c_str(), ENV_NAME);               \
    }
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
    _settings_are_configured() = true;

    static bool _once = false;
    if(_once) return;
    _once = true;

    if(get_state() < State::Init)
    {
        ::tim::print_demangled_backtrace<64>();
        OMNITRACE_CONDITIONAL_THROW(true,
                                    "config::configure_settings() called before "
                                    "omnitrace_init_library. state = %s",
                                    std::to_string(get_state()).c_str());
    }

    static auto _config = settings::shared_instance();

    // if using timemory, default to perfetto being off
    auto _default_perfetto_v =
        !tim::get_env<bool>("OMNITRACE_USE_TIMEMORY", false, false);

    auto _system_backend = tim::get_env("OMNITRACE_BACKEND_SYSTEM", false, false);

    auto _omnitrace_debug = _config->get<bool>("OMNITRACE_DEBUG");
    if(_omnitrace_debug) tim::set_env("TIMEMORY_DEBUG_SETTINGS", "1", 0);

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_PERFETTO", "Enable perfetto backend",
                             _default_perfetto_v, "backend", "perfetto",
                             "instrumentation");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_TIMEMORY", "Enable timemory backend",
                             !_config->get<bool>("OMNITRACE_USE_PERFETTO"), "backend",
                             "timemory", "instrumentation", "sampling");

#if defined(OMNITRACE_USE_ROCTRACER)
    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_ROCTRACER", "Enable ROCM tracing", true,
                             "backend", "roctracer", "rocm");
#endif

#if defined(OMNITRACE_USE_ROCM_SMI)
    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_USE_ROCM_SMI",
        "Enable sampling GPU power, temp, utilization, and memory usage", true, "backend",
        "rocm-smi", "rocm");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_ROCM_SMI_DEVICES",
                             "Devices to query when OMNITRACE_USE_ROCM_SMI=ON", "all",
                             "backend", "rocm-smi", "rocm");
#endif

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_SAMPLING",
                             "Enable statistical sampling of call-stack", false,
                             "backend", "sampling");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_USE_PID",
        "Enable tagging filenames with process identifier (either MPI rank or pid)", true,
        "io");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_KOKKOSP",
                             "Enable support for Kokkos Tools", false, "kokkos",
                             "backend");

#if defined(TIMEMORY_USE_OMPT)
    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_OMPT",
                             "Enable support for OpenMP-Tools", true, "openmp", "ompt",
                             "backend");
#endif

    OMNITRACE_CONFIG_SETTING(size_t, "OMNITRACE_INSTRUMENTATION_INTERVAL",
                             "Instrumentation only takes measurements once every N "
                             "function calls (not statistical)",
                             1, "instrumentation");

    OMNITRACE_CONFIG_SETTING(
        double, "OMNITRACE_SAMPLING_FREQ",
        "Number of software interrupts per second when OMNITTRACE_USE_SAMPLING=ON", 10.0,
        "sampling");

    OMNITRACE_CONFIG_SETTING(
        double, "OMNITRACE_SAMPLING_DELAY",
        "Number of seconds to wait before the first sampling signal is delivered, "
        "increasing this value can fix deadlocks during init",
        0.5, "sampling");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_SAMPLING_CPUS",
        "CPUs to collect frequency information for. Values should be separated by commas "
        "and can be explicit or ranges, e.g. 0,1,5-8. An empty value implies 'all' and "
        "'none' suppresses all CPU frequency sampling",
        "", "sampling");

    auto _backend = tim::get_env_choice<std::string>(
        "OMNITRACE_BACKEND",
        (_system_backend)
            ? "system"      // if OMNITRACE_BACKEND_SYSTEM is true, default to system.
            : "inprocess",  // Otherwise, default to inprocess
        { "inprocess", "system", "all" }, false);

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_BACKEND",
                             "Specify the perfetto backend to activate. Options are: "
                             "'inprocess', 'system', or 'all'",
                             _backend, "perfetto", "instrumentation", "sampling");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_CRITICAL_TRACE",
                             "Enable generation of the critical trace", false, "backend",
                             "critical-trace", "instrumentation");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_FLAT_SAMPLING",
                             "Ignore hierarchy in all statistical sampling entries",
                             _config->get_flat_profile(), "sampling", "data_layout");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_TIMELINE_SAMPLING",
        "Create unique entries for every sample when statistical sampling is enabled",
        _config->get_timeline_profile(), "sampling", "data_layout");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_ROCTRACER_FLAT_PROFILE",
        "Ignore hierarchy in all kernels entries with timemory backend",
        _config->get_flat_profile(), "roctracer", "data_layout", "rocm");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_ROCTRACER_TIMELINE_PROFILE",
        "Create unique entries for every kernel with timemory backend",
        _config->get_timeline_profile(), "roctracer", "data_layout", "rocm");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_ROCTRACER_HSA_ACTIVITY",
                             "Enable HSA activity tracing support", false, "roctracer",
                             "rocm");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_ROCTRACER_HSA_API",
                             "Enable HSA API tracing support", false, "roctracer",
                             "rocm");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_ROCTRACER_HSA_API_TYPES",
                             "HSA API type to collect", "", "roctracer", "rocm");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_CRITICAL_TRACE_DEBUG",
                             "Enable debugging for critical trace", _omnitrace_debug,
                             "debugging", "critical-trace");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_CRITICAL_TRACE_SERIALIZE_NAMES",
        "Include names in serialization of critical trace (mainly for debugging)",
        _omnitrace_debug, "debugging", "critical-trace");

    OMNITRACE_CONFIG_SETTING(size_t, "OMNITRACE_SHMEM_SIZE_HINT_KB",
                             "Hint for shared-memory buffer size in perfetto (in KB)",
                             40960, "perfetto", "data");

    OMNITRACE_CONFIG_SETTING(size_t, "OMNITRACE_BUFFER_SIZE_KB",
                             "Size of perfetto buffer (in KB)", 1024000, "perfetto",
                             "data");

    OMNITRACE_CONFIG_SETTING(int64_t, "OMNITRACE_CRITICAL_TRACE_COUNT",
                             "Number of critical trace to export (0 == all)", 0, "data",
                             "critical-trace");

    OMNITRACE_CONFIG_SETTING(uint64_t, "OMNITRACE_CRITICAL_TRACE_BUFFER_COUNT",
                             "Number of critical trace records to store in thread-local "
                             "memory before submitting to shared buffer",
                             2000, "data", "critical-trace");

    OMNITRACE_CONFIG_SETTING(
        uint64_t, "OMNITRACE_CRITICAL_TRACE_NUM_THREADS",
        "Number of threads to use when generating the critical trace",
        std::min<uint64_t>(8, std::thread::hardware_concurrency()), "parallelism",
        "critical-trace");

    OMNITRACE_CONFIG_SETTING(
        int64_t, "OMNITRACE_CRITICAL_TRACE_PER_ROW",
        "How many critical traces per row in perfetto (0 == all in one row)", 0, "io",
        "critical-trace");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_TIMEMORY_COMPONENTS",
        "List of components to collect via timemory (see timemory-avail)", "wall_clock",
        "timemory", "component", "instrumentation");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_OUTPUT_FILE", "Perfetto filename",
                             "", "perfetto", "io");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_SETTINGS_DESC",
                             "Provide descriptions when printing settings", false,
                             "debugging");

    _config->get_flamegraph_output()     = false;
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
    _config->get_papi_events()           = "PAPI_TOT_CYC, PAPI_TOT_INS";

    // settings native to timemory but critically and/or extensively used by omnitrace
    auto _add_omnitrace_category = [](auto itr) {
        if(itr != _config->end())
        {
            auto _categories = itr->second->get_categories();
            _categories.emplace("omnitrace");
            _categories.emplace("omnitrace-library");
            itr->second->set_categories(_categories);
        }
    };

    _add_omnitrace_category(_config->find("OMNITRACE_CONFIG_FILE"));
    _add_omnitrace_category(_config->find("OMNITRACE_DEBUG"));
    _add_omnitrace_category(_config->find("OMNITRACE_VERBOSE"));
    _add_omnitrace_category(_config->find("OMNITRACE_TIME_OUTPUT"));
    _add_omnitrace_category(_config->find("OMNITRACE_OUTPUT_PREFIX"));
    _add_omnitrace_category(_config->find("OMNITRACE_OUTPUT_PATH"));

#if defined(TIMEMORY_USE_PAPI)
    int _paranoid = 2;
    {
        std::ifstream _fparanoid{ "/proc/sys/kernel/perf_event_paranoid" };
        if(_fparanoid) _fparanoid >> _paranoid;
    }

    if(_paranoid > 1)
    {
        OMNITRACE_CONDITIONAL_BASIC_PRINT(
            get_verbose_env() >= 0,
            "/proc/sys/kernel/perf_event_paranoid has a value of %i. "
            "Disabling PAPI (requires a value <= 1)...\n",
            _paranoid);
        OMNITRACE_CONDITIONAL_BASIC_PRINT(
            get_verbose_env() >= 0,
            "In order to enable PAPI support, run 'echo N | sudo tee "
            "/proc/sys/kernel/perf_event_paranoid' where N is < 2\n");
        tim::trait::runtime_enabled<comp::papi_common>::set(false);
        tim::trait::runtime_enabled<comp::papi_array_t>::set(false);
        tim::trait::runtime_enabled<comp::papi_vector>::set(false);
        tim::trait::runtime_enabled<comp::cpu_roofline_flops>::set(false);
        tim::trait::runtime_enabled<comp::cpu_roofline_dp_flops>::set(false);
        tim::trait::runtime_enabled<comp::cpu_roofline_sp_flops>::set(false);
        _config->get_papi_events() = "";
    }
    else
    {
        _add_omnitrace_category(_config->find("OMNITRACE_PAPI_EVENTS"));
    }
#else
    _config->get_papi_quiet() = true;
#endif

    for(auto&& itr :
        tim::delimit(_config->get<std::string>("OMNITRACE_CONFIG_FILE"), ";:"))
    {
        OMNITRACE_CONDITIONAL_BASIC_PRINT(get_verbose_env() > 0,
                                          "Reading config file %s\n", itr.c_str());
        _config->read(itr);
    }

    _config->get_global_components() =
        _config->get<std::string>("OMNITRACE_TIMEMORY_COMPONENTS");

    // always initialize timemory because gotcha wrappers are always used
    auto _cmd     = tim::read_command_line(process::get_id());
    auto _cmd_env = tim::get_env<std::string>("OMNITRACE_COMMAND_LINE", "");
    if(!_cmd_env.empty()) _cmd = tim::delimit(_cmd_env, " ");
    auto _exe = (_cmd.empty()) ? "exe" : _cmd.front();
    auto _pos = _exe.find_last_of('/');
    if(_pos < _exe.length() - 1) _exe = _exe.substr(_pos + 1);
    get_exe_name() = _exe;

    scope::get_fields()[scope::flat::value]     = tim::settings::flat_profile();
    scope::get_fields()[scope::timeline::value] = tim::settings::timeline_profile();

    bool _found_sep = false;
    for(const auto& itr : _cmd)
    {
        if(itr == "--") _found_sep = true;
    }
    if(!_found_sep && _cmd.size() > 1) _cmd.insert(_cmd.begin() + 1, "--");

    if(_init)
    {
        using argparser_t = tim::argparse::argument_parser;
        argparser_t _parser{ _exe };
        tim::timemory_init(_cmd, _parser, "omnitrace-");
    }

    settings::suppress_parsing()  = true;
    settings::suppress_config()   = true;
    settings::use_output_suffix() = _config->get<bool>("OMNITRACE_USE_PID");
#if !defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_MPI_HEADERS)
    if(tim::dmp::is_initialized()) settings::default_process_suffix() = tim::dmp::rank();
#endif
    OMNITRACE_CONDITIONAL_BASIC_PRINT(get_verbose_env() > 0, "configuration complete\n");

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
            tim::sampling::block_signals({ SIGPROF, SIGALRM },
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
                OMNITRACE_CONDITIONAL_BASIC_PRINT(
                    get_verbose_env() > 0 || get_debug_env(),
                    "signal %s (%i) ignored (OMNITRACE_IGNORE_DYNINST_TRAMPOLINE=ON)",
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

    bool _md = tim::get_env<bool>("OMNITRACE_SETTINGS_DESC_MARKDOWN", false);

    constexpr size_t nfields = 3;
    using str_array_t        = std::array<std::string, nfields>;
    std::vector<str_array_t>    _data{};
    std::array<size_t, nfields> _widths{};
    _widths.fill(0);
    for(const auto& itr : *get_config())
    {
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

    bool _print_desc = get_debug() || get_config()->get<bool>("OMNITRACE_SETTINGS_DESC");

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
print_settings()
{
    if(dmp::rank() > 0) return;

    static std::set<tim::string_view_t> _sample_options = {
        "OMNITRACE_SAMPLING_FREQ",     "OMNITRACE_SAMPLING_DELAY",
        "OMNITRACE_SAMPLING_CPUS",     "OMNITRACE_FLAT_SAMPLING",
        "OMNITRACE_TIMELINE_SAMPLING", "OMNITRACE_FLAT_SAMPLING",
        "OMNITRACE_TIMELINE_SAMPLING",
    };
    static std::set<tim::string_view_t> _perfetto_options = {
        "OMNITRACE_OUTPUT_FILE",
        "OMNITRACE_BACKEND",
        "OMNITRACE_SHMEM_SIZE_HINT_KB",
        "OMNITRACE_BUFFER_SIZE_KB",
    };
    static std::set<tim::string_view_t> _timemory_options = {
        "OMNITRACE_ROCTRACER_FLAT_PROFILE", "OMNITRACE_ROCTRACER_TIMELINE_PROFILE"
    };
    // generic filter for filtering relevant options
    auto _is_omnitrace_option = [](const auto& _v, const auto& _c) {
        if(!get_use_roctracer() && _v.find("OMNITRACE_ROCTRACER_") == 0) return false;
        if(!get_use_critical_trace() && _v.find("OMNITRACE_CRITICAL_TRACE_") == 0)
            return false;
        if(!get_use_perfetto() && _perfetto_options.count(_v) > 0) return false;
        if(!get_use_timemory() && _timemory_options.count(_v) > 0) return false;
        if(!get_use_sampling() && _sample_options.count(_v) > 0) return false;
        const auto npos = std::string::npos;
        if(_v.find("WIDTH") != npos || _v.find("SEPARATOR_FREQ") != npos ||
           _v.find("AUTO_OUTPUT") != npos || _v.find("DART_OUTPUT") != npos ||
           _v.find("FILE_OUTPUT") != npos || _v.find("PLOT_OUTPUT") != npos ||
           _v.find("FLAMEGRAPH_OUTPUT") != npos)
            return false;
        if(!_c.empty())
        {
            if(_c.find("omnitrace") != _c.end()) return true;
            if(_c.find("debugging") != _c.end() && _v.find("DEBUG") != npos) return true;
            if(_c.find("config") != _c.end()) return true;
            if(_c.find("dart") != _c.end()) return false;
            if(_c.find("io") != _c.end() && _v.find("_OUTPUT") != npos) return true;
            if(_c.find("format") != _c.end()) return true;
            return false;
        }
        return (_v.find("OMNITRACE_") == 0);
    };

    tim::print_env(std::cerr, [_is_omnitrace_option](const std::string& _v) {
        return _is_omnitrace_option(_v, std::set<std::string>{});
    });

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
    static auto _v = []() {
        auto _mode = tim::get_env_choice<std::string>("OMNITRACE_MODE", "trace",
                                                      { "trace", "sampling" });
        if(_mode == "sampling") return Mode::Sampling;
        return Mode::Trace;
    }();
    return _v;
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
    return tim::get_env<bool>("OMNITRACE_DEBUG", false);
}

bool
get_is_continuous_integration()
{
    return tim::get_env<bool>("OMNITRACE_CI", false);
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
    static bool _v = tim::get_env<bool>("OMNITRACE_DEBUG_SAMPLING", get_debug_env());
    return (_v || get_debug());
}

int
get_verbose_env()
{
    return tim::get_env<int>("OMNITRACE_VERBOSE", 0);
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

bool&
get_use_roctracer()
{
#if defined(OMNITRACE_USE_ROCTRACER)
    static auto _v = get_config()->find("OMNITRACE_USE_ROCTRACER");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    static auto _v            = false;
    return _v;
#endif
}

bool&
get_use_rocm_smi()
{
#if defined(OMNITRACE_USE_ROCM_SMI)
    static auto _v = get_config()->find("OMNITRACE_USE_ROCM_SMI");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
#else
    static auto _v = false;
    return _v;
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
    static auto _v = get_config()->find("OMNITRACE_SHMEM_SIZE_HINT_KB");
    return static_cast<tim::tsettings<size_t>&>(*_v->second).get();
}

size_t
get_perfetto_buffer_size()
{
    static auto _v = get_config()->find("OMNITRACE_BUFFER_SIZE_KB");
    return static_cast<tim::tsettings<size_t>&>(*_v->second).get();
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
    static auto _v = get_config()->find("OMNITRACE_BACKEND");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
}

std::string&
get_perfetto_output_filename()
{
    static auto  _v        = get_config()->find("OMNITRACE_OUTPUT_FILE");
    static auto& _t        = static_cast<tim::tsettings<std::string>&>(*_v->second);
    static auto  _generate = []() {
        if(tim::dmp::is_initialized())
            settings::default_process_suffix() = tim::dmp::rank();
        // default name: perfetto-trace.<pid>.proto or perfetto-trace.<rank>.proto
        auto _default_fname =
            settings::compose_output_filename("perfetto-trace", "proto", get_use_pid());
        auto _pid_patch = std::string{ "/" } + std::to_string(tim::process::get_id()) +
                          "-perfetto-trace";
        auto _dpos = _default_fname.find(_pid_patch);
        if(_dpos != std::string::npos)
            _default_fname =
                _default_fname.replace(_dpos, _pid_patch.length(), "/perfetto-trace");
        // have the default display the full path to the output file
        return tim::get_env<std::string>(
            "OMNITRACE_OUTPUT_FILE",
            JOIN('/', tim::get_env<std::string>("PWD", ".", false), _default_fname),
            false);
    };
    static auto _generated = _generate();
    if(_t.get().empty() || _t.get() == _generated)
    {
        _t.set(_generate());
        _generated = _t.get();
    }
    return _t.get();
}

size_t&
get_instrumentation_interval()
{
    static auto _v = get_config()->find("OMNITRACE_INSTRUMENTATION_INTERVAL");
    return static_cast<tim::tsettings<size_t>&>(*_v->second).get();
}

double&
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

double&
get_thread_sampling_freq()
{
    static auto _v = std::min<double>(get_sampling_freq(), 1000.0);
    return _v;
}

std::string
get_rocm_smi_devices()
{
#if defined(OMNITRACE_USE_ROCM_SMI)
    static auto _v = get_config()->find("OMNITRACE_ROCM_SMI_DEVICES");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
#else
    return std::string{};
#endif
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
