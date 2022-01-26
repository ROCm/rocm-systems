// Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"
#include "library/thread_data.hpp"

#include <timemory/backends/dmp.hpp>
#include <timemory/backends/mpi.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/environment.hpp>
#include <timemory/settings.hpp>
#include <timemory/settings/types.hpp>
#include <timemory/utility/argparse.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <ostream>
#include <string>

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

#define OMNITRACE_CONFIG_SETTING(TYPE, ENV_NAME, DESCRIPTION, INITIAL_VALUE, ...)        \
    _config->insert<TYPE, TYPE>(                                                         \
        ENV_NAME, ENV_NAME, DESCRIPTION, INITIAL_VALUE,                                  \
        std::set<std::string>{ "custom", "omnitrace", __VA_ARGS__ })
}  // namespace

void
configure_settings()
{
    static bool _once = false;
    if(_once) return;
    _once = true;

    static auto _config = settings::shared_instance();
    // auto* _config = settings::instance();

    // if using timemory, default to perfetto being off
    auto _default_perfetto_v =
        !tim::get_env<bool>("OMNITRACE_USE_TIMEMORY", false, false);

    auto _default_config_file =
        JOIN("/", tim::get_env<std::string>("HOME", "."), "omnitrace.cfg");

    auto _system_backend = tim::get_env("OMNITRACE_BACKEND_SYSTEM", false, false);

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_CONFIG_FILE",
                             "Configuration file of omnitrace and timemory settings",
                             _default_config_file, "config");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_DEBUG", "Enable debugging output",
                             _config->get_debug(), "debugging");

    auto _omnitrace_debug = _config->get<bool>("OMNITRACE_DEBUG");
    if(_omnitrace_debug) tim::set_env("TIMEMORY_DEBUG_SETTINGS", "1", 0);

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_PERFETTO", "Enable perfetto backend",
                             _default_perfetto_v, "backend", "perfetto");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_TIMEMORY", "Enable timemory backend",
                             !_config->get<bool>("OMNITRACE_USE_PERFETTO"), "backend",
                             "timemory");

#if defined(OMNITRACE_USE_ROCTRACER)
    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_ROCTRACER", "Enable ROCM tracing", true,
                             "backend", "roctracer");
#endif

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_USE_SAMPLING",
                             "Enable statistical sampling of call-stack", false,
                             "backend", "sampling");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_USE_PID",
        "Enable tagging filenames with process identifier (either MPI rank or pid)", true,
        "io");

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
        "Number of seconds to delay activating the statistical sampling", 0.05,
        "sampling");

    auto _backend = tim::get_env_choice<std::string>(
        "OMNITRACE_BACKEND",
        (_system_backend)
            ? "system"      // if OMNITRACE_BACKEND_SYSTEM is true, default to system.
            : "inprocess",  // Otherwise, default to inprocess
        { "inprocess", "system", "all" }, false);

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_BACKEND",
                             "Specify the perfetto backend to activate. Options are: "
                             "'inprocess', 'system', or 'all'",
                             _backend, "perfetto");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_CRITICAL_TRACE",
                             "Enable generation of the critical trace", false, "feature");

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
        _config->get_flat_profile(), "roctracer", "data_layout");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_ROCTRACER_TIMELINE_PROFILE",
        "Create unique entries for every kernel with timemory backend",
        _config->get_timeline_profile(), "roctracer", "data_layout");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_ROCTRACER_HSA_ACTIVITY",
                             "Enable HSA activity tracing support", false, "roctracer");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_ROCTRACER_HSA_API",
                             "Enable HSA API tracing support", false, "roctracer");

    OMNITRACE_CONFIG_SETTING(std::string, "OMNITRACE_ROCTRACER_HSA_API_TYPES",
                             "HSA API type to collect", "", "roctracer");

    OMNITRACE_CONFIG_SETTING(bool, "OMNITRACE_CRITICAL_TRACE_DEBUG",
                             "Enable debugging for critical trace", _omnitrace_debug,
                             "debugging");

    OMNITRACE_CONFIG_SETTING(
        bool, "OMNITRACE_CRITICAL_TRACE_SERIALIZE_NAMES",
        "Include names in serialization of critical trace (mainly for debugging)",
        _omnitrace_debug, "debugging");

    OMNITRACE_CONFIG_SETTING(size_t, "OMNITRACE_SHMEM_SIZE_HINT_KB",
                             "Hint for shared-memory buffer size in perfetto (in KB)",
                             40960, "perfetto", "data");

    OMNITRACE_CONFIG_SETTING(size_t, "OMNITRACE_BUFFER_SIZE_KB",
                             "Size of perfetto buffer (in KB)", 1024000, "perfetto",
                             "data");

    OMNITRACE_CONFIG_SETTING(int64_t, "OMNITRACE_CRITICAL_TRACE_COUNT",
                             "Number of critical trace to export (0 == all)", 0, "data");

    OMNITRACE_CONFIG_SETTING(uint64_t, "OMNITRACE_CRITICAL_TRACE_BUFFER_COUNT",
                             "Number of critical trace records to store in thread-local "
                             "memory before submitting to shared buffer",
                             2000, "data");

    OMNITRACE_CONFIG_SETTING(
        uint64_t, "OMNITRACE_CRITICAL_TRACE_NUM_THREADS",
        "Number of threads to use when generating the critical trace",
        std::min<uint64_t>(8, std::thread::hardware_concurrency()), "parallelism");

    OMNITRACE_CONFIG_SETTING(
        int64_t, "OMNITRACE_CRITICAL_TRACE_PER_ROW",
        "How many critical traces per row in perfetto (0 == all in one row)", 0, "io");

    OMNITRACE_CONFIG_SETTING(
        std::string, "OMNITRACE_TIMEMORY_COMPONENTS",
        "List of components to collect via timemory (see timemory-avail)", "wall_clock",
        "timemory", "component");

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

    for(auto&& itr :
        tim::delimit(_config->get<std::string>("OMNITRACE_CONFIG_FILE"), ";:"))
    {
        OMNITRACE_CONDITIONAL_BASIC_PRINT(true, "Reading config file %s\n", itr.c_str());
        _config->read(itr);
    }

    _config->get_global_components() =
        _config->get<std::string>("OMNITRACE_TIMEMORY_COMPONENTS");

    // always initialize timemory because gotcha wrappers are always used
    auto _cmd = tim::read_command_line(process::get_id());
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

    using argparser_t = tim::argparse::argument_parser;
    argparser_t _parser{ _exe };
    tim::timemory_init(_cmd, _parser, "omnitrace-");

    settings::suppress_parsing()  = true;
    settings::suppress_config()   = true;
    settings::use_output_suffix() = _config->get<bool>("OMNITRACE_USE_PID");
#if !defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_MPI_HEADERS)
    if(tim::mpi::is_initialized()) settings::default_process_suffix() = tim::mpi::rank();
#endif
    OMNITRACE_CONDITIONAL_BASIC_PRINT(true, "configuration complete\n");
}

void
print_config_settings(
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

bool
get_debug_env()
{
    return tim::get_env<bool>("OMNITRACE_DEBUG", false);
}

bool
get_debug()
{
    static auto _v = get_config()->find("OMNITRACE_DEBUG");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
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
    static bool _v = false;
    if(_v)
        throw std::runtime_error("Error! sampling was enabled but omnitrace was not "
                                 "built with libunwind support");
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
    static auto  _v = get_config()->find("OMNITRACE_OUTPUT_FILE");
    static auto& _t = static_cast<tim::tsettings<std::string>&>(*_v->second);
    if(_t.get().empty())
    {
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
        _t.set(tim::get_env<std::string>(
            "OMNITRACE_OUTPUT_FILE",
            JOIN('/', tim::get_env<std::string>("PWD", ".", false), _default_fname),
            false));
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

int64_t
get_critical_trace_count()
{
    static auto _v = get_config()->find("OMNITRACE_CRITICAL_TRACE_COUNT");
    return static_cast<tim::tsettings<int64_t>&>(*_v->second).get();
}

State&
get_state()
{
    static State _v{ State::PreInit };
    return _v;
}

std::atomic<uint64_t>&
get_cpu_cid()
{
    static std::atomic<uint64_t> _v{ 0 };
    return _v;
}

std::unique_ptr<std::vector<uint64_t>>&
get_cpu_cid_stack(int64_t _tid)
{
    struct omnitrace_cpu_cid_stack
    {};
    using thread_data_t =
        omnitrace_thread_data<std::vector<uint64_t>, omnitrace_cpu_cid_stack>;
    static auto&             _v       = thread_data_t::instances();
    static thread_local auto _v_check = [_tid]() {
        thread_data_t::construct((_tid > 0) ? *thread_data_t::instances().at(0)
                                            : std::vector<uint64_t>{});
        return true;
    }();
    return _v.at(_tid);
    (void) _v_check;
}
}  // namespace omnitrace
