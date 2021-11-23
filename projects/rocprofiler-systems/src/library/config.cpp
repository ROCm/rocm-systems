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
#include "library/thread_data.hpp"
#include "timemory/backends/dmp.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/settings/types.hpp"
#include "timemory/utility/argparse.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <ostream>
#include <string>
#include <timemory/environment.hpp>
#include <timemory/settings.hpp>

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

#define HOSTTRACE_CONFIG_SETTING(TYPE, ENV_NAME, DESCRIPTION, INITIAL_VALUE)             \
    _config->insert<TYPE, TYPE>(ENV_NAME, ENV_NAME, DESCRIPTION, INITIAL_VALUE,          \
                                std::vector<std::string>{})
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
        !tim::get_env<bool>("HOSTTRACE_USE_TIMEMORY", false, false);

    auto _default_config_file =
        JOIN("/", tim::get_env<std::string>("HOME", "."), "hosttrace.cfg");

    auto _system_backend = tim::get_env("HOSTTRACE_BACKEND_SYSTEM", false, false);

    HOSTTRACE_CONFIG_SETTING(std::string, "HOSTTRACE_CONFIG_FILE",
                             "Configuration file of hosttrace and timemory settings",
                             _default_config_file);

    HOSTTRACE_CONFIG_SETTING(bool, "HOSTTRACE_DEBUG", "Enable debugging output",
                             _config->get_debug());

    auto _hosttrace_debug = _config->get<bool>("HOSTTRACE_DEBUG");
    if(_hosttrace_debug) tim::set_env("TIMEMORY_DEBUG_SETTINGS", "1", 0);

    HOSTTRACE_CONFIG_SETTING(bool, "HOSTTRACE_USE_PERFETTO", "Enable perfetto backend",
                             _default_perfetto_v);

    HOSTTRACE_CONFIG_SETTING(bool, "HOSTTRACE_USE_TIMEMORY", "Enable timemory backend",
                             !_config->get<bool>("HOSTTRACE_USE_PERFETTO"));

    HOSTTRACE_CONFIG_SETTING(
        bool, "HOSTTRACE_USE_PID",
        "Enable tagging filenames with process identifier (either MPI rank or pid)",
        true);

    HOSTTRACE_CONFIG_SETTING(
        size_t, "HOSTTRACE_SAMPLE_RATE",
        "Counts every function call (N), only record function if (N % <VALUE> == 0)", 1);

    auto _backend = tim::get_env_choice<std::string>(
        "HOSTTRACE_BACKEND",
        (_system_backend)
            ? "system"      // if HOSTTRACE_BACKEND_SYSTEM is true, default to system.
            : "inprocess",  // Otherwise, default to inprocess
        { "inprocess", "system", "all" }, false);

    HOSTTRACE_CONFIG_SETTING(std::string, "HOSTTRACE_BACKEND",
                             "Specify the perfetto backend to activate. Options are: "
                             "'inprocess', 'system', or 'all'",
                             _backend);

    HOSTTRACE_CONFIG_SETTING(bool, "HOSTTRACE_CRITICAL_TRACE",
                             "Enable generation of the critical trace", false);

    HOSTTRACE_CONFIG_SETTING(
        bool, "HOSTTRACE_ROCTRACER_TIMELINE_PROFILE",
        "Create unique entries for every kernel with timemory backend",
        _config->get_timeline_profile());

    HOSTTRACE_CONFIG_SETTING(
        bool, "HOSTTRACE_ROCTRACER_FLAT_PROFILE",
        "Ignore hierarchy in all kernels entries with timemory backend",
        _config->get_flat_profile());

    HOSTTRACE_CONFIG_SETTING(bool, "HOSTTRACE_ROCTRACER_HSA_ACTIVITY",
                             "Enable HSA activity tracing support", false);

    HOSTTRACE_CONFIG_SETTING(bool, "HOSTTRACE_ROCTRACER_HSA_API",
                             "Enable HSA API tracing support", false);

    HOSTTRACE_CONFIG_SETTING(std::string, "HOSTTRACE_ROCTRACER_HSA_API_TYPES",
                             "HSA API type to collect", "");

    HOSTTRACE_CONFIG_SETTING(bool, "HOSTTRACE_CRITICAL_TRACE_DEBUG",
                             "Enable debugging for critical trace", _hosttrace_debug);

    HOSTTRACE_CONFIG_SETTING(
        bool, "HOSTTRACE_CRITICAL_TRACE_SERIALIZE_NAMES",
        "Include names in serialization of critical trace (mainly for debugging)",
        _hosttrace_debug);

    HOSTTRACE_CONFIG_SETTING(size_t, "HOSTTRACE_SHMEM_SIZE_HINT_KB",
                             "Hint for shared-memory buffer size in perfetto (in KB)",
                             40960);

    HOSTTRACE_CONFIG_SETTING(size_t, "HOSTTRACE_BUFFER_SIZE_KB",
                             "Size of perfetto buffer (in KB)", 1024000);

    HOSTTRACE_CONFIG_SETTING(int64_t, "HOSTTRACE_CRITICAL_TRACE_COUNT",
                             "Number of critical trace to export (0 == all)", 0);

    HOSTTRACE_CONFIG_SETTING(uint64_t, "HOSTTRACE_CRITICAL_TRACE_BUFFER_COUNT",
                             "Number of critical trace records to store in thread-local "
                             "memory before submitting to shared buffer",
                             2000);

    HOSTTRACE_CONFIG_SETTING(
        uint64_t, "HOSTTRACE_CRITICAL_TRACE_NUM_THREADS",
        "Number of threads to use when generating the critical trace",
        std::min<uint64_t>(8, std::thread::hardware_concurrency()));

    HOSTTRACE_CONFIG_SETTING(
        int64_t, "HOSTTRACE_CRITICAL_TRACE_PER_ROW",
        "How many critical traces per row in perfetto (0 == all in one row)", 0);

    HOSTTRACE_CONFIG_SETTING(
        std::string, "HOSTTRACE_COMPONENTS",
        "List of components to collect via timemory (see timemory-avail)", "wall_clock");

    HOSTTRACE_CONFIG_SETTING(std::string, "HOSTTRACE_OUTPUT_FILE", "Perfetto filename",
                             "");

    HOSTTRACE_CONFIG_SETTING(bool, "HOSTTRACE_SETTINGS_DESC",
                             "Provide descriptions when printing settings", false);

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
        tim::delimit(_config->get<std::string>("HOSTTRACE_CONFIG_FILE"), ";:"))
    {
        HOSTTRACE_CONDITIONAL_BASIC_PRINT(true, "Reading config file %s\n", itr.c_str());
        _config->read(itr);
    }

    _config->get_global_components() = _config->get<std::string>("HOSTTRACE_COMPONENTS");

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
    tim::timemory_init(_cmd, _parser, "hosttrace-");

    settings::suppress_parsing()  = true;
    settings::suppress_config()   = true;
    settings::use_output_suffix() = _config->get<bool>("HOSTTRACE_USE_PID");
}

void
print_config_settings(std::ostream&                                  _os,
                      std::function<bool(const std::string_view&)>&& _filter)
{
    auto _flags = _os.flags();

    constexpr size_t nfields = 3;
    using str_array_t        = std::array<std::string, nfields>;
    std::vector<str_array_t>    _data{};
    std::array<size_t, nfields> _widths{};
    _widths.fill(0);
    for(const auto& itr : *get_config())
    {
        if(_filter(itr.first))
        {
            auto _disp = itr.second->get_display(std::ios::boolalpha);
            _data.emplace_back(str_array_t{ _disp.at("name"), _disp.at("value"),
                                            _disp.at("description") });
            for(size_t i = 0; i < nfields; ++i)
                _widths.at(i) =
                    std::max<size_t>(_widths.at(i), _data.back().at(i).length());
        }
    }

    std::sort(_data.begin(), _data.end(), [](const auto& lhs, const auto& rhs) {
        auto _npos = std::string::npos;
        // HOSTTRACE_CONFIG_FILE always first
        if(lhs.at(0).find("HOSTTRACE_CONFIG") != _npos) return true;
        if(rhs.at(0).find("HOSTTRACE_CONFIG") != _npos) return false;
        // HOSTTRACE_USE_* prioritized
        auto _lhs_use = lhs.at(0).find("HOSTTRACE_USE_");
        auto _rhs_use = rhs.at(0).find("HOSTTRACE_USE_");
        if(_lhs_use != _rhs_use && _lhs_use < _rhs_use) return true;
        if(_lhs_use != _rhs_use && _lhs_use > _rhs_use) return false;
        // length sort followed by alphabetical sort
        return (lhs.at(0).length() == rhs.at(0).length())
                   ? (lhs.at(0) < rhs.at(0))
                   : (lhs.at(0).length() < rhs.at(0).length());
    });

    bool _print_desc = get_debug() || get_config()->get<bool>("HOSTTRACE_SETTINGS_DESC");

    auto tot_width = std::accumulate(_widths.begin(), _widths.end(), 0);
    if(!_print_desc) tot_width -= _widths.back() + 4;

    std::stringstream _spacer{};
    _spacer.fill('-');
    _spacer << "#" << std::setw(tot_width + 11) << ""
            << "#";
    _os << _spacer.str() << "\n";
    // _os << "# Hosttrace settings:" << std::setw(tot_width - 8) << "#" << "\n";
    for(const auto& itr : _data)
    {
        _os << "# ";
        for(size_t i = 0; i < nfields; ++i)
        {
            switch(i)
            {
                case 0: _os << std::left; break;
                case 1: _os << std::left; break;
                case 2: _os << std::left; break;
            }
            _os << std::setw(_widths.at(i)) << itr.at(i) << " ";
            if(!_print_desc && i == 1) break;
            switch(i)
            {
                case 0: _os << "= "; break;
                case 1: _os << "[ "; break;
                case 2: _os << "]"; break;
            }
        }
        _os << "  #\n";
    }
    _os << _spacer.str() << "\n";

    _os.setf(_flags);
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
    static auto _v = get_config()->find("HOSTTRACE_CONFIG_FILE");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
}

bool
get_debug()
{
    static auto _v = get_config()->find("HOSTTRACE_DEBUG");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_perfetto()
{
    static auto _v = get_config()->find("HOSTTRACE_USE_PERFETTO");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_timemory()
{
    static auto _v = get_config()->find("HOSTTRACE_USE_TIMEMORY");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool&
get_use_pid()
{
    static auto _v = get_config()->find("HOSTTRACE_USE_PID");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_use_mpip()
{
    static bool _v = tim::get_env("HOSTTRACE_USE_MPIP", false, false);
    return _v;
}

bool
get_use_critical_trace()
{
    static auto _v = get_config()->find("HOSTTRACE_CRITICAL_TRACE");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_critical_trace_debug()
{
    static auto _v = get_config()->find("HOSTTRACE_CRITICAL_TRACE_DEBUG");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_critical_trace_serialize_names()
{
    static auto _v = get_config()->find("HOSTTRACE_CRITICAL_TRACE_SERIALIZE_NAMES");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_roctracer_timeline_profile()
{
    static auto _v = get_config()->find("HOSTTRACE_ROCTRACER_TIMELINE_PROFILE");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_roctracer_flat_profile()
{
    static auto _v = get_config()->find("HOSTTRACE_ROCTRACER_FLAT_PROFILE");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_trace_hsa_api()
{
    static auto _v = get_config()->find("HOSTTRACE_ROCTRACER_HSA_API");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

bool
get_trace_hsa_activity()
{
    static auto _v = get_config()->find("HOSTTRACE_ROCTRACER_HSA_ACTIVITY");
    return static_cast<tim::tsettings<bool>&>(*_v->second).get();
}

int64_t
get_critical_trace_per_row()
{
    static auto _v = get_config()->find("HOSTTRACE_CRITICAL_TRACE_PER_ROW");
    return static_cast<tim::tsettings<int64_t>&>(*_v->second).get();
}

size_t
get_perfetto_shmem_size_hint()
{
    static auto _v = get_config()->find("HOSTTRACE_SHMEM_SIZE_HINT_KB");
    return static_cast<tim::tsettings<size_t>&>(*_v->second).get();
}

size_t
get_perfetto_buffer_size()
{
    static auto _v = get_config()->find("HOSTTRACE_BUFFER_SIZE_KB");
    return static_cast<tim::tsettings<size_t>&>(*_v->second).get();
}

uint64_t
get_critical_trace_update_freq()
{
    static uint64_t _v =
        get_config()->get<uint64_t>("HOSTTRACE_CRITICAL_TRACE_BUFFER_COUNT");
    return _v;
}

uint64_t
get_critical_trace_num_threads()
{
    static uint64_t _v =
        get_config()->get<uint64_t>("HOSTTRACE_CRITICAL_TRACE_NUM_THREADS");
    return _v;
}

std::string
get_trace_hsa_api_types()
{
    static std::string _v =
        get_config()->get<std::string>("HOSTTRACE_ROCTRACER_HSA_API_TYPES");
    return _v;
}

std::string&
get_backend()
{
    // select inprocess, system, or both (i.e. all)
    static auto _v = get_config()->find("HOSTTRACE_BACKEND");
    return static_cast<tim::tsettings<std::string>&>(*_v->second).get();
}

std::string
get_perfetto_output_filename()
{
    static auto  _v = get_config()->find("HOSTTRACE_OUTPUT_FILE");
    static auto& _t = static_cast<tim::tsettings<std::string>&>(*_v->second);
    if(_t.get().empty())
    {
        // default name: perfetto-trace.<pid>.proto or perfetto-trace.<rank>.proto
        auto _default_fname = settings::compose_output_filename(
            "perfetto-trace", "proto", get_use_pid(),
            (tim::dmp::is_initialized()) ? tim::dmp::rank() : process::get_id());
        auto _pid_patch = std::string{ "/" } + std::to_string(tim::process::get_id()) +
                          "-perfetto-trace";
        auto _dpos = _default_fname.find(_pid_patch);
        if(_dpos != std::string::npos)
            _default_fname =
                _default_fname.replace(_dpos, _pid_patch.length(), "/perfetto-trace");
        // have the default display the full path to the output file
        _t.set(tim::get_env<std::string>(
            "HOSTTRACE_OUTPUT_FILE",
            JOIN('/', tim::get_env<std::string>("PWD", ".", false), _default_fname),
            false));
    }
    return _t.get();
}

size_t&
get_sample_rate()
{
    static auto _v = get_config()->find("HOSTTRACE_SAMPLE_RATE");
    return static_cast<tim::tsettings<size_t>&>(*_v->second).get();
}

int64_t
get_critical_trace_count()
{
    static auto _v = get_config()->find("HOSTTRACE_CRITICAL_TRACE_COUNT");
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
    struct hosttrace_cpu_cid_stack
    {};
    using thread_data_t =
        hosttrace_thread_data<std::vector<uint64_t>, hosttrace_cpu_cid_stack>;
    static auto&             _v       = thread_data_t::instances();
    static thread_local auto _v_check = [_tid]() {
        thread_data_t::construct((_tid > 0) ? *thread_data_t::instances().at(0)
                                            : std::vector<uint64_t>{});
        return true;
    }();
    return _v.at(_tid);
    (void) _v_check;
}
