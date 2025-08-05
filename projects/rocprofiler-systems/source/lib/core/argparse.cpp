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

#include "argparse.hpp"
#include "common/join.hpp"
#include "config.hpp"
#include "defines.hpp"
#include "exception.hpp"
#include "gpu.hpp"
#include "state.hpp"

#include <timemory/settings/types.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/join.hpp>

namespace rocprofsys
{
namespace argparse
{
namespace
{
namespace filepath   = ::tim::filepath;
using array_config_t = ::timemory::join::array_config;
using ::tim::get_env;
using ::timemory::join::join;

auto
get_clock_id_choices()
{
    auto clock_name = [](std::string _v) {
        constexpr auto _clock_prefix = std::string_view{ "clock_" };
        for(auto& itr : _v)
            itr = tolower(itr);
        auto _pos = _v.find(_clock_prefix);
        if(_pos == 0) _v = _v.substr(_pos + _clock_prefix.length());
        if(_v == "process_cputime_id") _v = "cputime";
        return _v;
    };

#define ROCPROFSYS_CLOCK_IDENTIFIER(VAL)                                                 \
    std::make_tuple(clock_name(#VAL), VAL, std::string_view{ #VAL })

    auto _choices = strvec_t{};
    auto _aliases = std::map<std::string, strvec_t>{};
    for(auto itr : { ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_REALTIME),
                     ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_MONOTONIC),
                     ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_PROCESS_CPUTIME_ID),
                     ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_MONOTONIC_RAW),
                     ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_REALTIME_COARSE),
                     ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_MONOTONIC_COARSE),
                     ROCPROFSYS_CLOCK_IDENTIFIER(CLOCK_BOOTTIME) })
    {
        auto _choice = std::to_string(std::get<1>(itr));
        _choices.emplace_back(_choice);
        _aliases[_choice] = { std::get<0>(itr), std::string{ std::get<2>(itr) } };
    }

#undef ROCPROFSYS_CLOCK_IDENTIFIER

    return std::make_pair(_choices, _aliases);
}

auto
get_realpath(const std::string& _path)
{
    return filepath::realpath(_path, nullptr, false);
}

enum update_mode : int
{
    UPD_REPLACE = 0x1,
    UPD_PREPEND = 0x2,
    UPD_APPEND  = 0x3,
    UPD_WEAK    = 0x4,
};

template <typename Tp>
void
update_env(parser_data& _data, std::string_view _env_var, Tp&& _env_val,
           update_mode&& _mode = UPD_REPLACE, std::string_view _join_delim = ":")
{
    _data.updated.emplace(_env_var);

    auto _prepend  = (_mode & UPD_PREPEND) == UPD_PREPEND;
    auto _append   = (_mode & UPD_APPEND) == UPD_APPEND;
    auto _weak_upd = (_mode & UPD_WEAK) == UPD_WEAK;

    auto _key = join("", _env_var, "=");
    for(auto& itr : _data.current)
    {
        if(!itr) continue;
        if(std::string_view{ itr }.find(_key) == 0)
        {
            if(_weak_upd)
            {
                // if the value has changed, do not update but allow overridding the value
                // inherited from the initial env
                if(_data.initial.find(std::string{ itr }) == _data.initial.end()) return;
            }

            if(_prepend || _append)
            {
                if(std::string_view{ itr }.find(join("", _env_val)) ==
                   std::string_view::npos)
                {
                    auto _val = std::string{ itr }.substr(_key.length());
                    free(itr);
                    if(_prepend)
                        itr =
                            strdup(join('=', _env_var, join(_join_delim, _val, _env_val))
                                       .c_str());
                    else
                        itr =
                            strdup(join('=', _env_var, join(_join_delim, _env_val, _val))
                                       .c_str());
                }
            }
            else
            {
                free(itr);
                itr = strdup(rocprofsys::common::join('=', _env_var, _env_val).c_str());
            }
            return;
        }
    }
    _data.current.emplace_back(
        strdup(rocprofsys::common::join('=', _env_var, _env_val).c_str()));
}

void
remove_env(parser_data& _data, std::string_view _env_var)
{
    auto _key   = join("", _env_var, "=");
    auto _match = [&_key](auto itr) { return std::string_view{ itr }.find(_key) == 0; };

    auto& _environ = _data.current;
    _environ.erase(std::remove_if(_environ.begin(), _environ.end(), _match),
                   _environ.end());

    auto& _initial = _data.initial;
    for(const auto& itr : _initial)
    {
        if(std::string_view{ itr }.find(_key) == 0)
            _environ.emplace_back(strdup(itr.c_str()));
    }
}

std::string
get_internal_libpath(const std::string& _lib)
{
    auto _exe = filepath::realpath("/proc/self/exe", nullptr, false);
    auto _pos = _exe.find_last_of('/');
    auto _dir = filepath::get_cwd();
    if(_pos != std::string_view::npos) _dir = _exe.substr(0, _pos);
    return filepath::realpath(rocprofsys::common::join("/", _dir, "..", "lib", _lib),
                              nullptr, false);
}
std::string
get_internal_script_path(void)
{
    auto _exe = std::string_view{ realpath("/proc/self/exe", nullptr) };
    auto _pos = _exe.find_last_of('/');
    auto _dir = std::string{ "./" };
    if(_pos != std::string_view::npos) _dir = _exe.substr(0, _pos);

    auto _script_dir =
        rocprofsys::common::join("/", _dir, "..", "libexec", "rocprofiler-systems");

    return _script_dir;
}

}  // namespace

bool
default_setting_filter(vsetting_t* _v, const parser_data& _data)
{
    return (_data.processed_settings.count(_v) == 0 &&
            _data.processed_environs.count(_v->get_name()) == 0 &&
            _data.processed_environs.count(_v->get_env_name()) == 0);
}

bool
default_environ_filter(std::string_view _v, const parser_data& _data)
{
    return (_data.processed_environs.count(_v.data()) == 0);
}

bool
default_grouping_filter(std::string_view _v, const parser_data& _data)
{
    return (_data.processed_groups.count(_v.data()) == 0);
}

parser_data&
init_parser(parser_data& _data)
{
    tim::settings::suppress_config()  = true;
    tim::settings::suppress_parsing() = true;

    set_state(State::Init);
    config::configure_settings(false);

    auto& _current = _data.current;
    auto& _initial = _data.initial;

    if(environ != nullptr)
    {
        int idx = 0;
        while(environ[idx] != nullptr)
        {
            auto* _v = environ[idx++];
            _initial.emplace(_v);
            _current.emplace_back(strdup(_v));
        }
    }

    _data.dl_libpath = get_realpath(get_internal_libpath("librocprof-sys-dl.so").c_str());
    _data.omni_libpath = get_realpath(get_internal_libpath("librocprof-sys.so").c_str());

    auto _libexecpath = get_realpath(get_internal_script_path());
    update_env(_data, "ROCPROFSYS_SCRIPT_PATH", _libexecpath, UPD_REPLACE);

#if defined(ROCPROFSYS_USE_OMPT)
    if(!getenv("OMP_TOOL_LIBRARIES"))
        update_env(_data, "OMP_TOOL_LIBRARIES", _data.dl_libpath, UPD_PREPEND);
#endif

    return _data;
}

parser_data&
add_ld_preload(parser_data& _data)
{
    update_env(_data, "LD_PRELOAD", _data.dl_libpath, UPD_APPEND);
    return _data;
}

parser_data&
add_ld_library_path(parser_data& _data)
{
    auto _libdir = filepath::dirname(_data.dl_libpath);
    if(filepath::exists(_libdir))
        update_env(_data, "LD_LIBRARY_PATH", _libdir, UPD_APPEND);
    return _data;
}

parser_data&
add_core_arguments(parser_t& _parser, parser_data& _data)
{
    const auto* _cputime_desc =
        R"(Sample based on a CPU-clock timer (default). Accepts zero or more arguments:
    %{INDENT}%0. Enables sampling based on CPU-clock timer.
    %{INDENT}%1. Interrupts per second. E.g., 100 == sample every 10 milliseconds of CPU-time.
    %{INDENT}%2. Delay (in seconds of CPU-clock time). I.e., how long each thread should wait before taking first sample.
    %{INDENT}%3+ Thread IDs to target for sampling, starting at 0 (the main thread).
    %{INDENT}%   May be specified as index or range, e.g., '0 2-4' will be interpreted as:
    %{INDENT}%      sample the main thread (0), do not sample the first child thread but sample the 2nd, 3rd, and 4th child threads)";

    const auto* _realtime_desc =
        R"(Sample based on a real-clock timer. Accepts zero or more arguments:
    %{INDENT}%0. Enables sampling based on real-clock timer.
    %{INDENT}%1. Interrupts per second. E.g., 100 == sample every 10 milliseconds of realtime.
    %{INDENT}%2. Delay (in seconds of real-clock time). I.e., how long each thread should wait before taking first sample.
    %{INDENT}%3+ Thread IDs to target for sampling, starting at 0 (the main thread).
    %{INDENT}%   May be specified as index or range, e.g., '0 2-4' will be interpreted as:
    %{INDENT}%      sample the main thread (0), do not sample the first child thread but sample the 2nd, 3rd, and 4th child threads
    %{INDENT}%   When sampling with a real-clock timer, please note that enabling this will cause threads which are typically "idle"
    %{INDENT}%   to consume more resources since, while idle, the real-clock time increases (and therefore triggers taking samples)
    %{INDENT}%   whereas the CPU-clock time does not.)";

    const auto* _overflow_desc =
        R"(Sample based on an overflow event. Accepts zero or more arguments:
    %{INDENT}%0. Enables sampling based on overflow.
    %{INDENT}%1. Overflow metric, e.g. PERF_COUNT_HW_INSTRUCTIONS
    %{INDENT}%2. Overflow value. E.g., if metric == PERF_COUNT_HW_INSTRUCTIONS, then 10000000 == sample every 10,000,000 instructions.
    %{INDENT}%3+ Thread IDs to target for sampling, starting at 0 (the main thread).
    %{INDENT}%   May be specified as index or range, e.g., '0 2-4' will be interpreted as:
    %{INDENT}%      sample the main thread (0), do not sample the first child thread but sample the 2nd, 3rd, and 4th child threads)";

    const auto* _hsa_interrupt_desc =
        R"(Set the value of the HSA_ENABLE_INTERRUPT environment variable.
%{INDENT}%  ROCm version 5.2 and older have a bug which will cause a deadlock if a sample is taken while waiting for the signal
%{INDENT}%  that a kernel completed -- which happens when sampling with a real-clock timer. We require this option to be set to
%{INDENT}%  when --realtime is specified to make users aware that, while this may fix the bug, it can have a negative impact on
%{INDENT}%  performance.
%{INDENT}%  Values:
%{INDENT}%    0     avoid triggering the bug, potentially at the cost of reduced performance
%{INDENT}%    1     do not modify how ROCm is notified about kernel completion)";

    const auto* _trace_policy_desc =
        R"(Policy for new data when the buffer size limit is reached:
    %{INDENT}%- discard     : new data is ignored
    %{INDENT}%- ring_buffer : new data overwrites oldest data)";

    _parser.start_group("DEBUG OPTIONS", "");

    if(_data.environ_filter("monochrome", _data))
    {
        _parser.add_argument({ "--monochrome" }, "Disable colorized output")
            .max_count(1)
            .dtype("bool")
            .action([&](parser_t& p) {
                auto _monochrome = p.get<bool>("monochrome");
                _data.monochrome = _monochrome;
                p.set_use_color(!_monochrome);
                update_env(_data, "ROCPROFSYS_MONOCHROME", (_monochrome) ? "1" : "0");
                update_env(_data, "MONOCHROME", (_monochrome) ? "1" : "0");
            });

        _data.processed_environs.emplace("monochrome");
    }

    if(_data.environ_filter("debug", _data))
    {
        _parser.add_argument({ "--debug" }, "Debug output")
            .max_count(1)
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_DEBUG", p.get<bool>("debug"));
            });

        _data.processed_environs.emplace("debug");
    }

    if(_data.environ_filter("verbose", _data))
    {
        _parser.add_argument({ "-v", "--verbose" }, "Verbose output")
            .count(1)
            .dtype("integral")
            .action([&](parser_t& p) {
                auto _v       = p.get<int>("verbose");
                _data.verbose = _v;
                update_env(_data, "ROCPROFSYS_VERBOSE", _v);
            });

        _data.processed_environs.emplace("verbose");
    }

    add_group_arguments(_parser, "debugging", _data);
    add_group_arguments(_parser, "mode", _data, true);

    _parser.start_group("GENERAL OPTIONS",
                        "These are options which are ubiquitously applied");

    if(_data.environ_filter("config", _data))
    {
        _parser.add_argument({ "-c", "--config" }, "Configuration file")
            .min_count(1)
            .dtype("filepath")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_CONFIG_FILE",
                           join(array_config_t{ ":" }, p.get<strvec_t>("config")));
            });

        _data.processed_environs.emplace("config");
        _data.processed_environs.emplace("config_file");
    }

    if(_data.environ_filter("output", _data))
    {
        _parser
            .add_argument(
                { "-o", "--output" },
                "Output path. Accepts 1-2 parameters corresponding to the output "
                "path and the output prefix")
            .min_count(1)
            .max_count(2)
            .dtype("path [prefix]")
            .action([&](parser_t& p) {
                auto _v = p.get<strvec_t>("output");
                update_env(_data, "ROCPROFSYS_OUTPUT_PATH", _v.at(0));
                if(_v.size() > 1) update_env(_data, "ROCPROFSYS_OUTPUT_PREFIX", _v.at(1));
            });

        _data.processed_environs.emplace("output");
        _data.processed_environs.emplace("output_path");
        _data.processed_environs.emplace("output_prefix");
    }

    if(_data.environ_filter("trace", _data))
    {
        _parser
            .add_argument({ "-T", "--trace" },
                          "Generate a detailed trace (perfetto output)")
            .max_count(1)
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_TRACE", p.get<bool>("trace"));
            });

        _data.processed_environs.emplace("trace");
    }

    if(_data.environ_filter("profile", _data))
    {
        _parser
            .add_argument(
                { "-P", "--profile" },
                "Generate a call-stack-based profile (conflicts with --flat-profile)")
            .max_count(1)
            .conflicts({ "flat-profile" })
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_PROFILE", p.get<bool>("profile"));
            });

        _data.processed_environs.emplace("profile");
    }

    if(_data.environ_filter("flat_profile", _data))
    {
        _parser
            .add_argument({ "-F", "--flat-profile" },
                          "Generate a flat profile (conflicts with --profile)")
            .max_count(1)
            .conflicts({ "profile" })
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_PROFILE", p.get<bool>("flat-profile"));
                update_env(_data, "ROCPROFSYS_FLAT_PROFILE", p.get<bool>("flat-profile"));
            });

        _data.processed_environs.emplace("flat_profile");
    }

    if(_data.environ_filter("sampling", _data))
    {
        _parser
            .add_argument({ "-S", "--sample" },
                          "Enable statistical sampling of call-stack")
            .min_count(0)
            .max_count(2)
            .dtype("timer-type")
            .choices({ "cputime", "realtime" })
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_USE_SAMPLING", true);
                auto _modes = p.get<strset_t>("sample");
                if(!_modes.empty())
                {
                    update_env(_data, "ROCPROFSYS_SAMPLING_CPUTIME",
                               _modes.count("cputime") > 0, UPD_WEAK);
                    update_env(_data, "ROCPROFSYS_SAMPLING_REALTIME",
                               _modes.count("realtime") > 0, UPD_WEAK);
                }
            });

        _data.processed_environs.emplace("cpu_freq");
    }

    if(_data.environ_filter("host", _data))
    {
        _parser
            .add_argument({ "-H", "--host" },
                          "Enable sampling host-based metrics for the process. E.g. CPU "
                          "frequency, memory usage, etc.")
            .max_count(1)
            .action([&](parser_t& p) {
                auto _h = p.get<bool>("host");
                auto _d = p.get<bool>("device");
                update_env(_data, "ROCPROFSYS_USE_PROCESS_SAMPLING", _h || _d);
                update_env(_data, "ROCPROFSYS_CPU_FREQ_ENABLED", _h);
                if(_h) update_env(_data, "ROCPROFSYS_USE_AMD_SMI", _d);
            });

        _data.processed_environs.emplace("host");
        _data.processed_environs.emplace("cpu_freq");
    }

    if(_data.environ_filter("device", _data))
    {
        _parser
            .add_argument(
                { "-D", "--device" },
                "Enable sampling device-based metrics for the process. E.g. GPU "
                "temperature, memory usage, etc.")
            .max_count(1)
            .action([&](parser_t& p) {
                auto _h = p.get<bool>("host");
                auto _d = p.get<bool>("device");
                update_env(_data, "ROCPROFSYS_USE_PROCESS_SAMPLING", _h || _d);
                update_env(_data, "ROCPROFSYS_USE_AMD_SMI", _d);
                if(_d) update_env(_data, "ROCPROFSYS_CPU_FREQ_ENABLED", _h);
            });

        _data.processed_environs.emplace("device");
        _data.processed_environs.emplace("amd_smi");
    }

    if(_data.environ_filter("wait", _data))
    {
        _parser
            .add_argument(
                { "-w", "--wait" },
                "This option is a combination of '--trace-wait' and "
                "'--sampling-wait'. See the descriptions for those two options.")
            .count(1)
            .dtype("seconds")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_TRACE_DELAY", p.get<double>("wait"),
                           UPD_WEAK);
                update_env(_data, "ROCPROFSYS_SAMPLING_DELAY", p.get<double>("wait"),
                           UPD_WEAK);
                update_env(_data, "ROCPROFSYS_CAUSAL_DELAY", p.get<double>("wait"),
                           UPD_WEAK);
            });

        _data.processed_environs.emplace("wait");
    }

    if(_data.environ_filter("duration", _data))
    {
        _parser
            .add_argument(
                { "-d", "--duration" },
                "This option is a combination of '--trace-duration' and "
                "'--sampling-duration'. See the descriptions for those two options.")
            .count(1)
            .dtype("seconds")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_TRACE_DURATION", p.get<double>("duration"),
                           UPD_WEAK);
                update_env(_data, "ROCPROFSYS_SAMPLING_DURATION",
                           p.get<double>("duration"), UPD_WEAK);
                update_env(_data, "ROCPROFSYS_CAUSAL_DURATION", p.get<double>("duration"),
                           UPD_WEAK);
            });

        _data.processed_environs.emplace("duration");
    }

    if(_data.environ_filter("periods", _data))
    {
        _parser
            .add_argument({ "--periods" },
                          "Similar to specifying delay and/or duration except in "
                          "the form <DELAY>:<DURATION>, <DELAY>:<DURATION>:<REPEAT>, "
                          "and/or <DELAY>:<DURATION>:<REPEAT>:<CLOCK_ID>")
            .min_count(1)
            .dtype("period-spec(s)")
            .action([&](parser_t& p) {
                update_env(
                    _data, "ROCPROFSYS_TRACE_PERIODS",
                    join(array_config_t{ " ", "", "" }, p.get<strvec_t>("periods")),
                    UPD_WEAK);
            });

        _data.processed_environs.emplace("periods");
    }

    strset_t _backend_choices = { "all",        "kokkosp", "mpip", "ompt",
                                  "rcclp",      "amd-smi", "rocm", "mutex-locks",
                                  "spin-locks", "rw-locks" };

#if !defined(ROCPROFSYS_USE_MPI) && !defined(ROCPROFSYS_USE_MPI_HEADERS)
    _backend_choices.erase("mpip");
#endif

#if !defined(ROCPROFSYS_USE_OMPT)
    _backend_choices.erase("ompt");
#endif

#if !defined(ROCPROFSYS_USE_ROCM)
    _backend_choices.erase("amd-smi");
    _backend_choices.erase("rocm");
    _backend_choices.erase("rcclp");
#endif

    if(gpu::device_count() == 0)
    {
        // remove GPU-specific backends
        _backend_choices.erase("rcclp");
        _backend_choices.erase("amd-smi");
        _backend_choices.erase("rocm");

#if defined(ROCPROFSYS_USE_ROCM)
        update_env(_data, "ROCPROFSYS_USE_AMD_SMI", false);
        update_env(_data, "ROCPROFSYS_USE_ROCM", false);
#endif
    }

    _parser.start_group("BACKEND OPTIONS",
                        "These options control region information captured "
                        "w/o sampling or instrumentation");

    if(_data.environ_filter("include", _data))
    {
        _parser.add_argument({ "-I", "--include" }, "Include data from these backends")
            .min_count(1)
            .max_count(_backend_choices.size())
            .dtype("[backend...]")
            .choices(_backend_choices)
            .action([&](parser_t& p) {
                auto _v      = p.get<strset_t>("include");
                auto _update = [&](const auto& _opt, bool _cond) {
                    if(_cond || _v.count("all") > 0) update_env(_data, _opt, true);
                };
                _update("ROCPROFSYS_USE_KOKKOSP", _v.count("kokkosp") > 0);
                _update("ROCPROFSYS_USE_MPIP", _v.count("mpip") > 0);
                _update("ROCPROFSYS_USE_OMPT", _v.count("ompt") > 0);
                _update("ROCPROFSYS_USE_ROCM", _v.count("rocm") > 0);
                _update("ROCPROFSYS_USE_RCCLP", _v.count("rcclp") > 0);
                _update("ROCPROFSYS_USE_AMD_SMI", _v.count("amd-smi") > 0);
                _update("ROCPROFSYS_TRACE_THREAD_LOCKS", _v.count("mutex-locks") > 0);
                _update("ROCPROFSYS_TRACE_THREAD_RW_LOCKS", _v.count("rw-locks") > 0);
                _update("ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS", _v.count("spin-locks") > 0);

                if(_v.count("all") > 0 || _v.count("ompt") > 0)
                    update_env(_data, "OMP_TOOL_LIBRARIES", _data.dl_libpath,
                               UPD_PREPEND);

                if(_v.count("all") > 0 || _v.count("kokkosp") > 0)
                    update_env(_data, "KOKKOS_TOOLS_LIBS", _data.omni_libpath,
                               UPD_PREPEND);
            });

        _data.processed_environs.emplace("include");
    }

    if(_data.environ_filter("exclude", _data))
    {
        _parser.add_argument({ "-E", "--exclude" }, "Exclude data from these backends")
            .min_count(1)
            .max_count(_backend_choices.size())
            .dtype("[backend...]")
            .choices(_backend_choices)
            .action([&](parser_t& p) {
                auto _v      = p.get<strset_t>("exclude");
                auto _update = [&](const auto& _opt, bool _cond) {
                    if(_cond || _v.count("all") > 0) update_env(_data, _opt, false);
                };
                _update("ROCPROFSYS_USE_KOKKOSP", _v.count("kokkosp") > 0);
                _update("ROCPROFSYS_USE_MPIP", _v.count("mpip") > 0);
                _update("ROCPROFSYS_USE_OMPT", _v.count("ompt") > 0);
                _update("ROCPROFSYS_USE_ROCM", _v.count("rocm") > 0);
                _update("ROCPROFSYS_USE_RCCLP", _v.count("rcclp") > 0);
                _update("ROCPROFSYS_USE_AMD_SMI", _v.count("amd-smi") > 0);
                _update("ROCPROFSYS_TRACE_THREAD_LOCKS", _v.count("mutex-locks") > 0);
                _update("ROCPROFSYS_TRACE_THREAD_RW_LOCKS", _v.count("rw-locks") > 0);
                _update("ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS", _v.count("spin-locks") > 0);

                if(_v.count("all") > 0 || _v.count("ompt") > 0)
                    remove_env(_data, "OMP_TOOL_LIBRARIES");

                if(_v.count("all") > 0 || _v.count("kokkosp") > 0)
                    remove_env(_data, "KOKKOS_TOOLS_LIBS");
            });

        _data.processed_environs.emplace("exclude");
    }

    add_group_arguments(_parser, "backend", _data);
    add_group_arguments(_parser, "parallelism", _data, true);

    if(_data.environ_filter("launcher", _data))
    {
        _parser
            .add_argument(
                { "-l", "--launcher" },
                "When running MPI jobs, typically the associated '--' for this "
                "executable should be right before the target executable, e.g. `mpirun "
                "-n 2 <THIS_EXE> -- <TARGET_EXE> <TARGET_EXE_ARGS...>`. This options "
                "enables prefixing the entire command (i.e. before `mpirun`, `srun`, "
                "etc.). Pass the name of the target executable (or a regex for matching "
                "to the name of the target) as the argument to this option and this "
                "executable will insert itself a second time in the appropriate "
                "location, e.g. `<THIS_EXE> --launcher sleep -- mpirun -n 2 sleep 10` is "
                "equivalent to `mpirun -n 2 <THIS_EXE> -- sleep 10`")
            .count(1)
            .dtype("target-exe")
            .action(
                [&](parser_t& p) { _data.launcher = p.get<std::string>("launcher"); });

        _data.processed_environs.emplace("launcher");
    }

    _parser.start_group("TRACING OPTIONS", "Specific options controlling tracing (i.e. "
                                           "deterministic measurements of every event)");

    if(_data.environ_filter("trace_file", _data))
    {
        _parser
            .add_argument(
                { "--trace-file" },
                "Specify the trace output filename. Relative filepath will be with "
                "respect to output path and output prefix.")
            .count(1)
            .dtype("filepath")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_PERFETTO_FILE",
                           p.get<std::string>("trace-file"));
            });

        _data.processed_environs.emplace("trace_file");
        _data.processed_environs.emplace("perfetto_file");
    }

    if(_data.environ_filter("trace_buffer_size", _data))
    {
        _parser
            .add_argument({ "--trace-buffer-size" },
                          "Size limit for the trace output (in KB)")
            .count(1)
            .dtype("KB")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB",
                           p.get<int64_t>("trace-buffer-size"));
            });

        _data.processed_environs.emplace("trace_buffer_size");
        _data.processed_environs.emplace("perfetto_buffer_size_kb");
    }

    if(_data.environ_filter("trace_fill_policy", _data))
    {
        _parser.add_argument({ "--trace-fill-policy" }, _trace_policy_desc)
            .count(1)
            .dtype("policy")
            .choices({ "discard", "ring_buffer" })
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_PERFETTO_FILL_POLICY",
                           p.get<std::string>("trace-fill-policy"));
            });

        _data.processed_environs.emplace("trace_fill_policy");
        _data.processed_environs.emplace("perfetto_fill_policy");
    }

    if(_data.environ_filter("trace_wait", _data))
    {
        _parser
            .add_argument(
                { "--trace-wait" },
                "Set the wait time (in seconds) "
                "before collecting trace and/or profiling data"
                "(in seconds). By default, the duration is in seconds of realtime "
                "but that can changed via --trace-clock-id.")
            .count(1)
            .dtype("seconds")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_TRACE_DELAY", p.get<double>("trace-wait"));
            });

        _data.processed_environs.emplace("trace_delay");
    }

    if(_data.environ_filter("trace_duration", _data))
    {
        _parser
            .add_argument(
                { "--trace-duration" },
                "Set the duration of the trace and/or profile data collection (in "
                "seconds). By default, the duration is in seconds of realtime but "
                "that can changed via --trace-clock-id.")
            .count(1)
            .dtype("seconds")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_TRACE_DURATION",
                           p.get<double>("trace-duration"));
            });

        _data.processed_environs.emplace("trace_duration");
    }

    if(_data.environ_filter("trace_periods", _data))
    {
        _parser
            .add_argument(
                { "--trace-periods" },
                "More powerful version of specifying trace delay and/or duration. Format "
                "is one or more groups of: <DELAY>:<DURATION>, "
                "<DELAY>:<DURATION>:<REPEAT>, "
                "and/or <DELAY>:<DURATION>:<REPEAT>:<CLOCK_ID>.")
            .min_count(1)
            .dtype("period-spec(s)")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_TRACE_PERIODS",
                           join(array_config_t{ ",", "", "" },
                                p.get<strvec_t>("trace-periods")));
            });

        _data.processed_environs.emplace("trace_periods");
    }

    if(_data.environ_filter("trace_clock_id", _data))
    {
        auto _clock_id_choices = get_clock_id_choices();
        _parser
            .add_argument(
                { "--trace-clock-id" },
                "Set the default clock ID for for trace delay/duration. Note: "
                "\"cputime\" is "
                "the *process* CPU time and might need to be scaled based on the number "
                "of "
                "threads, i.e. 4 seconds of CPU-time for an application with 4 fully "
                "active "
                "threads would equate to ~1 second of realtime. If this proves to be "
                "difficult to handle in practice, please file a feature request for "
                "rocprof-sys to auto-scale based on the number of threads.")
            .count(1)
            .dtype("clock-id")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_TRACE_PERIOD_CLOCK_ID",
                           p.get<double>("trace-clock-id"));
            })
            .choices(_clock_id_choices.first)
            .choice_aliases(_clock_id_choices.second);

        _data.processed_environs.emplace("trace_clock_id");
        _data.processed_environs.emplace("trace_period_clock_id");
    }

    _parser.start_group("PROFILE OPTIONS",
                        "Specific options controlling profiling (i.e. deterministic "
                        "measurements which are aggregated into a summary)");

    if(_data.environ_filter("profile_format", _data))
    {
        _parser.add_argument({ "--profile-format" }, "Data formats for profiling results")
            .min_count(1)
            .max_count(3)
            .dtype("string")
            .required({ "profile|flat-profile" })
            .choices({ "text", "json", "console" })
            .action([&](parser_t& p) {
                auto _v = p.get<strset_t>("profile-format");
                update_env(_data, "ROCPROFSYS_PROFILE", true);
                if(!_v.empty())
                {
                    update_env(_data, "ROCPROFSYS_TEXT_OUTPUT", _v.count("text") != 0);
                    update_env(_data, "ROCPROFSYS_JSON_OUTPUT", _v.count("json") != 0);
                    update_env(_data, "ROCPROFSYS_COUT_OUTPUT", _v.count("console") != 0);
                }
            });

        _data.processed_environs.emplace("profile_format");
        _data.processed_environs.emplace("text_output");
        _data.processed_environs.emplace("json_output");
        _data.processed_environs.emplace("cout_output");
    }

    if(_data.environ_filter("profile_diff", _data))
    {
        _parser
            .add_argument(
                { "--profile-diff" },
                "Generate a diff output b/t the profile collected and an existing "
                "profile from another run Accepts 1-2 parameters corresponding to "
                "the input path and the input prefix")
            .min_count(1)
            .max_count(2)
            .dtype("path [prefix]")
            .action([&](parser_t& p) {
                auto _v = p.get<strvec_t>("profile-diff");
                update_env(_data, "ROCPROFSYS_DIFF_OUTPUT", true);
                update_env(_data, "ROCPROFSYS_INPUT_PATH", _v.at(0));
                if(_v.size() > 1) update_env(_data, "ROCPROFSYS_INPUT_PREFIX", _v.at(1));
            });

        _data.processed_environs.emplace("profile_diff");
        _data.processed_environs.emplace("diff_output");
        _data.processed_environs.emplace("input_path");
        _data.processed_environs.emplace("input_prefix");
    }

    _parser.start_group(
        "HOST/DEVICE (PROCESS SAMPLING) OPTIONS",
        "Process sampling is background measurements for resources available to the "
        "entire process. These samples are not tied to specific lines/regions of code");

    if(_data.environ_filter("process_freq", _data))
    {
        _parser
            .add_argument({ "--process-freq" },
                          "Set the default host/device sampling frequency "
                          "(number of interrupts per second)")
            .count(1)
            .dtype("floating-point")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_PROCESS_SAMPLING_FREQ",
                           p.get<double>("process-freq"));
            });

        _data.processed_environs.emplace("process_freq");
        _data.processed_environs.emplace("process_sampling_freq");
    }

    if(_data.environ_filter("process_wait", _data))
    {
        _parser
            .add_argument({ "--process-wait" }, "Set the default wait time (i.e. delay) "
                                                "before taking first host/device sample "
                                                "(in seconds of realtime)")
            .count(1)
            .dtype("seconds")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_PROCESS_SAMPLING_DELAY",
                           p.get<double>("process-wait"));
            });

        _data.processed_environs.emplace("process_wait");
        _data.processed_environs.emplace("process_sampling_delay");
    }

    if(_data.environ_filter("process_duration", _data))
    {
        _parser
            .add_argument(
                { "--process-duration" },
                "Set the duration of the host/device sampling (in seconds of realtime)")
            .count(1)
            .dtype("seconds")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_SAMPLING_PROCESS_DURATION",
                           p.get<double>("process-duration"));
            });

        _data.processed_environs.emplace("process_duration");
        _data.processed_environs.emplace("process_sampling_duration");
    }

    if(_data.environ_filter("cpus", _data))
    {
        _parser
            .add_argument(
                { "--cpus" },
                "CPU IDs for frequency sampling. Supports integers and/or ranges")
            .dtype("int and/or range")
            .required({ "host" })
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_SAMPLING_CPUS",
                           join(array_config_t{ "," }, p.get<strvec_t>("cpus")));
            });

        _data.processed_environs.emplace("cpus");
        _data.processed_environs.emplace("sampling_cpus");
    }

    if(_data.environ_filter("gpus", _data))
    {
        _parser
            .add_argument({ "--gpus" },
                          "GPU IDs for SMI queries. Supports integers and/or ranges")
            .dtype("int and/or range")
            .required({ "device" })
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_SAMPLING_GPUS",
                           join(array_config_t{ "," }, p.get<strvec_t>("gpus")));
            });

        _data.processed_environs.emplace("gpus");
        _data.processed_environs.emplace("sampling_gpus");
    }

    _parser.start_group("GENERAL SAMPLING OPTIONS",
                        "General options for timer-based sampling per-thread");

    if(_data.environ_filter("sampling_freq", _data))
    {
        _parser
            .add_argument({ "-f", "--sampling-freq" },
                          "Set the default sampling frequency "
                          "(number of interrupts per second)")
            .count(1)
            .dtype("floating-point")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_SAMPLING_FREQ",
                           p.get<double>("sampling-freq"));
            });

        _data.processed_environs.emplace("sampling_freq");
    }

    if(_data.environ_filter("tids", _data))
    {
        _parser
            .add_argument(
                { "-t", "--tids" },
                "Specify the default thread IDs for sampling, where 0 (zero) is "
                "the main thread and each thread created by the target application "
                "is assigned an atomically incrementing value.")
            .min_count(1)
            .dtype("int and/or range")
            .action([&](parser_t& p) {
                update_env(
                    _data, "ROCPROFSYS_SAMPLING_TIDS",
                    join(array_config_t{ ", " }, p.get<std::vector<int64_t>>("tids")));
            });

        _data.processed_environs.emplace("tids");
        _data.processed_environs.emplace("sampling_tids");
    }

    if(_data.environ_filter("sampling_wait", _data))
    {
        _parser
            .add_argument(
                { "--sampling-wait" },
                "Set the default wait time (i.e. delay) before taking first sample "
                "(in seconds). This delay time is based on the clock of the sampler, "
                "i.e., a "
                "delay of 1 second for CPU-clock sampler may not equal 1 second of "
                "realtime")
            .count(1)
            .dtype("seconds")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_SAMPLING_DELAY",
                           p.get<double>("sampling-wait"));
            });

        _data.processed_environs.emplace("sampling_wait");
        _data.processed_environs.emplace("sampling_delay");
    }

    if(_data.environ_filter("sampling_duration", _data))
    {
        _parser
            .add_argument(
                { "--sampling-duration" },
                "Set the duration of the sampling (in seconds of realtime). I.e., it is "
                "possible (currently) to set a CPU-clock time delay that exceeds the "
                "real-time duration... resulting in zero samples being taken")
            .count(1)
            .dtype("seconds")
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_SAMPLING_DURATION",
                           p.get<double>("sampling-duration"));
            });

        _data.processed_environs.emplace("sampling_duration");
    }

    _parser.start_group(
        "SAMPLING TIMER OPTIONS",
        "These options determine the heuristic for deciding when to take a sample");

    if(_data.environ_filter("sampling_cputime", _data))
    {
        _parser.add_argument({ "--sample-cputime" }, _cputime_desc)
            .min_count(0)
            .dtype("[freq] [delay] [tids...]")
            .action([&](parser_t& p) {
                auto _v = p.get<std::deque<std::string>>("sample-cputime");
                update_env(_data, "ROCPROFSYS_SAMPLING_CPUTIME", true);
                if(!_v.empty())
                {
                    update_env(_data, "ROCPROFSYS_SAMPLING_CPUTIME_FREQ", _v.front());
                    _v.pop_front();
                }
                if(!_v.empty())
                {
                    update_env(_data, "ROCPROFSYS_SAMPLING_CPUTIME_DELAY", _v.front());
                    _v.pop_front();
                }
                if(!_v.empty())
                {
                    update_env(_data, "ROCPROFSYS_SAMPLING_CPUTIME_TIDS",
                               join(array_config_t{ "," }, _v));
                }
            });

        _data.processed_environs.emplace("sampling_cputime");
    }

    if(_data.environ_filter("sampling_realtime", _data))
    {
        _parser.add_argument({ "--sample-realtime" }, _realtime_desc)
            .min_count(0)
            .dtype("[freq] [delay] [tids...]")
            .action([&](parser_t& p) {
                auto _v = p.get<std::deque<std::string>>("sample-realtime");
                update_env(_data, "ROCPROFSYS_SAMPLING_REALTIME", true);
                if(!_v.empty())
                {
                    update_env(_data, "ROCPROFSYS_SAMPLING_REALTIME_FREQ", _v.front());
                    _v.pop_front();
                }
                if(!_v.empty())
                {
                    update_env(_data, "ROCPROFSYS_SAMPLING_REALTIME_DELAY", _v.front());
                    _v.pop_front();
                }
                if(!_v.empty())
                {
                    update_env(_data, "ROCPROFSYS_SAMPLING_REALTIME_TIDS",
                               join(array_config_t{ "," }, _v));
                }
            });

        _data.processed_environs.emplace("sampling_realtime");
    }

    if(_data.environ_filter("sampling_overflow", _data))
    {
        _parser.add_argument({ "--sample-overflow" }, _overflow_desc)
            .min_count(0)
            .dtype("[event] [freq] [tids...]")
            .action([&](parser_t& p) {
                auto _v = p.get<std::deque<std::string>>("sample-overflow");
                update_env(_data, "ROCPROFSYS_SAMPLING_OVERFLOW", true);

                if(!_v.empty())
                {
                    if(p.exists("sampling-overflow-event") &&
                       _v.front() != p.get<std::string>("sampling-overflow-event"))
                        throw exception<std::runtime_error>(join(
                            "", "'--sample-overflow ", _v.front(),
                            " ...' conflicts with '--sampling-overflow-event ",
                            p.get<std::string>("sampling-overflow-event"), "' option"));
                    update_env(_data, "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT", _v.front());
                    _v.pop_front();
                }
                if(!_v.empty())
                {
                    update_env(_data, "ROCPROFSYS_SAMPLING_OVERFLOW_FREQ", _v.front());
                    _v.pop_front();
                }
                if(!_v.empty())
                {
                    update_env(_data, "ROCPROFSYS_SAMPLING_OVERFLOW_TIDS",
                               join(array_config_t{ "," }, _v));
                }
            });

        _data.processed_environs.emplace("sampling_overflow");
    }

    _parser.start_group(
        "ADVANCED SAMPLING OPTIONS",
        "These options determine the heuristic for deciding when to take a sample");

    add_group_arguments(_parser, "sampling", _data);

    _parser.start_group("HARDWARE COUNTER OPTIONS", "See also: rocprof-sys-avail -H");

    if(_data.environ_filter("cpu_events", _data))
    {
        _parser
            .add_argument({ "-C", "--cpu-events" },
                          "Set the CPU hardware counter events to record (ref: "
                          "`rocprof-sys-avail -H -c CPU`)")
            .min_count(1)
            .dtype("[EVENT ...]")
            .action([&](parser_t& p) {
                auto _events = join(array_config_t{ "," }, p.get<strvec_t>("cpu-events"));
                update_env(_data, "ROCPROFSYS_PAPI_EVENTS", _events);
            });

        _data.processed_environs.emplace("cpu_events");
        _data.processed_environs.emplace("papi_events");
    }

    add_group_arguments(_parser, "category", _data, true);
    add_group_arguments(_parser, "io", _data, true);
    add_group_arguments(_parser, "perfetto", _data, true);
    add_group_arguments(_parser, "timemory", _data, true);
    add_group_arguments(_parser, "rocm", _data, true);

    _parser.start_group("MISCELLANEOUS OPTIONS", "");

    if(_data.environ_filter("inlines", _data))
    {
        _parser
            .add_argument({ "-i", "--inlines" },
                          "Include inline info in output when available")
            .max_count(1)
            .action([&](parser_t& p) {
                update_env(_data, "ROCPROFSYS_SAMPLING_INCLUDE_INLINES",
                           p.get<bool>("inlines"));
            });

        _data.processed_environs.emplace("inlines");
        _data.processed_environs.emplace("sampling_include_inlines");
    }

    if(_data.environ_filter("hsa_interrupt", _data))
    {
        _parser.add_argument({ "--hsa-interrupt" }, _hsa_interrupt_desc)
            .count(1)
            .dtype("int")
            .choices({ 0, 1 })
            .action([&](parser_t& p) {
                update_env(_data, "HSA_ENABLE_INTERRUPT", p.get<int>("hsa-interrupt"));
            });

        _data.processed_environs.emplace("hsa_interrupt");
    }

    _parser.end_group();

    return _data;
}

parser_data&
add_group_arguments(parser_t& _parser, const std::string& _group_name, parser_data& _data,
                    bool _add_group)
{
    if(!_data.grouping_filter(_group_name, _data)) return _data;

    auto _get_name = [](const std::shared_ptr<tim::vsettings>& itr) {
        auto _name = itr->get_name();
        auto _pos  = std::string::npos;
        while((_pos = _name.find('_')) != std::string::npos)
            _name = _name.replace(_pos, 1, "-");
        return _name;
    };

    auto _add_option = [&_parser, &_data](const std::string&                     _name,
                                          const std::shared_ptr<tim::vsettings>& itr) {
        if(!_data.setting_filter(itr.get(), _data)) return false;

        if(_name.empty())
            throw exception<std::runtime_error>("Error! empty name for " +
                                                itr->get_name());

        _data.processed_settings.emplace(itr.get());

        auto _opt_name = std::string{ "--" } + _name;
        itr->set_command_line({ _opt_name });
        auto* _arg = static_cast<parser_t::argument*>(itr->add_argument(_parser));
        if(_arg)
        {
            _arg->action([&_data, itr, _name](parser_t& p) {
                using namespace timemory::join;
                auto _value = join(array_config{ " ", "", "" }, p.get<strvec_t>(_name));
                if(_value.empty()) _value = p.get<std::string>(_name);
                if(_value.empty()) _value = join("", std::boolalpha, p.get<bool>(_name));
                if(_value.empty())
                    throw exception<std::runtime_error>("Error! no value for " + _name);
                update_env(_data, itr->get_env_name(), _value);
            });
        }
        else
        {
            TIMEMORY_PRINTF_WARNING(stderr, "Warning! Option %s (%s) is not enabled\n",
                                    _name.c_str(), itr->get_env_name().c_str());
            _parser.add_argument({ _opt_name }, itr->get_description())
                .action([&](parser_t& p) {
                    using namespace timemory::join;
                    auto _value =
                        join(array_config{ " ", "", "" }, p.get<strvec_t>(_name));
                    if(_value.empty())
                        throw exception<std::runtime_error>("Error! no value for " +
                                                            _name);
                    update_env(_data, itr->get_env_name(), _value);
                });
        }
        return true;
    };

    auto _settings = std::vector<std::shared_ptr<tim::vsettings>>{};
    for(auto& itr : *rocprofsys::settings::instance())
    {
        if(itr.second->get_categories().count("rocprofsys") == 0) continue;
        if(itr.second->get_categories().count("deprecated") > 0) continue;
        if(itr.second->get_hidden()) continue;
        if(!_data.setting_filter(itr.second.get(), _data)) continue;
        if(!_data.environ_filter(itr.second->get_name(), _data)) continue;
        if(itr.second->get_categories().count(_group_name) == 0) continue;

        itr.second->set_enabled(true);
        _settings.emplace_back(itr.second);

        if(itr.second->get_name() == "papi_events")
        {
            auto _choices = itr.second->get_choices();
            _choices.erase(
                std::remove_if(_choices.begin(), _choices.end(),
                               [](const auto& citr) {
                                   return std::regex_search(
                                              citr,
                                              std::regex{ "[A-Za-z0-9]:([A-Za-z_]+)" }) ||
                                          std::regex_search(citr, std::regex{ "io:::" });
                               }),
                _choices.end());
            _choices.emplace_back(
                "... run `rocprof-sys-avail -H -c CPU` for full list ...");
            itr.second->set_choices(_choices);
        }
    }

    std::sort(_settings.begin(), _settings.end(), [](const auto& _lhs, const auto& _rhs) {
        auto _lhs_v = _lhs->get_name();
        auto _rhs_v = _rhs->get_name();
        if(_lhs_v.length() > 4 && _rhs_v.length() > 4 &&
           _lhs_v.substr(0, 4) == _rhs_v.substr(0, 4))
            return _lhs_v < _rhs_v;
        return _lhs_v.length() < _rhs_v.length();
    });

    if(_add_group)
    {
        auto _group_label = _group_name;
        for(auto& c : _group_label)
            c = toupper(c);
        _parser.start_group(_group_label);
    }

    for(const auto& itr : _settings)
    {
        _add_option(_get_name(itr), itr);
    }

    if(_add_group) _parser.end_group();

    return _data;
}

parser_data&
add_extended_arguments(parser_t& _parser, parser_data& _data)
{
    auto _category_count_map = std::unordered_map<std::string, uint32_t>{};
    auto _settings           = std::vector<std::shared_ptr<tim::vsettings>>{};
    for(auto& itr : *rocprofsys::settings::instance())
    {
        if(itr.second->get_categories().count("rocprofsys") == 0) continue;
        if(itr.second->get_categories().count("deprecated") > 0) continue;
        if(itr.second->get_hidden()) continue;
        if(!_data.setting_filter(itr.second.get(), _data)) continue;
        if(!_data.environ_filter(itr.second->get_name(), _data)) continue;

        itr.second->set_enabled(true);
        _settings.emplace_back(itr.second);

        if(itr.second->get_name() == "papi_events")
        {
            auto _choices = itr.second->get_choices();
            _choices.erase(
                std::remove_if(_choices.begin(), _choices.end(),
                               [](const auto& citr) {
                                   return std::regex_search(
                                              citr,
                                              std::regex{ "[A-Za-z0-9]:([A-Za-z_]+)" }) ||
                                          std::regex_search(citr, std::regex{ "io:::" });
                               }),
                _choices.end());
            _choices.emplace_back(
                "... run `rocprof-sys-avail -H -c CPU` for full list ...");
            itr.second->set_choices(_choices);
        }

        for(const auto& citr : itr.second->get_categories())
        {
            if(std::regex_search(citr, std::regex{ "rocprofsys|timemory|^("
                                                   "native|custom|advanced|analysis)$" }))
                continue;
            _category_count_map[citr] += 1;
        }
    }

    auto _category_count_vec = strvec_t{};
    for(const auto& itr : _category_count_map)
        _category_count_vec.emplace_back(itr.first);

    std::sort(_category_count_vec.begin(), _category_count_vec.end(),
              [&_category_count_map](const auto& _lhs, const auto& _rhs) {
                  auto _lhs_v = _category_count_map.at(_lhs);
                  auto _rhs_v = _category_count_map.at(_rhs);
                  if(_lhs_v == _rhs_v) return _lhs < _rhs;
                  return _lhs_v > _rhs_v;
              });

    auto _groups =
        std::unordered_map<std::string, std::vector<std::shared_ptr<tim::vsettings>>>{};
    for(const auto& citr : _category_count_vec)
    {
        _groups[citr] = {};
        for(const auto& itr : _settings)
        {
            if(itr->get_categories().count(citr) > 0) _groups[citr].emplace_back(itr);
        }
        _settings.erase(std::remove_if(_settings.begin(), _settings.end(),
                                       [&citr](const auto& itr) {
                                           return itr->get_categories().count(citr) > 0;
                                       }),
                        _settings.end());
    }

    for(const auto& citr : _category_count_vec)
    {
        auto _group = _groups.at(citr);
        if(_group.empty()) continue;

        add_group_arguments(_parser, citr, _data, true);
    }

    return _data;
}
}  // namespace argparse
}  // namespace rocprofsys
