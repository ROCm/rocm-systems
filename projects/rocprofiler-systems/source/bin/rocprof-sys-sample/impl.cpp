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

#include "rocprof-sys-sample.hpp"

#include "common/delimit.hpp"
#include "common/environment.hpp"
#include "common/join.hpp"
#include "common/setup.hpp"

#include <timemory/environment.hpp>
#include <timemory/log/color.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/console.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/join.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace color = tim::log::color;
using namespace timemory::join;
using tim::get_env;
using tim::log::monochrome;
using tim::log::stream;

namespace
{
int  verbose          = 0;
auto updated_envs     = std::set<std::string_view>{};
auto original_envs    = std::set<std::string>{};
auto clock_id_choices = []() {
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

    auto _choices = std::vector<std::string>{};
    auto _aliases = std::map<std::string, std::vector<std::string>>{};
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
}();
}  // namespace

std::string
get_realpath(const std::string& _v)
{
    auto* _tmp = realpath(_v.c_str(), nullptr);
    auto  _ret = std::string{ _tmp };
    free(_tmp);
    return _ret;
}

void
print_command(const std::vector<char*>& _argv)
{
    if(verbose >= 1)
        stream(std::cout, color::info())
            << "Executing '" << join(array_config{ " " }, _argv) << "'...\n";
}

std::vector<char*>
get_initial_environment()
{
    auto _env = std::vector<char*>{};
    if(environ != nullptr)
    {
        int idx = 0;
        while(environ[idx] != nullptr)
        {
            auto* _v = environ[idx++];
            original_envs.emplace(_v);
            _env.emplace_back(strdup(_v));
        }
    }

    auto _dl_libpath   = get_realpath(get_internal_libpath("librocprof-sys-dl.so"));
    auto _omni_libpath = get_realpath(get_internal_libpath("librocprof-sys.so"));

    update_env(_env, "LD_PRELOAD", _dl_libpath, UPD_APPEND);
    update_env(_env, "LD_LIBRARY_PATH", tim::filepath::dirname(_dl_libpath), UPD_APPEND);

    auto _mode = get_env<std::string>("ROCPROFSYS_MODE", "sampling", false);

    update_env(_env, "ROCPROFSYS_USE_SAMPLING", (_mode != "causal"));

#if defined(ROCPROFSYS_USE_OMPT)
    if(!getenv("OMP_TOOL_LIBRARIES"))
        update_env(_env, "OMP_TOOL_LIBRARIES", _dl_libpath, UPD_APPEND);
#endif

    return _env;
}

std::string
get_internal_libpath(const std::string& _lib)
{
    auto _exe = std::string_view{ realpath("/proc/self/exe", nullptr) };
    auto _pos = _exe.find_last_of('/');
    auto _dir = std::string{ "./" };
    if(_pos != std::string_view::npos) _dir = _exe.substr(0, _pos);
    return rocprofsys::common::join("/", _dir, "..", "lib", _lib);
}

void
print_updated_environment(std::vector<char*> _env)
{
    if(get_env<int>("ROCPROFSYS_VERBOSE", 0) < 0) return;

    std::sort(_env.begin(), _env.end(), [](auto* _lhs, auto* _rhs) {
        if(!_lhs) return false;
        if(!_rhs) return true;
        return std::string_view{ _lhs } < std::string_view{ _rhs };
    });

    std::vector<char*> _updates = {};
    std::vector<char*> _general = {};

    for(auto* itr : _env)
    {
        if(itr == nullptr) continue;

        auto _is_omni = (std::string_view{ itr }.find("ROCPROFSYS") == 0);
        auto _updated = false;
        for(const auto& vitr : updated_envs)
        {
            if(std::string_view{ itr }.find(vitr) == 0)
            {
                _updated = true;
                break;
            }
        }

        if(_updated)
            _updates.emplace_back(itr);
        else if(verbose >= 1 && _is_omni)
            _general.emplace_back(itr);
    }

    if(_general.size() + _updates.size() == 0 || verbose < 0) return;

    std::cerr << std::endl;

    for(auto& itr : _general)
        stream(std::cerr, color::source()) << itr << "\n";
    for(auto& itr : _updates)
        stream(std::cerr, color::source()) << itr << "\n";

    std::cerr << std::endl;
}

template <typename Tp>
void
update_env(std::vector<char*>& _environ, std::string_view _env_var, Tp&& _env_val,
           update_mode&& _mode, std::string_view _join_delim)
{
    updated_envs.emplace(_env_var);

    auto _prepend  = (_mode & UPD_PREPEND) == UPD_PREPEND;
    auto _append   = (_mode & UPD_APPEND) == UPD_APPEND;
    auto _weak_upd = (_mode & UPD_WEAK) == UPD_WEAK;

    auto _key = join("", _env_var, "=");
    for(auto& itr : _environ)
    {
        if(!itr) continue;
        if(std::string_view{ itr }.find(_key) == 0)
        {
            if(_weak_upd)
            {
                // if the value has changed, do not update but allow overridding the value
                // inherited from the initial env
                if(original_envs.find(std::string{ itr }) == original_envs.end()) return;
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
    _environ.emplace_back(
        strdup(rocprofsys::common::join('=', _env_var, _env_val).c_str()));
}

void
remove_env(std::vector<char*>& _environ, std::string_view _env_var)
{
    auto _key   = join("", _env_var, "=");
    auto _match = [&_key](auto itr) { return std::string_view{ itr }.find(_key) == 0; };

    _environ.erase(std::remove_if(_environ.begin(), _environ.end(), _match),
                   _environ.end());

    for(const auto& itr : original_envs)
    {
        if(std::string_view{ itr }.find(_key) == 0)
            _environ.emplace_back(strdup(itr.c_str()));
    }
}

std::vector<char*>
parse_args(int argc, char** argv, std::vector<char*>& _env)
{
    using parser_t     = tim::argparse::argument_parser;
    using parser_err_t = typename parser_t::result_type;

    auto help_check = [](parser_t& p, int _argc, char** _argv) {
        std::set<std::string> help_args = { "-h", "--help", "-?" };
        return (p.exists("help") || _argc == 1 ||
                (_argc > 1 && help_args.find(_argv[1]) != help_args.end()));
    };

    auto _pec        = EXIT_SUCCESS;
    auto help_action = [&_pec, argc, argv](parser_t& p) {
        if(_pec != EXIT_SUCCESS)
        {
            std::stringstream msg;
            msg << "Error in command:";
            for(int i = 0; i < argc; ++i)
                msg << " " << argv[i];
            msg << "\n\n";
            stream(std::cerr, color::fatal()) << msg.str();
            std::cerr << std::flush;
        }

        p.print_help();
        exit(_pec);
    };

    auto* _dl_libpath =
        realpath(get_internal_libpath("librocprof-sys-dl.so").c_str(), nullptr);
    auto* _omni_libpath =
        realpath(get_internal_libpath("librocprof-sys.so").c_str(), nullptr);

    auto parser = parser_t(argv[0]);

    parser.on_error([](parser_t&, const parser_err_t& _err) {
        stream(std::cerr, color::fatal()) << _err << "\n";
        exit(EXIT_FAILURE);
    });

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

    parser.set_use_color(true);
    parser.enable_help();
    parser.enable_version("rocprof-sys-sample", ROCPROFSYS_ARGPARSE_VERSION_INFO);

    auto _cols = std::get<0>(tim::utility::console::get_columns());
    if(_cols > parser.get_help_width() + 8)
        parser.set_description_width(
            std::min<int>(_cols - parser.get_help_width() - 8, 120));

    parser.start_group("DEBUG OPTIONS", "");
    parser.add_argument({ "--monochrome" }, "Disable colorized output")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            auto _monochrome = p.get<bool>("monochrome");
            monochrome()     = _monochrome;
            p.set_use_color(!_monochrome);
            update_env(_env, "ROCPROFSYS_MONOCHROME", (_monochrome) ? "1" : "0");
            update_env(_env, "MONOCHROME", (_monochrome) ? "1" : "0");
        });
    parser.add_argument({ "--debug" }, "Debug output")
        .max_count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_DEBUG", p.get<bool>("debug"));
        });
    parser.add_argument({ "-v", "--verbose" }, "Verbose output")
        .count(1)
        .action([&](parser_t& p) {
            auto _v = p.get<int>("verbose");
            verbose = _v;
            update_env(_env, "ROCPROFSYS_VERBOSE", _v);
        });

    parser.start_group("GENERAL OPTIONS",
                       "These are options which are ubiquitously applied");
    parser.add_argument({ "-c", "--config" }, "Configuration file")
        .min_count(0)
        .dtype("filepath")
        .action([&](parser_t& p) {
            update_env(
                _env, "ROCPROFSYS_CONFIG_FILE",
                join(array_config{ ":" }, p.get<std::vector<std::string>>("config")));
        });
    parser
        .add_argument({ "-o", "--output" },
                      "Output path. Accepts 1-2 parameters corresponding to the output "
                      "path and the output prefix")
        .min_count(1)
        .max_count(2)
        .action([&](parser_t& p) {
            auto _v = p.get<std::vector<std::string>>("output");
            update_env(_env, "ROCPROFSYS_OUTPUT_PATH", _v.at(0));
            if(_v.size() > 1) update_env(_env, "ROCPROFSYS_OUTPUT_PREFIX", _v.at(1));
        });
    parser
        .add_argument({ "-T", "--trace" }, "Generate a detailed trace (perfetto output)")
        .max_count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_TRACE", p.get<bool>("trace"));
        });
    parser
        .add_argument(
            { "-P", "--profile" },
            "Generate a call-stack-based profile (conflicts with --flat-profile)")
        .max_count(1)
        .conflicts({ "flat-profile" })
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_PROFILE", p.get<bool>("profile"));
        });
    parser
        .add_argument({ "-F", "--flat-profile" },
                      "Generate a flat profile (conflicts with --profile)")
        .max_count(1)
        .conflicts({ "profile" })
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_PROFILE", p.get<bool>("flat-profile"));
            update_env(_env, "ROCPROFSYS_FLAT_PROFILE", p.get<bool>("flat-profile"));
        });
    parser
        .add_argument({ "-H", "--host" },
                      "Enable sampling host-based metrics for the process. E.g. CPU "
                      "frequency, memory usage, etc.")
        .max_count(1)
        .action([&](parser_t& p) {
            auto _h = p.get<bool>("host");
            auto _d = p.get<bool>("device");
            update_env(_env, "ROCPROFSYS_USE_PROCESS_SAMPLING", _h || _d);
            update_env(_env, "ROCPROFSYS_CPU_FREQ_ENABLED", _h);
            if(_h) update_env(_env, "ROCPROFSYS_USE_AMD_SMI", _d);
        });
    parser
        .add_argument({ "-D", "--device" },
                      "Enable sampling device-based metrics for the process. E.g. GPU "
                      "temperature, memory usage, etc.")
        .max_count(1)
        .action([&](parser_t& p) {
            auto _h = p.get<bool>("host");
            auto _d = p.get<bool>("device");
            update_env(_env, "ROCPROFSYS_USE_PROCESS_SAMPLING", _h || _d);
            update_env(_env, "ROCPROFSYS_USE_AMD_SMI", _d);
            if(_d) update_env(_env, "ROCPROFSYS_CPU_FREQ_ENABLED", _h);
        });
    parser
        .add_argument({ "-w", "--wait" },
                      "This option is a combination of '--trace-wait' and "
                      "'--sampling-wait'. See the descriptions for those two options.")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_TRACE_DELAY", p.get<double>("wait"));
            update_env(_env, "ROCPROFSYS_SAMPLING_DELAY", p.get<double>("wait"));
        });
    parser
        .add_argument(
            { "-d", "--duration" },
            "This option is a combination of '--trace-duration' and "
            "'--sampling-duration'. See the descriptions for those two options.")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_TRACE_DURATION", p.get<double>("duration"));
            update_env(_env, "ROCPROFSYS_SAMPLING_DURATION", p.get<double>("duration"));
        });

    parser.start_group("TRACING OPTIONS", "Specific options controlling tracing (i.e. "
                                          "deterministic measurements of every event)");
    parser
        .add_argument({ "--trace-file" },
                      "Specify the trace output filename. Relative filepath will be with "
                      "respect to output path and output prefix.")
        .count(1)
        .dtype("filepath")
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_PERFETTO_FILE",
                       p.get<std::string>("trace-file"));
        });
    parser
        .add_argument({ "--trace-buffer-size" },
                      "Size limit for the trace output (in KB)")
        .count(1)
        .dtype("KB")
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB",
                       p.get<int64_t>("trace-buffer-size"));
        });
    parser.add_argument({ "--trace-fill-policy" }, _trace_policy_desc)
        .count(1)
        .choices({ "discard", "ring_buffer" })
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_PERFETTO_FILL_POLICY",
                       p.get<std::string>("trace-fill-policy"));
        });
    parser
        .add_argument({ "--trace-wait" },
                      "Set the wait time (in seconds) "
                      "before collecting trace and/or profiling data"
                      "(in seconds). By default, the duration is in seconds of realtime "
                      "but that can changed via --trace-clock-id.")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_TRACE_DELAY", p.get<double>("trace-wait"));
        });
    parser
        .add_argument({ "--trace-duration" },
                      "Set the duration of the trace and/or profile data collection (in "
                      "seconds). By default, the duration is in seconds of realtime but "
                      "that can changed via --trace-clock-id.")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_TRACE_DURATION",
                       p.get<double>("trace-duration"));
        });
    parser
        .add_argument(
            { "--trace-periods" },
            "More powerful version of specifying trace delay and/or duration. Format is "
            "one or more groups of: <DELAY>:<DURATION>, <DELAY>:<DURATION>:<REPEAT>, "
            "and/or <DELAY>:<DURATION>:<REPEAT>:<CLOCK_ID>.")
        .min_count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_TRACE_PERIODS",
                       join(array_config{ ",", "", "" },
                            p.get<std::vector<std::string>>("trace-periods")));
        });
    parser
        .add_argument(
            { "--trace-clock-id" },
            "Set the default clock ID for for trace delay/duration. Note: \"cputime\" is "
            "the *process* CPU time and might need to be scaled based on the number of "
            "threads, i.e. 4 seconds of CPU-time for an application with 4 fully active "
            "threads would equate to ~1 second of realtime. If this proves to be "
            "difficult to handle in practice, please file a feature request for "
            "rocprof-sys to auto-scale based on the number of threads.")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_TRACE_PERIOD_CLOCK_ID",
                       p.get<double>("trace-clock-id"));
        })
        .choices(clock_id_choices.first)
        .choice_aliases(clock_id_choices.second);

    parser.start_group("PROFILE OPTIONS",
                       "Specific options controlling profiling (i.e. deterministic "
                       "measurements which are aggregated into a summary)");
    parser.add_argument({ "--profile-format" }, "Data formats for profiling results")
        .min_count(1)
        .max_count(3)
        .required({ "profile|flat-profile" })
        .choices({ "text", "json", "console" })
        .action([&](parser_t& p) {
            auto _v = p.get<std::set<std::string>>("profile");
            update_env(_env, "ROCPROFSYS_PROFILE", true);
            if(!_v.empty())
            {
                update_env(_env, "ROCPROFSYS_TEXT_OUTPUT", _v.count("text") != 0);
                update_env(_env, "ROCPROFSYS_JSON_OUTPUT", _v.count("json") != 0);
                update_env(_env, "ROCPROFSYS_COUT_OUTPUT", _v.count("console") != 0);
            }
        });

    parser
        .add_argument({ "--profile-diff" },
                      "Generate a diff output b/t the profile collected and an existing "
                      "profile from another run Accepts 1-2 parameters corresponding to "
                      "the input path and the input prefix")
        .min_count(1)
        .max_count(2)
        .action([&](parser_t& p) {
            auto _v = p.get<std::vector<std::string>>("profile-diff");
            update_env(_env, "ROCPROFSYS_DIFF_OUTPUT", true);
            update_env(_env, "ROCPROFSYS_INPUT_PATH", _v.at(0));
            if(_v.size() > 1) update_env(_env, "ROCPROFSYS_INPUT_PREFIX", _v.at(1));
        });

    parser.start_group(
        "HOST/DEVICE (PROCESS SAMPLING) OPTIONS",
        "Process sampling is background measurements for resources available to the "
        "entire process. These samples are not tied to specific lines/regions of code");
    parser
        .add_argument({ "--process-freq" },
                      "Set the default host/device sampling frequency "
                      "(number of interrupts per second)")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_PROCESS_SAMPLING_FREQ",
                       p.get<double>("process-freq"));
        });
    parser
        .add_argument({ "--process-wait" }, "Set the default wait time (i.e. delay) "
                                            "before taking first host/device sample "
                                            "(in seconds of realtime)")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_PROCESS_SAMPLING_DELAY",
                       p.get<double>("process-wait"));
        });
    parser
        .add_argument(
            { "--process-duration" },
            "Set the duration of the host/device sampling (in seconds of realtime)")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_SAMPLING_PROCESS_DURATION",
                       p.get<double>("process-duration"));
        });
    parser
        .add_argument({ "--cpus" },
                      "CPU IDs for frequency sampling. Supports integers and/or ranges")
        .dtype("int or range")
        .required({ "host" })
        .action([&](parser_t& p) {
            update_env(
                _env, "ROCPROFSYS_SAMPLING_CPUS",
                join(array_config{ "," }, p.get<std::vector<std::string>>("cpus")));
        });
    parser
        .add_argument({ "--gpus" },
                      "GPU IDs for SMI queries. Supports integers and/or ranges")
        .dtype("int or range")
        .required({ "device" })
        .action([&](parser_t& p) {
            update_env(
                _env, "ROCPROFSYS_SAMPLING_GPUS",
                join(array_config{ "," }, p.get<std::vector<std::string>>("gpus")));
        });

    parser.start_group("GENERAL SAMPLING OPTIONS",
                       "General options for timer-based sampling per-thread");
    parser
        .add_argument({ "-f", "--freq" }, "Set the default sampling frequency "
                                          "(number of interrupts per second)")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_SAMPLING_FREQ", p.get<double>("freq"));
        });
    parser
        .add_argument(
            { "--sampling-wait" },
            "Set the default wait time (i.e. delay) before taking first sample "
            "(in seconds). This delay time is based on the clock of the sampler, i.e., a "
            "delay of 1 second for CPU-clock sampler may not equal 1 second of realtime")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_SAMPLING_DELAY", p.get<double>("sampling-wait"));
        });
    parser
        .add_argument(
            { "--sampling-duration" },
            "Set the duration of the sampling (in seconds of realtime). I.e., it is "
            "possible (currently) to set a CPU-clock time delay that exceeds the "
            "real-time duration... resulting in zero samples being taken")
        .count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_SAMPLING_DURATION",
                       p.get<double>("sampling-duration"));
        });
    parser
        .add_argument({ "-t", "--tids" },
                      "Specify the default thread IDs for sampling, where 0 (zero) is "
                      "the main thread and each thread created by the target application "
                      "is assigned an atomically incrementing value.")
        .min_count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_SAMPLING_TIDS",
                       join(array_config{ ", " }, p.get<std::vector<int64_t>>("tids")));
        });

    parser.start_group(
        "SAMPLING TIMER OPTIONS",
        "These options determine the heuristic for deciding when to take a sample");
    parser.add_argument({ "--cputime" }, _cputime_desc)
        .min_count(0)
        .action([&](parser_t& p) {
            auto _v = p.get<std::deque<std::string>>("cputime");
            update_env(_env, "ROCPROFSYS_SAMPLING_CPUTIME", true);
            if(!_v.empty())
            {
                update_env(_env, "ROCPROFSYS_SAMPLING_CPUTIME_FREQ", _v.front());
                _v.pop_front();
            }
            if(!_v.empty())
            {
                update_env(_env, "ROCPROFSYS_SAMPLING_CPUTIME_DELAY", _v.front());
                _v.pop_front();
            }
            if(!_v.empty())
            {
                update_env(_env, "ROCPROFSYS_SAMPLING_CPUTIME_TIDS",
                           join(array_config{ "," }, _v));
            }
        });

    parser.add_argument({ "--realtime" }, _realtime_desc)
        .min_count(0)
        .action([&](parser_t& p) {
            auto _v = p.get<std::deque<std::string>>("realtime");
            update_env(_env, "ROCPROFSYS_SAMPLING_REALTIME", true);
            if(!_v.empty())
            {
                update_env(_env, "ROCPROFSYS_SAMPLING_REALTIME_FREQ", _v.front());
                _v.pop_front();
            }
            if(!_v.empty())
            {
                update_env(_env, "ROCPROFSYS_SAMPLING_REALTIME_DELAY", _v.front());
                _v.pop_front();
            }
            if(!_v.empty())
            {
                update_env(_env, "ROCPROFSYS_SAMPLING_REALTIME_TIDS",
                           join(array_config{ "," }, _v));
            }
        });

    std::set<std::string> _backend_choices = {
        "all",      "kokkosp",         "mpip",        "ompt",
        "rcclp",    "amd-smi",         "mutex-locks", "spin-locks",
        "rw-locks", "rocprofiler-sdk", "rocm"
    };

#if !defined(ROCPROFSYS_USE_MPI) && !defined(ROCPROFSYS_USE_MPI_HEADERS)
    _backend_choices.erase("mpip");
#endif

#if !defined(ROCPROFSYS_USE_OMPT)
    _backend_choices.erase("ompt");
#endif

#if !defined(ROCPROFSYS_USE_RCCL)
    _backend_choices.erase("rcclp");
#endif

#if !defined(ROCPROFSYS_USE_ROCM)
    _backend_choices.erase("rocm");
    _backend_choices.erase("amd-smi");
    _backend_choices.erase("rocprofiler-sdk");
#endif

    parser.start_group("BACKEND OPTIONS",
                       "These options control region information captured "
                       "w/o sampling or instrumentation");
    parser.add_argument({ "-I", "--include" }, "Include data from these backends")
        .choices(_backend_choices)
        .action([&](parser_t& p) {
            auto _v      = p.get<std::set<std::string>>("include");
            auto _update = [&](const auto& _opt, bool _cond) {
                if(_cond || _v.count("all") > 0) update_env(_env, _opt, true);
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
                update_env(_env, "OMP_TOOL_LIBRARIES", _dl_libpath, UPD_APPEND);

            if(_v.count("all") > 0 || _v.count("kokkosp") > 0)
                update_env(_env, "KOKKOS_TOOLS_LIBS", _omni_libpath, UPD_APPEND);
        });

    parser.add_argument({ "-E", "--exclude" }, "Exclude data from these backends")
        .choices(_backend_choices)
        .action([&](parser_t& p) {
            auto _v      = p.get<std::set<std::string>>("exclude");
            auto _update = [&](const auto& _opt, bool _cond) {
                if(_cond || _v.count("all") > 0) update_env(_env, _opt, false);
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

            // if(_v.count("all") > 0 || _v.count("rocprofiler") > 0)
            // {
            //     remove_env(_env, "ROCP_TOOL_LIB");
            //     remove_env(_env, "ROCP_HSA_INTERCEPT");
            // }

            if(_v.count("all") > 0 || _v.count("ompt") > 0)
                remove_env(_env, "OMP_TOOL_LIBRARIES");

            if(_v.count("all") > 0 || _v.count("kokkosp") > 0)
                remove_env(_env, "KOKKOS_TOOLS_LIBS");
        });

    parser.start_group("HARDWARE COUNTER OPTIONS", "See also: rocprof-sys-avail -H");
    parser
        .add_argument({ "-C", "--cpu-events" },
                      "Set the CPU hardware counter events to record (ref: "
                      "`rocprof-sys-avail -H -c CPU`)")
        .action([&](parser_t& p) {
            auto _events =
                join(array_config{ "," }, p.get<std::vector<std::string>>("cpu-events"));
            update_env(_env, "ROCPROFSYS_PAPI_EVENTS", _events);
        });

    parser.start_group("MISCELLANEOUS OPTIONS", "");
    parser
        .add_argument({ "-i", "--inlines" },
                      "Include inline info in output when available")
        .max_count(1)
        .action([&](parser_t& p) {
            update_env(_env, "ROCPROFSYS_SAMPLING_INCLUDE_INLINES",
                       p.get<bool>("inlines"));
        });

    parser.add_argument({ "--hsa-interrupt" }, _hsa_interrupt_desc)
        .count(1)
        .dtype("int")
        .choices({ 0, 1 })
        .action([&](parser_t& p) {
            update_env(_env, "HSA_ENABLE_INTERRUPT", p.get<int>("hsa-interrupt"));
        });

    parser.end_group();

    auto _inpv = std::vector<char*>{};
    auto _outv = std::vector<char*>{};
    bool _hash = false;
    for(int i = 0; i < argc; ++i)
    {
        if(_hash)
        {
            _outv.emplace_back(argv[i]);
        }
        else if(std::string_view{ argv[i] } == "--")
        {
            _hash = true;
        }
        else
        {
            _inpv.emplace_back(argv[i]);
        }
    }

    auto _cerr = parser.parse_args(_inpv.size(), _inpv.data());
    if(help_check(parser, argc, argv))
        help_action(parser);
    else if(_cerr)
        throw std::runtime_error(_cerr.what());

    if(parser.exists("realtime") && !parser.exists("cputime"))
        update_env(_env, "ROCPROFSYS_SAMPLING_CPUTIME", false);
    if(parser.exists("profile") && parser.exists("flat-profile"))
        throw std::runtime_error(
            "Error! '--profile' argument conflicts with '--flat-profile' argument");

    free(_dl_libpath);
    free(_omni_libpath);

    return _outv;
}
