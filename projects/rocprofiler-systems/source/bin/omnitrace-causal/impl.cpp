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

#include "omnitrace-causal.hpp"

#include "common/defines.h"
#include "common/delimit.hpp"
#include "common/environment.hpp"
#include "common/join.hpp"
#include "common/setup.hpp"

#include <regex>
#include <timemory/environment.hpp>
#include <timemory/log/color.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/console.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/join.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace color    = ::tim::log::color;
namespace filepath = ::tim::filepath;
namespace console  = ::tim::utility::console;
namespace argparse = ::tim::argparse;
using namespace timemory::join;
using tim::get_env;
using tim::log::colorized;
using tim::log::stream;

namespace std
{
std::string
to_string(bool _v)
{
    return (_v) ? "true" : "false";
}
}  // namespace std

namespace
{
int  verbose       = 0;
auto updated_envs  = std::set<std::string_view>{};
auto original_envs = std::set<std::string>{};
auto child_pids    = std::set<pid_t>{};
auto launcher      = std::string{};

inline signal_handler&
get_signal_handler(int _sig)
{
    static auto _v  = std::unordered_map<int, signal_handler>{};
    auto        itr = _v.emplace(_sig, signal_handler{});
    return itr.first->second;
}

void
create_signal_handler(int sig, signal_handler& sh, void (*func)(int))
{
    if(sig < 1) return;
    sh.m_custom_sigaction.sa_handler = func;
    sigemptyset(&sh.m_custom_sigaction.sa_mask);
    sh.m_custom_sigaction.sa_flags = SA_RESTART;
    if(sigaction(sig, &sh.m_custom_sigaction, &sh.m_original_sigaction) == -1)
    {
        std::cerr << "Failed to create signal handler for " << sig << std::endl;
    }
}

void
forward_signal(int sig)
{
    for(auto itr : child_pids)
    {
        TIMEMORY_PRINTF_WARNING(stderr, "Killing pid=%i with signal %i...\n", itr, sig);
        kill(itr, sig);
        diagnose_status(itr, wait_pid(itr));
    }
    signal(sig, SIG_DFL);
    kill(getpid(), sig);
}
}  // namespace

int
get_verbose()
{
    verbose = get_env("OMNITRACE_CAUSAL_VERBOSE",
                      get_env<int>("OMNITRACE_VERBOSE", verbose, false));
    auto _debug =
        get_env("OMNITRACE_CAUSAL_DEBUG", get_env<bool>("OMNITRACE_DEBUG", false, false));
    if(_debug) verbose += 8;
    return verbose;
}

void
forward_signals(const std::set<int>& _signals)
{
    for(auto itr : _signals)
        create_signal_handler(itr, get_signal_handler(itr), &forward_signal);
}

void
add_child_pid(pid_t _v)
{
    child_pids.emplace(_v);
}

void
remove_child_pid(pid_t _v)
{
    child_pids.erase(_v);
}

int
wait_pid(pid_t _pid, int _opts)
{
    int   _status = 0;
    pid_t _pid_v  = -1;
    _opts |= WUNTRACED;
    do
    {
        if((_opts & WNOHANG) > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
        _pid_v = waitpid(_pid, &_status, _opts);
    } while(_pid <= 0);
    return _status;
}

int
diagnose_status(pid_t _pid, int _status)
{
    auto _verbose = get_verbose();
    if(_verbose >= 3)
    {
        fflush(stderr);
        fflush(stdout);
        std::cout << std::flush;
        std::cerr << std::flush;
    }

    bool _normal_exit      = (WIFEXITED(_status) > 0);
    bool _unhandled_signal = (WIFSIGNALED(_status) > 0);
    bool _core_dump        = (WCOREDUMP(_status) > 0);
    bool _stopped          = (WIFSTOPPED(_status) > 0);
    int  _exit_status      = WEXITSTATUS(_status);
    int  _stop_signal      = (_stopped) ? WSTOPSIG(_status) : 0;
    int  _ec               = (_unhandled_signal) ? WTERMSIG(_status) : 0;

    if(_verbose >= 4)
    {
        TIMEMORY_PRINTF_INFO(
            stderr,
            "diagnosing status for process %i :: status: %i... normal exit: %s, "
            "unhandled signal: %s, core dump: %s, stopped: %s, exit status: %i, stop "
            "signal: %i, exit code: %i\n",
            _pid, _status, std::to_string(_normal_exit).c_str(),
            std::to_string(_unhandled_signal).c_str(), std::to_string(_core_dump).c_str(),
            std::to_string(_stopped).c_str(), _exit_status, _stop_signal, _ec);
    }
    else if(_verbose >= 3)
    {
        TIMEMORY_PRINTF_INFO(stderr,
                             "diagnosing status for process %i :: status: %i ...\n", _pid,
                             _status);
    }

    if(!_normal_exit)
    {
        if(_ec == 0) _ec = EXIT_FAILURE;
        if(_verbose >= 0)
        {
            TIMEMORY_PRINTF_FATAL(
                stderr, "process %i terminated abnormally. exit code: %i\n", _pid, _ec);
        }
    }

    if(_stopped)
    {
        if(_verbose >= 0)
        {
            TIMEMORY_PRINTF_FATAL(stderr,
                                  "process %i stopped with signal %i. exit code: %i\n",
                                  _pid, _stop_signal, _ec);
        }
    }

    if(_core_dump)
    {
        if(_verbose >= 0)
        {
            TIMEMORY_PRINTF_FATAL(
                stderr, "process %i terminated and produced a core dump. exit code: %i\n",
                _pid, _ec);
        }
    }

    if(_unhandled_signal)
    {
        if(_verbose >= 0)
        {
            TIMEMORY_PRINTF_FATAL(stderr,
                                  "process %i terminated because it received a signal "
                                  "(%i) that was not handled. exit code: %i\n",
                                  _pid, _ec, _ec);
        }
    }

    if(!_normal_exit && _exit_status > 0)
    {
        if(_verbose >= 0)
        {
            if(_exit_status == 127)
            {
                TIMEMORY_PRINTF_FATAL(
                    stderr, "execv in process %i failed. exit code: %i\n", _pid, _ec);
            }
            else
            {
                TIMEMORY_PRINTF_FATAL(
                    stderr,
                    "process %i terminated with a non-zero status. exit code: %i\n", _pid,
                    _ec);
            }
        }
    }

    return _ec;
}

std::string
get_realpath(const std::string& _v)
{
    auto* _tmp = realpath(_v.c_str(), nullptr);
    auto  _ret = std::string{ _tmp };
    free(_tmp);
    return _ret;
}

void
print_command(const std::vector<char*>& _argv, std::string_view _prefix)
{
    if(verbose >= 1)
        stream(std::cout, color::info())
            << _prefix << "Executing '" << join(array_config{ " " }, _argv) << "'...\n";

    std::cerr << color::end() << std::flush;
}

std::vector<char*>
get_initial_environment()
{
    std::vector<char*> _env;
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

    update_env(_env, "OMNITRACE_MODE", "causal");
    update_env(_env, "OMNITRACE_USE_CAUSAL", true);
    update_env(_env, "OMNITRACE_USE_SAMPLING", false);
    update_env(_env, "OMNITRACE_USE_PERFETTO", false);
    update_env(_env, "OMNITRACE_USE_TIMEMORY", false);
    update_env(_env, "OMNITRACE_USE_PROCESS_SAMPLING", false);
    update_env(_env, "OMNITRACE_CRITICAL_TRACE", false);

    return _env;
}

void
prepare_command_for_run(char* _exe, std::vector<char*>& _argv)
{
    if(!launcher.empty())
    {
        bool _injected = false;
        auto _new_argv = std::vector<char*>{};
        for(auto* itr : _argv)
        {
            if(!_injected && std::regex_search(itr, std::regex{ launcher }))
            {
                _new_argv.emplace_back(_exe);
                _new_argv.emplace_back(strdup("--"));
                _injected = true;
            }
            _new_argv.emplace_back(itr);
        }

        if(!_injected)
        {
            throw std::runtime_error(
                join("", "omnitrace-causal was unable to match \"", launcher,
                     "\" to any arguments on the command line: \"",
                     join(array_config{ " ", "", "" }, _argv), "\""));
        }

        std::swap(_argv, _new_argv);
    }
}

void
prepare_environment_for_run(std::vector<char*>& _env)
{
    if(launcher.empty())
    {
        update_env(_env, "LD_PRELOAD",
                   get_realpath(get_internal_libpath("libomnitrace-dl.so")), true);
    }
}

std::string
get_internal_libpath(const std::string& _lib)
{
    auto _exe = std::string_view{ realpath("/proc/self/exe", nullptr) };
    auto _pos = _exe.find_last_of('/');
    auto _dir = std::string{ "./" };
    if(_pos != std::string_view::npos) _dir = _exe.substr(0, _pos);
    return omnitrace::common::join("/", _dir, "..", "lib", _lib);
}

void
print_updated_environment(std::vector<char*> _env, std::string_view _prefix)
{
    if(get_verbose() < 0) return;

    std::sort(_env.begin(), _env.end(), [](auto* _lhs, auto* _rhs) {
        if(!_lhs) return false;
        if(!_rhs) return true;
        return std::string_view{ _lhs } < std::string_view{ _rhs };
    });

    std::vector<std::string_view> _updates = {};
    std::vector<std::string_view> _general = {};

    for(auto* itr : _env)
    {
        if(itr == nullptr) continue;

        auto _is_omni = (std::string_view{ itr }.find("OMNITRACE") == 0);
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
        stream(std::cerr, color::source()) << _prefix << itr << "\n";
    for(auto& itr : _updates)
        stream(std::cerr, color::source()) << _prefix << itr << "\n";

    std::cerr << color::end() << std::flush;
}

template <typename Tp>
void
update_env(std::vector<char*>& _environ, std::string_view _env_var, Tp&& _env_val,
           bool _append, std::string_view _join_delim)
{
    updated_envs.emplace(_env_var);

    auto _key = join("", _env_var, "=");
    for(auto& itr : _environ)
    {
        if(!itr) continue;
        if(std::string_view{ itr }.find(_key) == 0)
        {
            if(_append)
            {
                if(std::string_view{ itr }.find(join("", _env_val)) ==
                   std::string_view::npos)
                {
                    auto _val = std::string{ itr }.substr(_key.length());
                    free(itr);
                    itr = strdup(
                        join('=', _env_var, join(_join_delim, _env_val, _val)).c_str());
                }
            }
            else
            {
                free(itr);
                itr = strdup(omnitrace::common::join('=', _env_var, _env_val).c_str());
            }
            return;
        }
    }
    _environ.emplace_back(
        strdup(omnitrace::common::join('=', _env_var, _env_val).c_str()));
}

template <typename Tp>
void
add_default_env(std::vector<char*>& _environ, std::string_view _env_var, Tp&& _env_val)
{
    auto _key = join("", _env_var, "=");
    for(auto& itr : _environ)
    {
        if(!itr) continue;
        if(std::string_view{ itr }.find(_key) == 0) return;
    }

    updated_envs.emplace(_env_var);
    _environ.emplace_back(
        strdup(omnitrace::common::join('=', _env_var, _env_val).c_str()));
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
parse_args(int argc, char** argv, std::vector<char*>& _env,
           std::vector<std::map<std::string_view, std::string>>& _causal_envs)
{
    using parser_t     = argparse::argument_parser;
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

    const auto* _desc = R"desc(
    Causal profiling usually requires multiple runs to reliably resolve the speedup estimates.
    This executable is designed to streamline that process.
    For example (assume all commands end with '-- <exe> <args>'):

        omnitrace-causal -n 5 -- <exe>                  # runs <exe> 5x with causal profiling enabled

        omnitrace-causal -s 0 5,10,15,20                # runs <exe> 2x with virtual speedups:
                                                        #   - 0
                                                        #   - randomly selected from 5, 10, 15, and 20

        omnitrace-causal -F func_A func_B func_(A|B)    # runs <exe> 3x with the function scope limited to:
                                                        #   1. func_A
                                                        #   2. func_B
                                                        #   3. func_A or func_B
    General tips:
    - Insert progress points at hotspots in your code or use omnitrace's runtime instrumentation
        - Note: binary rewrite will produce a incompatible new binary
    - Collect a flat profile via sampling
        - E.g., omnitrace-sample -F -- <exe> <args>
        - Inspect sampling_wall_clock.txt and sampling_cpu_clock.txt for functions to target
    - Run omnitrace-causal in "function" mode first (does not require debug info)
    - Run omnitrace-causal in "line" mode when you are targeting one function (requires debug info)
        - Preferably, use predictions from the "function" mode to determine which function to target
    - Limit the virtual speedups to a smaller pool, e.g., 0,5,10,25,50, to get reliable predictions quicker
    - Make use of the binary, source, and function scope to limit the functions/lines selected for experiments
        - Note: source scope requires debug info
    )desc";

    auto parser = parser_t{ basename(argv[0]), _desc };

    parser.on_error([](parser_t&, const parser_err_t& _err) {
        stream(std::cerr, color::fatal()) << _err << "\n";
        exit(EXIT_FAILURE);
    });

    auto _add_separator = [&](std::string _v, const std::string& _desc) {
        parser.add_argument({ "" }, "");
        parser
            .add_argument({ join("", "[", _v, "]") },
                          (_desc.empty()) ? _desc : join({ "", "(", ")" }, _desc))
            .color(color::info());
        parser.add_argument({ "" }, "");
    };

    parser.enable_help();
    parser.enable_version("omnitrace-causal", "v" OMNITRACE_VERSION_STRING,
                          OMNITRACE_GIT_DESCRIBE, OMNITRACE_GIT_REVISION);

    auto _cols = std::get<0>(console::get_columns());
    if(_cols > parser.get_help_width() + 8)
        parser.set_description_width(
            std::min<int>(_cols - parser.get_help_width() - 8, 120));

    _add_separator("DEBUG OPTIONS", "");
    parser.add_argument({ "--monochrome" }, "Disable colorized output")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            auto _colorized = !p.get<bool>("monochrome");
            colorized()     = _colorized;
            p.set_use_color(_colorized);
            update_env(_env, "OMNITRACE_COLORIZED_LOG", (_colorized) ? "1" : "0");
            update_env(_env, "COLORIZED_LOG", (_colorized) ? "1" : "0");
        });
    parser.add_argument({ "--debug" }, "Debug output")
        .max_count(1)
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_DEBUG", p.get<bool>("debug"));
        });
    parser.add_argument({ "-v", "--verbose" }, "Verbose output")
        .count(1)
        .action([&](parser_t& p) {
            auto _v = p.get<int>("verbose");
            verbose = _v;
            update_env(_env, "OMNITRACE_VERBOSE", _v);
        });

    std::string _config_file      = {};
    std::string _config_folder    = "omnitrace-causal-config";
    bool        _generate_configs = false;
    bool        _add_defaults     = true;

    _add_separator("GENERAL OPTIONS", "");
    parser.add_argument({ "-c", "--config" }, "Base configuration file")
        .min_count(0)
        .dtype("filepath")
        .action([&](parser_t& p) {
            _config_file =
                join(array_config{ ":" }, p.get<std::vector<std::string>>("config"));
        });
    parser
        .add_argument(
            { "-l", "--launcher" },
            "When running MPI jobs, omnitrace-causal needs to be *before* the executable "
            "which launches the MPI processes (i.e. before `mpirun`, `srun`, etc.). Pass "
            "the name of the target executable (or a regex for matching to the name of "
            "the target) for causal profiling, e.g., `omnitrace-causal -l foo -- mpirun "
            "-n 4 foo`. This ensures that the omnitrace library is LD_PRELOADed on the "
            "proper target")
        .count(1)
        .dtype("executable")
        .action([&](parser_t& p) { launcher = p.get<std::string>("launcher"); });
    parser
        .add_argument({ "-g", "--generate-configs" },
                      "Generate config files instead of passing environment variables "
                      "directly. If no arguments are provided, the config files will be "
                      "placed in ${PWD}/omnitrace-causal-config folder")
        .min_count(0)
        .max_count(1)
        .dtype("folder")
        .action([&](parser_t& p) {
            _generate_configs = true;
            auto _dir         = p.get<std::string>("generate-configs");
            if(!_dir.empty()) _config_folder = std::move(_dir);
            if(!filepath::exists(_config_folder)) filepath::makedir(_config_folder);
        });
    parser
        .add_argument({ "--no-defaults" },
                      "Do not activate default features which are recommended for causal "
                      "profiling. For example: PID-tagging of output files and "
                      "timestamped subdirectories are disabled by default. Kokkos tools "
                      "support is added by default (OMNITRACE_USE_KOKKOSP=ON) because, "
                      "for Kokkos applications, the Kokkos-Tools callbacks are used for "
                      "progress points. Activation of OpenMP tools support is similar")
        .min_count(0)
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) { _add_defaults = !p.get<bool>("no-defaults"); });

    _add_separator("CAUSAL PROFILING OPTIONS (General)",
                   "These settings will be applied to all causal profiling runs");
    parser.add_argument({ "-m", "--mode" }, "Causal profiling mode")
        .count(1)
        .dtype("string")
        .choices({ "function", "line" })
        .choice_alias("function", { "func" })
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_CAUSAL_MODE", p.get<std::string>("mode"));
        });

    parser
        .add_argument({ "-o", "--output-name" },
                      "Output filename of causal profiling data w/o extension")
        .min_count(1)
        .dtype("filename")
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_CAUSAL_FILE", p.get<std::string>("output-name"));
        });

    bool _reset = false;

    parser
        .add_argument({ "-r", "--reset" },
                      "Overwrite any existing experiment results during the first run")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) { _reset = p.get<bool>("reset"); });

    parser
        .add_argument({ "-e", "--end-to-end" },
                      "Single causal experiment for the entire application runtime")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_CAUSAL_END_TO_END", p.get<bool>("end-to-end"));
        });

    parser
        .add_argument({ "-w", "--wait" },
                      "Set the wait time (i.e. delay) before starting the first causal "
                      "experiment (in seconds)")
        .count(1)
        .dtype("seconds")
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_CAUSAL_DELAY", p.get<double>("wait"));
        });

    parser
        .add_argument(
            { "-d", "--duration" },
            "Set the length of time (in seconds) to perform causal experimentationafter "
            "the first experiment is started. Once this amount of time has elapsed, no "
            "more causal experiments will be started but any currently running "
            "experiment will be allowed to finish.")
        .count(1)
        .dtype("seconds")
        .action([&](parser_t& p) {
            update_env(_env, "OMNITRACE_CAUSAL_DURATION", p.get<double>("duration"));
        });

    int64_t _niterations       = 1;
    auto    _virtual_speedups  = std::vector<std::string>{};
    auto    _function_scopes   = std::vector<std::string>{};
    auto    _binary_scopes     = std::vector<std::string>{};
    auto    _source_scopes     = std::vector<std::string>{};
    auto    _function_excludes = std::vector<std::string>{};
    auto    _binary_excludes   = std::vector<std::string>{};
    auto    _source_excludes   = std::vector<std::string>{};

    parser
        .add_argument({ "-n", "--iterations" },
                      "Number of times to repeat the combination of run configurations")
        .count(1)
        .dtype("int")
        .action([&](parser_t& p) { _niterations = p.get<int64_t>("iterations"); });

    _add_separator(
        "CAUSAL PROFILING OPTIONS (Combinatorial)",
        "Each individual argument to these options will multiply the number runs by the "
        "number of arguments and the number of iterations. E.g. -n 2 -B \"MAIN\" -F "
        "\"foo\" \"bar\" will produce 4 runs: 2 iterations x 1 binary scope x 2 function "
        "scopes (MAIN+foo, MAIN+bar, MAIN+foo, MAIN+bar)");

    parser
        .add_argument({ "-s", "--speedups" },
                      "Pool of virtual speedups to sample from during experimentation. "
                      "Each space designates a group and multiple speedups can be "
                      "grouped together by commas, e.g. -s 0 0,10,20-50 is two groups: "
                      "group #1 is '0' and group #2 is '0 10 20 25 30 35 40 45 50'")
        .min_count(0)
        .max_count(-1)
        .dtype("integers")
        .action([&](parser_t& p) {
            _virtual_speedups = p.get<std::vector<std::string>>("speedups");
        });

    parser
        .add_argument({ "-B", "--binary-scope" },
                      "Restricts causal experiments to the binaries matching the list of "
                      "regular expressions. Each space designates a group and multiple "
                      "scopes can be grouped together with a semi-colon")
        .min_count(0)
        .max_count(-1)
        .dtype("integers")
        .action([&](parser_t& p) {
            _binary_scopes = p.get<std::vector<std::string>>("binary-scope");
        });

    parser
        .add_argument({ "-S", "--source-scope" },
                      "Restricts causal experiments to the source files or source file + "
                      "lineno pairs (i.e. <file> or <file>:<line>) matching the list of "
                      "regular expressions. Each space designates a group and multiple "
                      "scopes can be grouped together with a semi-colon")
        .min_count(0)
        .max_count(-1)
        .dtype("integers")
        .action([&](parser_t& p) {
            _source_scopes = p.get<std::vector<std::string>>("source-scope");
        });

    parser
        .add_argument(
            { "-F", "--function-scope" },
            "Restricts causal experiments to the functions matching the list of "
            "regular expressions. Each space designates a group and multiple "
            "scopes can be grouped together with a semi-colon")
        .min_count(0)
        .max_count(-1)
        .dtype("regex-list")
        .action([&](parser_t& p) {
            _function_scopes = p.get<std::vector<std::string>>("function-scope");
        });

    parser
        .add_argument(
            { "-BE", "--binary-exclude" },
            "Excludes causal experiments from being performed on the binaries matching "
            "the list of regular expressions. Each space designates a group and multiple "
            "excludes can be grouped together with a semi-colon")
        .min_count(0)
        .max_count(-1)
        .dtype("integers")
        .action([&](parser_t& p) {
            _binary_excludes = p.get<std::vector<std::string>>("binary-exclude");
        });

    parser
        .add_argument(
            { "-SE", "--source-exclude" },
            "Excludes causal experiments from being performed on the code from the "
            "source files or source file + lineno pair (i.e. <file> or <file>:<line>) "
            "matching the list of regular expressions. Each space designates a group and "
            "multiple excludes can be grouped together with a semi-colon")
        .min_count(0)
        .max_count(-1)
        .dtype("integers")
        .action([&](parser_t& p) {
            _source_excludes = p.get<std::vector<std::string>>("source-exclude");
        });

    parser
        .add_argument(
            { "-FE", "--function-exclude" },
            "Excludes causal experiments from being performed on the functions matching "
            "the list of regular expressions. Each space designates a group and multiple "
            "excludes can be grouped together with a semi-colon")
        .min_count(0)
        .max_count(-1)
        .dtype("regex-list")
        .action([&](parser_t& p) {
            _function_excludes = p.get<std::vector<std::string>>("function-exclude");
        });

#if OMNITRACE_HIP_VERSION > 0 && OMNITRACE_HIP_VERSION < 50300
    update_env(_env, "HSA_ENABLE_INTERRUPT", 0);
#endif

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

    if(_niterations < 1) _niterations = 1;
    auto _get_size = [](const auto& _v) { return std::max<size_t>(_v.size(), 1); };

    auto _causal_envs_tmp = std::vector<std::map<std::string_view, std::string>>{};
    auto _fill = [&_causal_envs_tmp](std::string_view _env_var, const auto& _data,
                                     bool _quote) {
        if(_data.empty()) return;
        if(_causal_envs_tmp.empty()) _causal_envs_tmp.emplace_back();
        auto _tmp = _causal_envs_tmp;
        _causal_envs_tmp.clear();
        _causal_envs_tmp.reserve(_data.size() * _tmp.size());
        for(auto ditr : _data)
        {
            if(_quote)
            {
                ditr.insert(0, "\"");
                ditr += "\"";
            }

            // duplicate the env, add the env variable, emplace back
            for(auto itr : _tmp)
            {
                itr[_env_var] = ditr;
                _causal_envs_tmp.emplace_back(itr);
            }
        }
    };

    if(_add_defaults)
    {
        add_default_env(_env, "OMNITRACE_TIME_OUTPUT", false);
        add_default_env(_env, "OMNITRACE_USE_PID", false);
        add_default_env(_env, "OMNITRACE_USE_KOKKOSP", true);

#if defined(OMNITRACE_USE_OMPT) && OMNITRACE_USE_OMPT > 0
        add_default_env(_env, "OMNITRACE_USE_OMPT", true);
#endif

#if(defined(OMNITRACE_USE_MPI) && OMNITRACE_USE_MPI > 0) ||                              \
    (defined(OMNITRACE_USE_MPI_HEADERS) && OMNITRACE_USE_MPI_HEADERS > 0)
        add_default_env(_env, "OMNITRACE_USE_MPIP", true);
#endif

#if defined(OMNITRACE_USE_ROCTRACER) && OMNITRACE_USE_ROCTRACER > 0
        add_default_env(_env, "OMNITRACE_ROCTRACER_HIP_API", true);
        add_default_env(_env, "OMNITRACE_ROCTRACER_HSA_API", true);
#endif

#if defined(OMNITRACE_USE_RCCL) && OMNITRACE_USE_RCCL > 0
        add_default_env(_env, "OMNITRACE_USE_RCCLP", true);
#endif
    }

    _fill("OMNITRACE_CAUSAL_BINARY_EXCLUDE", _binary_excludes, _generate_configs);
    _fill("OMNITRACE_CAUSAL_SOURCE_EXCLUDE", _source_excludes, _generate_configs);
    _fill("OMNITRACE_CAUSAL_FUNCTION_EXCLUDE", _function_excludes, _generate_configs);

    _fill("OMNITRACE_CAUSAL_BINARY_SCOPE", _binary_scopes, _generate_configs);
    _fill("OMNITRACE_CAUSAL_SOURCE_SCOPE", _source_scopes, _generate_configs);
    _fill("OMNITRACE_CAUSAL_FUNCTION_SCOPE", _function_scopes, _generate_configs);

    _fill("OMNITRACE_CAUSAL_FIXED_SPEEDUP", _virtual_speedups, false);

    // make sure at least one env exists
    if(_causal_envs_tmp.empty()) _causal_envs_tmp.emplace_back();

    // duplicate for the number of iterations
    _causal_envs.clear();
    _causal_envs.reserve(_niterations * _causal_envs_tmp.size());
    for(int64_t i = 0; i < _niterations; ++i)
    {
        for(const auto& itr : _causal_envs_tmp)
            _causal_envs.emplace_back(itr);
    }

    if(_generate_configs)
    {
        auto _is_omni_cfg = [](std::string_view itr) {
            return (itr.find("OMNITRACE") == 0 && itr.find("OMNITRACE_MODE") != 0 &&
                    itr.find("OMNITRACE_CONFIG_FILE") != 0 &&
                    itr.find('=') < itr.length());
        };

        auto _omni_env = std::map<std::string, std::string>{};
        for(auto* itr : _env)
        {
            if(_is_omni_cfg(itr))
            {
                auto _env_var = std::string{ itr };
                auto _pos     = _env_var.find('=');
                auto _env_val = _env_var.substr(_pos + 1);
                _env_var      = _env_var.substr(0, _pos);
                _omni_env.emplace(_env_var, _env_val);
            }
        }

        _env.erase(std::remove_if(_env.begin(), _env.end(), _is_omni_cfg), _env.end());

        _causal_envs_tmp = std::move(_causal_envs);
        _causal_envs.clear();
        auto _write_config =
            [_omni_env](std::ostream&                                  _os,
                        const std::map<std::string_view, std::string>& _data) {
                size_t _width = 0;
                for(const auto& itr : _omni_env)
                    _width = std::max(_width, itr.first.length());

                for(const auto& itr : _data)
                    _width = std::max(_width, itr.first.length());

                _os << "# omnitrace common settings\n";
                for(const auto& itr : _omni_env)
                    _os << std::setw(_width + 1) << std::left << itr.first << " = "
                        << itr.second << "\n";

                _os << "\n# omnitrace causal settings\n";
                for(const auto& itr : _data)
                    _os << std::setw(_width + 1) << std::left << itr.first << " = "
                        << itr.second << "\n";
            };

        int nwidth = (std::log10(_causal_envs_tmp.size()) + 1);
        for(size_t i = 0; i < _causal_envs_tmp.size(); ++i)
        {
            std::stringstream fname{};
            fname.fill('0');
            fname << _config_folder << "/causal-" << std::setw(nwidth) << i << ".cfg";
            std::ofstream _ofs{ fname.str() };
            _write_config(_ofs, _causal_envs_tmp.at(i));
            auto _cfg_name = (_config_file.empty())
                                 ? fname.str()
                                 : join(array_config{ ":" }, _config_file, fname.str());
            auto _cfg =
                std::map<std::string_view, std::string>{ { "OMNITRACE_CONFIG_FILE",
                                                           _cfg_name } };
            _causal_envs.emplace_back(_cfg);
        }
    }

    if(_reset)
        _causal_envs.front().emplace(std::string_view{ "OMNITRACE_CAUSAL_FILE_RESET" },
                                     std::string{ "true" });

    return _outv;
}

// explicit instantiation for usage in omnitrace-causal.cpp
template void
update_env(std::vector<char*>&, std::string_view, const std::string& _env_val,
           bool _append, std::string_view);
