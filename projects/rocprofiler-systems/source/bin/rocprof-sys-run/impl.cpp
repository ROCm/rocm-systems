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

#include "rocprof-sys-run.hpp"

#include "common/defines.h"
#include "common/delimit.hpp"
#include "common/environment.hpp"
#include "common/join.hpp"
#include "common/setup.hpp"
#include "core/argparse.hpp"
#include "core/config.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"

#include <timemory/environment.hpp>
#include <timemory/environment/types.hpp>
#include <timemory/log/color.hpp>
#include <timemory/settings/types.hpp>
#include <timemory/settings/vsettings.hpp>
#include <timemory/signals/signal_handlers.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/console.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/join.hpp>

#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace color    = ::tim::log::color;
namespace filepath = ::tim::filepath;  // NOLINT
namespace console  = ::tim::utility::console;
namespace argparse = ::tim::argparse;
namespace signals  = ::tim::signals;
using settings     = ::rocprofsys::settings;
using namespace ::timemory::join;
using ::tim::get_env;
using ::tim::log::stream;

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
std::string
get_internal_libpath(const std::string& _lib)
{
    auto _exe = std::string_view{ realpath("/proc/self/exe", nullptr) };
    auto _pos = _exe.find_last_of('/');
    auto _dir = std::string{ "./" };
    if(_pos != std::string_view::npos) _dir = _exe.substr(0, _pos);
    return rocprofsys::common::join("/", _dir, "..", "lib", _lib);
}

parser_data_t&
get_initial_environment(parser_data_t& _data)
{
    if(environ != nullptr)
    {
        int idx = 0;
        while(environ[idx] != nullptr)
        {
            auto* _v = environ[idx++];
            _data.initial.emplace(_v);
            _data.current.emplace_back(strdup(_v));
        }
    }

    return _data;
}

int
get_verbose(parser_data_t& _data)
{
    auto& verbose = _data.verbose;
    verbose       = get_env("ROCPROFSYS_CAUSAL_VERBOSE",
                            get_env<int>("ROCPROFSYS_VERBOSE", verbose, false));
    auto _debug   = get_env("ROCPROFSYS_CAUSAL_DEBUG",
                            get_env<bool>("ROCPROFSYS_DEBUG", false, false));
    if(_debug) verbose += 8;
    return verbose;
}

std::string
get_realpath(const std::string& _v)
{
    auto* _tmp = realpath(_v.c_str(), nullptr);
    auto  _ret = std::string{ _tmp };
    free(_tmp);
    return _ret;
}

auto
toggle_suppression(std::tuple<bool, bool> _inp)
{
    auto _out =
        std::make_tuple(settings::suppress_config(), settings::suppress_parsing());
    std::tie(settings::suppress_config(), settings::suppress_parsing()) = _inp;
    return _out;
}

// disable suppression when exe loads but store original values for restoration later
auto initial_suppression = toggle_suppression({ true, true });
}  // namespace

void
print_command(const parser_data_t& _data, std::string_view _prefix)
{
    auto        verbose = _data.verbose;
    const auto& _argv   = _data.command;
    if(verbose >= 1)
        stream(std::cout, color::info())
            << _prefix << "Executing '" << join(array_config{ " " }, _argv) << "'...\n";

    std::cerr << color::end() << std::flush;
}

void
prepare_command_for_run(char* _exe, parser_data_t& _data)
{
    if(!_data.launcher.empty())
    {
        bool _injected = false;
        auto _new_argv = std::vector<char*>{};
        for(auto* itr : _data.command)
        {
            if(!_injected && std::regex_search(itr, std::regex{ _data.launcher }))
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
                join("", "rocprof-sys-run was unable to match \"", _data.launcher,
                     "\" to any arguments on the command line: \"",
                     join(array_config{ " ", "", "" }, _data.command), "\""));
        }

        std::swap(_data.command, _new_argv);
    }
}

void
prepare_environment_for_run(parser_data_t& _data)
{
    if(_data.launcher.empty())
    {
        rocprofsys::argparse::add_ld_preload(_data);
        rocprofsys::argparse::add_ld_library_path(_data);
    }
}

void
print_updated_environment(parser_data_t& _data, std::string_view _prefix)
{
    auto _verbose = get_verbose(_data);

    if(_verbose < 0) return;

    auto        _env          = _data.current;
    const auto& _updated_envs = _data.updated;

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

        auto _is_omni = (std::string_view{ itr }.find("ROCPROFSYS") == 0);
        auto _updated = false;
        for(const auto& vitr : _updated_envs)
        {
            if(std::string_view{ itr }.find(vitr) == 0)
            {
                _updated = true;
                break;
            }
        }

        if(_updated)
            _updates.emplace_back(itr);
        else if(_verbose >= 1 && _is_omni)
            _general.emplace_back(itr);
    }

    if(_general.size() + _updates.size() == 0 || _verbose < 0) return;

    std::cerr << std::endl;

    for(auto& itr : _general)
        stream(std::cerr, color::source()) << _prefix << itr << "\n";
    for(auto& itr : _updates)
        stream(std::cerr, color::source()) << _prefix << itr << "\n";

    std::cerr << color::end() << std::flush;
}

parser_data_t&
parse_args(int argc, char** argv, parser_data_t& _parser_data, bool& _fork_exec)
{
    get_initial_environment(_parser_data);

    bool _do_parse_args = false;
    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string_view{ argv[i] };
        if(_arg == "--" || _arg == "-?" || _arg == "-h" || _arg == "--help" ||
           _arg == "--version")
            _do_parse_args = true;
    }

    if(!_do_parse_args && argc > 1 && std::string_view{ argv[1] }.find('-') == 0)
        _do_parse_args = true;

    if(!_do_parse_args) return parse_command(argc, argv, _parser_data);

    using parser_t     = argparse::argument_parser;
    using parser_err_t = typename parser_t::result_type;

    toggle_suppression(initial_suppression);
    rocprofsys::argparse::init_parser(_parser_data);

    // no need for backtraces
    signals::disable_signal_detection(signals::signal_settings::get_enabled());

    auto help_check = [](parser_t& p, int _argc, char** _argv) {
        std::set<std::string> help_args = { "-h", "--help", "-?" };
        return (p.exists("help") || _argc == 1 ||
                (_argc > 1 && help_args.find(_argv[1]) != help_args.end()));
    };

    const auto* _desc = R"desc(
    Command line interface to rocprof-sys configuration.
    )desc";

    auto parser = parser_t{ basename(argv[0]), _desc };

    parser.on_error([](parser_t&, const parser_err_t& _err) {
        stream(std::cerr, color::fatal()) << _err << "\n";
        exit(EXIT_FAILURE);
    });

    parser.enable_help("", "Usage: rocprof-sys-run <OPTIONS> -- <COMMAND> <ARGS>");
    parser.enable_version("rocprof-sys-run", ROCPROFSYS_ARGPARSE_VERSION_INFO);

    auto _cols = std::get<0>(console::get_columns());
    if(_cols > parser.get_help_width() + 8)
        parser.set_description_width(
            std::min<int>(_cols - parser.get_help_width() - 8, 120));

    // disable options related to causal profiling
    _parser_data.processed_groups.emplace("causal");

    rocprofsys::argparse::add_core_arguments(parser, _parser_data);
    rocprofsys::argparse::add_extended_arguments(parser, _parser_data);

    parser.start_group("EXECUTION OPTIONS", "");
    parser.add_argument({ "--fork" }, "Execute via fork + execvpe instead of execvpe")
        .min_count(0)
        .max_count(1)
        .dtype("boolean")
        .action([&](parser_t& p) { _fork_exec = p.get<bool>("fork"); });

    auto  _inpv = std::vector<char*>{};
    auto& _outv = _parser_data.command;
    bool  _hash = false;
    for(int i = 0; i < argc; ++i)
    {
        if(argv[i] == nullptr)
        {
            continue;
        }
        else if(_hash)
        {
            _outv.emplace_back(strdup(argv[i]));
        }
        else if(std::string_view{ argv[i] } == "--")
        {
            _hash = true;
        }
        else
        {
            _inpv.emplace_back(strdup(argv[i]));
        }
    }

    auto _cerr = parser.parse_args(_inpv.size(), _inpv.data());
    if(_cerr)
    {
        std::cerr << _cerr.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    tim::log::monochrome() = _parser_data.monochrome;

    return _parser_data;
}

parser_data_t&
parse_command(int argc, char** argv, parser_data_t& _parser_data)
{
    toggle_suppression(initial_suppression);
    rocprofsys::argparse::init_parser(_parser_data);

    // no need for backtraces
    signals::disable_signal_detection(signals::signal_settings::get_enabled());

    auto& _outv = _parser_data.command;
    bool  _hash = false;
    for(int i = 1; i < argc; ++i)
    {
        _outv.emplace_back(strdup(argv[i]));
    }

    return _parser_data;
}
