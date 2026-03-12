// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "preset_config.hpp"
#include "rocprof-sys-run.hpp"

#include "common/common_utils.hpp"
#include "common/defines.h"
#include "common/environment.hpp"
#include "common/path.hpp"
#include "core/argparse.hpp"
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

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>

namespace color    = ::tim::log::color;
namespace filepath = ::tim::filepath;  // NOLINT
namespace console  = ::tim::utility::console;
namespace argparse = ::tim::argparse;
namespace signals  = ::tim::signals;
namespace path     = rocprofsys::common::path;
using settings     = ::rocprofsys::settings;
using namespace ::timemory::join;
using ::tim::get_env;
using ::tim::log::stream;

namespace
{
using rocprofsys::common::update_mode;

auto original_envs = std::unordered_set<std::string>{};

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
            original_envs.emplace(_v);
        }
    }

    auto _libexecpath = path::realpath(path::get_internal_script_path());
    if(!_libexecpath.empty())
    {
        rocprofsys::common::update_env(_data.current, "ROCPROFSYS_SCRIPT_PATH",
                                       _libexecpath, update_mode::REPLACE, ":",
                                       _data.updated, original_envs);
    }

    const bool verbose = (get_verbose(_data) > 0);
    if(auto llvm_dir = rocprofsys::common::discover_llvm_libdir_for_ompt(verbose);
       !llvm_dir.empty())
    {
        rocprofsys::common::update_env(_data.current, "LD_LIBRARY_PATH", llvm_dir,
                                       update_mode::APPEND, ":", _data.updated,
                                       original_envs);
        auto        current_ld = getenv("LD_LIBRARY_PATH");
        std::string new_ld     = current_ld ? (llvm_dir + ":" + current_ld) : llvm_dir;
        setenv("LD_LIBRARY_PATH", new_ld.c_str(), 1);
    }

    return _data;
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

    rocprofsys::argparse::add_torch_library_path(_data, _data.verbose > 0);

    rocprofsys::common::consolidate_env_entries(_data.current);
}

parser_data_t&
parse_args(int argc, char** argv, parser_data_t& _parser_data, bool& _fork_exec)
{
    using parser_t     = argparse::argument_parser;
    using parser_err_t = typename parser_t::result_type;

    auto help_check = [](parser_t& p, int _argc, char** _argv) {
        std::unordered_set<std::string> help_args = { "-h", "--help", "-?" };
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

    toggle_suppression(initial_suppression);
    rocprofsys::argparse::init_parser(_parser_data);

    // no need for backtraces
    signals::disable_signal_detection(signals::signal_settings::get_enabled());

    auto parser = parser_t{ basename(argv[0]), rocprofsys::run::descriptions::main };

    parser.on_error([](parser_t&, const parser_err_t& _err) {
        stream(std::cerr, color::fatal()) << _err << "\n";
        exit(EXIT_FAILURE);
    });

    parser.enable_help();
    parser.enable_version("rocprof-sys-run", ROCPROFSYS_ARGPARSE_VERSION_INFO);

    auto _cols = std::get<0>(console::get_columns());
    if(_cols > parser.get_help_width() + 8)
        parser.set_description_width(
            std::min<int>(_cols - parser.get_help_width() - 8, 120));

    // disable options related to causal profiling
    _parser_data.processed_groups.emplace("causal");

    rocprofsys::argparse::add_core_arguments(parser, _parser_data);
    rocprofsys::argparse::add_extended_arguments(parser, _parser_data);

    parser.start_group("PRESET MODES",
                       "Simplified profiling presets for common use cases");

    for(const auto& preset : rocprofsys::run::PRESETS)
    {
        auto preset_name = std::string(preset.name);
        parser
            .add_argument({ std::string("--") + preset_name },
                          std::string(preset.description))
            .max_count(1)
            .dtype("bool")
            .action([&, preset_name, env_vars = preset.env_vars,
                     env_settings = preset.env_settings](parser_t& p) {
                if(p.get<bool>(preset_name))
                {
                    for(const auto& var : env_vars)
                        _parser_data.updated.emplace(var);

                    for(const auto& [key, value] : env_settings)
                        tim::set_env(std::string(key).c_str(), std::string(value).c_str(),
                                     0);

                    if(preset_name == "detailed" || preset_name == "workload-trace")
                    {
                        auto* hip_visible_devices = getenv("HIP_VISIBLE_DEVICES");
                        if(hip_visible_devices && strlen(hip_visible_devices) > 0)
                        {
                            _parser_data.updated.emplace("ROCPROFSYS_SAMPLING_GPUS");
                            tim::set_env("ROCPROFSYS_SAMPLING_GPUS",
                                         std::string(hip_visible_devices).c_str(), 0);
                        }
                    }
                }
            });
    }

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
    if(help_check(parser, argc, argv))
        help_action(parser);
    else if(_cerr)
        throw std::runtime_error(_cerr.what());

    tim::log::monochrome() = _parser_data.monochrome;

    auto active_presets = rocprofsys::common_utils::collect_active_presets(
        parser, { "balanced", "profile-only", "detailed", "trace-hpc", "workload-trace",
                  "sys-trace", "runtime-trace", "trace-gpu", "trace-openmp",
                  "profile-mpi", "trace-hw-counters" });

    const auto are_valid_presets =
        rocprofsys::common_utils::validate_preset_modes(active_presets);

    if(!are_valid_presets)
    {
        exit(EXIT_FAILURE);
    }

    rocprofsys::common_utils::warn_if_gpu_preset_without_rocm(active_presets);

    if(!active_presets.empty() && _parser_data.verbose >= 1)
    {
        rocprofsys::common_utils::print_pre_execution_info("run", active_presets[0]);
    }

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
