// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys-run.hpp"

#include "common/argument_registration.hpp"
#include "common/common_utils.hpp"
#include "common/defines.h"
#include "common/env_vars.hpp"
#include "common/environment.hpp"
#include "common/json_config.hpp"
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
#include <fstream>
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
namespace env      = rocprofsys::env_vars;
using settings     = ::rocprofsys::settings;
using namespace ::timemory::join;
using ::tim::get_env;
using ::tim::log::stream;

int
get_verbose(parser_data_t& _data)
{
    auto&       verbose    = _data.verbose;
    const auto* _log_level = std::getenv(env::LOG_LEVEL.data());
    if(_log_level != nullptr) verbose = env::log_level_to_verbose(_log_level);
    return verbose;
}

namespace
{
using rocprofsys::common::update_mode;

auto original_envs = std::unordered_set<std::string>{};

// Export configuration to JSON file or stdout
void
export_config(const parser_data_t& _parser_data, const std::string& preset_name,
              const std::string& output_file = "")
{
    rocprofsys::common_utils::export_config(_parser_data.current, original_envs,
                                            preset_name, "run", output_file);
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
        rocprofsys::common::update_env(_data.current, env::SCRIPT_PATH, _libexecpath,
                                       update_mode::REPLACE, ":", _data.updated,
                                       original_envs);
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

        rocprofsys::common_utils::dispatch_help(p, "run", _pec);
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

    const auto* _desc = R"desc(
Execute instrumented binaries with ROCm Systems Profiler configuration.
QUICK REFERENCE:
  Presets:  --preset=balanced (default), --preset=profile-only, --preset=trace-hpc, --preset=workload-trace
  Domains:  --gpu, --rocm, --cpu, --parallel (composable with presets)
  Output:   Results saved to rocprof-sys-output/ directory
  Visualize: Open perfetto-trace.proto in https://ui.perfetto.dev
EXAMPLES:
  Quick Start:
    rocprof-sys-run --preset=balanced -- ./myapp.inst
  Workload-Specific Presets:
    rocprof-sys-run --preset=trace-hpc -- ./hpc_app.inst         # HPC/MPI/OpenMP
    rocprof-sys-run --preset=workload-trace -- ./gpu_app.inst    # AI/ML/GPU workloads
    rocprof-sys-run --preset=profile-only -- ./myapp.inst        # Minimal overhead
  Domain Flags (composable):
    rocprof-sys-run --gpu -- ./myapp.inst                        # GPU metrics
    rocprof-sys-run --preset=balanced --gpu=temp,power -- ./app  # Preset + specific GPU metrics
    rocprof-sys-run --rocm=hip,kernel --cpu=100 -- ./app         # ROCm APIs + CPU sampling
    rocprof-sys-run --parallel=mpi,openmp -- ./app               # MPI + OpenMP profiling
  Custom Configuration File:
    rocprof-sys-run --preset=./my-config.json -- ./myapp.inst
  Export Configuration:
    rocprof-sys-run --preset=balanced --gpu --export-config > my-config.json
INSTRUMENTATION WORKFLOW:
  1. Instrument: rocprof-sys-instrument -o app.inst -- ./app
  2. Run:        rocprof-sys-run --preset=balanced -- ./app.inst
  3. Analyze:    cat rocprof-sys-output/wall_clock.txt
    )desc";

    auto parser = parser_t{ basename(argv[0]), _desc };

    parser.on_error([](parser_t&, const parser_err_t& _err) {
        stream(std::cerr, color::fatal()) << _err << "\n";
        exit(EXIT_FAILURE);
    });

    parser.enable_help().count(-1).min_count(0).max_count(1).dtype("topic");
    parser.enable_version("rocprof-sys-run", ROCPROFSYS_ARGPARSE_VERSION_INFO);

    auto _cols = std::get<0>(console::get_columns());
    if(_cols > parser.get_help_width() + 8)
        parser.set_description_width(
            std::min<int>(_cols - parser.get_help_width() - 8, 120));

    // disable options related to causal profiling
    _parser_data.processed_groups.emplace("causal");

    rocprofsys::argparse::add_core_arguments(parser, _parser_data);
    rocprofsys::argparse::add_extended_arguments(parser, _parser_data);

    // Track preset and domain flag state for validation and export
    rocprofsys::common_utils::domain_flag_state domain_state;

    // Register shared preset and domain arguments
    rocprofsys::common_utils::register_preset_and_domain_arguments(
        parser, "run", domain_state, [&](std::string_view key, std::string_view val) {
            rocprofsys::common::update_env(_parser_data.current, std::string{ key },
                                           std::string{ val }, update_mode::REPLACE, ":",
                                           _parser_data.updated, original_envs);
        });

    parser.start_group("EXECUTION OPTIONS", "");
    parser.add_argument({ "--fork" }, "Execute via fork + execvpe instead of execvpe")
        .min_count(0)
        .max_count(1)
        .dtype("boolean")
        .action([&](parser_t& p) { _fork_exec = p.get<bool>("fork"); });

    auto args =
        rocprofsys::common_utils::translate_arguments(argc, argv, domain_state.registry);
    _parser_data.command = std::move(args.command);

    auto _cerr = parser.parse_args(args.argv_ptrs.size(), args.argv_ptrs.data());
    if(help_check(parser, argc, argv))
        help_action(parser);
    else if(_cerr)
        throw std::runtime_error(_cerr.what());

    tim::log::monochrome() = _parser_data.monochrome;

    // Handle export-config: output configuration and exit
    if(domain_state.export_config_requested)
    {
        export_config(_parser_data, domain_state.active_preset_name,
                      domain_state.export_config_file);
        throw rocprofsys::common_utils::cli_done{ EXIT_SUCCESS };
    }

    rocprofsys::common_utils::run_post_parse_validation(
        "run", domain_state.active_preset_name, domain_state.gpu_domain_enabled,
        domain_state.rocm_domain_enabled, domain_state.cpu_domain_enabled,
        domain_state.parallel_domain_enabled, _parser_data.verbose,
        domain_state.registry);

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
    }

    return _parser_data;
}
