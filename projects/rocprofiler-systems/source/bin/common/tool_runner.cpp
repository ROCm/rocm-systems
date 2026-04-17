// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/tool_runner.hpp"

#include "common/argument_registration.hpp"
#include "common/common_utils.hpp"
#include "common/defines.h"
#include "common/env_vars.hpp"
#include "common/environment.hpp"
#include "common/json_config.hpp"
#include "common/path.hpp"
#include "core/argparse.hpp"
#include "core/mproc.hpp"
#include "core/timemory.hpp"

#include <timemory/environment.hpp>
#include <timemory/environment/types.hpp>
#include <timemory/log/color.hpp>
#include <timemory/log/macros.hpp>
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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace color     = ::tim::log::color;
namespace console   = ::tim::utility::console;
namespace argparse  = ::tim::argparse;
namespace signals   = ::tim::signals;
namespace path      = rocprofsys::common::path;
namespace env       = rocprofsys::env_vars;
namespace utils     = rocprofsys::common_utils;
using settings      = ::rocprofsys::settings;
using parser_data_t = rocprofsys::argparse::parser_data;
using namespace ::timemory::join;
using ::tim::get_env;
using ::tim::log::stream;

namespace
{
using rocprofsys::common::update_mode;

std::string
replace_all(std::string str, std::string_view from, std::string_view replacement)
{
    for(auto pos = str.find(from); pos != std::string::npos;
        pos      = str.find(from, pos + replacement.size()))
        str.replace(pos, from.size(), replacement);
    return str;
}

std::string
build_description(const rocprofsys::common_utils::tool_config& cfg)
{
    auto cmd = std::string{ "rocprof-sys-" }.append(cfg.tool_name);

    auto desc = replace_all(
        R"(
@SUMMARY@
QUICK REFERENCE:
  Presets:  --preset=balanced (default), --preset=profile-only, --preset=trace-hpc, --preset=workload-trace
  Domains:  --gpu, --rocm, --cpu, --parallel (composable with presets)
  Output:   Results saved to rocprof-sys-output/ directory
  Visualize: Open perfetto-trace.proto in https://ui.perfetto.dev
EXAMPLES:
  Quick Start:
    @CMD@ --preset=balanced -- ./myapp
  Workload-Specific Presets:
    @CMD@ --preset=trace-hpc -- ./hpc_app             # HPC/MPI/OpenMP
    @CMD@ --preset=workload-trace -- python train.py  # AI/ML/GPU workloads
    @CMD@ --preset=profile-only -- ./myapp            # Minimal overhead
  Domain Flags (composable):
    @CMD@ --gpu -- ./myapp                            # GPU metrics
    @CMD@ --preset=balanced --gpu=temp,power -- ./app # Preset + specific GPU metrics
    @CMD@ --rocm=hip,kernel --cpu=100 -- ./app        # ROCm APIs + CPU sampling
  Custom Configuration File:
    @CMD@ --preset=./my-config.json -- ./myapp
  Export Configuration:
    @CMD@ --preset=balanced --gpu --export-config > my-config.json
@WORKFLOW@
    )",
        "@CMD@", cmd);

    desc = replace_all(std::move(desc), "@SUMMARY@", cfg.summary);
    desc = replace_all(std::move(desc), "@WORKFLOW@", cfg.workflow);
    return desc;
}

auto
toggle_suppression(std::tuple<bool, bool> _inp)
{
    auto _out =
        std::make_tuple(settings::suppress_config(), settings::suppress_parsing());
    std::tie(settings::suppress_config(), settings::suppress_parsing()) = _inp;
    return _out;
}

// Suppresses timemory env-parsing during static init; replayed in parse_args.
// Captured value holds the prior settings so they can be restored once we
// have control inside main() and can decide when parsing should run.
auto pre_main_suppression_guard = toggle_suppression({ true, true });

int
get_verbose(parser_data_t& _data)
{
    auto&       verbose    = _data.verbose;
    const auto* _log_level = std::getenv(env::LOG_LEVEL.data());
    if(_log_level != nullptr) verbose = env::log_level_to_verbose(_log_level);
    return verbose;
}

parser_data_t&
get_initial_environment(parser_data_t&                               _data,
                        const rocprofsys::common_utils::tool_config& _config)
{
    if(environ != nullptr)
    {
        int idx = 0;
        while(environ[idx] != nullptr)
        {
            auto* env_entry = environ[idx++];
            _data.initial.emplace(env_entry);
            _data.current.emplace_back(strdup(env_entry));
        }
    }

    auto _libexecpath = path::realpath(path::get_internal_script_path());
    if(!_libexecpath.empty())
    {
        rocprofsys::common::update_env(_data.current, env::SCRIPT_PATH, _libexecpath,
                                       update_mode::REPLACE, ":", _data.updated,
                                       _data.initial);
    }

    const bool verbose = (get_verbose(_data) > 0);
    if(auto llvm_dir = rocprofsys::common::discover_llvm_libdir_for_ompt(verbose);
       !llvm_dir.empty())
    {
        rocprofsys::common::update_env(_data.current, "LD_LIBRARY_PATH", llvm_dir,
                                       update_mode::APPEND, ":", _data.updated,
                                       _data.initial);
        auto        current_ld = getenv("LD_LIBRARY_PATH");
        std::string new_ld     = current_ld ? (llvm_dir + ":" + current_ld) : llvm_dir;
        setenv("LD_LIBRARY_PATH", new_ld.c_str(), 1);
    }

    if(_config.force_sampling())
    {
        auto _mode = get_env<std::string>(std::string{ env::MODE }, "sampling", false);
        rocprofsys::common::update_env(_data.current, env::USE_SAMPLING,
                                       (_mode != "causal"), update_mode::REPLACE, ":",
                                       _data.updated, _data.initial);
    }

    return _data;
}

void
prepare_command(char* _exe, parser_data_t& _data)
{
    if(_data.launcher.empty()) return;

    bool _injected = false;
    auto _new_argv = std::vector<char*>{};
    for(auto* itr : _data.command)
    {
        if(!_injected &&
           std::string_view{ itr }.find(_data.launcher) != std::string_view::npos)
        {
            _new_argv.emplace_back(_exe);
            _new_argv.emplace_back(strdup("--"));
            _injected = true;
        }
        _new_argv.emplace_back(itr);
    }

    if(!_injected)
    {
        throw std::runtime_error(join("", "Unable to match launcher \"", _data.launcher,
                                      "\" to any arguments on the command line: \"",
                                      join(array_config{ " ", "", "" }, _data.command),
                                      "\""));
    }

    std::swap(_data.command, _new_argv);
}

void
prepare_environment(parser_data_t&                               _data,
                    const rocprofsys::common_utils::tool_config& _config)
{
    // launcher mode re-injects LD_PRELOAD itself, so skip it here
    if(!_config.enable_launcher() || _data.launcher.empty())
    {
        rocprofsys::argparse::add_ld_preload(_data);
        rocprofsys::argparse::add_ld_library_path(_data);
    }

    rocprofsys::argparse::add_torch_library_path(_data, _data.verbose > 0);
    rocprofsys::common::consolidate_env_entries(_data.current);
}

void
parse_command_fast_path(int argc, char** argv, parser_data_t& _data)
{
    toggle_suppression(pre_main_suppression_guard);
    rocprofsys::argparse::init_parser(_data);

    signals::disable_signal_detection(signals::signal_settings::get_enabled());

    auto& _outv           = _data.command;
    bool  _past_separator = false;
    for(int arg_idx = 1; arg_idx < argc; ++arg_idx)
    {
        if(argv[arg_idx] == nullptr) continue;

        if(_past_separator)
            _outv.emplace_back(strdup(argv[arg_idx]));
        else if(std::string_view{ argv[arg_idx] } == "--")
            _past_separator = true;
    }
}

using parser_t     = argparse::argument_parser;
using parser_err_t = typename parser_t::result_type;

constexpr int HELP_PADDING   = 8;
constexpr int MAX_DESC_WIDTH = 120;

bool
needs_full_parse(int argc, char** argv)
{
    for(int arg_idx = 1; arg_idx < argc; ++arg_idx)
    {
        auto arg = std::string_view{ argv[arg_idx] };
        if(arg == "--" || arg == "-?" || arg == "-h" || arg == "--help" ||
           arg == "--version" || arg == "--export-config" ||
           arg.find("--export-config=") == 0 || arg == "--list-presets" ||
           arg == "--explain" || arg.find("--explain=") == 0)
        {
            return true;
        }
    }
    return argc > 1 && std::string_view{ argv[1] }.find('-') == 0;
}

bool
help_requested(const parser_t& parser, int argc, char** argv)
{
    static const std::unordered_set<std::string> help_args = { "-h", "--help", "-?" };
    return parser.exists("help") || argc == 1 ||
           (argc > 1 && help_args.find(argv[1]) != help_args.end());
}

void
configure_parser(parser_t& parser, parser_data_t& _data,
                 const rocprofsys::common_utils::tool_config& _config,
                 rocprofsys::common_utils::domain_flag_state& domain_state,
                 bool&                                        _fork_exec)
{
    parser.on_error([](parser_t&, const parser_err_t& _err) {
        stream(std::cerr, color::fatal()) << _err << "\n";
        exit(EXIT_FAILURE);
    });

    parser.enable_help().count(-1).min_count(0).max_count(1).dtype("topic");
    parser.enable_version(std::string{ _config.version_name },
                          ROCPROFSYS_ARGPARSE_VERSION_INFO);

    auto cols = std::get<0>(console::get_columns());
    if(cols > parser.get_help_width() + HELP_PADDING)
        parser.set_description_width(
            std::min<int>(cols - parser.get_help_width() - HELP_PADDING, MAX_DESC_WIDTH));

    _data.processed_groups.emplace("causal");
    if(!_config.show_sample_flag()) _data.processed_environs.emplace("sampling");
    if(!_config.enable_launcher()) _data.processed_environs.emplace("launcher");

    rocprofsys::argparse::add_core_arguments(parser, _data);
    rocprofsys::argparse::add_extended_arguments(parser, _data);

    rocprofsys::common_utils::register_preset_and_domain_arguments(
        parser, _config.tool_name, domain_state,
        [&](std::string_view key, std::string_view val) {
            rocprofsys::common::update_env(_data.current, std::string{ key },
                                           std::string{ val }, update_mode::REPLACE, ":",
                                           _data.updated, _data.initial);
        });

    if(_config.enable_fork())
    {
        parser.start_group("EXECUTION OPTIONS", "");
        parser.add_argument({ "--fork" }, "Execute via fork + execvpe instead of execvpe")
            .min_count(0)
            .max_count(1)
            .dtype("boolean")
            .action([&_fork_exec](parser_t& parser_ref) {
                _fork_exec = parser_ref.get<bool>("fork");
            });
    }
}

std::optional<int>
apply_post_parse(parser_t& parser, parser_data_t& _data,
                 const rocprofsys::common_utils::tool_config& _config,
                 rocprofsys::common_utils::domain_flag_state& domain_state)
{
    if(_config.disable_cputime_on_realtime_only())
    {
        if(parser.exists("sample-realtime") && !parser.exists("sample-cputime"))
            rocprofsys::common::update_env(_data.current, env::SAMPLING_CPUTIME, false,
                                           update_mode::REPLACE, ":", _data.updated,
                                           _data.initial);
    }

    if(parser.exists("profile") && parser.exists("flat-profile"))
        throw std::runtime_error(
            "Error! '--profile' argument conflicts with '--flat-profile' argument");

    if(domain_state.export_config_requested)
    {
        rocprofsys::common_utils::export_config(
            _data.current, _data.initial, domain_state.active_preset_name,
            _config.tool_name, domain_state.export_config_file);
        return EXIT_SUCCESS;
    }

    rocprofsys::common_utils::run_post_parse_validation(
        _config.tool_name, domain_state.active_preset_name,
        domain_state.gpu_domain_enabled, domain_state.rocm_domain_enabled,
        domain_state.cpu_domain_enabled, domain_state.parallel_domain_enabled,
        _data.verbose, domain_state.registry);

    return std::nullopt;
}

std::optional<int>
parse_args(int argc, char** argv, parser_data_t& _data,
           const rocprofsys::common_utils::tool_config& _config, bool& _fork_exec)
{
    get_initial_environment(_data, _config);

    if(!needs_full_parse(argc, argv))
    {
        parse_command_fast_path(argc, argv, _data);
        return std::nullopt;
    }

    toggle_suppression(pre_main_suppression_guard);
    rocprofsys::argparse::init_parser(_data);
    signals::disable_signal_detection(signals::signal_settings::get_enabled());

    auto parser = parser_t{ basename(argv[0]), build_description(_config) };

    rocprofsys::common_utils::domain_flag_state domain_state;
    configure_parser(parser, _data, _config, domain_state, _fork_exec);

    auto args = rocprofsys::common_utils::translate_arguments(
        argc, argv, domain_state.registry, _config.deprecated_flags);
    _data.command = std::move(args.command);

    auto parse_err = parser.parse_args(args.argv_ptrs.size(), args.argv_ptrs.data());
    if(help_requested(parser, argc, argv))
        return rocprofsys::common_utils::dispatch_help(parser, _config.tool_name,
                                                       EXIT_SUCCESS);
    if(parse_err) throw std::runtime_error(parse_err.what());
    if(domain_state.early_exit) return domain_state.early_exit;

    tim::log::monochrome() = _data.monochrome;

    return apply_post_parse(parser, _data, _config, domain_state);
}
}  // namespace

namespace rocprofsys::common_utils
{

tool_config
make_run_config()
{
    return tool_config{
        tool_mode::run,
        "run",
        "rocprof-sys-run",
        "Execute instrumented binaries with ROCm Systems Profiler configuration.",
        R"(INSTRUMENTATION WORKFLOW:
  1. Instrument: rocprof-sys-instrument -o app.inst -- ./app
  2. Run:        rocprof-sys-run --preset=balanced -- ./app.inst
  3. Analyze:    cat rocprof-sys-output/wall_clock.txt)",
        "ROCPROFSYS: ",
    };
}

tool_config
make_sample_config()
{
    return tool_config{
        tool_mode::sample,
        "sample",
        "rocprof-sys-sample",
        "Call-stack sampling profiler for applications without binary instrumentation.",
        R"(PROFILING WORKFLOW:
  1. Profile:   rocprof-sys-sample --preset=balanced -- ./app
  2. Analyze:   cat rocprof-sys-output/wall_clock.txt
  3. Visualize: Open rocprof-sys-output/perfetto-trace.proto in ui.perfetto.dev)",
        {},
        {
            { "--cputime", "--sample-cputime" },
            { "--realtime", "--sample-realtime" },
            { "--freq", "--sampling-freq" },
        },
    };
}

int
run_tool(int argc, char** argv, const tool_config& config)
{
    auto _print_usage = [argv]() {
        std::cerr << tim::log::color::fatal() << "Usage: " << argv[0]
                  << " [OPTIONS] -- <COMMAND> <ARGS>" << tim::log::color::end()
                  << std::endl;
    };

    if(argc == 1)
    {
        _print_usage();
        return EXIT_FAILURE;
    }

    auto _parse_data = parser_data_t{};
    auto _fork_exec  = false;

    if(auto exit_code = parse_args(argc, argv, _parse_data, config, _fork_exec))
        return *exit_code;

    if(config.enable_launcher()) prepare_command(argv[0], _parse_data);

    prepare_environment(_parse_data, config);

    auto& _argv = _parse_data.command;
    auto& _envp = _parse_data.current;

    if(_argv.empty())
    {
        _print_usage();
        return EXIT_FAILURE;
    }

    auto _verbose = get_verbose(_parse_data);
    if(_verbose >= 0)
        utils::print_environment(_parse_data.current, _parse_data.updated, _verbose >= 1,
                                 config.output_prefix);
    if(_verbose >= 1) utils::print_command(_parse_data.command, config.output_prefix);

    _argv.emplace_back(nullptr);
    _envp.emplace_back(nullptr);

    if(_fork_exec)
    {
        auto _pid = fork();
        if(_pid == 0) return execvpe(_argv.front(), _argv.data(), _envp.data());

        auto _status = rocprofsys::mproc::wait_pid(_pid);
        auto _ec     = rocprofsys::mproc::diagnose_status(_pid, _status);
        if(_ec != 0 && _parse_data.verbose >= 0)
        {
            TIMEMORY_PRINTF_FATAL(
                stderr, "process %i exiting with non-zero exit code: %i\n", _pid, _ec);
        }
        else if(_parse_data.verbose >= 2)
        {
            TIMEMORY_PRINTF_FATAL(
                stderr, "rocprof-sys run in process %i completed. exit code: %i\n", _pid,
                _ec);
        }
        return _ec;
    }

    return execvpe(_argv.front(), _argv.data(), _envp.data());
}

}  // namespace rocprofsys::common_utils
