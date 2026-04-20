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
    auto pos = str.find(from);
    while(pos != std::string::npos)
    {
        str.replace(pos, from.size(), replacement);
        pos = str.find(from, pos + replacement.size());
    }
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
toggle_suppression(std::tuple<bool, bool> incoming)
{
    auto previous =
        std::make_tuple(settings::suppress_config(), settings::suppress_parsing());
    std::tie(settings::suppress_config(), settings::suppress_parsing()) = incoming;
    return previous;
}

// Suppresses timemory env-parsing during static init; replayed in parse_args.
// Captured value holds the prior settings so they can be restored once we
// have control inside main() and can decide when parsing should run.
auto pre_main_suppression_guard = toggle_suppression({ true, true });

int
refresh_verbose_from_env(parser_data_t& data)
{
    const auto* log_level = std::getenv(env::LOG_LEVEL.data());
    if(log_level != nullptr) data.out.verbose = env::log_level_to_verbose(log_level);
    return data.out.verbose;
}

void
get_initial_environment(parser_data_t&                               data,
                        const rocprofsys::common_utils::tool_config& config)
{
    if(environ != nullptr)
    {
        for(int idx = 0; environ[idx] != nullptr; ++idx)
        {
            data.env.initial.emplace(environ[idx]);
            data.env.current.emplace_back(environ[idx]);
        }
    }

    auto libexec_path = path::realpath(path::get_internal_script_path());
    if(!libexec_path.empty())
    {
        rocprofsys::common::update_env(data.env.current, env::SCRIPT_PATH, libexec_path,
                                       update_mode::REPLACE, ":", data.env.updated,
                                       data.env.initial);
    }

    const bool verbose = (refresh_verbose_from_env(data) > 0);
    if(auto llvm_dir = rocprofsys::common::discover_llvm_libdir_for_ompt(verbose);
       !llvm_dir.empty())
    {
        rocprofsys::common::update_env(data.env.current, "LD_LIBRARY_PATH", llvm_dir,
                                       update_mode::APPEND, ":", data.env.updated,
                                       data.env.initial);
        auto        current_ld = getenv("LD_LIBRARY_PATH");
        std::string new_ld     = current_ld ? (llvm_dir + ":" + current_ld) : llvm_dir;
        setenv("LD_LIBRARY_PATH", new_ld.c_str(), 1);
    }

    if(config.force_sampling)
    {
        auto mode = get_env<std::string>(std::string{ env::MODE }, "sampling", false);
        rocprofsys::common::update_env(data.env.current, env::USE_SAMPLING,
                                       (mode != "causal"), update_mode::REPLACE, ":",
                                       data.env.updated, data.env.initial);
    }
}

// Slots reserved when injecting the launcher prefix into the command: exe + "--".
constexpr size_t LAUNCHER_INJECT_SLOTS = 2;

void
prepare_command(const char* exe, parser_data_t& data)
{
    if(data.out.launcher.empty()) return;

    const auto basename_of = [](std::string_view path) {
        const auto slash = path.rfind('/');
        return (slash == std::string_view::npos) ? path : path.substr(slash + 1);
    };

    bool                     injected = false;
    std::vector<std::string> new_argv;
    new_argv.reserve(data.out.command.size() + LAUNCHER_INJECT_SLOTS);
    for(const auto& arg : data.out.command)
    {
        if(!injected &&
           basename_of(arg).find(data.out.launcher) != std::string_view::npos)
        {
            new_argv.emplace_back(exe);
            new_argv.emplace_back("--");
            injected = true;
        }
        new_argv.emplace_back(arg);
    }

    if(!injected)
    {
        throw std::runtime_error(
            join("", "Unable to match launcher \"", data.out.launcher,
                 "\" to any arguments on the command line: \"",
                 join(array_config{ " ", "", "" }, data.out.command), "\""));
    }

    data.out.command = std::move(new_argv);
}

void
prepare_environment(parser_data_t&                               data,
                    const rocprofsys::common_utils::tool_config& config)
{
    // launcher mode re-injects LD_PRELOAD itself, so skip it here
    if(!config.enable_launcher || data.out.launcher.empty())
    {
        rocprofsys::argparse::add_ld_preload(data);
        rocprofsys::argparse::add_ld_library_path(data);
    }

    rocprofsys::argparse::add_torch_library_path(data, data.out.verbose > 0);
    rocprofsys::common::consolidate_env_entries(data.env.current);
}

void
parse_command_fast_path(int argc, char** argv, parser_data_t& data)
{
    toggle_suppression(pre_main_suppression_guard);
    rocprofsys::argparse::init_parser(data);

    signals::disable_signal_detection(signals::signal_settings::get_enabled());

    bool past_separator = false;
    for(int arg_idx = 1; arg_idx < argc; ++arg_idx)
    {
        if(argv[arg_idx] == nullptr) continue;

        if(past_separator)
            data.out.command.emplace_back(argv[arg_idx]);
        else if(std::string_view{ argv[arg_idx] } == "--")
            past_separator = true;
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
configure_parser(parser_t& parser, parser_data_t& data,
                 const rocprofsys::common_utils::tool_config& config,
                 rocprofsys::common_utils::domain_flag_state& domain_state,
                 bool&                                        fork_exec)
{
    parser.on_error([](parser_t&, const parser_err_t& err) {
        stream(std::cerr, color::fatal()) << err << "\n";
        exit(EXIT_FAILURE);
    });

    parser.enable_help().count(-1).min_count(0).max_count(1).dtype("topic");
    parser.enable_version(std::string{ config.version_name },
                          ROCPROFSYS_ARGPARSE_VERSION_INFO);

    auto cols = std::get<0>(console::get_columns());
    if(cols > parser.get_help_width() + HELP_PADDING)
        parser.set_description_width(
            std::min<int>(cols - parser.get_help_width() - HELP_PADDING, MAX_DESC_WIDTH));

    data.reg.processed_groups.emplace("causal");
    if(!config.show_sample_flag) data.reg.processed_environs.emplace("sampling");
    if(!config.enable_launcher) data.reg.processed_environs.emplace("launcher");

    rocprofsys::argparse::add_core_arguments(parser, data);
    rocprofsys::argparse::add_extended_arguments(parser, data);

    rocprofsys::common_utils::register_preset_and_domain_arguments(
        parser, config.tool_name, domain_state,
        [&](std::string_view key, std::string_view val) {
            rocprofsys::common::update_env(data.env.current, std::string{ key },
                                           std::string{ val }, update_mode::REPLACE, ":",
                                           data.env.updated, data.env.initial);
        });

    if(config.enable_fork)
    {
        parser.start_group("EXECUTION OPTIONS", "");
        parser.add_argument({ "--fork" }, "Execute via fork + execvpe instead of execvpe")
            .min_count(0)
            .max_count(1)
            .dtype("boolean")
            .action([&fork_exec](parser_t& parser_ref) {
                fork_exec = parser_ref.get<bool>("fork");
            });
    }
}

std::optional<int>
apply_post_parse(parser_t& parser, parser_data_t& data,
                 const rocprofsys::common_utils::tool_config& config,
                 rocprofsys::common_utils::domain_flag_state& domain_state)
{
    if(config.disable_cputime_on_realtime_only)
    {
        if(parser.exists("sample-realtime") && !parser.exists("sample-cputime"))
            rocprofsys::common::update_env(data.env.current, env::SAMPLING_CPUTIME, false,
                                           update_mode::REPLACE, ":", data.env.updated,
                                           data.env.initial);
    }

    if(parser.exists("profile") && parser.exists("flat-profile"))
        throw std::runtime_error(
            "Error! '--profile' argument conflicts with '--flat-profile' argument");

    if(domain_state.export_config_requested)
    {
        rocprofsys::common_utils::export_config(
            data.env.current, data.env.initial, domain_state.active_preset_name,
            config.tool_name, domain_state.export_config_file);
        return EXIT_SUCCESS;
    }

    rocprofsys::common_utils::run_post_parse_validation(
        config.tool_name, domain_state.active_preset_name,
        domain_state.gpu_domain_enabled, domain_state.rocm_domain_enabled,
        domain_state.cpu_domain_enabled, domain_state.parallel_domain_enabled,
        data.out.verbose, domain_state.registry);

    return std::nullopt;
}

std::optional<int>
do_full_parse(int argc, char** argv, parser_data_t& data,
              const rocprofsys::common_utils::tool_config& config, bool& fork_exec)
{
    toggle_suppression(pre_main_suppression_guard);
    rocprofsys::argparse::init_parser(data);
    signals::disable_signal_detection(signals::signal_settings::get_enabled());

    auto parser = parser_t{ basename(argv[0]), build_description(config) };

    rocprofsys::common_utils::domain_flag_state domain_state;
    configure_parser(parser, data, config, domain_state, fork_exec);

    auto args = rocprofsys::common_utils::translate_arguments(
        argc, argv, domain_state.registry, config.deprecated_flags);
    data.out.command = std::move(args.command);

    auto parse_err = parser.parse_args(args.argv_ptrs.size(), args.argv_ptrs.data());
    if(help_requested(parser, argc, argv))
        return rocprofsys::common_utils::dispatch_help(parser, config.tool_name,
                                                       EXIT_SUCCESS);
    if(parse_err) throw std::runtime_error(parse_err.what());
    if(domain_state.early_exit) return domain_state.early_exit;

    tim::log::monochrome() = data.out.monochrome;

    return apply_post_parse(parser, data, config, domain_state);
}

std::optional<int>
parse_args(int argc, char** argv, parser_data_t& data,
           const rocprofsys::common_utils::tool_config& config, bool& fork_exec)
{
    get_initial_environment(data, config);

    if(!needs_full_parse(argc, argv))
    {
        parse_command_fast_path(argc, argv, data);
        return std::nullopt;
    }

    return do_full_parse(argc, argv, data, config, fork_exec);
}
}  // namespace

namespace rocprofsys::common_utils
{

tool_config
make_run_config()
{
    tool_config cfg{};
    cfg.mode             = tool_mode::run;
    cfg.tool_name        = "run";
    cfg.version_name     = "rocprof-sys-run";
    cfg.summary          = "Execute instrumented binaries with ROCm Systems Profiler "
                           "configuration.";
    cfg.workflow         = R"(INSTRUMENTATION WORKFLOW:
  1. Instrument: rocprof-sys-instrument -o app.inst -- ./app
  2. Run:        rocprof-sys-run --preset=balanced -- ./app.inst
  3. Analyze:    cat rocprof-sys-output/wall_clock.txt)";
    cfg.output_prefix    = "ROCPROFSYS: ";
    cfg.enable_fork      = true;
    cfg.enable_launcher  = true;
    cfg.show_sample_flag = true;
    return cfg;
}

tool_config
make_sample_config()
{
    tool_config cfg{};
    cfg.mode           = tool_mode::sample;
    cfg.tool_name      = "sample";
    cfg.version_name   = "rocprof-sys-sample";
    cfg.summary        = "Call-stack sampling profiler for applications without binary "
                         "instrumentation.";
    cfg.workflow       = R"(PROFILING WORKFLOW:
  1. Profile:   rocprof-sys-sample --preset=balanced -- ./app
  2. Analyze:   cat rocprof-sys-output/wall_clock.txt
  3. Visualize: Open rocprof-sys-output/perfetto-trace.proto in ui.perfetto.dev)";
    cfg.force_sampling = true;
    cfg.disable_cputime_on_realtime_only = true;
    cfg.deprecated_flags                 = {
        { "--cputime", "--sample-cputime" },
        { "--realtime", "--sample-realtime" },
        { "--freq", "--sampling-freq" },
    };
    return cfg;
}

int
run_tool(int argc, char** argv, const tool_config& config)
{
    auto print_usage = [argv]() {
        std::cerr << tim::log::color::fatal() << "Usage: " << argv[0]
                  << " [OPTIONS] -- <COMMAND> <ARGS>" << tim::log::color::end()
                  << std::endl;
    };

    if(argc == 1)
    {
        print_usage();
        return EXIT_FAILURE;
    }

    auto parse_data = parser_data_t{};
    auto fork_exec  = false;

    if(auto exit_code = parse_args(argc, argv, parse_data, config, fork_exec))
        return *exit_code;

    if(config.enable_launcher) prepare_command(argv[0], parse_data);

    prepare_environment(parse_data, config);

    if(parse_data.out.command.empty())
    {
        print_usage();
        return EXIT_FAILURE;
    }

    auto verbose = refresh_verbose_from_env(parse_data);
    if(verbose >= 0)
        utils::print_environment(parse_data.env.current, parse_data.env.updated,
                                 verbose >= 1, config.output_prefix);
    if(verbose >= 1) utils::print_command(parse_data.out.command, config.output_prefix);

    auto argv_ptrs = utils::to_c_argv(parse_data.out.command);
    auto envp_ptrs = utils::to_c_argv(parse_data.env.current);

    if(fork_exec)
    {
        auto pid = fork();
        if(pid == 0)
            return execvpe(argv_ptrs.front(), argv_ptrs.data(), envp_ptrs.data());

        auto status    = rocprofsys::mproc::wait_pid(pid);
        auto exit_code = rocprofsys::mproc::diagnose_status(pid, status);
        if(exit_code != 0 && parse_data.out.verbose >= 0)
        {
            TIMEMORY_PRINTF_FATAL(stderr,
                                  "process %i exiting with non-zero exit code: %i\n", pid,
                                  exit_code);
        }
        else if(parse_data.out.verbose >= 2)
        {
            TIMEMORY_PRINTF_FATAL(
                stderr, "rocprof-sys run in process %i completed. exit code: %i\n", pid,
                exit_code);
        }
        return exit_code;
    }

    return execvpe(argv_ptrs.front(), argv_ptrs.data(), envp_ptrs.data());
}

}  // namespace rocprofsys::common_utils
