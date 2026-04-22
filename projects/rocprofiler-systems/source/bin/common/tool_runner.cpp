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

#include <timemory/log/macros.hpp>
#include <timemory/signals/signal_handlers.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/join.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace argparse  = ::tim::argparse;
namespace signals   = ::tim::signals;
namespace path      = rocprofsys::common::path;
namespace env       = rocprofsys::env_vars;
namespace utils     = rocprofsys::common_utils;
using settings      = ::rocprofsys::settings;
using parser_data_t = rocprofsys::argparse::parser_data;
using namespace ::timemory::join;

namespace
{
using rocprofsys::common::update_mode;
using parser_t     = argparse::argument_parser;
using parser_err_t = typename parser_t::result_type;

constexpr int    HELP_PADDING          = 8;
constexpr int    MAX_DESC_WIDTH        = 120;
constexpr size_t LAUNCHER_INJECT_SLOTS = 2;
constexpr int    EXEC_FAILURE_STATUS   = 127;

constexpr std::string_view ANSI_FATAL = "\033[31m";  // red
constexpr std::string_view ANSI_RESET = "\033[0m";

std::atomic<bool>&
monochrome_flag()
{
    static std::atomic<bool> flag{ false };
    return flag;
}

std::string_view
fatal_color()
{
    return monochrome_flag().load(std::memory_order_relaxed) ? std::string_view{}
                                                             : ANSI_FATAL;
}

std::string_view
reset_color()
{
    return monochrome_flag().load(std::memory_order_relaxed) ? std::string_view{}
                                                             : ANSI_RESET;
}

std::string_view
basename_of(std::string_view path)
{
    const auto slash = path.rfind('/');
    return (slash == std::string_view::npos) ? path : path.substr(slash + 1);
}

int
terminal_columns()
{
    struct winsize ws
    {};
    if(ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) return ws.ws_col;
    return 0;  // unknown / not a tty
}

std::string
getenv_string(std::string_view key, std::string_view default_value)
{
    const auto* val = std::getenv(std::string{ key }.c_str());
    return (val != nullptr) ? std::string{ val } : std::string{ default_value };
}

// All string_view fields point at static-storage data (string literals from the
// make_*_config factories). Do not assign from temporaries.
struct tool_config
{
    rocprofsys::common_utils::tool_mode mode = rocprofsys::common_utils::tool_mode::run;
    std::string_view                    tool_name;
    std::string_view                    version_name;
    std::string_view                    summary;
    std::string_view                    workflow;
    std::string_view                    output_prefix = {};

    bool force_sampling                   = false;
    bool enable_fork                      = false;
    bool enable_launcher                  = false;
    bool show_sample_flag                 = false;
    bool disable_cputime_on_realtime_only = false;

    std::unordered_map<std::string, std::string> deprecated_flags = {};
};

tool_config
make_run_config()
{
    tool_config cfg{};
    cfg.mode             = rocprofsys::common_utils::tool_mode::run;
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
    cfg.mode           = rocprofsys::common_utils::tool_mode::sample;
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

auto
toggle_suppression(std::tuple<bool, bool> incoming)
{
    auto previous =
        std::make_tuple(settings::suppress_config(), settings::suppress_parsing());
    std::tie(settings::suppress_config(), settings::suppress_parsing()) = incoming;
    return previous;
}

// Suppresses timemory env-parsing during static init; restored in parse_args.
//
// Why a global: timemory reads ROCPROFSYS_CONFIG_FILE / settings during its
// own translation-unit init, which fires before main(). We must beat that
// init by setting suppress_{config,parsing} as early as possible. Moving this
// into run() is too late.
//
// Static-init order safety: tim::settings::suppress_{config,parsing}() are
// function-local statics inside timemory, so accessing them here triggers
// their construction — no order-of-init UB across translation units.
//
// Captured value holds the prior settings so they can be restored once we
// have control inside main() and can decide when parsing should run.
const auto pre_main_suppression_guard = toggle_suppression({ true, true });

bool
needs_full_parse(int argc, char** argv)
{
    for(int arg_idx = 1; arg_idx < argc; ++arg_idx)
    {
        if(argv[arg_idx] == nullptr) continue;
        auto arg = std::string_view{ argv[arg_idx] };
        if(arg == "--" || arg == "-?" || arg == "-h" || arg == "--help" ||
           arg == "--version" || arg == "--export-config" ||
           arg.find("--export-config=") == 0 || arg == "--list-presets" ||
           arg == "--explain" || arg.find("--explain=") == 0)
        {
            return true;
        }
    }
    return argc > 1 && argv[1] != nullptr && std::string_view{ argv[1] }.find('-') == 0;
}

bool
help_requested(const parser_t& parser, int argc, char** argv)
{
    constexpr std::array<std::string_view, 3> help_args{ "-h", "--help", "-?" };
    if(parser.exists("help") || argc == 1) return true;
    if(argc <= 1 || argv[1] == nullptr) return false;
    return std::find(help_args.begin(), help_args.end(), std::string_view{ argv[1] }) !=
           help_args.end();
}

class tool_runner
{
public:
    tool_runner(int argc_, char** argv_, tool_config cfg)
    : argc{ argc_ }
    , argv{ argv_ }
    , config{ std::move(cfg) }
    {}

    [[nodiscard]] int run();

private:
    void               print_usage() const;
    void               update_verbose_from_env();
    void               get_initial_environment();
    void               prepare_command(const char* exe);
    void               prepare_environment();
    void               parse_command_fast_path();
    void               configure_parser(parser_t& parser);
    std::optional<int> apply_post_parse(parser_t& parser);
    std::optional<int> do_full_parse();
    std::optional<int> parse_args();
    std::string        build_description() const;

    int                                         argc;
    char**                                      argv;
    tool_config                                 config;
    parser_data_t                               data{};
    rocprofsys::common_utils::domain_flag_state domain_state{};
};

std::string
tool_runner::build_description() const
{
    auto cmd = std::string{ "rocprof-sys-" }.append(config.tool_name);

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

    desc = replace_all(std::move(desc), "@SUMMARY@", config.summary);
    desc = replace_all(std::move(desc), "@WORKFLOW@", config.workflow);
    return desc;
}

void
tool_runner::print_usage() const
{
    std::cerr << fatal_color() << "Usage: " << argv[0] << " [OPTIONS] -- <COMMAND> <ARGS>"
              << reset_color() << '\n';
}

void
tool_runner::update_verbose_from_env()
{
    const auto* log_level = std::getenv(env::LOG_LEVEL.data());
    if(log_level != nullptr) data.out.verbose = env::log_level_to_verbose(log_level);
}

void
tool_runner::get_initial_environment()
{
    // Single-threaded assumption: this runs from main() before any worker thread
    // is spawned, so iterating `environ` and calling getenv/setenv is safe.
    if(environ != nullptr)
    {
        for(int idx = 0; environ[idx] != nullptr; ++idx)
        {
            data.env.initial.emplace(environ[idx]);
            data.env.current.emplace_back(environ[idx]);
        }
    }

    auto libexec_path = path::realpath(path::get_internal_script_path());
    if(!libexec_path.empty()) data.env.set(env::SCRIPT_PATH, libexec_path);

    update_verbose_from_env();
    const bool verbose = (data.out.verbose > 0);
    if(auto llvm_dir = rocprofsys::common::discover_llvm_libdir_for_ompt(verbose);
       !llvm_dir.empty())
    {
        data.env.set("LD_LIBRARY_PATH", llvm_dir, update_mode::APPEND);
        // Also mutate the live process env: any dlopen() that happens before
        // execvpe (e.g. OMPT runtime discovery) reads the real environ, not
        // data.env.current.
        const auto* current_ld = std::getenv("LD_LIBRARY_PATH");
        const auto  new_ld =
            (current_ld != nullptr) ? (llvm_dir + ":" + current_ld) : llvm_dir;
        setenv("LD_LIBRARY_PATH", new_ld.c_str(), 1);
    }

    if(config.force_sampling)
    {
        auto mode = getenv_string(env::MODE, "sampling");
        // Bool value flows through update_env's to_env_string(bool) overload,
        // becoming "true"/"false" in the env string.
        data.env.set(env::USE_SAMPLING, (mode != "causal"));
    }
}

void
tool_runner::prepare_command(const char* exe)
{
    if(data.out.launcher.empty()) return;

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
tool_runner::prepare_environment()
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
tool_runner::parse_command_fast_path()
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

void
tool_runner::configure_parser(parser_t& parser)
{
    parser.on_error([](parser_t&, const parser_err_t& err) {
        // Throw rather than exit() so RAII destructors run (parse_data, parser, args).
        // Caught at the top of run() and converted to EXIT_FAILURE.
        throw std::runtime_error(err.what());
    });

    parser.enable_help().count(-1).min_count(0).max_count(1).dtype("topic");
    parser.enable_version(std::string{ config.version_name },
                          ROCPROFSYS_ARGPARSE_VERSION_INFO);

    if(auto cols = terminal_columns(); cols > parser.get_help_width() + HELP_PADDING)
        parser.set_description_width(
            std::min<int>(cols - parser.get_help_width() - HELP_PADDING, MAX_DESC_WIDTH));

    data.reg.processed_groups.emplace("causal");
    if(!config.show_sample_flag) data.reg.processed_environs.emplace("sampling");
    if(!config.enable_launcher) data.reg.processed_environs.emplace("launcher");

    rocprofsys::argparse::add_core_arguments(parser, data);
    rocprofsys::argparse::add_extended_arguments(parser, data);

    rocprofsys::common_utils::register_preset_and_domain_arguments(
        parser, config.tool_name, domain_state,
        [this](std::string_view key, std::string_view val) {
            data.env.set(key, std::string{ val });
        });

    if(config.enable_fork)
    {
        parser.start_group("EXECUTION OPTIONS", "");
        parser.add_argument({ "--fork" }, "Execute via fork + execvpe instead of execvpe")
            .min_count(0)
            .max_count(1)
            .dtype("boolean")
            .action([this](parser_t& parser_ref) {
                data.out.fork_exec = parser_ref.get<bool>("fork");
            });
    }
}

std::optional<int>
tool_runner::apply_post_parse(parser_t& parser)
{
    if(config.disable_cputime_on_realtime_only)
    {
        if(parser.exists("sample-realtime") && !parser.exists("sample-cputime"))
            data.env.set(env::SAMPLING_CPUTIME, false);
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

    rocprofsys::common_utils::run_post_parse_validation(config.tool_name, domain_state,
                                                        data.out.verbose);

    return std::nullopt;
}

std::optional<int>
tool_runner::do_full_parse()
{
    toggle_suppression(pre_main_suppression_guard);
    rocprofsys::argparse::init_parser(data);
    signals::disable_signal_detection(signals::signal_settings::get_enabled());

    auto parser = parser_t{ std::string{ basename_of(argv[0]) }, build_description() };

    configure_parser(parser);

    auto args = rocprofsys::common_utils::translate_arguments(
        argc, argv, domain_state.registry, config.deprecated_flags);
    data.out.command = std::move(args.command);

    auto parse_err =
        parser.parse_args(static_cast<int>(args.argv_ptrs.size()), args.argv_ptrs.data());
    if(help_requested(parser, argc, argv))
        return rocprofsys::common_utils::dispatch_help(parser, config.tool_name,
                                                       EXIT_SUCCESS);
    if(parse_err) throw std::runtime_error(parse_err.what());
    if(domain_state.early_exit) return domain_state.early_exit;

    monochrome_flag().store(data.out.monochrome, std::memory_order_relaxed);
    // Keep timemory's own logger in sync — its background paths still use its color.
    tim::log::monochrome() = data.out.monochrome;

    return apply_post_parse(parser);
}

std::optional<int>
tool_runner::parse_args()
{
    get_initial_environment();

    if(!needs_full_parse(argc, argv))
    {
        parse_command_fast_path();
        return std::nullopt;
    }

    return do_full_parse();
}

int
tool_runner::run()
try
{
    if(argc == 1)
    {
        print_usage();
        return EXIT_FAILURE;
    }

    if(auto exit_code = parse_args()) return *exit_code;

    if(config.enable_launcher) prepare_command(argv[0]);

    prepare_environment();

    if(data.out.command.empty())
    {
        print_usage();
        return EXIT_FAILURE;
    }

    update_verbose_from_env();
    const auto verbose = data.out.verbose;
    if(verbose >= 0)
        utils::print_environment(data.env.current, data.env.updated, verbose >= 1,
                                 config.output_prefix);
    if(verbose >= 1) utils::print_command(data.out.command, config.output_prefix);

    auto argv_ptrs = utils::to_c_argv(data.out.command);
    auto envp_ptrs = utils::to_c_argv(data.env.current);

    if(data.out.fork_exec)
    {
        auto pid = fork();
        if(pid < 0)
        {
            std::perror("fork");
            return EXIT_FAILURE;
        }
        if(pid == 0)
        {
            execvpe(argv_ptrs.front(), argv_ptrs.data(), envp_ptrs.data());
            // execvpe only returns on failure; use _exit to skip parent atexit handlers
            std::perror("execvpe");
            _exit(EXEC_FAILURE_STATUS);
        }

        auto status    = rocprofsys::mproc::wait_pid(pid);
        auto exit_code = rocprofsys::mproc::diagnose_status(pid, status);
        if(exit_code != 0 && data.out.verbose >= 0)
        {
            std::fprintf(stderr,
                         "[rocprof-sys][fatal] process %i exiting with non-zero exit "
                         "code: %i\n",
                         pid, exit_code);
        }
        else if(data.out.verbose >= 2)
        {
            std::fprintf(stderr,
                         "[rocprof-sys][fatal] rocprof-sys run in process %i completed. "
                         "exit code: %i\n",
                         pid, exit_code);
        }
        return exit_code;
    }

    execvpe(argv_ptrs.front(), argv_ptrs.data(), envp_ptrs.data());
    std::perror("execvpe");
    return EXEC_FAILURE_STATUS;
} catch(const std::exception& ex)
{
    std::cerr << "[rocprof-sys] error: " << ex.what() << '\n';
    return EXIT_FAILURE;
}

}  // namespace

namespace rocprofsys::common_utils
{

int
run_tool(int argc, char** argv, tool_mode mode)
{
    auto config = (mode == tool_mode::sample) ? make_sample_config() : make_run_config();
    return tool_runner{ argc, argv, std::move(config) }.run();
}

}  // namespace rocprofsys::common_utils
