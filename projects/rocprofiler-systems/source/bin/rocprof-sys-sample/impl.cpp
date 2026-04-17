// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys-sample.hpp"

#include "common/argument_registration.hpp"
#include "common/common_utils.hpp"
#include "common/env_vars.hpp"
#include "common/environment.hpp"
#include "common/json_config.hpp"
#include "common/path.hpp"

#include <timemory/environment.hpp>
#include <timemory/log/color.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/console.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/join.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <unistd.h>

namespace color = tim::log::color;
namespace path  = rocprofsys::common::path;
namespace env   = rocprofsys::env_vars;
using namespace timemory::join;
using rocprofsys::common::remove_env;
using rocprofsys::common::update_mode;
using tim::get_env;
using tim::log::monochrome;
using tim::log::stream;

namespace
{
int  verbose          = 0;
auto updated_envs     = std::unordered_set<std::string_view>{};
auto original_envs    = std::unordered_set<std::string>{};
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

const std::unordered_set<std::string_view>&
get_updated_envs()
{
    return updated_envs;
}

int
get_verbose_level()
{
    const auto* _log_level = std::getenv(env::LOG_LEVEL.data());
    if(_log_level != nullptr) verbose = env::log_level_to_verbose(_log_level);
    return verbose;
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

    auto _dl_libpath = path::realpath(path::get_internal_libpath("librocprof-sys-dl.so"));
    auto _omni_libpath = path::realpath(path::get_internal_libpath("librocprof-sys.so"));
    auto _libexecpath  = path::realpath(path::get_internal_script_path());
    auto _rootpath     = path::realpath(path::get_rocprofsys_root());

    rocprofsys::common::update_env(_env, env::ROOT, _rootpath, update_mode::REPLACE, ":",
                                   updated_envs, original_envs);
    rocprofsys::common::update_env(_env, "LD_PRELOAD", _dl_libpath, update_mode::APPEND,
                                   ":", updated_envs, original_envs);
    rocprofsys::common::update_env(_env, "LD_LIBRARY_PATH",
                                   tim::filepath::dirname(_dl_libpath),
                                   update_mode::APPEND, ":", updated_envs, original_envs);
    rocprofsys::common::update_env(_env, env::SCRIPT_PATH, _libexecpath,
                                   update_mode::REPLACE, ":", updated_envs,
                                   original_envs);

    // Discover LLVM libdir containing libomptarget.so and append to LD_LIBRARY_PATH
    if(auto llvm_dir = rocprofsys::common::discover_llvm_libdir_for_ompt(verbose > 0);
       !llvm_dir.empty())
    {
        rocprofsys::common::update_env(_env, "LD_LIBRARY_PATH", llvm_dir,
                                       update_mode::APPEND, ":", updated_envs,
                                       original_envs);
    }

    auto _mode = get_env<std::string>(std::string{ env::MODE }, "sampling", false);

    rocprofsys::common::update_env(_env, env::USE_SAMPLING, (_mode != "causal"),
                                   update_mode::REPLACE, ":", updated_envs,
                                   original_envs);

    return _env;
}

std::vector<char*>
parse_args(int argc, char** argv, std::vector<char*>& _env)
{
    using parser_t     = tim::argparse::argument_parser;
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

        rocprofsys::common_utils::dispatch_help(p, "sample", _pec);
    };

    auto _dl_libpath = path::realpath(path::get_internal_libpath("librocprof-sys-dl.so"));
    auto _omni_libpath = path::realpath(path::get_internal_libpath("librocprof-sys.so"));

    const auto* _desc = R"(
Call-stack sampling profiler for applications without binary instrumentation.
QUICK REFERENCE:
  Presets:  --preset=balanced (default), --preset=profile-only, --preset=trace-hpc, --preset=workload-trace
  Domains:  --gpu, --rocm, --cpu, --parallel (composable with presets)
  Output:   Results saved to rocprof-sys-output/ directory
  Visualize: Open perfetto-trace.proto in https://ui.perfetto.dev
EXAMPLES:
  Quick Start:
    rocprof-sys-sample --preset=balanced -- ./myapp
  Workload-Specific Presets:
    rocprof-sys-sample --preset=trace-hpc -- ./hpc_app             # HPC/MPI/OpenMP
    rocprof-sys-sample --preset=workload-trace -- python train.py  # AI/ML/GPU workloads
    rocprof-sys-sample --preset=profile-only -- ./myapp            # Minimal overhead
  Domain Flags (composable):
    rocprof-sys-sample --gpu -- ./myapp                            # GPU metrics
    rocprof-sys-sample --preset=balanced --gpu=temp,power -- ./app # Preset + specific GPU metrics
    rocprof-sys-sample --rocm=hip,kernel --cpu=100 -- ./app        # ROCm APIs + CPU sampling
  Custom Configuration File:
    rocprof-sys-sample --preset=./my-config.json -- ./myapp
  Export Configuration:
    rocprof-sys-sample --preset=balanced --gpu --export-config > my-config.json
PROFILING WORKFLOW:
  1. Profile:   rocprof-sys-sample --preset=balanced -- ./app
  2. Analyze:   cat rocprof-sys-output/wall_clock.txt
  3. Visualize: Open rocprof-sys-output/perfetto-trace.proto in ui.perfetto.dev
)";

    auto parser = parser_t(argv[0], _desc);

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
    parser.enable_help().count(-1).min_count(0).max_count(1).dtype("topic");
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
            rocprofsys::common::update_env(
                _env, env::MONOCHROME, (_monochrome) ? "1" : "0", update_mode::REPLACE,
                ":", updated_envs, original_envs);
            rocprofsys::common::update_env(_env, "MONOCHROME", (_monochrome) ? "1" : "0",
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
        });

    parser.add_argument({ "--log-level" }, "Log level")
        .max_count(1)
        .dtype("string")
        .choices({ "trace", "debug", "info", "warn", "error", "critical", "off" })
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::LOG_LEVEL, p.get<std::string>("log-level"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });

    parser.add_argument({ "--debug" }, "[DEPRECATED Use --log-level=debug] Debug output")
        .max_count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(_env, env::DEBUG, p.get<bool>("debug"),
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);

            rocprofsys::common::update_env(_env, env::LOG_LEVEL, "debug",
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
        });
    parser
        .add_argument({ "-v", "--verbose" },
                      "[DEPRECATED Use --log-level=trace] Verbose output")
        .count(1)
        .action([&](parser_t& p) {
            auto _v = p.get<int>("verbose");
            verbose = _v;

            rocprofsys::common::update_env(_env, env::VERBOSE, _v, update_mode::REPLACE,
                                           ":", updated_envs, original_envs);

            constexpr std::array<const char*, 5> log_levels = { "off", "info", "debug",
                                                                "debug", "trace" };

            auto index = std::clamp(_v + 1, 0, static_cast<int>(log_levels.size() - 1));
            rocprofsys::common::update_env(_env, env::LOG_LEVEL, log_levels[index],
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
        });

    // Track preset and domain flag state for validation and export
    rocprofsys::common_utils::domain_flag_state domain_state;

    // Register shared preset and domain arguments
    rocprofsys::common_utils::register_preset_and_domain_arguments(
        parser, "sample", domain_state, [&](std::string_view key, std::string_view val) {
            updated_envs.emplace(key);
            rocprofsys::common::update_env(_env, std::string{ key }, std::string{ val },
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
        });

    parser.start_group("GENERAL OPTIONS",
                       "These are options which are ubiquitously applied");
    parser.add_argument({ "-c", "--config" }, "Configuration file")
        .min_count(0)
        .dtype("filepath")
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::CONFIG_FILE,
                join(array_config{ ":" }, p.get<std::vector<std::string>>("config")),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });
    parser
        .add_argument({ "-o", "--output" },
                      "Output path. Accepts 1-2 parameters corresponding to the output "
                      "path and the output prefix")
        .min_count(1)
        .max_count(2)
        .action([&](parser_t& p) {
            auto _v = p.get<std::vector<std::string>>("output");
            rocprofsys::common::update_env(_env, env::OUTPUT_PATH, _v.at(0),
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
            if(_v.size() > 1)
                rocprofsys::common::update_env(_env, env::OUTPUT_PREFIX, _v.at(1),
                                               update_mode::REPLACE, ":", updated_envs,
                                               original_envs);
        });
    parser
        .add_argument({ "-T", "--trace" }, "Generate a detailed trace (perfetto output)")
        .max_count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(_env, env::TRACE, p.get<bool>("trace"),
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
        });
    parser
        .add_argument({ "-L", "--trace-legacy" },
                      "Use legacy direct mode for tracing instead of deferred trace "
                      "generation (higher overhead)")
        .max_count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::TRACE_LEGACY, p.get<bool>("trace-legacy"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });
    parser
        .add_argument(
            { "-P", "--profile" },
            "Generate a call-stack-based profile (conflicts with --flat-profile)")
        .max_count(1)
        .conflicts({ "flat-profile" })
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(_env, env::PROFILE, p.get<bool>("profile"),
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
        });
    parser
        .add_argument({ "-F", "--flat-profile" },
                      "Generate a flat profile (conflicts with --profile)")
        .max_count(1)
        .conflicts({ "profile" })
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::PROFILE, p.get<bool>("flat-profile"), update_mode::REPLACE,
                ":", updated_envs, original_envs);
            rocprofsys::common::update_env(
                _env, env::FLAT_PROFILE, p.get<bool>("flat-profile"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });
    parser
        .add_argument({ "-H", "--host" },
                      "Enable sampling host-based metrics for the process. E.g. CPU "
                      "frequency, memory usage, etc.")
        .max_count(1)
        .action([&](parser_t& p) {
            auto _h = p.get<bool>("host");
            auto _d = p.get<bool>("device");
            rocprofsys::common::update_env(_env, env::USE_PROCESS_SAMPLING, _h || _d,
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
            rocprofsys::common::update_env(_env, env::CPU_FREQ_ENABLED, _h,
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
            if(_h)
                rocprofsys::common::update_env(_env, env::USE_AMD_SMI, _d,
                                               update_mode::REPLACE, ":", updated_envs,
                                               original_envs);
        });
    parser
        .add_argument({ "-D", "--device" },
                      "Enable sampling device-based metrics for the process. E.g. GPU "
                      "temperature, memory usage, etc.")
        .max_count(1)
        .action([&](parser_t& p) {
            auto _h = p.get<bool>("host");
            auto _d = p.get<bool>("device");
            rocprofsys::common::update_env(_env, env::USE_PROCESS_SAMPLING, _h || _d,
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
            rocprofsys::common::update_env(_env, env::USE_AMD_SMI, _d,
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
            if(_d)
                rocprofsys::common::update_env(_env, env::CPU_FREQ_ENABLED, _h,
                                               update_mode::REPLACE, ":", updated_envs,
                                               original_envs);
        });
    parser
        .add_argument({ "-w", "--wait" },
                      "This option is a combination of '--trace-wait' and "
                      "'--sampling-wait'. See the descriptions for those two options.")
        .count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(_env, env::TRACE_DELAY, p.get<double>("wait"),
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
            rocprofsys::common::update_env(_env, env::SAMPLING_DELAY,
                                           p.get<double>("wait"), update_mode::REPLACE,
                                           ":", updated_envs, original_envs);
        });
    parser
        .add_argument(
            { "-d", "--duration" },
            "This option is a combination of '--trace-duration' and "
            "'--sampling-duration'. See the descriptions for those two options.")
        .count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::TRACE_DURATION, p.get<double>("duration"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
            rocprofsys::common::update_env(
                _env, env::SAMPLING_DURATION, p.get<double>("duration"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });

    parser
        .add_argument({ "--selected-regions" },
                      "Comma-separated list of roctx region names. When set, only "
                      "activity inside matching roctx regions is traced (matched against "
                      "roctxRangeStartA message)")
        .count(1)
        .dtype("string")
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::SELECTED_REGIONS, p.get<std::string>("selected-regions"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
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
            rocprofsys::common::update_env(
                _env, env::PERFETTO_FILE, p.get<std::string>("trace-file"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });
    parser
        .add_argument({ "--trace-buffer-size" },
                      "Size limit for the trace output (in KB)")
        .count(1)
        .dtype("KB")
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::PERFETTO_BUFFER_SIZE_KB, p.get<int64_t>("trace-buffer-size"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });
    parser.add_argument({ "--trace-fill-policy" }, _trace_policy_desc)
        .count(1)
        .choices({ "discard", "ring_buffer" })
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::PERFETTO_FILL_POLICY, p.get<std::string>("trace-fill-policy"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });
    parser
        .add_argument({ "--trace-wait" },
                      "Set the wait time (in seconds) "
                      "before collecting trace and/or profiling data"
                      "(in seconds). By default, the duration is in seconds of realtime "
                      "but that can changed via --trace-clock-id.")
        .count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::TRACE_DELAY, p.get<double>("trace-wait"), update_mode::REPLACE,
                ":", updated_envs, original_envs);
        });
    parser
        .add_argument({ "--trace-duration" },
                      "Set the duration of the trace and/or profile data collection (in "
                      "seconds). By default, the duration is in seconds of realtime but "
                      "that can changed via --trace-clock-id.")
        .count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::TRACE_DURATION, p.get<double>("trace-duration"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });
    parser
        .add_argument(
            { "--trace-periods" },
            "More powerful version of specifying trace delay and/or duration. Format is "
            "one or more groups of: <DELAY>:<DURATION>, <DELAY>:<DURATION>:<REPEAT>, "
            "and/or <DELAY>:<DURATION>:<REPEAT>:<CLOCK_ID>.")
        .min_count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::TRACE_PERIODS,
                join(array_config{ ",", "", "" },
                     p.get<std::vector<std::string>>("trace-periods")),
                update_mode::REPLACE, ":", updated_envs, original_envs);
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
            rocprofsys::common::update_env(
                _env, env::TRACE_PERIOD_CLOCK_ID, p.get<double>("trace-clock-id"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
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
            rocprofsys::common::update_env(_env, env::PROFILE, true, update_mode::REPLACE,
                                           ":", updated_envs, original_envs);
            if(!_v.empty())
            {
                rocprofsys::common::update_env(
                    _env, env::TEXT_OUTPUT, _v.count("text") != 0, update_mode::REPLACE,
                    ":", updated_envs, original_envs);
                rocprofsys::common::update_env(
                    _env, env::JSON_OUTPUT, _v.count("json") != 0, update_mode::REPLACE,
                    ":", updated_envs, original_envs);
                rocprofsys::common::update_env(
                    _env, env::COUT_OUTPUT, _v.count("console") != 0,
                    update_mode::REPLACE, ":", updated_envs, original_envs);
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
            rocprofsys::common::update_env(_env, env::DIFF_OUTPUT, true,
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
            rocprofsys::common::update_env(_env, env::INPUT_PATH, _v.at(0),
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
            if(_v.size() > 1)
                rocprofsys::common::update_env(_env, env::INPUT_PREFIX, _v.at(1),
                                               update_mode::REPLACE, ":", updated_envs,
                                               original_envs);
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
            rocprofsys::common::update_env(
                _env, env::PROCESS_SAMPLING_FREQ, p.get<double>("process-freq"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });
    parser
        .add_argument({ "--process-wait" }, "Set the default wait time (i.e. delay) "
                                            "before taking first host/device sample "
                                            "(in seconds of realtime)")
        .count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::PROCESS_SAMPLING_DELAY, p.get<double>("process-wait"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });
    parser
        .add_argument(
            { "--process-duration" },
            "Set the duration of the host/device sampling (in seconds of realtime)")
        .count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::SAMPLING_PROCESS_DURATION, p.get<double>("process-duration"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });
    parser
        .add_argument({ "--cpus" },
                      "CPU IDs for frequency sampling. Supports integers and/or ranges")
        .dtype("int or range")
        .required({ "host" })
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::SAMPLING_CPUS,
                join(array_config{ "," }, p.get<std::vector<std::string>>("cpus")),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });
    parser
        .add_argument({ "--gpus" },
                      "GPU IDs for SMI queries. Supports integers and/or ranges")
        .dtype("int or range")
        .required({ "device" })
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::SAMPLING_GPUS,
                join(array_config{ "," }, p.get<std::vector<std::string>>("gpus")),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });
    parser
        .add_argument({ "--ai-nics" },
                      "AI NIC IDs for SMI queries. Comma-separated list.")
        .dtype("string")
        .required({ "device" })
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::SAMPLING_AINICS,
                join(array_config{ "," }, p.get<std::vector<std::string>>("ai-nics")),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });

    parser.start_group("GENERAL SAMPLING OPTIONS",
                       "General options for timer-based sampling per-thread");
    parser
        .add_argument({ "-f", "--freq" }, "Set the default sampling frequency "
                                          "(number of interrupts per second)")
        .count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(_env, env::SAMPLING_FREQ,
                                           p.get<double>("freq"), update_mode::REPLACE,
                                           ":", updated_envs, original_envs);
        });
    parser
        .add_argument(
            { "--sampling-wait" },
            "Set the default wait time (i.e. delay) before taking first sample "
            "(in seconds). This delay time is based on the clock of the sampler, i.e., a "
            "delay of 1 second for CPU-clock sampler may not equal 1 second of realtime")
        .count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::SAMPLING_DELAY, p.get<double>("sampling-wait"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });
    parser
        .add_argument(
            { "--sampling-duration" },
            "Set the duration of the sampling (in seconds of realtime). I.e., it is "
            "possible (currently) to set a CPU-clock time delay that exceeds the "
            "real-time duration... resulting in zero samples being taken")
        .count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::SAMPLING_DURATION, p.get<double>("sampling-duration"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });
    parser
        .add_argument({ "-t", "--tids" },
                      "Specify the default thread IDs for sampling, where 0 (zero) is "
                      "the main thread and each thread created by the target application "
                      "is assigned an atomically incrementing value.")
        .min_count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, env::SAMPLING_TIDS,
                join(array_config{ ", " }, p.get<std::vector<int64_t>>("tids")),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });

    parser.start_group(
        "SAMPLING TIMER OPTIONS",
        "These options determine the heuristic for deciding when to take a sample");
    parser.add_argument({ "--cputime" }, _cputime_desc)
        .min_count(0)
        .action([&](parser_t& p) {
            auto _v = p.get<std::deque<std::string>>("cputime");
            rocprofsys::common::update_env(_env, env::SAMPLING_CPUTIME, true,
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
            if(!_v.empty())
            {
                rocprofsys::common::update_env(_env, env::SAMPLING_CPUTIME_FREQ,
                                               _v.front(), update_mode::REPLACE, ":",
                                               updated_envs, original_envs);
                _v.pop_front();
            }
            if(!_v.empty())
            {
                rocprofsys::common::update_env(_env, env::SAMPLING_CPUTIME_DELAY,
                                               _v.front(), update_mode::REPLACE, ":",
                                               updated_envs, original_envs);
                _v.pop_front();
            }
            if(!_v.empty())
            {
                rocprofsys::common::update_env(
                    _env, env::SAMPLING_CPUTIME_TIDS, join(array_config{ "," }, _v),
                    update_mode::REPLACE, ":", updated_envs, original_envs);
            }
        });

    parser.add_argument({ "--realtime" }, _realtime_desc)
        .min_count(0)
        .action([&](parser_t& p) {
            auto _v = p.get<std::deque<std::string>>("realtime");
            rocprofsys::common::update_env(_env, env::SAMPLING_REALTIME, true,
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
            if(!_v.empty())
            {
                rocprofsys::common::update_env(_env, env::SAMPLING_REALTIME_FREQ,
                                               _v.front(), update_mode::REPLACE, ":",
                                               updated_envs, original_envs);
                _v.pop_front();
            }
            if(!_v.empty())
            {
                rocprofsys::common::update_env(_env, env::SAMPLING_REALTIME_DELAY,
                                               _v.front(), update_mode::REPLACE, ":",
                                               updated_envs, original_envs);
                _v.pop_front();
            }
            if(!_v.empty())
            {
                rocprofsys::common::update_env(
                    _env, env::SAMPLING_REALTIME_TIDS, join(array_config{ "," }, _v),
                    update_mode::REPLACE, ":", updated_envs, original_envs);
            }
        });

    std::set<std::string> _backend_choices = { "all",         "kokkosp",    "mpip",
                                               "ompt",        "rcclp",      "amd-smi",
                                               "mutex-locks", "spin-locks", "rw-locks",
                                               "rocm" };

#if(!defined(ROCPROFSYS_USE_MPI) || ROCPROFSYS_USE_MPI == 0) &&                          \
    (!defined(ROCPROFSYS_USE_MPI_HEADERS) || ROCPROFSYS_USE_MPI_HEADERS == 0)
    _backend_choices.erase("mpip");
#endif

#if !defined(ROCPROFSYS_USE_OMPT) || ROCPROFSYS_USE_OMPT == 0
    _backend_choices.erase("ompt");
#endif

    parser.start_group("BACKEND OPTIONS",
                       "These options control region information captured "
                       "w/o sampling or instrumentation");
    parser.add_argument({ "-I", "--include" }, "Include data from these backends")
        .choices(_backend_choices)
        .action([&](parser_t& p) {
            auto _v      = p.get<std::set<std::string>>("include");
            auto _update = [&](const auto& _opt, bool _cond) {
                if(_cond || _v.count("all") > 0)
                    rocprofsys::common::update_env(_env, _opt, true, update_mode::REPLACE,
                                                   ":", updated_envs, original_envs);
            };
            _update(env::USE_KOKKOSP, _v.count("kokkosp") > 0);
            _update(env::USE_MPIP, _v.count("mpip") > 0);
            _update(env::USE_OMPT, _v.count("ompt") > 0);
            _update(env::USE_RCCLP, _v.count("rcclp") > 0);
            _update(env::USE_AMD_SMI, _v.count("amd-smi") > 0);
            _update(env::TRACE_THREAD_LOCKS, _v.count("mutex-locks") > 0);
            _update(env::TRACE_THREAD_RW_LOCKS, _v.count("rw-locks") > 0);
            _update(env::TRACE_THREAD_SPIN_LOCKS, _v.count("spin-locks") > 0);

            if(_v.count("all") > 0 || _v.count("kokkosp") > 0)
                rocprofsys::common::update_env(_env, "KOKKOS_TOOLS_LIBS", _omni_libpath,
                                               update_mode::APPEND, ":", updated_envs,
                                               original_envs);
        });

    parser.add_argument({ "-E", "--exclude" }, "Exclude data from these backends")
        .choices(_backend_choices)
        .action([&](parser_t& p) {
            auto _v      = p.get<std::set<std::string>>("exclude");
            auto _update = [&](const auto& _opt, bool _cond) {
                if(_cond || _v.count("all") > 0)
                    rocprofsys::common::update_env(_env, _opt, false,
                                                   update_mode::REPLACE, ":",
                                                   updated_envs, original_envs);
            };
            _update(env::USE_KOKKOSP, _v.count("kokkosp") > 0);
            _update(env::USE_MPIP, _v.count("mpip") > 0);
            _update(env::USE_OMPT, _v.count("ompt") > 0);
            _update(env::USE_RCCLP, _v.count("rcclp") > 0);
            _update(env::USE_AMD_SMI, _v.count("amd-smi") > 0);
            _update(env::TRACE_THREAD_LOCKS, _v.count("mutex-locks") > 0);
            _update(env::TRACE_THREAD_RW_LOCKS, _v.count("rw-locks") > 0);
            _update(env::TRACE_THREAD_SPIN_LOCKS, _v.count("spin-locks") > 0);

            if(_v.count("all") > 0 || _v.count("kokkosp") > 0)
                remove_env(_env, "KOKKOS_TOOLS_LIBS", original_envs);
        });

    parser.start_group("HARDWARE COUNTER OPTIONS", "See also: rocprof-sys-avail -H");
    parser
        .add_argument({ "-C", "--cpu-events" },
                      "Set the CPU hardware counter events to record (ref: "
                      "`rocprof-sys-avail -H -c CPU`)")
        .action([&](parser_t& p) {
            auto _events =
                join(array_config{ "," }, p.get<std::vector<std::string>>("cpu-events"));
            rocprofsys::common::update_env(_env, env::PAPI_EVENTS, _events,
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
        });

    parser
        .add_argument({ "-G", "--gpu-events" },
                      "Set the GPU hardware counter events to record (ref: "
                      "`rocprof-sys-avail -H -c GPU`)")
        .action([&](parser_t& p) {
            auto _events =
                join(array_config{ "," }, p.get<std::vector<std::string>>("gpu-events"));
            rocprofsys::common::update_env(_env, env::ROCM_EVENTS, _events,
                                           update_mode::REPLACE, ":", updated_envs,
                                           original_envs);
        });

    parser.start_group("MISCELLANEOUS OPTIONS", "");
    parser
        .add_argument({ "-i", "--inlines" },
                      "Include inline info in output when available")
        .max_count(1)
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(_env, env::SAMPLING_INCLUDE_INLINES,
                                           p.get<bool>("inlines"), update_mode::REPLACE,
                                           ":", updated_envs, original_envs);
        });

    parser.add_argument({ "--hsa-interrupt" }, _hsa_interrupt_desc)
        .count(1)
        .dtype("int")
        .choices({ 0, 1 })
        .action([&](parser_t& p) {
            rocprofsys::common::update_env(
                _env, "HSA_ENABLE_INTERRUPT", p.get<int>("hsa-interrupt"),
                update_mode::REPLACE, ":", updated_envs, original_envs);
        });

    parser.end_group();

    auto args =
        rocprofsys::common_utils::translate_arguments(argc, argv, domain_state.registry);

    auto _cerr = parser.parse_args(args.argv_ptrs.size(), args.argv_ptrs.data());
    if(help_check(parser, argc, argv))
        help_action(parser);
    else if(_cerr)
        throw std::runtime_error(_cerr.what());

    if(parser.exists("realtime") && !parser.exists("cputime"))
        rocprofsys::common::update_env(_env, env::SAMPLING_CPUTIME, false,
                                       update_mode::REPLACE, ":", updated_envs,
                                       original_envs);
    if(parser.exists("profile") && parser.exists("flat-profile"))
        throw std::runtime_error(
            "Error! '--profile' argument conflicts with '--flat-profile' argument");

    // Handle export-config: output configuration and exit
    if(domain_state.export_config_requested)
    {
        rocprofsys::common_utils::export_config(_env, original_envs,
                                                domain_state.active_preset_name, "sample",
                                                domain_state.export_config_file);
        throw rocprofsys::common_utils::cli_done{ EXIT_SUCCESS };
    }

    rocprofsys::common_utils::run_post_parse_validation(
        "sample", domain_state.active_preset_name, domain_state.gpu_domain_enabled,
        domain_state.rocm_domain_enabled, domain_state.cpu_domain_enabled,
        domain_state.parallel_domain_enabled, verbose, domain_state.registry);

    return args.command;
}

void
add_torch_library_path(std::vector<char*>& envp, const std::vector<char*>& argv)
{
    rocprofsys::common::add_torch_library_path(envp, argv, verbose > 0, updated_envs);
}
