// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "rocprof-sys-run.hpp"

#include "common/common_utils.hpp"
#include "common/defines.h"
#include "common/environment.hpp"
#include "common/json_config.hpp"
#include "common/path.hpp"
#include "common/preset_loader.hpp"
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

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

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

bool
apply_preset_from_json(std::string_view preset_name, parser_data_t& _parser_data)
{
    return rocprofsys::common_utils::apply_preset_from_json(
        preset_name, [&](const std::string& key, const std::string& val) {
            rocprofsys::common::update_env(_parser_data.current, key, val,
                                           update_mode::REPLACE, ":",
                                           _parser_data.updated, original_envs);
        });
}

// Export configuration to JSON file or stdout
void
export_config(const parser_data_t& _parser_data, const std::string& preset_name,
              const std::string& output_file = "")
{
    rocprofsys::common_utils::export_config(_parser_data.current, original_envs,
                                            preset_name, output_file);
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

    rocprofsys::argparse::add_torch_library_path(_data, _data.verbose > 0);

    rocprofsys::common::consolidate_env_entries(_data.current);
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

    // Track preset and domain flag state for validation and export
    std::string active_preset_name;
    bool        export_config_requested = false;
    std::string export_config_file;
    bool        gpu_domain_enabled      = false;
    bool        rocm_domain_enabled     = false;
    bool        cpu_domain_enabled      = false;
    bool        parallel_domain_enabled = false;

    parser.start_group("PRESET OPTIONS",
                       "Load a profiling preset by name or from a JSON file");

    parser
        .add_argument(
            { "--preset" },
            "Load a preset configuration by name or file path. Available presets: "
            "balanced, profile-only, detailed, trace-hpc, workload-trace, sys-trace, "
            "runtime-trace, trace-gpu, trace-openmp, profile-mpi, trace-hw-counters. "
            "For custom configs, provide a path containing '/' or ending with '.json'")
        .max_count(1)
        .dtype("string")
        .action([&](parser_t& p) {
            auto preset = p.get<std::string>("preset");
            if(preset.empty()) return;
            active_preset_name = preset;
            if(!apply_preset_from_json(preset, _parser_data))
            {
                stream(std::cerr, color::warning())
                    << "[rocprof-sys] WARNING: Could not load preset '" << preset
                    << "'. Check preset name or file path.\n";
            }
        });

    parser
        .add_argument({ "--list-presets" },
                      "List all available presets with descriptions and exit")
        .max_count(0)
        .action([&](parser_t&) {
            rocprofsys::common_utils::list_presets("run");
            exit(EXIT_SUCCESS);
        });

    parser
        .add_argument({ "--explain" },
                      "Show detailed information about a preset and exit")
        .max_count(1)
        .dtype("string")
        .action([&](parser_t& p) {
            auto preset_name = p.get<std::string>("explain");
            if(preset_name.empty())
            {
                std::cerr << "[rocprof-sys] --explain requires a preset name\n";
                exit(EXIT_FAILURE);
            }
            if(!rocprofsys::common_utils::explain_preset(preset_name, "run"))
            {
                exit(EXIT_FAILURE);
            }
            exit(EXIT_SUCCESS);
        });

    parser.start_group("DOMAIN OPTIONS", "High-level domain flags for composable "
                                         "configuration (can combine with presets)");

    parser
        .add_argument({ "--gpu" },
                      "Enable GPU metrics collection. Optional value specifies metrics: "
                      "--gpu (all defaults) or --gpu=temp,power,busy,mem_usage")
        .min_count(0)
        .max_count(1)
        .dtype("string")
        .action([&](parser_t& p) {
            gpu_domain_enabled = true;
            rocprofsys::common::update_env(_parser_data.current, "ROCPROFSYS_USE_AMD_SMI",
                                           true, update_mode::REPLACE, ":",
                                           _parser_data.updated, original_envs);
            rocprofsys::common::update_env(
                _parser_data.current, "ROCPROFSYS_USE_PROCESS_SAMPLING", true,
                update_mode::REPLACE, ":", _parser_data.updated, original_envs);

            if(p.exists("gpu"))
            {
                auto metrics_str = p.get<std::string>("gpu");
                if(!metrics_str.empty())
                {
                    auto expanded =
                        rocprofsys::json_config::expand_gpu_metrics(metrics_str);
                    if(!expanded.empty())
                    {
                        rocprofsys::common::update_env(
                            _parser_data.current, "ROCPROFSYS_AMD_SMI_METRICS", expanded,
                            update_mode::REPLACE, ":", _parser_data.updated,
                            original_envs);
                    }
                }
            }
        });

    parser
        .add_argument(
            { "--rocm" },
            "Enable ROCm API tracing. Optional value specifies domains: "
            "--rocm (all defaults) or --rocm=hip,kernel,memory. "
            "Shortcuts: hip->hip_runtime_api, kernel->kernel_dispatch, "
            "memory->memory_copy, hsa->hsa_api, marker->marker_api, rccl->rccl_api")
        .min_count(0)
        .max_count(1)
        .dtype("string")
        .action([&](parser_t& p) {
            rocm_domain_enabled = true;
            std::string domains_str =
                "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,scratch_memory";

            if(p.exists("rocm"))
            {
                auto input = p.get<std::string>("rocm");
                if(!input.empty())
                {
                    domains_str = rocprofsys::json_config::expand_rocm_domains(input);
                }
            }

            rocprofsys::common::update_env(
                _parser_data.current, "ROCPROFSYS_ROCM_DOMAINS", domains_str,
                update_mode::REPLACE, ":", _parser_data.updated, original_envs);
        });

    parser
        .add_argument({ "--cpu" },
                      "Enable CPU sampling. Optional value specifies frequency in Hz: "
                      "--cpu (default 100Hz) or --cpu=50")
        .min_count(0)
        .max_count(1)
        .dtype("string")
        .action([&](parser_t& p) {
            cpu_domain_enabled = true;
            rocprofsys::common::update_env(
                _parser_data.current, "ROCPROFSYS_USE_SAMPLING", true,
                update_mode::REPLACE, ":", _parser_data.updated, original_envs);

            std::string freq = "100";  // default
            if(p.exists("cpu"))
            {
                auto input = p.get<std::string>("cpu");
                if(!input.empty()) freq = input;
            }
            rocprofsys::common::update_env(
                _parser_data.current, "ROCPROFSYS_SAMPLING_FREQ", freq,
                update_mode::REPLACE, ":", _parser_data.updated, original_envs);
        });

    parser
        .add_argument(
            { "--parallel" },
            "Enable parallel runtime profiling. Optional value specifies runtimes: "
            "--parallel (all) or --parallel=mpi,openmp,kokkos,rccl")
        .min_count(0)
        .max_count(1)
        .dtype("string")
        .action([&](parser_t& p) {
            parallel_domain_enabled = true;
            std::string runtimes_str;

            if(p.exists("parallel"))
            {
                runtimes_str = p.get<std::string>("parallel");
            }

            auto env_vars =
                rocprofsys::json_config::expand_parallel_runtimes(runtimes_str);
            for(const auto& [key, val] : env_vars)
            {
                rocprofsys::common::update_env(_parser_data.current, key, val,
                                               update_mode::REPLACE, ":",
                                               _parser_data.updated, original_envs);
            }
        });

    parser.start_group("EXPORT OPTIONS", "Export resolved configuration");

    parser
        .add_argument({ "--export-config" },
                      "Export the resolved configuration as JSON instead of running. "
                      "Optional value specifies output file: --export-config (stdout) or "
                      "--export-config=config.json")
        .min_count(0)
        .max_count(1)
        .dtype("filepath")
        .action([&](parser_t& p) {
            export_config_requested = true;
            if(p.exists("export-config"))
            {
                export_config_file = p.get<std::string>("export-config");
            }
        });

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

    // Handle export-config: output configuration and exit
    if(export_config_requested)
    {
        export_config(_parser_data, active_preset_name, export_config_file);
        exit(EXIT_SUCCESS);
    }

    rocprofsys::common_utils::run_post_parse_validation(
        "run", active_preset_name, gpu_domain_enabled, rocm_domain_enabled,
        cpu_domain_enabled, parallel_domain_enabled, _parser_data.verbose);

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
