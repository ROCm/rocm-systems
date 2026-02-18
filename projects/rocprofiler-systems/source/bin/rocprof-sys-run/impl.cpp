// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

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

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  Presets:  --balanced (default), --profile-only (minimal), --trace-hpc (HPC/MPI), --workload-trace (GPU/ML)
  Output:   Results saved to rocprof-sys-output/ directory
  Visualize: Open perfetto-trace.proto in https://ui.perfetto.dev
EXAMPLES:
  Quick Start:
    rocprof-sys-run --balanced -- ./myapp.inst
  Workload-Specific Presets:
    rocprof-sys-run --trace-hpc -- ./hpc_app.inst         # HPC/MPI/OpenMP
    rocprof-sys-run --workload-trace -- ./gpu_app.inst    # AI/ML/GPU workloads
    rocprof-sys-run --profile-only -- ./myapp.inst        # Minimal overhead
  Custom Configuration:
    rocprof-sys-run --trace-buffer-size=500000 -- ./myapp.inst
    rocprof-sys-run -o ./results -- ./myapp.inst
    mpirun -n 4 rocprof-sys-run --trace-hpc -- ./mpi_app.inst
INSTRUMENTATION WORKFLOW:
  1. Instrument: rocprof-sys-instrument -o app.inst -- ./app
  2. Run:        rocprof-sys-run --balanced -- ./app.inst
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

    parser.start_group("PRESET MODES",
                       "Simplified profiling presets for common use cases");
    parser
        .add_argument(
            { "--balanced" },
            "Balanced profiling mode: moderate overhead with comprehensive data "
            "(tracing, call-stack profiling, and sampling)")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            if(p.get<bool>("balanced"))
            {
                _parser_data.updated.emplace("ROCPROFSYS_TRACE");
                _parser_data.updated.emplace("ROCPROFSYS_PROFILE");
                _parser_data.updated.emplace("ROCPROFSYS_USE_SAMPLING");
                _parser_data.updated.emplace("ROCPROFSYS_USE_PROCESS_SAMPLING");
                tim::set_env("ROCPROFSYS_TRACE", "ON", 0);
                tim::set_env("ROCPROFSYS_PROFILE", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_SAMPLING", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_PROCESS_SAMPLING", "ON", 0);
                tim::set_env("ROCPROFSYS_SAMPLING_FREQ", "50", 0);
            }
        });
    parser
        .add_argument({ "--profile-only" },
                      "Profiling-only mode: lightweight profiling without tracing "
                      "(flat profile, minimal overhead)")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            if(p.get<bool>("profile-only"))
            {
                _parser_data.updated.emplace("ROCPROFSYS_TRACE");
                _parser_data.updated.emplace("ROCPROFSYS_PROFILE");
                _parser_data.updated.emplace("ROCPROFSYS_FLAT_PROFILE");
                tim::set_env("ROCPROFSYS_TRACE", "OFF", 0);
                tim::set_env("ROCPROFSYS_PROFILE", "ON", 0);
                tim::set_env("ROCPROFSYS_FLAT_PROFILE", "ON", 0);
            }
        });
    parser
        .add_argument({ "--detailed" },
                      "Detailed profiling mode: full trace, profile, and system metrics")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            if(p.get<bool>("detailed"))
            {
                _parser_data.updated.emplace("ROCPROFSYS_TRACE");
                _parser_data.updated.emplace("ROCPROFSYS_PROFILE");
                _parser_data.updated.emplace("ROCPROFSYS_USE_SAMPLING");
                _parser_data.updated.emplace("ROCPROFSYS_USE_PROCESS_SAMPLING");
                _parser_data.updated.emplace("ROCPROFSYS_SAMPLING_GPUS");
                tim::set_env("ROCPROFSYS_TRACE", "ON", 0);
                tim::set_env("ROCPROFSYS_PROFILE", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_SAMPLING", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_PROCESS_SAMPLING", "ON", 0);
                tim::set_env("ROCPROFSYS_SAMPLING_CPUS", "all", 0);
                auto* hip_visible_devices = getenv("HIP_VISIBLE_DEVICES");
                if(hip_visible_devices && strlen(hip_visible_devices) > 0)
                {
                    tim::set_env("ROCPROFSYS_SAMPLING_GPUS",
                                 std::string(hip_visible_devices).c_str(), 0);
                }
            }
        });
    parser
        .add_argument(
            { "--trace-hpc" },
            "HPC workload preset: optimized for MPI, OpenMP, and compute-intensive "
            "applications with hardware counter collection")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            if(p.get<bool>("trace-hpc"))
            {
                _parser_data.updated.emplace("ROCPROFSYS_TRACE");
                _parser_data.updated.emplace("ROCPROFSYS_PROFILE");
                _parser_data.updated.emplace("ROCPROFSYS_USE_SAMPLING");
                _parser_data.updated.emplace("ROCPROFSYS_USE_PROCESS_SAMPLING");
                _parser_data.updated.emplace("ROCPROFSYS_USE_OMPT");
                _parser_data.updated.emplace("ROCPROFSYS_USE_KOKKOSP");
                _parser_data.updated.emplace("ROCPROFSYS_USE_RCCL");
                _parser_data.updated.emplace("ROCPROFSYS_USE_MPIP");
                _parser_data.updated.emplace("ROCPROFSYS_SAMPLING_CPUS");
                _parser_data.updated.emplace("ROCPROFSYS_ROCM_DOMAINS");
                _parser_data.updated.emplace("ROCPROFSYS_AMD_SMI_METRICS");
                _parser_data.updated.emplace("ROCPROFSYS_PAPI_EVENTS");
                tim::set_env("ROCPROFSYS_TRACE", "ON", 0);
                tim::set_env("ROCPROFSYS_PROFILE", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_SAMPLING", "OFF", 0);
                tim::set_env("ROCPROFSYS_SAMPLING_FREQ", "100", 0);
                tim::set_env("ROCPROFSYS_USE_PROCESS_SAMPLING", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_OMPT", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_RCCL", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_KOKKOSP", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_MPIP", "true", 0);
                tim::set_env("ROCPROFSYS_SAMPLING_CPUS", "none", 0);
                tim::set_env("ROCPROFSYS_ROCM_DOMAINS",
                             "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,"
                             "scratch_memory",
                             0);
                tim::set_env("ROCPROFSYS_AMD_SMI_METRICS", "busy,temp,power,mem_usage",
                             0);
                tim::set_env("ROCPROFSYS_PAPI_EVENTS",
                             "PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_L3_TCM", 0);
            }
        });
    parser
        .add_argument({ "--workload-trace" },
                      "General compute workload preset: optimized for AI/ML, HPC, and "
                      "GPU workloads with "
                      "comprehensive tracing and increased buffer size")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            if(p.get<bool>("workload-trace"))
            {
                _parser_data.updated.emplace("ROCPROFSYS_TRACE");
                _parser_data.updated.emplace("ROCPROFSYS_PROFILE");
                _parser_data.updated.emplace("ROCPROFSYS_USE_SAMPLING");
                _parser_data.updated.emplace("ROCPROFSYS_USE_PROCESS_SAMPLING");
                _parser_data.updated.emplace("ROCPROFSYS_USE_MPIP");
                _parser_data.updated.emplace("ROCPROFSYS_SAMPLING_CPUS");
                _parser_data.updated.emplace("ROCPROFSYS_ROCM_DOMAINS");
                _parser_data.updated.emplace("ROCPROFSYS_AMD_SMI_METRICS");
                _parser_data.updated.emplace("ROCPROFSYS_SAMPLING_GPUS");
                _parser_data.updated.emplace("ROCPROFSYS_USE_ROCTRACER");
                _parser_data.updated.emplace("ROCPROFSYS_TRACE_HIP_API");
                _parser_data.updated.emplace("ROCPROFSYS_TRACE_HIP_ACTIVITY");
                _parser_data.updated.emplace("ROCPROFSYS_USE_RCCL");
                _parser_data.updated.emplace("ROCPROFSYS_USE_ROCPD");
                _parser_data.updated.emplace("ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB");
                tim::set_env("ROCPROFSYS_TRACE", "ON", 0);
                tim::set_env("ROCPROFSYS_PROFILE", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_SAMPLING", "OFF", 0);
                tim::set_env("ROCPROFSYS_SAMPLING_FREQ", "50", 0);
                tim::set_env("ROCPROFSYS_USE_PROCESS_SAMPLING", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_MPIP", "true", 0);
                tim::set_env("ROCPROFSYS_SAMPLING_CPUS", "none", 0);
                tim::set_env("ROCPROFSYS_ROCM_DOMAINS",
                             "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,"
                             "scratch_memory",
                             0);
                tim::set_env("ROCPROFSYS_AMD_SMI_METRICS", "busy,temp,power,mem_usage",
                             0);
                auto* hip_visible_devices = getenv("HIP_VISIBLE_DEVICES");
                if(hip_visible_devices && strlen(hip_visible_devices) > 0)
                {
                    tim::set_env("ROCPROFSYS_SAMPLING_GPUS",
                                 std::string(hip_visible_devices).c_str(), 0);
                }
                tim::set_env("ROCPROFSYS_USE_ROCTRACER", "ON", 0);
                tim::set_env("ROCPROFSYS_TRACE_HIP_API", "ON", 0);
                tim::set_env("ROCPROFSYS_TRACE_HIP_ACTIVITY", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_RCCL", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_ROCPD", "ON", 0);
                tim::set_env("ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB", "2048000", 0);
            }
        });
    parser
        .add_argument({ "--sys-trace" },
                      "Comprehensive system API tracing: HIP API, HSA API, ROCTx, RCCL, "
                      "rocDecode, rocJPEG, memory operations, and kernel dispatches")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            if(p.get<bool>("sys-trace"))
            {
                _parser_data.updated.emplace("ROCPROFSYS_TRACE");
                _parser_data.updated.emplace("ROCPROFSYS_PROFILE");
                _parser_data.updated.emplace("ROCPROFSYS_USE_ROCM");
                _parser_data.updated.emplace("ROCPROFSYS_ROCM_DOMAINS");
                tim::set_env("ROCPROFSYS_TRACE", "ON", 0);
                tim::set_env("ROCPROFSYS_PROFILE", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_ROCM", "ON", 0);
                tim::set_env("ROCPROFSYS_ROCM_DOMAINS",
                             "hip_api,hsa_api,marker_api,rccl_api,memory_copy,"
                             "scratch_memory,kernel_dispatch",
                             0);
            }
        });
    parser
        .add_argument(
            { "--runtime-trace" },
            "Runtime API tracing: HIP runtime API, ROCTx, RCCL, rocDecode, rocJPEG, "
            "memory operations, and kernel dispatches (excludes HIP compiler and HSA "
            "APIs)")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            if(p.get<bool>("runtime-trace"))
            {
                _parser_data.updated.emplace("ROCPROFSYS_TRACE");
                _parser_data.updated.emplace("ROCPROFSYS_PROFILE");
                _parser_data.updated.emplace("ROCPROFSYS_USE_ROCM");
                _parser_data.updated.emplace("ROCPROFSYS_ROCM_DOMAINS");
                tim::set_env("ROCPROFSYS_TRACE", "ON", 0);
                tim::set_env("ROCPROFSYS_PROFILE", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_ROCM", "ON", 0);
                tim::set_env("ROCPROFSYS_ROCM_DOMAINS",
                             "hip_runtime_api,marker_api,rccl_api,memory_copy,"
                             "scratch_memory,kernel_dispatch",
                             0);
            }
        });
    parser
        .add_argument(
            { "--trace-gpu" },
            "GPU workload analysis: trace with host functions, MPI, and device activity")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            if(p.get<bool>("trace-gpu"))
            {
                _parser_data.updated.emplace("ROCPROFSYS_TRACE");
                _parser_data.updated.emplace("ROCPROFSYS_PROFILE");
                _parser_data.updated.emplace("ROCPROFSYS_USE_ROCM");
                _parser_data.updated.emplace("ROCPROFSYS_USE_AMD_SMI");
                _parser_data.updated.emplace("ROCPROFSYS_SAMPLING_CPUS");
                _parser_data.updated.emplace("ROCPROFSYS_ROCM_DOMAINS");
                tim::set_env("ROCPROFSYS_TRACE", "ON", 0);
                tim::set_env("ROCPROFSYS_PROFILE", "OFF", 0);
                tim::set_env("ROCPROFSYS_USE_ROCM", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_AMD_SMI", "ON", 0);
                tim::set_env("ROCPROFSYS_SAMPLING_CPUS", "none", 0);
                tim::set_env("ROCPROFSYS_ROCM_DOMAINS",
                             "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,"
                             "scratch_memory",
                             0);
            }
        });
    parser
        .add_argument({ "--trace-openmp" },
                      "OpenMP offload workloads: tracing with HSA domains enabled")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            if(p.get<bool>("trace-openmp"))
            {
                _parser_data.updated.emplace("ROCPROFSYS_TRACE");
                _parser_data.updated.emplace("ROCPROFSYS_PROFILE");
                _parser_data.updated.emplace("ROCPROFSYS_USE_ROCM");
                _parser_data.updated.emplace("ROCPROFSYS_ROCM_DOMAINS");
                _parser_data.updated.emplace("ROCPROFSYS_USE_OMPT");
                tim::set_env("ROCPROFSYS_TRACE", "ON", 0);
                tim::set_env("ROCPROFSYS_PROFILE", "OFF", 0);
                tim::set_env("ROCPROFSYS_USE_ROCM", "ON", 0);
                tim::set_env("ROCPROFSYS_ROCM_DOMAINS",
                             "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,"
                             "hsa_api",
                             0);
                tim::set_env("ROCPROFSYS_USE_OMPT", "YES", 0);
            }
        });
    parser
        .add_argument({ "--profile-mpi" }, "MPI communication latency profiling: flat "
                                           "profiling with wall-clock per rank")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            if(p.get<bool>("profile-mpi"))
            {
                _parser_data.updated.emplace("ROCPROFSYS_TRACE");
                _parser_data.updated.emplace("ROCPROFSYS_PROFILE");
                _parser_data.updated.emplace("ROCPROFSYS_FLAT_PROFILE");
                _parser_data.updated.emplace("ROCPROFSYS_USE_AMD_SMI");
                _parser_data.updated.emplace("ROCPROFSYS_USE_ROCM");
                tim::set_env("ROCPROFSYS_TRACE", "OFF", 0);
                tim::set_env("ROCPROFSYS_PROFILE", "ON", 0);
                tim::set_env("ROCPROFSYS_FLAT_PROFILE", "ON", 0);
                tim::set_env("ROCPROFSYS_USE_AMD_SMI", "OFF", 0);
                tim::set_env("ROCPROFSYS_USE_ROCM", "OFF", 0);
            }
        });
    parser
        .add_argument(
            { "--trace-hw-counters" },
            "Hardware counter collection: GPU performance counters during execution")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            if(p.get<bool>("trace-hw-counters"))
            {
                _parser_data.updated.emplace("ROCPROFSYS_PROFILE");
                _parser_data.updated.emplace("ROCPROFSYS_SAMPLING_CPUS");
                _parser_data.updated.emplace("ROCPROFSYS_ROCM_EVENTS");
                tim::set_env("ROCPROFSYS_PROFILE", "ON", 0);
                tim::set_env("ROCPROFSYS_SAMPLING_CPUS", "none", 0);
                tim::set_env("ROCPROFSYS_ROCM_EVENTS", "VALUUtilization,Occupancy", 0);
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
