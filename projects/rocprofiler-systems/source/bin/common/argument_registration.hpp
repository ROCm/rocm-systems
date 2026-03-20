// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "common/common_utils.hpp"
#include "common/env_vars.hpp"
#include "common/json_config.hpp"

#include <string>
#include <string_view>

namespace rocprofsys
{
namespace common_utils
{

/**
 * State tracking for domain flags and preset options.
 * Used to track which options were specified on the command line
 * for validation and export purposes.
 */
struct DomainFlagState
{
    std::string active_preset_name;
    bool        export_config_requested = false;
    std::string export_config_file;
    bool        gpu_domain_enabled      = false;
    bool        rocm_domain_enabled     = false;
    bool        cpu_domain_enabled      = false;
    bool        parallel_domain_enabled = false;
};

/**
 * Registers all preset and domain arguments on an argument parser.
 *
 * This template function extracts the ~180 lines of duplicated argument
 * registration code from rocprof-sys-run and rocprof-sys-sample.
 *
 * @tparam ParserT The argument parser type (tim::argparse::argument_parser)
 * @tparam EnvUpdater Callable with signature void(std::string_view key, std::string_view
 * val)
 * @tparam PresetApplier Callable with signature bool(std::string_view preset_name)
 *
 * @param parser The argument parser to register arguments on
 * @param tool_name The name of the tool ("run" or "sample") for help messages
 * @param state DomainFlagState struct to track which options were specified
 * @param update_env Callback to update environment variable
 * @param apply_preset Callback to apply a preset by name
 */
template <typename ParserT, typename EnvUpdater, typename PresetApplier>
void
register_preset_and_domain_arguments(ParserT& parser, std::string_view tool_name,
                                     DomainFlagState& state, EnvUpdater&& update_env,
                                     PresetApplier&& apply_preset)
{
    namespace env = rocprofsys::env_vars;

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
        .action([&state, apply_preset = std::forward<PresetApplier>(apply_preset)](
                    ParserT& p) mutable {
            auto preset = p.template get<std::string>("preset");
            if(preset.empty()) return;
            state.active_preset_name = preset;
            if(!apply_preset(preset))
            {
                std::cerr << "[rocprof-sys] WARNING: Could not load preset '" << preset
                          << "'. Check preset name or file path.\n";
            }
        });

    parser
        .add_argument({ "--list-presets" },
                      "List all available presets with descriptions and exit")
        .max_count(0)
        .action([tool_name](ParserT&) {
            rocprofsys::common_utils::list_presets(tool_name);
            exit(EXIT_SUCCESS);
        });

    parser
        .add_argument({ "--explain" },
                      "Show detailed information about a preset and exit")
        .max_count(1)
        .dtype("string")
        .action([tool_name](ParserT& p) {
            auto preset_name = p.template get<std::string>("explain");
            if(preset_name.empty())
            {
                std::cerr << "[rocprof-sys] --explain requires a preset name\n";
                exit(EXIT_FAILURE);
            }
            if(!rocprofsys::common_utils::explain_preset(preset_name, tool_name))
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
        .action([&state,
                 update_env = std::forward<EnvUpdater>(update_env)](ParserT& p) mutable {
            state.gpu_domain_enabled = true;
            update_env(env::USE_AMD_SMI, "true");
            update_env(env::USE_PROCESS_SAMPLING, "true");

            if(p.exists("gpu"))
            {
                auto metrics_str = p.template get<std::string>("gpu");
                if(!metrics_str.empty())
                {
                    auto expanded =
                        rocprofsys::json_config::expand_gpu_metrics(metrics_str);
                    if(!expanded.empty())
                    {
                        update_env(env::AMD_SMI_METRICS, expanded);
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
        .action([&state,
                 update_env = std::forward<EnvUpdater>(update_env)](ParserT& p) mutable {
            state.rocm_domain_enabled = true;
            std::string domains_str =
                "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,scratch_memory";

            if(p.exists("rocm"))
            {
                auto input = p.template get<std::string>("rocm");
                if(!input.empty())
                {
                    domains_str = rocprofsys::json_config::expand_rocm_domains(input);
                }
            }

            update_env(env::ROCM_DOMAINS, domains_str);
        });

    parser
        .add_argument({ "--cpu" },
                      "Enable CPU sampling. Optional value specifies frequency in Hz: "
                      "--cpu (default 100Hz) or --cpu=50")
        .min_count(0)
        .max_count(1)
        .dtype("string")
        .action([&state,
                 update_env = std::forward<EnvUpdater>(update_env)](ParserT& p) mutable {
            state.cpu_domain_enabled = true;
            update_env(env::USE_SAMPLING, "true");

            std::string freq = "100";  // default
            if(p.exists("cpu"))
            {
                auto input = p.template get<std::string>("cpu");
                if(!input.empty()) freq = input;
            }
            update_env(env::SAMPLING_FREQ, freq);
        });

    parser
        .add_argument(
            { "--parallel" },
            "Enable parallel runtime profiling. Optional value specifies runtimes: "
            "--parallel (all) or --parallel=mpi,openmp,kokkos,rccl")
        .min_count(0)
        .max_count(1)
        .dtype("string")
        .action([&state,
                 update_env = std::forward<EnvUpdater>(update_env)](ParserT& p) mutable {
            state.parallel_domain_enabled = true;
            std::string runtimes_str;

            if(p.exists("parallel"))
            {
                runtimes_str = p.template get<std::string>("parallel");
            }

            auto env_vars =
                rocprofsys::json_config::expand_parallel_runtimes(runtimes_str);
            for(const auto& [key, val] : env_vars)
            {
                update_env(key, val);
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
        .action([&state](ParserT& p) {
            state.export_config_requested = true;
            if(p.exists("export-config"))
            {
                state.export_config_file = p.template get<std::string>("export-config");
            }
        });
}

}  // namespace common_utils
}  // namespace rocprofsys
