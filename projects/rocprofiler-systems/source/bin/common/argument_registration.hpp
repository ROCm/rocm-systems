// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/common_utils.hpp"
#include "common/domain_flag_state.hpp"
#include "common/env_vars.hpp"
#include "common/json_config.hpp"

#include <timemory/utility/argparse.hpp>

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace rocprofsys
{
namespace common_utils
{

using argument_parser = tim::argparse::argument_parser;

/**
 * Registers all preset and domain arguments on an argument parser.
 *
 * @note Terminal argument actions (--list-presets, --explain) record their
 *       exit code in domain_flag_state::early_exit; the caller short-circuits
 *       after parser.parse_args() returns.
 *
 * @tparam EnvUpdater Callable with signature void(std::string_view key, std::string_view
 * val)
 *
 * @param parser The argument parser to register arguments on
 * @param tool_name The name of the tool ("run" or "sample") for help messages
 * @param state domain_flag_state struct to track which options were specified
 * @param update_env Callback to update environment variable
 */
template <typename EnvUpdater>
void
register_preset_and_domain_arguments(argument_parser& parser, std::string_view tool_name,
                                     domain_flag_state& state, EnvUpdater&& update_env)
{
    namespace env = rocprofsys::env_vars;

    // Capture callable once to avoid multiple std::forward (use-after-move).
    auto env_updater = std::forward<EnvUpdater>(update_env);

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
        .action([&state, env_updater](argument_parser& parser_ref) mutable {
            auto preset = parser_ref.get<std::string>("preset");
            if(preset.empty()) return;
            state.active_preset_name = preset;
            auto settings            = state.registry.get_settings(preset);
            if(settings)
            {
                for(const auto& [key, val] : *settings)
                    env_updater(key, val);
            }
            else
            {
                std::cerr << "[rocprof-sys] WARNING: Could not load preset '" << preset
                          << "'. Check preset name or file path.\n";
            }
        });

    parser
        .add_argument({ "--list-presets" },
                      "List all available presets with descriptions and exit")
        .max_count(0)
        .action([&state, tool_name](argument_parser&) {
            state.registry.list(tool_name);
            state.early_exit = EXIT_SUCCESS;
        });

    parser
        .add_argument({ "--explain" },
                      "Show detailed information about a preset and exit")
        .max_count(1)
        .dtype("string")
        .action([&state, tool_name](argument_parser& parser_ref) {
            auto preset_name = parser_ref.get<std::string>("explain");
            if(preset_name.empty())
            {
                std::cerr << "[rocprof-sys] --explain requires a preset name\n";
                state.early_exit = EXIT_FAILURE;
                return;
            }
            state.early_exit = state.registry.explain(preset_name, tool_name)
                                   ? EXIT_SUCCESS
                                   : EXIT_FAILURE;
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
        .action([&state, env_updater](argument_parser& parser_ref) mutable {
            state.gpu_domain_enabled = true;
            env_updater(env::USE_AMD_SMI, "true");
            env_updater(env::USE_PROCESS_SAMPLING, "true");

            if(parser_ref.exists("gpu"))
            {
                auto metrics_str = parser_ref.get<std::string>("gpu");
                if(!metrics_str.empty())
                {
                    auto expanded =
                        rocprofsys::json_config::expand_gpu_metrics(metrics_str);
                    if(!expanded.empty())
                    {
                        env_updater(env::AMD_SMI_METRICS, expanded);
                    }
                }
            }
        });

    parser
        .add_argument(
            { "--rocm" },
            "Enable ROCm API tracing. Optional value specifies domains: "
            "--rocm (all defaults) or --rocm=hip,kernel,memory,kfd_events. "
            "Shortcuts: hip->hip_runtime_api, kernel->kernel_dispatch, "
            "memory->memory_copy, hsa->hsa_api, marker->marker_api, rccl->rccl_api. "
            "Opt-in domains (not in defaults): kfd_events (requires ROCm 7.13+).")
        .min_count(0)
        .max_count(1)
        .dtype("string")
        .action([&state, env_updater](argument_parser& parser_ref) mutable {
            state.rocm_domain_enabled = true;
            std::string domains_str =
                "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,scratch_memory";

            if(parser_ref.exists("rocm"))
            {
                auto input = parser_ref.get<std::string>("rocm");
                if(!input.empty())
                {
                    domains_str = rocprofsys::json_config::expand_rocm_domains(input);
                }
            }

            env_updater(env::ROCM_DOMAINS, domains_str);
        });

    parser
        .add_argument({ "--cpu" },
                      "Enable CPU sampling. Optional value specifies frequency in Hz: "
                      "--cpu (default 100Hz) or --cpu=50")
        .min_count(0)
        .max_count(1)
        .dtype("string")
        .action([&state, env_updater](argument_parser& parser_ref) mutable {
            state.cpu_domain_enabled = true;
            env_updater(env::USE_SAMPLING, "true");

            std::string freq = "100";  // default
            if(parser_ref.exists("cpu"))
            {
                auto input = parser_ref.get<std::string>("cpu");
                if(!input.empty())
                {
                    try
                    {
                        std::stoi(input);
                        freq = input;
                    } catch(...)
                    {
                        std::cerr
                            << "[rocprof-sys] WARNING: Invalid CPU sampling frequency '"
                            << input << "', using default 100 Hz\n";
                    }
                }
            }
            env_updater(env::SAMPLING_FREQ, freq);
        });

    parser
        .add_argument(
            { "--parallel" },
            "Enable parallel runtime profiling. Optional value specifies runtimes: "
            "--parallel (all) or --parallel=mpi,openmp,kokkos,rccl")
        .min_count(0)
        .max_count(1)
        .dtype("string")
        .action([&state, env_updater](argument_parser& parser_ref) mutable {
            state.parallel_domain_enabled = true;
            std::string runtimes_str;

            if(parser_ref.exists("parallel"))
            {
                runtimes_str = parser_ref.get<std::string>("parallel");
            }

            auto env_vars =
                rocprofsys::json_config::expand_parallel_runtimes(runtimes_str);
            for(const auto& [key, val] : env_vars)
            {
                env_updater(key, val);
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
        .action([&state](argument_parser& parser_ref) {
            state.export_config_requested = true;
            if(parser_ref.exists("export-config"))
            {
                state.export_config_file = parser_ref.get<std::string>("export-config");
            }
        });
}

}  // namespace common_utils
}  // namespace rocprofsys
