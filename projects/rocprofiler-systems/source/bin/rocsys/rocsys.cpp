// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cli_dispatcher.hpp"
#include "common/defines.h"
#include "delivery.hpp"
#include "diagnostic.hpp"
#include "dry_run.hpp"
#include "help/renderer.hpp"
#include "launcher.hpp"
#include "options/avail_traces.hpp"
#include "options/context.hpp"
#include "options/dry_run.hpp"
#include "options/parser.hpp"
#include "options/table.hpp"
#include "options/traces.hpp"
#include "options/validator.hpp"
#include "trace/catalog.hpp"

#include "backends/rocprofiler_sdk/backend.hpp"
#include "backends/rocprofiler_sdk/wrapper.hpp"
#include "sdk/tracing-config-deps.hpp"
#include "sdk/tracing-config.hpp"

#include <timemory/utility/argparse.hpp>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
void
print_diagnostic(std::ostream& out, const rocprofsys::rocsys::diagnostic& diag)
{
    out << "error: " << diag.error << '\n';
    if(!diag.hint.empty())
    {
        out << "hint: " << diag.hint << '\n';
    }
}

// enable_version exits on success. The exit below runs only if that action did not.
[[noreturn]] int
exit_with_version(std::string_view program)
{
    auto name   = std::string{ program };
    auto parser = tim::argparse::argument_parser{ name };
    parser.enable_version(name, ROCPROFSYS_ARGPARSE_VERSION_INFO);

    auto flag = std::string{ "--version" };
    auto argv = std::array<char*, 2>{ name.data(), flag.data() };
    // enable_version's action calls exit(EXIT_SUCCESS) from parse_args.
    parser.parse_args(2, argv.data());
    std::exit(EXIT_FAILURE);
}

[[nodiscard]] std::vector<std::string>
copy_argv(int argc, char** argv)
{
    std::vector<std::string> tokens;
    tokens.reserve(static_cast<std::size_t>(argc));
    for(int index = 0; index < argc; ++index)
    {
        tokens.emplace_back(argv[index]);
    }
    return tokens;
}

struct prepared_launch
{
    rocprofsys::rocsys::launch_plan plan      = {};
    rocprofsys::rocsys::env_plan    env       = {};
    bool                            dry_run   = {};
    bool                            tool_help = {};
    bool                            traces    = {};
};

[[nodiscard]] bool
command_has_option(const rocprofsys::rocsys::options::parsed_command& command,
                   std::string_view                                   name)
{
    for(const auto& option : command.options)
    {
        if(option.spec != nullptr && option.spec->name == name)
        {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool
is_avail_traces_query(const rocprofsys::rocsys::options::parsed_command& command)
{
    for(const auto& option : command.options)
    {
        if(option.spec != nullptr &&
           option.spec->kind == rocprofsys::rocsys::options::value_kind::flag &&
           option.spec->name == rocprofsys::rocsys::options::k_avail_traces.name)
        {
            return true;
        }
    }
    return false;
}

void
print_trace_notes(std::ostream& out, const rocprofsys::rocsys::env_plan& ops)
{
    for(const auto& note :
        rocprofsys::rocsys::options::disabled_runtime_notes(ops, std::getenv))
    {
        out << "note: " << note << '\n';
    }
}

[[nodiscard]] std::vector<std::string>
sdk_domain_choices()
{
    using tracing_config_t = rocprofsys::rocprofiler_sdk::tracing_config<
        rocprofsys::backends::rocprofiler_sdk::backend<
            rocprofsys::rocprofiler_sdk::wrapper>,
        rocprofsys::rocprofiler_sdk::default_externals>;
    return tracing_config_t::get_domain_choices();
}

[[nodiscard]] std::vector<std::string>
sdk_supported_domains()
{
    using tracing_config_t = rocprofsys::rocprofiler_sdk::tracing_config<
        rocprofsys::backends::rocprofiler_sdk::backend<
            rocprofsys::rocprofiler_sdk::wrapper>,
        rocprofsys::rocprofiler_sdk::default_externals>;
    return tracing_config_t::get_supported_domain_names();
}

[[nodiscard]] bool
runtime_compiled(std::string_view name)
{
    if(name == "mpi")
    {
#if(defined(ROCPROFSYS_USE_MPI) && ROCPROFSYS_USE_MPI > 0) ||                            \
    (defined(ROCPROFSYS_USE_MPI_HEADERS) && ROCPROFSYS_USE_MPI_HEADERS > 0)
        return true;
#else
        return false;
#endif
    }
    if(name == "openmp")
    {
#if defined(ROCPROFSYS_USE_OMPT) && ROCPROFSYS_USE_OMPT > 0
        return true;
#else
        return false;
#endif
    }
    return name == "rccl" || name == "ucx" || name == "openshmem";
}

[[nodiscard]] rocprofsys::rocsys::options::domain_query
production_domains()
{
    return { .choices          = sdk_domain_choices,
             .supported        = sdk_supported_domains,
             .runtime_compiled = runtime_compiled };
}

[[nodiscard]] rocprofsys::rocsys::result<prepared_launch>
prepare_launch(const rocprofsys::cli::dispatch_result& routed,
               const std::vector<std::string>&         argv)
{
    if(routed.spec == nullptr || argv.empty())
    {
        return rocprofsys::rocsys::diagnostic{ .error = "missing command arguments" };
    }
    const auto ctx =
        rocprofsys::rocsys::options::context{ .domains = production_domains() };
    auto parsed =
        rocprofsys::rocsys::options::parse(rocprofsys::rocsys::options::parse_request{
            .verb             = routed.spec,
            .strip_subcommand = routed.strip_subcommand,
            .argv             = argv,
            .options          = rocprofsys::rocsys::options::option_table(),
            .ctx              = &ctx,
        });
    if(!parsed)
    {
        return parsed.error();
    }
    auto checked = rocprofsys::rocsys::options::validate(std::move(parsed.value()));
    if(!checked)
    {
        return checked.error();
    }
    auto env = rocprofsys::rocsys::options::resolve(checked.value(), ctx);
    if(!env)
    {
        return env.error();
    }
    // The verb is already removed. Delivery sees argv0, legacy flags, then the app.
    auto planned = rocprofsys::rocsys::plan(
        *routed.spec, false,
        rocprofsys::rocsys::options::command_argv(argv.front(), checked.value()),
        env.value());
    if(!planned)
    {
        return planned.error();
    }
    const auto& command = checked.value();
    return prepared_launch{
        .plan = std::move(planned.value()),
        .env  = std::move(env.value()),
        .dry_run =
            command_has_option(command, rocprofsys::rocsys::options::k_dry_run.name),
        .tool_help = rocprofsys::rocsys::help::requests_tool_help(command.passthrough),
        .traces    = is_avail_traces_query(command),
    };
}

[[nodiscard]] int
launch_command(const rocprofsys::cli::dispatch_result& routed,
               const std::vector<std::string>& argv, std::string_view program)
{
    const auto prepared = prepare_launch(routed, argv);
    if(!prepared)
    {
        print_diagnostic(std::cerr, prepared.error());
        return EXIT_FAILURE;
    }
    const auto& ready = prepared.value();
    if(ready.tool_help)
    {
        rocprofsys::rocsys::help::write_verb_options(
            std::cout, *routed.spec, rocprofsys::rocsys::options::option_table());
        std::cout.flush();
    }
    else if(ready.traces)
    {
        const auto catalog =
            rocprofsys::rocsys::trace::trace_catalog{ production_domains() };
        rocprofsys::rocsys::options::write_avail_traces(std::cout, catalog);
        return EXIT_SUCCESS;
    }
    else if(ready.dry_run)
    {
        print_trace_notes(std::cerr, ready.env);
        rocprofsys::rocsys::write_dry_run(std::cout, routed.spec->name, ready.plan.argv,
                                          ready.env);
        return EXIT_SUCCESS;
    }
    print_trace_notes(std::cerr, ready.env);
    return rocprofsys::rocsys::launcher<>{}.launch(ready.plan, program);
}
}  // namespace

int
main(int argc, char** argv)
{
    const auto program = std::filesystem::path{ argv[0] }.filename().string();
    const auto parsed  = rocprofsys::cli::parse_dispatch(argc, argv, program);

    switch(parsed.kind)
    {
        case rocprofsys::cli::dispatch_kind::show_help:
            rocprofsys::cli::print_help(std::cout, program);
            return EXIT_SUCCESS;
        case rocprofsys::cli::dispatch_kind::show_version:
            return exit_with_version(program);
        case rocprofsys::cli::dispatch_kind::error:
            std::cerr << parsed.error_message << '\n';
            return EXIT_FAILURE;
        case rocprofsys::cli::dispatch_kind::in_process:
        case rocprofsys::cli::dispatch_kind::exec_tool:
            return launch_command(parsed, copy_argv(argc, argv), program);
        default: return EXIT_FAILURE;
    }
}
