// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cli_dispatcher.hpp"
#include "common/defines.h"
#include "delivery.hpp"
#include "diagnostic.hpp"
#include "dry_run.hpp"
#include "help/renderer.hpp"
#include "launcher.hpp"
#include "options/dry_run.hpp"
#include "options/parser.hpp"
#include "options/table.hpp"
#include "options/validator.hpp"

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

[[nodiscard]] rocprofsys::rocsys::result<prepared_launch>
prepare_launch(const rocprofsys::cli::dispatch_result& routed,
               const std::vector<std::string>&         argv)
{
    if(routed.spec == nullptr || argv.empty())
    {
        return rocprofsys::rocsys::diagnostic{ .error = "missing command arguments" };
    }
    const rocprofsys::rocsys::options::context ctx{};
    auto                                       parsed =
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
    else if(ready.dry_run)
    {
        rocprofsys::rocsys::write_dry_run(std::cout, routed.spec->name, ready.plan.argv,
                                          ready.env);
        return EXIT_SUCCESS;
    }
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
