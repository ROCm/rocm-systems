// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/cli_dispatcher.hpp"
#include "common/tool_runner.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <initializer_list>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using rocprofsys::cli::directory_of;
using rocprofsys::cli::dispatch_kind;
using rocprofsys::cli::find_subcommand;
using rocprofsys::cli::forward_options;
using rocprofsys::cli::join_sibling_path;
using rocprofsys::cli::k_subcommands;
using rocprofsys::cli::make_forwarded_argv;
using rocprofsys::cli::parse_dispatch;
using rocprofsys::cli::print_help;
using rocprofsys::cli::print_version;
using rocprofsys::cli::program_name;
using rocprofsys::common_utils::tool_mode;

namespace
{
struct argv_builder
{
    std::vector<std::string> storage;
    std::vector<char*>       ptrs;

    explicit argv_builder(std::initializer_list<const char*> args)
    {
        storage.reserve(args.size());
        for(const auto* arg : args)
        {
            storage.emplace_back(arg);
        }
        ptrs.reserve(storage.size());
        for(auto& entry : storage)
        {
            ptrs.push_back(entry.data());
        }
    }

    argv_builder(const argv_builder&)            = delete;
    argv_builder& operator=(const argv_builder&) = delete;
    argv_builder(argv_builder&&)                 = delete;
    argv_builder& operator=(argv_builder&&)      = delete;
    ~argv_builder()                              = default;

    [[nodiscard]] int argc() const noexcept { return static_cast<int>(storage.size()); }

    [[nodiscard]] char** argv() noexcept { return ptrs.data(); }
};

[[nodiscard]] forward_options
instrument_forward(std::string_view extra_flag = {})
{
    return {
        .strip_subcommand = true,
        .argv0_override   = "rocprof-sys-instrument",
        .extra_flag       = extra_flag,
    };
}

[[nodiscard]] int
invalid_subcommand_count()
{
    int invalid = 0;
    for(const auto& spec : k_subcommands)
    {
        const bool unnamed = spec.name.empty();
        const bool missing = find_subcommand(spec.name) == nullptr;
        if(unnamed || missing)
        {
            ++invalid;
        }
    }
    return invalid;
}

[[nodiscard]] int
default_subcommand_count()
{
    int defaults = 0;
    for(const auto& spec : k_subcommands)
    {
        if(spec.is_default)
        {
            ++defaults;
        }
    }
    return defaults;
}

[[nodiscard]] std::string
missing_help_entries(std::string_view text)
{
    std::string missing;
    for(const auto& spec : k_subcommands)
    {
        if(text.find(spec.name) != std::string_view::npos)
        {
            continue;
        }
        if(!missing.empty())
        {
            missing += ", ";
        }
        missing.append(spec.name);
    }
    return missing;
}

std::vector<std::string>
forwarded_args(int argc, char** argv, forward_options options)
{
    auto                     forwarded = make_forwarded_argv(argc, argv, options);
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(forwarded.argc()));
    for(int idx = 0; idx < forwarded.argc(); ++idx)
    {
        out.emplace_back(forwarded.argv()[idx]);
    }
    return out;
}
}  // namespace

TEST(cli_dispatcher_test, no_args_shows_help)
{
    auto args   = argv_builder{ "rocsys" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::show_help);
}

TEST(cli_dispatcher_test, help_flags_show_help)
{
    for(const char* flag : { "--help", "-h", "-?", "--help=all" })
    {
        auto args   = argv_builder{ "rocsys", flag };
        auto result = parse_dispatch(args.argc(), args.argv());
        EXPECT_EQ(result.kind, dispatch_kind::show_help) << flag;
    }
}

TEST(cli_dispatcher_test, version_flag_shows_version)
{
    auto args   = argv_builder{ "rocsys", "--version" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::show_version);
}

TEST(cli_dispatcher_test, implicit_default_with_separator)
{
    auto args   = argv_builder{ "rocsys", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::in_process);
    EXPECT_EQ(result.mode, tool_mode::run);
    EXPECT_FALSE(result.strip_subcommand);
    EXPECT_EQ(result.subcommand_name, "profile");
    EXPECT_EQ(result.binary_name, "rocprof-sys-run");
}

TEST(cli_dispatcher_test, implicit_default_with_flags)
{
    auto args   = argv_builder{ "rocsys", "--preset=quick", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::in_process);
    EXPECT_EQ(result.mode, tool_mode::run);
    EXPECT_FALSE(result.strip_subcommand);
}

TEST(cli_dispatcher_test, profile_is_in_process_run)
{
    auto args   = argv_builder{ "rocsys", "profile", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::in_process);
    EXPECT_EQ(result.mode, tool_mode::run);
    EXPECT_TRUE(result.strip_subcommand);
    EXPECT_EQ(result.subcommand_name, "profile");
    EXPECT_EQ(result.binary_name, "rocprof-sys-run");
}

TEST(cli_dispatcher_test, sample_and_trace_are_unknown_verbs)
{
    for(const char* verb : { "sample", "trace" })
    {
        auto args   = argv_builder{ "rocsys", verb, "--", "./app" };
        auto result = parse_dispatch(args.argc(), args.argv());
        EXPECT_EQ(result.kind, dispatch_kind::error) << verb;
        EXPECT_NE(result.error_message.find("unknown subcommand"), std::string::npos)
            << verb;
        EXPECT_NE(result.error_message.find("error:"), std::string::npos) << verb;
        EXPECT_NE(result.error_message.find("hint:"), std::string::npos) << verb;
    }
}

TEST(cli_dispatcher_test, instrument_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "instrument", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_TRUE(result.strip_subcommand);
    EXPECT_EQ(result.binary_name, "rocprof-sys-instrument");
    EXPECT_TRUE(result.extra_flag.empty());
}

TEST(cli_dispatcher_test, rewrite_execs_instrument_with_output_flag)
{
    auto args   = argv_builder{ "rocsys", "rewrite", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_TRUE(result.strip_subcommand);
    EXPECT_EQ(result.binary_name, "rocprof-sys-instrument");
    EXPECT_EQ(result.extra_flag, "-o");
}

TEST(cli_dispatcher_test, causal_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "causal", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_EQ(result.binary_name, "rocprof-sys-causal");
}

TEST(cli_dispatcher_test, avail_execs_without_requiring_app)
{
    auto args   = argv_builder{ "rocsys", "avail" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_EQ(result.binary_name, "rocprof-sys-avail");
    EXPECT_TRUE(result.strip_subcommand);
}

TEST(cli_dispatcher_test, python_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "python", "--", "script.py" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_EQ(result.binary_name, "rocprof-sys-python");
}

TEST(cli_dispatcher_test, attach_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "attach", "--pid=1234" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_EQ(result.binary_name, "rocprof-sys-attach");
}

TEST(cli_dispatcher_test, all_named_subcommands_are_registered)
{
    EXPECT_EQ(invalid_subcommand_count(), 0);
    EXPECT_EQ(default_subcommand_count(), 1);
    EXPECT_TRUE(rocprofsys::cli::default_subcommand().is_default);
    EXPECT_EQ(rocprofsys::cli::default_subcommand().name, "profile");
    EXPECT_EQ(rocprofsys::cli::default_subcommand().mode, tool_mode::run);
}

TEST(cli_dispatcher_test, unknown_subcommand_is_error)
{
    auto args   = argv_builder{ "rocsys", "foobar" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::error);
    EXPECT_NE(result.error_message.find("error: unknown subcommand"), std::string::npos);
    EXPECT_NE(result.error_message.find("hint: run 'rocsys --help'"), std::string::npos);
}

TEST(cli_dispatcher_test, profile_without_app_is_error)
{
    auto args   = argv_builder{ "rocsys", "profile" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::error);
    EXPECT_NE(result.error_message.find("error: missing application argument"),
              std::string::npos);
    EXPECT_NE(result.error_message.find("hint:"), std::string::npos);
}

TEST(cli_dispatcher_test, profile_help_is_forwarded)
{
    auto args   = argv_builder{ "rocsys", "profile", "--help" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::in_process);
    EXPECT_TRUE(result.strip_subcommand);
}

TEST(cli_dispatcher_test, make_forwarded_argv_strips_subcommand)
{
    auto args = argv_builder{ "rocsys", "profile", "--preset=quick", "--", "./app" };
    auto fwd  = forwarded_args(args.argc(), args.argv(),
                               forward_options{ .strip_subcommand = true });
    ASSERT_EQ(fwd.size(), 4U);
    EXPECT_EQ(fwd[0], "rocsys");
    EXPECT_EQ(fwd[1], "--preset=quick");
    EXPECT_EQ(fwd[2], "--");
    EXPECT_EQ(fwd[3], "./app");
}

TEST(cli_dispatcher_test, make_forwarded_argv_prepends_rewrite_output_flag)
{
    auto args = argv_builder{ "rocsys", "rewrite", "--", "./app" };
    auto fwd  = forwarded_args(args.argc(), args.argv(), instrument_forward("-o"));
    ASSERT_EQ(fwd.size(), 4U);
    EXPECT_EQ(fwd[0], "rocprof-sys-instrument");
    EXPECT_EQ(fwd[1], "-o");
    EXPECT_EQ(fwd[2], "--");
    EXPECT_EQ(fwd[3], "./app");
}

TEST(cli_dispatcher_test, make_forwarded_argv_skips_rewrite_flag_if_output_present)
{
    auto args = argv_builder{ "rocsys", "rewrite", "-o", "app.inst", "--", "./app" };
    auto fwd  = forwarded_args(args.argc(), args.argv(), instrument_forward("-o"));
    ASSERT_EQ(fwd.size(), 5U);
    EXPECT_EQ(fwd[0], "rocprof-sys-instrument");
    EXPECT_EQ(fwd[1], "-o");
    EXPECT_EQ(fwd[2], "app.inst");
    EXPECT_EQ(fwd[3], "--");
    EXPECT_EQ(fwd[4], "./app");
}

TEST(cli_dispatcher_test, make_forwarded_argv_skips_rewrite_flag_if_long_output_present)
{
    auto args = argv_builder{ "rocsys", "rewrite", "--output=app.inst", "--", "./app" };
    auto fwd  = forwarded_args(args.argc(), args.argv(), instrument_forward("-o"));
    ASSERT_EQ(fwd.size(), 4U);
    EXPECT_EQ(fwd[0], "rocprof-sys-instrument");
    EXPECT_EQ(fwd[1], "--output=app.inst");
}

TEST(cli_dispatcher_test, make_forwarded_argv_keeps_implicit_args)
{
    auto args = argv_builder{ "rocsys", "--", "./app" };
    auto fwd  = forwarded_args(args.argc(), args.argv(), forward_options{});
    ASSERT_EQ(fwd.size(), 3U);
    EXPECT_EQ(fwd[0], "rocsys");
    EXPECT_EQ(fwd[1], "--");
    EXPECT_EQ(fwd[2], "./app");
}

TEST(cli_dispatcher_test, make_forwarded_argv_overrides_argv0)
{
    auto args = argv_builder{ "rocsys", "instrument", "--help" };
    auto fwd  = forwarded_args(args.argc(), args.argv(), instrument_forward());
    ASSERT_EQ(fwd.size(), 2U);
    EXPECT_EQ(fwd[0], "rocprof-sys-instrument");
    EXPECT_EQ(fwd[1], "--help");
}

TEST(cli_dispatcher_test, directory_of_and_join_sibling_path)
{
    EXPECT_EQ(directory_of("/usr/bin/rocsys"), "/usr/bin");
    EXPECT_EQ(directory_of("rocsys"), "");
    EXPECT_EQ(directory_of("./rocsys"), ".");
    EXPECT_EQ(directory_of("/rocsys"), "/");
    EXPECT_EQ(join_sibling_path("/usr/bin", "rocprof-sys-instrument"),
              "/usr/bin/rocprof-sys-instrument");
    EXPECT_EQ(join_sibling_path("/", "rocsys"), "/rocsys");
    EXPECT_EQ(join_sibling_path("", "rocprof-sys-avail"), "rocprof-sys-avail");
}

TEST(cli_dispatcher_test, program_name_uses_basename)
{
    EXPECT_EQ(program_name("/opt/rocm/bin/rocsys"), "rocsys");
    EXPECT_EQ(program_name("rocsys"), "rocsys");
    EXPECT_EQ(program_name(""), "rocsys");
}

TEST(cli_dispatcher_test, print_help_lists_subcommands_and_example)
{
    std::ostringstream out;
    print_help(out, "rocsys");
    const auto text = out.str();
    EXPECT_NE(text.find("Usage:"), std::string::npos);
    EXPECT_NE(text.find("rocsys -- ./app"), std::string::npos);
    EXPECT_NE(text.find("rocsys profile -- ./app"), std::string::npos);
    EXPECT_NE(text.find("rocsys rewrite -- ./app"), std::string::npos);
    EXPECT_EQ(text.find("  sample"), std::string::npos);
    EXPECT_EQ(text.find("  trace"), std::string::npos);
    const auto missing = missing_help_entries(text);
    EXPECT_TRUE(missing.empty()) << missing;
}

TEST(cli_dispatcher_test, print_version_includes_program_and_version)
{
    std::ostringstream out;
    print_version(out, "rocsys", "1.9.0");
    EXPECT_EQ(out.str(), "rocsys version 1.9.0\n");
}

TEST(cli_dispatcher_test, forwarded_argv_is_null_terminated)
{
    auto args = argv_builder{ "rocsys", "profile", "--", "./app" };
    auto fwd  = make_forwarded_argv(args.argc(), args.argv(),
                                    forward_options{ .strip_subcommand = true });
    ASSERT_GE(fwd.argc(), 1);
    EXPECT_EQ(fwd.argv()[fwd.argc()], nullptr);
}

TEST(cli_dispatcher_test, forwarded_argv_survives_move)
{
    auto args  = argv_builder{ "rocsys", "rewrite", "--", "./app" };
    auto fwd   = make_forwarded_argv(args.argc(), args.argv(), instrument_forward("-o"));
    auto moved = std::move(fwd);
    ASSERT_EQ(moved.argc(), 4);
    EXPECT_STREQ(moved.argv()[0], "rocprof-sys-instrument");
    EXPECT_STREQ(moved.argv()[1], "-o");
    EXPECT_EQ(moved.argv()[moved.argc()], nullptr);
}
