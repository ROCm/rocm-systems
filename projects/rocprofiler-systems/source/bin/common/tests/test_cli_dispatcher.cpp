// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/cli_dispatcher.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using rocprofsys::cli::directory_of;
using rocprofsys::cli::dispatch_kind;
using rocprofsys::cli::find_subcommand;
using rocprofsys::cli::join_sibling_path;
using rocprofsys::cli::make_forwarded_argv;
using rocprofsys::cli::parse_dispatch;
using rocprofsys::cli::print_help;
using rocprofsys::cli::print_version;
using rocprofsys::cli::program_name;
using rocprofsys::cli::subcommands;
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
        for(auto arg : args)
            storage.emplace_back(arg);
        ptrs.reserve(storage.size());
        for(auto& entry : storage)
            ptrs.push_back(entry.data());
    }

    [[nodiscard]] int    argc() noexcept { return static_cast<int>(storage.size()); }
    [[nodiscard]] char** argv() noexcept { return ptrs.data(); }
};

std::vector<std::string>
forwarded_args(int argc, char** argv, bool strip, std::string_view argv0 = {})
{
    auto                     fwd = make_forwarded_argv(argc, argv, strip, argv0);
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(fwd.argc()));
    for(int i = 0; i < fwd.argc(); ++i)
        out.emplace_back(fwd.argv()[i]);
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

TEST(cli_dispatcher_test, implicit_profile_with_separator)
{
    auto args   = argv_builder{ "rocsys", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::in_process);
    EXPECT_EQ(result.mode, tool_mode::sample);
    EXPECT_FALSE(result.strip_subcommand);
    EXPECT_EQ(result.subcommand_name, "profile");
}

TEST(cli_dispatcher_test, implicit_profile_with_flags)
{
    auto args   = argv_builder{ "rocsys", "--preset=quick", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::in_process);
    EXPECT_EQ(result.mode, tool_mode::sample);
    EXPECT_FALSE(result.strip_subcommand);
}

TEST(cli_dispatcher_test, profile_is_in_process_sample)
{
    auto args   = argv_builder{ "rocsys", "profile", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::in_process);
    EXPECT_EQ(result.mode, tool_mode::sample);
    EXPECT_TRUE(result.strip_subcommand);
    EXPECT_EQ(result.binary_name, "rocprof-sys-sample");
}

TEST(cli_dispatcher_test, trace_is_in_process_run)
{
    auto args   = argv_builder{ "rocsys", "trace", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::in_process);
    EXPECT_EQ(result.mode, tool_mode::run);
    EXPECT_TRUE(result.strip_subcommand);
    EXPECT_EQ(result.binary_name, "rocprof-sys-run");
}

TEST(cli_dispatcher_test, instrument_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "instrument", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_TRUE(result.strip_subcommand);
    EXPECT_EQ(result.binary_name, "rocprof-sys-instrument");
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

TEST(cli_dispatcher_test, all_seven_subcommands_are_registered)
{
    EXPECT_EQ(subcommands.size(), 7u);
    for(const auto& spec : subcommands)
        EXPECT_NE(find_subcommand(spec.name), nullptr) << spec.name;
}

TEST(cli_dispatcher_test, unknown_subcommand_is_error)
{
    auto args   = argv_builder{ "rocsys", "foobar" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::error);
    EXPECT_NE(result.error_message.find("unknown subcommand"), std::string::npos);
    EXPECT_NE(result.error_message.find("rocsys --help"), std::string::npos);
}

TEST(cli_dispatcher_test, profile_without_app_is_error)
{
    auto args   = argv_builder{ "rocsys", "profile" };
    auto result = parse_dispatch(args.argc(), args.argv());
    EXPECT_EQ(result.kind, dispatch_kind::error);
    EXPECT_NE(result.error_message.find("missing application argument"),
              std::string::npos);
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
    auto fwd  = forwarded_args(args.argc(), args.argv(), true);
    ASSERT_EQ(fwd.size(), 4u);
    EXPECT_EQ(fwd[0], "rocsys");
    EXPECT_EQ(fwd[1], "--preset=quick");
    EXPECT_EQ(fwd[2], "--");
    EXPECT_EQ(fwd[3], "./app");
}

TEST(cli_dispatcher_test, make_forwarded_argv_keeps_implicit_args)
{
    auto args = argv_builder{ "rocsys", "--", "./app" };
    auto fwd  = forwarded_args(args.argc(), args.argv(), false);
    ASSERT_EQ(fwd.size(), 3u);
    EXPECT_EQ(fwd[0], "rocsys");
    EXPECT_EQ(fwd[1], "--");
    EXPECT_EQ(fwd[2], "./app");
}

TEST(cli_dispatcher_test, make_forwarded_argv_overrides_argv0)
{
    auto args = argv_builder{ "rocsys", "instrument", "--help" };
    auto fwd  = forwarded_args(args.argc(), args.argv(), true, "rocprof-sys-instrument");
    ASSERT_EQ(fwd.size(), 2u);
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
    for(const auto& spec : subcommands)
        EXPECT_NE(text.find(std::string{ spec.name }), std::string::npos) << spec.name;
}

TEST(cli_dispatcher_test, print_version_includes_program_and_version)
{
    std::ostringstream out;
    print_version(out, "rocsys", "1.9.0");
    EXPECT_EQ(out.str(), "rocsys version 1.9.0\n");
}

TEST(cli_dispatcher_test, forwarded_argv_is_null_terminated)
{
    auto args = argv_builder{ "rocsys", "trace", "--", "./app" };
    auto fwd  = make_forwarded_argv(args.argc(), args.argv(), true);
    ASSERT_GE(fwd.argc(), 1);
    EXPECT_EQ(fwd.argv()[fwd.argc()], nullptr);
}
