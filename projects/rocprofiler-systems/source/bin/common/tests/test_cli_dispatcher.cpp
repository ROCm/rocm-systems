// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/cli_dispatcher.hpp"
#include "common/tool_runner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <initializer_list>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using rocprofsys::cli::dispatch_kind;
using rocprofsys::cli::forward_options;
using rocprofsys::cli::k_subcommands;
using rocprofsys::cli::make_forwarded_argv;
using rocprofsys::cli::parse_dispatch;
using rocprofsys::cli::print_help;
using rocprofsys::cli::subcommand_spec;
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
        const bool missing =
            std::ranges::find(k_subcommands, spec.name, &subcommand_spec::name) ==
            k_subcommands.end();
        if(unnamed || missing)
        {
            ++invalid;
        }
    }
    return invalid;
}

[[nodiscard]] int
in_process_verb_count()
{
    int count = 0;
    for(const auto& spec : k_subcommands)
    {
        if(spec.mode.has_value())
        {
            ++count;
        }
    }
    return count;
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
    return make_forwarded_argv(argc, argv, options);
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
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::in_process);
    EXPECT_EQ(result.spec->mode, tool_mode::run);
    EXPECT_FALSE(result.strip_subcommand);
    EXPECT_EQ(result.spec->name, "profile");
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-run");
}

TEST(cli_dispatcher_test, implicit_default_with_flags)
{
    auto args   = argv_builder{ "rocsys", "--preset=quick", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::in_process);
    EXPECT_EQ(result.spec->mode, tool_mode::run);
    EXPECT_FALSE(result.strip_subcommand);
}

TEST(cli_dispatcher_test, profile_is_in_process_run)
{
    auto args   = argv_builder{ "rocsys", "profile", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::in_process);
    EXPECT_EQ(result.spec->mode, tool_mode::run);
    EXPECT_TRUE(result.strip_subcommand);
    EXPECT_EQ(result.spec->name, "profile");
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-run");
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
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_TRUE(result.strip_subcommand);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-instrument");
    EXPECT_TRUE(result.spec->extra_flag.empty());
    EXPECT_FALSE(result.spec->mode.has_value());
}

TEST(cli_dispatcher_test, rewrite_execs_instrument_with_output_flag)
{
    auto args   = argv_builder{ "rocsys", "rewrite", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_TRUE(result.strip_subcommand);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-instrument");
    EXPECT_EQ(result.spec->extra_flag, "-o");
    EXPECT_FALSE(result.spec->mode.has_value());
}

TEST(cli_dispatcher_test, causal_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "causal", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv());
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-causal");
}

TEST(cli_dispatcher_test, avail_execs_without_requiring_app)
{
    auto args   = argv_builder{ "rocsys", "avail" };
    auto result = parse_dispatch(args.argc(), args.argv());
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-avail");
    EXPECT_TRUE(result.strip_subcommand);
}

TEST(cli_dispatcher_test, python_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "python", "--", "script.py" };
    auto result = parse_dispatch(args.argc(), args.argv());
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-python");
}

TEST(cli_dispatcher_test, attach_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "attach", "--pid=1234" };
    auto result = parse_dispatch(args.argc(), args.argv());
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-attach");
}

TEST(cli_dispatcher_test, all_named_subcommands_are_registered)
{
    EXPECT_EQ(invalid_subcommand_count(), 0);
    EXPECT_EQ(k_subcommands.front().name, "profile");
    EXPECT_EQ(k_subcommands.front().mode, tool_mode::run);
    EXPECT_EQ(in_process_verb_count(), 1);
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
