// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cli_dispatcher.hpp"
#include "common/tool_runner.hpp"
#include "rocsys/schema.hpp"

#include "parser/schema.hpp"

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
using rocprofsys::cli::print_subcommand_help;
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
        const bool duplicate =
            std::ranges::count(k_subcommands, spec.name, &subcommand_spec::name) > 1;
        if(unnamed || duplicate)
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

std::vector<std::string>
forwarded_args(int argc, char** argv, forward_options options)
{
    return make_forwarded_argv(argc, argv, options);
}

void
expect_message_contains(std::string_view                        text,
                        std::initializer_list<std::string_view> parts,
                        const char*                             context)
{
    for(const auto part : parts)
    {
        EXPECT_NE(text.find(part), std::string_view::npos) << context << ": " << part;
    }
}

void
expect_subcommand_help(std::initializer_list<const char*> tokens, std::string_view name)
{
    auto args   = argv_builder(tokens);
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    EXPECT_EQ(result.kind, dispatch_kind::show_subcommand_help) << args.storage.back();
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.spec->name, name);
}
}  // namespace

TEST(cli_dispatcher_test, no_args_shows_help)
{
    auto args   = argv_builder{ "rocsys" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    EXPECT_EQ(result.kind, dispatch_kind::show_help);
}

TEST(cli_dispatcher_test, help_flags_show_help)
{
    for(const char* flag : { "--help", "-h", "-?", "--help=all" })
    {
        auto args   = argv_builder{ "rocsys", flag };
        auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
        EXPECT_EQ(result.kind, dispatch_kind::show_help) << flag;
    }
}

TEST(cli_dispatcher_test, a_help_topic_as_the_first_token_is_a_rocsys_topic)
{
    auto bare   = argv_builder{ "rocsys", "--help" };
    auto result = parse_dispatch(bare.argc(), bare.argv(), "rocsys");
    EXPECT_EQ(result.kind, dispatch_kind::show_help);
    EXPECT_EQ(result.topic, "");

    auto topic = argv_builder{ "rocsys", "--help=tracing" };
    result     = parse_dispatch(topic.argc(), topic.argv(), "rocsys");
    EXPECT_EQ(result.kind, dispatch_kind::show_help);
    EXPECT_EQ(result.topic, "tracing");
}

TEST(cli_dispatcher_test, a_word_that_is_not_a_subcommand_asks_for_an_options_help)
{
    for(const char* flag : { "--help", "-h", "-?" })
    {
        auto args   = argv_builder{ "rocsys", "traces", flag };
        auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
        EXPECT_EQ(result.kind, dispatch_kind::show_option_help) << flag;
        EXPECT_EQ(result.topic, "traces") << flag;
        EXPECT_NE(result.error_message.find("error: unknown subcommand 'traces'"),
                  std::string::npos)
            << flag;
    }

    auto after_separator = argv_builder{ "rocsys", "traces", "--", "./app", "--help" };
    EXPECT_EQ(
        parse_dispatch(after_separator.argc(), after_separator.argv(), "rocsys").kind,
        dispatch_kind::error);
}

TEST(cli_dispatcher_test, no_rocsys_option_is_named_like_a_subcommand)
{
    constexpr std::string_view k_long_prefix = "--";
    const auto                 catalog       = rocprofsys::rocsys::make_schema();
    for(const auto& param : catalog.params())
    {
        if(!param.arg)
        {
            ADD_FAILURE() << param.id << " has no command-line spelling";
            continue;
        }
        const auto name = std::string_view{ *param.arg }.substr(k_long_prefix.size());
        EXPECT_EQ(std::ranges::find(k_subcommands, name, &subcommand_spec::name),
                  k_subcommands.end())
            << name;
    }
}

TEST(cli_dispatcher_test, version_flag_shows_version)
{
    auto args   = argv_builder{ "rocsys", "--version" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    EXPECT_EQ(result.kind, dispatch_kind::show_version);
}

TEST(cli_dispatcher_test, implicit_default_with_separator)
{
    auto args   = argv_builder{ "rocsys", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::run_subcommand);
    EXPECT_EQ(result.spec->mode, tool_mode::run);
    EXPECT_FALSE(result.strip_subcommand);
    EXPECT_EQ(result.spec->name, "profile");
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-run");
}

TEST(cli_dispatcher_test, implicit_default_with_flags)
{
    auto args   = argv_builder{ "rocsys", "--preset=quick", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::run_subcommand);
    EXPECT_EQ(result.spec->mode, tool_mode::run);
    EXPECT_FALSE(result.strip_subcommand);
}

TEST(cli_dispatcher_test, profile_is_in_process_run)
{
    auto args   = argv_builder{ "rocsys", "profile", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::run_subcommand);
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
        auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
        EXPECT_EQ(result.kind, dispatch_kind::error) << verb;
        expect_message_contains(
            result.error_message,
            { "error:", "unknown subcommand",
              "Usage: rocsys [subcommand] [options] -- <app> [app-args]", "hint:" },
            verb);
    }
}

TEST(cli_dispatcher_test, instrument_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "instrument", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::run_subcommand);
    EXPECT_TRUE(result.strip_subcommand);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-instrument");
    EXPECT_TRUE(result.spec->extra_flag.empty());
    EXPECT_FALSE(result.spec->mode.has_value());
}

TEST(cli_dispatcher_test, rewrite_execs_instrument_with_output_flag)
{
    auto args   = argv_builder{ "rocsys", "rewrite", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::run_subcommand);
    EXPECT_TRUE(result.strip_subcommand);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-instrument");
    EXPECT_EQ(result.spec->extra_flag, "-o");
    EXPECT_FALSE(result.spec->mode.has_value());
}

TEST(cli_dispatcher_test, causal_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "causal", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::run_subcommand);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-causal");
}

TEST(cli_dispatcher_test, avail_execs_without_requiring_app)
{
    auto args   = argv_builder{ "rocsys", "avail" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::run_subcommand);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-avail");
    EXPECT_TRUE(result.strip_subcommand);
}

TEST(cli_dispatcher_test, python_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "python", "--", "script.py" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::run_subcommand);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-python");
}

TEST(cli_dispatcher_test, attach_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "attach", "--pid=1234" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::run_subcommand);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-attach");
}

TEST(cli_dispatcher_test, subcommand_names_are_unique_and_nonempty)
{
    EXPECT_EQ(invalid_subcommand_count(), 0);
    EXPECT_EQ(k_subcommands.front().name, "profile");
    EXPECT_EQ(k_subcommands.front().mode, tool_mode::run);
    EXPECT_EQ(in_process_verb_count(), 1);
}

TEST(cli_dispatcher_test, unknown_subcommand_is_error)
{
    auto args   = argv_builder{ "rocsys", "foobar" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    EXPECT_EQ(result.kind, dispatch_kind::error);
    EXPECT_NE(result.error_message.find("error: unknown subcommand"), std::string::npos);
    EXPECT_NE(result.error_message.find(
                  "Usage: rocsys [subcommand] [options] -- <app> [app-args]"),
              std::string::npos);
    EXPECT_NE(result.error_message.find("hint: run 'rocsys --help'"), std::string::npos);
}

TEST(cli_dispatcher_test, profile_without_app_is_error)
{
    auto args   = argv_builder{ "rocsys", "profile" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    EXPECT_EQ(result.kind, dispatch_kind::error);
    EXPECT_NE(result.error_message.find("error: missing application argument"),
              std::string::npos);
    EXPECT_NE(result.error_message.find(
                  "Usage: rocsys [subcommand] [options] -- <app> [app-args]"),
              std::string::npos);
    EXPECT_NE(result.error_message.find("hint:"), std::string::npos);
}

TEST(cli_dispatcher_test, an_application_without_the_separator_is_an_error)
{
    const auto expect_refused = [](std::initializer_list<const char*> tokens,
                                   std::string_view                   shown) {
        auto args   = argv_builder{ tokens };
        auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
        EXPECT_EQ(result.kind, dispatch_kind::error);
        EXPECT_EQ(result.error_message.find(shown), 0U) << result.error_message;
        EXPECT_NE(result.error_message.find(
                      "Usage: rocsys [subcommand] [options] -- <app> [app-args]"),
                  std::string::npos);
    };
    expect_refused({ "rocsys", "python", "./script.py", "--log-level", "info" },
                   "error: put '--' before the application: rocsys python [options] -- "
                   "./script.py [app-args]\n");
    expect_refused({ "rocsys", "python", "./script.py", "--help" },
                   "error: put '--' before the application: rocsys python [options] -- "
                   "./script.py [app-args]\n");
    expect_refused({ "rocsys", "profile", "./app" },
                   "error: put '--' before the application: rocsys profile [options] -- "
                   "./app [app-args]\n");
    expect_refused({ "rocsys", "causal", "./app" },
                   "error: put '--' before the application: rocsys causal [options] -- "
                   "./app [app-args]\n");
    expect_refused({ "rocsys", "python", "-v", "./script.py", "--log-level", "info" },
                   "error: put '--' before the application: rocsys python [options] -- "
                   "<app> [app-args]\n");
    expect_refused({ "rocsys", "rewrite", "-o", "out.inst", "./transpose" },
                   "error: put '--' before the application: rocsys rewrite [options] -- "
                   "<app> [app-args]\n");
    expect_refused({ "rocsys", "--traces", "hip", "./app" },
                   "error: put '--' before the application: rocsys [options] -- <app> "
                   "[app-args]\n");
    expect_refused({ "rocsys", "profile", "--preset", "balanced", "--export-config" },
                   "error: put '--' before the application: rocsys profile [options] -- "
                   "<app> [app-args]\n");
}

TEST(cli_dispatcher_test, a_subcommand_without_an_application_refuses_the_separator)
{
    const auto expect_refused = [](std::initializer_list<const char*> tokens,
                                   std::string_view                   message) {
        auto args   = argv_builder{ tokens };
        auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
        EXPECT_EQ(result.kind, dispatch_kind::error);
        EXPECT_EQ(result.error_message, message);
    };
    expect_refused({ "rocsys", "attach", "--log-level", "info", "--", "-p", "1" },
                   "error: rocsys attach runs no application, so it takes no '--'\n"
                   "hint: run 'rocsys attach --help' for its options.");
    expect_refused({ "rocsys", "avail", "--", "gpu" },
                   "error: rocsys avail runs no application, so it takes no '--'\n"
                   "hint: run 'rocsys avail --help' for its options.");
}

TEST(cli_dispatcher_test, the_separator_rule_leaves_other_forms_alone)
{
    const auto expect_kind = [](std::initializer_list<const char*> tokens,
                                dispatch_kind                      expected) {
        auto args = argv_builder{ tokens };
        EXPECT_EQ(parse_dispatch(args.argc(), args.argv(), "rocsys").kind, expected)
            << ::testing::PrintToString(args.storage);
    };
    constexpr auto k_run = dispatch_kind::run_subcommand;
    expect_kind({ "rocsys", "python", "--", "./script.py", "--log-level", "info" },
                k_run);
    expect_kind({ "rocsys", "rewrite", "-o", "out", "--", "./app" }, k_run);
    // --export-config prints the configuration and exits; "--" still ends the options.
    expect_kind({ "rocsys", "profile", "--preset", "balanced", "--export-config", "--" },
                k_run);
    expect_kind({ "rocsys", "python", "--help" }, dispatch_kind::show_subcommand_help);
    expect_kind({ "rocsys", "python", "--help=all" }, k_run);
    // rocprof-sys-avail reads a plain first token as its REGEX_FILTER.
    expect_kind({ "rocsys", "avail", "gpu" }, k_run);
    expect_kind({ "rocsys", "attach", "-p", "1" }, k_run);
    // rocprof-sys-attach answers a missing or plain pid with its own -p <pid> usage.
    expect_kind({ "rocsys", "attach" }, k_run);
    expect_kind({ "rocsys", "attach", "1234" }, k_run);
}

TEST(cli_dispatcher_test, subcommand_help_flags_show_rocsys_help)
{
    for(const char* name :
        { "profile", "avail", "attach", "causal", "instrument", "rewrite", "python" })
    {
        for(const char* flag : { "--help", "-h", "-?" })
        {
            SCOPED_TRACE(name);
            expect_subcommand_help({ "rocsys", name, flag }, name);
        }
    }
}

TEST(cli_dispatcher_test, help_flag_anywhere_before_the_separator)
{
    expect_subcommand_help(
        { "rocsys", "instrument", "--traces", "hip", "-h", "--", "./app" }, "instrument");
}

TEST(cli_dispatcher_test, implicit_profile_help)
{
    auto args   = argv_builder{ "rocsys", "--traces", "hip", "--help" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    EXPECT_EQ(result.kind, dispatch_kind::show_subcommand_help);
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.spec->name, "profile");
    EXPECT_FALSE(result.strip_subcommand);
}

TEST(cli_dispatcher_test, help_after_the_separator_goes_to_the_app)
{
    auto args   = argv_builder{ "rocsys", "python", "--", "./script.py", "--help" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    EXPECT_EQ(result.kind, dispatch_kind::run_subcommand);
}

TEST(cli_dispatcher_test, help_topic_goes_to_the_tool)
{
    auto args   = argv_builder{ "rocsys", "profile", "--help=all" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    EXPECT_EQ(result.kind, dispatch_kind::run_subcommand);
    EXPECT_TRUE(result.strip_subcommand);
}

TEST(cli_dispatcher_test, print_subcommand_help_layout)
{
    const auto instrument =
        std::ranges::find(k_subcommands, "instrument", &subcommand_spec::name);
    ASSERT_NE(instrument, k_subcommands.end());
    std::ostringstream out;
    print_subcommand_help(out, "rocsys", *instrument, "<rocsys options>\n\n");
    EXPECT_EQ(out.str(),
              "Usage:\n"
              "  rocsys instrument [options]\n"
              "\n"
              "Runtime instrumentation profiling. Runs rocprof-sys-instrument.\n"
              "\n"
              "<rocsys options>\n"
              "\n"
              "rocprof-sys-instrument\n"
              "  Every other option is passed to rocprof-sys-instrument unchanged.\n"
              "  Run 'rocprof-sys-instrument --help' to list them.\n");
}

TEST(cli_dispatcher_test, rewrite_help_says_what_it_writes)
{
    const auto rewrite =
        std::ranges::find(k_subcommands, "rewrite", &subcommand_spec::name);
    ASSERT_NE(rewrite, k_subcommands.end());
    std::ostringstream out;
    print_subcommand_help(out, "rocsys", *rewrite, "<rocsys options>\n\n");
    EXPECT_EQ(
        out.str(),
        "Usage:\n"
        "  rocsys rewrite [options]\n"
        "\n"
        "Binary rewrite application. Runs rocprof-sys-instrument with the -o option "
        "to generate an instrumented binary, such as app.inst for ./app.\n"
        "\n"
        "rocprof-sys-instrument\n"
        "  Every other option is passed to rocprof-sys-instrument unchanged.\n"
        "  Run 'rocprof-sys-instrument --help' to list them.\n");
}

TEST(cli_dispatcher_test, rewrite_avail_and_causal_help_leave_out_the_rocsys_options)
{
    for(const auto& spec : k_subcommands)
    {
        std::ostringstream out;
        print_subcommand_help(out, "rocsys", spec, "<rocsys options>\n\n");
        const bool left_out =
            spec.name == "rewrite" || spec.name == "avail" || spec.name == "causal";
        EXPECT_EQ(out.str().find("<rocsys options>") == std::string::npos, left_out)
            << spec.name;
    }
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
    print_help(out, "rocsys", "<rocsys options>\n\n");
    const auto text = out.str();
    EXPECT_NE(text.find("Usage:\n  rocsys [subcommand] [options] -- <app> [app-args]\n"),
              std::string::npos);
    EXPECT_NE(text.find("rocsys options:\n<rocsys options>\n\nExamples:\n"),
              std::string::npos)
        << text;
    EXPECT_NE(
        text.find("Use 'rocsys <subcommand> --help' for subcommand-specific options.\n"
                  "Use 'rocsys <option> --help' for one rocsys option in full, such "
                  "as\n'rocsys traces --help', 'rocsys --help=<category>' for one "
                  "category,\nand 'rocsys --help=all' for every option.\n\n"
                  "Documentation: "),
        std::string::npos)
        << text;
    EXPECT_NE(text.find("rocsys --traces hip mpi -- ./app"), std::string::npos);
    EXPECT_NE(text.find("ROCm Systems Profiler."), std::string::npos);
    EXPECT_NE(text.find("Experimental: This is a preview of the rocsys command-line "
                        "tool."),
              std::string::npos);
    EXPECT_NE(text.find("Commands and workflows are subject to change."),
              std::string::npos);
    EXPECT_EQ(text.find("unified command-line entry point"), std::string::npos);
    EXPECT_NE(text.find("rocsys -- ./app"), std::string::npos);
    EXPECT_NE(text.find("rocsys profile -- ./app"), std::string::npos);
    EXPECT_NE(text.find("rocsys rewrite -- ./app"), std::string::npos);
    EXPECT_NE(
        text.find("Documentation: "
                  "https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/"),
        std::string::npos);
    EXPECT_EQ(text.find("  sample"), std::string::npos);
    EXPECT_EQ(text.find("  trace"), std::string::npos);
    const auto profile    = text.find("\n  profile       ");
    const auto avail      = text.find("\n  avail         ");
    const auto attach     = text.find("\n  attach        ");
    const auto causal     = text.find("\n  causal        ");
    const auto instrument = text.find("\n  instrument    ");
    const auto rewrite    = text.find("\n  rewrite       ");
    const auto python     = text.find("\n  python        ");
    ASSERT_NE(profile, std::string::npos);
    ASSERT_NE(avail, std::string::npos);
    ASSERT_NE(attach, std::string::npos);
    ASSERT_NE(causal, std::string::npos);
    ASSERT_NE(instrument, std::string::npos);
    ASSERT_NE(rewrite, std::string::npos);
    ASSERT_NE(python, std::string::npos);
    EXPECT_LT(profile, avail);
    EXPECT_LT(avail, attach);
    EXPECT_LT(attach, causal);
    EXPECT_LT(causal, instrument);
    EXPECT_LT(instrument, rewrite);
    EXPECT_LT(rewrite, python);
}
