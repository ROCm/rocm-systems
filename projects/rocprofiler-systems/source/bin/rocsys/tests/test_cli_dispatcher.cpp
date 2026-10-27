// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cli_dispatcher.hpp"
#include "verb.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <initializer_list>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using rocprofsys::cli::dispatch_kind;
using rocprofsys::cli::parse_dispatch;
using rocprofsys::cli::print_help;
using rocprofsys::common_utils::tool_mode;
using rocprofsys::rocsys::capability;
using rocprofsys::rocsys::has_capability;
using rocprofsys::rocsys::k_verbs;
using rocprofsys::rocsys::verb_spec;

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

[[nodiscard]] int
invalid_subcommand_count()
{
    int invalid = 0;
    for(const auto& spec : k_verbs)
    {
        const bool unnamed = spec.name.empty();
        const bool duplicate =
            std::ranges::count(k_verbs, spec.name, &verb_spec::name) > 1;
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
    for(const auto& spec : k_verbs)
    {
        if(has_capability(spec.caps, capability::in_process))
        {
            ++count;
        }
    }
    return count;
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
    EXPECT_EQ(result.kind, dispatch_kind::in_process);
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
    EXPECT_EQ(result.kind, dispatch_kind::in_process);
    EXPECT_EQ(result.spec->mode, tool_mode::run);
    EXPECT_FALSE(result.strip_subcommand);
}

TEST(cli_dispatcher_test, profile_is_in_process_run)
{
    auto args   = argv_builder{ "rocsys", "profile", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
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
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_TRUE(result.strip_subcommand);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-instrument");
    EXPECT_FALSE(has_capability(result.spec->caps, capability::embeds_env_in_binary));
    EXPECT_FALSE(result.spec->mode.has_value());
}

TEST(cli_dispatcher_test, rewrite_execs_instrument_with_output_flag)
{
    auto args   = argv_builder{ "rocsys", "rewrite", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_TRUE(result.strip_subcommand);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-instrument");
    EXPECT_TRUE(has_capability(result.spec->caps, capability::embeds_env_in_binary));
    EXPECT_TRUE(has_capability(result.spec->caps, capability::honours_rocm_tracing));
    EXPECT_FALSE(result.spec->mode.has_value());
}

TEST(cli_dispatcher_test, causal_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "causal", "--", "./app" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-causal");
}

TEST(cli_dispatcher_test, avail_execs_without_requiring_app)
{
    auto args   = argv_builder{ "rocsys", "avail" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-avail");
    EXPECT_TRUE(result.strip_subcommand);
}

TEST(cli_dispatcher_test, python_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "python", "--", "script.py" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-python");
}

TEST(cli_dispatcher_test, attach_execs_sibling_binary)
{
    auto args   = argv_builder{ "rocsys", "attach", "--pid=1234" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    ASSERT_NE(result.spec, nullptr);
    EXPECT_EQ(result.kind, dispatch_kind::exec_tool);
    EXPECT_EQ(result.spec->binary_name, "rocprof-sys-attach");
}

TEST(cli_dispatcher_test, subcommand_names_are_unique_and_nonempty)
{
    EXPECT_EQ(invalid_subcommand_count(), 0);
    EXPECT_EQ(k_verbs.front().name, "profile");
    EXPECT_EQ(k_verbs.front().mode, tool_mode::run);
    EXPECT_TRUE(has_capability(k_verbs.front().caps, capability::in_process));
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

TEST(cli_dispatcher_test, profile_help_is_forwarded)
{
    auto args   = argv_builder{ "rocsys", "profile", "--help" };
    auto result = parse_dispatch(args.argc(), args.argv(), "rocsys");
    EXPECT_EQ(result.kind, dispatch_kind::in_process);
    EXPECT_TRUE(result.strip_subcommand);
}

TEST(cli_dispatcher_test, attach_needs_an_operand_but_not_an_application)
{
    const auto* attach =
        std::ranges::find(k_verbs, std::string_view{ "attach" }, &verb_spec::name);
    ASSERT_NE(attach, k_verbs.end());
    EXPECT_TRUE(has_capability(attach->caps, capability::needs_operand));
    EXPECT_TRUE(has_capability(attach->caps, capability::honours_rocm_tracing));
    EXPECT_FALSE(has_capability(attach->caps, capability::takes_app));

    const auto* avail =
        std::ranges::find(k_verbs, std::string_view{ "avail" }, &verb_spec::name);
    ASSERT_NE(avail, k_verbs.end());
    EXPECT_TRUE(has_capability(avail->caps, capability::query_only));
    EXPECT_FALSE(has_capability(avail->caps, capability::needs_operand));
}

TEST(cli_dispatcher_test, print_help_lists_subcommands_and_example)
{
    std::ostringstream out;
    print_help(out, "rocsys");
    const auto text = out.str();
    EXPECT_NE(text.find("Usage:\n  rocsys [subcommand] [options] -- <app> [app-args]\n"),
              std::string::npos);
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
    EXPECT_NE(text.find("\nCommon options:\n"), std::string::npos);
    EXPECT_NE(text.find("  --dry-run     Show the launch plan and exit\n"),
              std::string::npos);
    EXPECT_NE(
        text.find("Common options:\n  --dry-run     Show the launch plan and exit\n"),
        std::string::npos);
    EXPECT_LT(text.find("Subcommands:\n"), text.find("Common options:\n"));
    EXPECT_LT(text.find("Common options:\n"), text.find("Examples:\n"));
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
