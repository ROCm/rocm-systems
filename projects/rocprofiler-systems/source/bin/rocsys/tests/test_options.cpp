// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocsys/options.hpp"
#include "rocsys/schema.hpp"

#include "common/env_vars.hpp"
#include "parser/schema.hpp"

#include <gtest/gtest.h>

#include <stdlib.h>  // POSIX ::setenv and ::unsetenv

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::rocsys::testing
{
namespace
{
using tokens = std::vector<std::string>;

/// One command line, the tokens after the subcommand, and what rocsys makes of it.
struct example
{
    tokens              args;
    parser::value_map_t environment;
    tokens              tool_args;
};

/// The domain of every --traces ROCm API name, as an SDK that supports them all lists
/// them.
tokens
every_domain()
{
    return { "hip_api",     "hsa_api",           "marker_api",  "kernel_dispatch",
             "memory_copy", "memory_allocation", "rocshmem_api" };
}

/// parse_options() with an SDK that supports every --traces domain.
[[nodiscard]] options
parse(const tokens& args)
{
    return parse_options(args, every_domain());
}

/// options_help() with an SDK that supports every --traces domain.
[[nodiscard]] std::optional<std::string>
help(std::string_view topic)
{
    return options_help(topic, every_domain());
}

void
expect_options(const example& item)
{
    const auto found = parse(item.args);
    EXPECT_TRUE(found.errors.empty()) << found.errors.front();
    EXPECT_EQ(found.environment, item.environment);
    EXPECT_EQ(found.tool_args, item.tool_args);
    EXPECT_EQ(found.command, tokens{ "./app" });
}

[[nodiscard]] bool
reports(const tokens& args, std::string_view message)
{
    const auto found = parse(args);
    return std::ranges::any_of(found.errors, [message](const std::string& error) {
        return error.find(message) != std::string::npos;
    });
}

TEST(rocsys_options, accepts_the_jira_examples)
{
    const auto hip      = traces_environment({ "hip" });
    const auto hip_mpi  = traces_environment({ "hip", "mpi" });
    const auto examples = std::vector<example>{
        { .args        = { "--traces", "hip", "--", "./app" },
          .environment = hip,
          .tool_args   = {} },
        { .args        = { "--traces", "hip", "mpi", "--", "./app" },
          .environment = hip_mpi,
          .tool_args   = {} },
        { .args        = { "--traces", "hip,mpi", "--", "./app" },
          .environment = hip_mpi,
          .tool_args   = {} },
        { .args        = { "--traces", "hip,", "mpi", "--", "./app" },
          .environment = hip_mpi,
          .tool_args   = {} },
        { .args        = { "--traces", "hip", "--traces", "mpi", "--", "./app" },
          .environment = hip_mpi,
          .tool_args   = {} },
        { .args        = { "--traces", "hip", "--traces", "hip", "--", "./app" },
          .environment = hip,
          .tool_args   = {} },
        { .args        = { "--traces", "mpi", "--", "./app" },
          .environment = traces_environment({ "mpi" }),
          .tool_args   = {} },
        { .args = { "--output-path", "out", "--log-level", "debug", "--", "./app" },
          .environment = { { env_vars::LOG_LEVEL, "debug" },
                           { env_vars::OUTPUT_PATH, "out" } },
          .tool_args   = {} },
        { .args        = { "-T", "--traces", "hip", "-o", "out", "--", "./app" },
          .environment = hip,
          .tool_args   = { "-T", "-o", "out" } },
        { .args        = { "--gpu=temp", "--traces", "hip", "--", "./app" },
          .environment = hip,
          .tool_args   = { "--gpu=temp" } },
        { .args        = { "--traces", "hip", "--gpu=temp", "--", "./app" },
          .environment = hip,
          .tool_args   = { "--gpu=temp" } },
        { .args        = { "-t", "0", "--traces", "hip", "--", "./app" },
          .environment = hip,
          .tool_args   = { "-t", "0" } },
        { .args        = { "--traces", "kernel-dispatch", "-E", "foo", "--", "./app" },
          .environment = traces_environment({ "kernel-dispatch" }),
          .tool_args   = { "-E", "foo" } },
    };
    for(const auto& item : examples)
    {
        SCOPED_TRACE(::testing::PrintToString(item.args));
        expect_options(item);
    }
}

TEST(rocsys_options, forwards_tool_options_in_order)
{
    const auto found = parse({ "-T", "--traces", "hip", "-o", "out", "prefix",
                               "--preset=balanced", "--", "./app", "--traces", "x" });
    EXPECT_TRUE(found.errors.empty());
    EXPECT_EQ(found.tool_args,
              (tokens{ "-T", "-o", "out", "prefix", "--preset=balanced" }));
    EXPECT_EQ(found.command, (tokens{ "./app", "--traces", "x" }));
    EXPECT_EQ(found.environment, traces_environment({ "hip" }));
}

TEST(rocsys_options, writes_only_given_options)
{
    const auto found = parse({ "-T", "--", "./app" });
    EXPECT_TRUE(found.errors.empty());
    EXPECT_TRUE(found.environment.empty());
    EXPECT_EQ(found.tool_args, tokens{ "-T" });
}

TEST(rocsys_options, does_not_read_the_shell_environment)
{
    ASSERT_EQ(::setenv(env_vars::LOG_LEVEL, "WARNING", 1), 0);
    const auto found = parse({ "--", "./app" });
    ::unsetenv(env_vars::LOG_LEVEL);
    EXPECT_TRUE(found.errors.empty());
    EXPECT_TRUE(found.environment.empty());
}

TEST(rocsys_options, log_level_takes_every_spelling_the_tool_reads)
{
    for(const auto* level : { "trace", "debug", "info", "warn", "warning", "error", "err",
                              "critical", "off" })
    {
        SCOPED_TRACE(level);
        const auto found = parse({ "--log-level", level, "--", "./app" });
        EXPECT_TRUE(found.errors.empty());
        EXPECT_EQ(found.environment,
                  (parser::value_map_t{ { env_vars::LOG_LEVEL, level } }));
    }
}

TEST(rocsys_options, reports_bad_values)
{
    EXPECT_TRUE(reports({ "--traces", "hip", "bogus", "--", "./app" },
                        "invalid value 'bogus' for parameter 'traces'"));
    EXPECT_TRUE(reports({ "--traces", "hip", "./app" },
                        "invalid value './app' for parameter 'traces'"));
    EXPECT_TRUE(reports({ "--log-level", "loud", "--", "./app" },
                        "invalid value 'loud' for parameter 'log_level'"));
    EXPECT_TRUE(
        reports({ "--traces", "--", "./app" }, "option '--traces' requires a value"));
}

TEST(rocsys_options, equals_form_takes_names_separated_by_commas)
{
    const auto found = parse({ "--traces=hip", "mpi", "--", "./app" });
    ASSERT_EQ(found.errors.size(), 1U);
    EXPECT_EQ(found.errors.front(),
              "unexpected argument 'mpi'; did you mean '--traces=hip,mpi'?");
    EXPECT_TRUE(found.tool_args.empty());
    EXPECT_TRUE(parse({ "--traces=hip,mpi", "--", "./app" }).errors.empty());
}

TEST(rocsys_options, a_word_right_after_a_rocsys_value_is_refused)
{
    for(const auto& args : { tokens{ "--log-level", "debug", "extra", "--", "env" },
                             tokens{ "--output-path=out", "extra", "--", "env" } })
    {
        SCOPED_TRACE(args.front());
        const auto found = parse(args);
        ASSERT_EQ(found.errors.size(), 1U);
        EXPECT_EQ(found.errors.front(), "unexpected argument 'extra'");
        EXPECT_TRUE(found.tool_args.empty());
    }
}

TEST(rocsys_options, refuses_a_trace_name_the_installed_sdk_does_not_support)
{
    const auto refused =
        parse_options({ "--traces", "hip", "rocshmem", "--", "./app" }, { "hip_api" });
    EXPECT_EQ(
        refused.errors,
        (tokens{ "invalid value 'rocshmem' for parameter 'traces' from command line" }));
    EXPECT_TRUE(refused.environment.empty());

    const auto accepted =
        parse_options({ "--traces", "hip", "mpi", "--", "./app" }, { "hip_api" });
    EXPECT_TRUE(accepted.errors.empty());
    EXPECT_EQ(accepted.environment, traces_environment({ "hip", "mpi" }));
}

TEST(rocsys_options, help_lists_only_the_trace_names_the_installed_sdk_supports)
{
    const auto traces =
        options_help("--traces", { "hip_api", "kernel_dispatch" }).value_or("");
    EXPECT_NE(traces.find("      choices: rccl mpi openmp ucx openshmem hip-api hip "
                          "kernel-dispatch\n"),
              std::string::npos)
        << traces;
}

TEST(rocsys_options, summary_lists_each_option_on_one_line)
{
    EXPECT_EQ(
        options_summary(),
        "tracing\n"
        "  --traces       APIs and runtimes to trace; names left out are not traced\n"
        "\n"
        "output\n"
        "  --output-path  Directory for output files\n"
        "\n"
        "logging\n"
        "  --log-level    Log level\n"
        "\n");
}

TEST(rocsys_options, help_prints_a_category_or_one_option_in_full)
{
    const auto* logging =
        "logging\n"
        "  --log-level <value>\n"
        "      Log level\n"
        "      choices: trace debug info warn warning error err critical off\n"
        "      env: ROCPROFSYS_LOG_LEVEL\n"
        "\n";
    EXPECT_EQ(help("logging"), logging);
    EXPECT_EQ(help("--log-level"), logging);

    const auto traces = help("--traces").value_or("");
    EXPECT_NE(traces.find("  --traces <value>\n"), std::string::npos) << traces;
    EXPECT_NE(
        traces.find("choices: rccl mpi openmp ucx openshmem hip-api hip hsa-api hsa "
                    "roctx kernel-dispatch memory-copy memory-allocation rocshmem\n"),
        std::string::npos)
        << traces;
    EXPECT_EQ(traces.find("--output-path"), std::string::npos) << traces;
}

TEST(rocsys_options, help_all_prints_every_option_in_full)
{
    const auto all = help("all").value_or("");
    for(const auto* usage : { "  --traces <value>\n", "  --output-path <value>\n",
                              "  --log-level <value>\n" })
    {
        EXPECT_NE(all.find(usage), std::string::npos) << usage;
    }
}

TEST(rocsys_options, help_has_nothing_for_a_topic_the_catalog_lacks)
{
    EXPECT_EQ(help("bogus"), std::nullopt);
    EXPECT_EQ(help("--bogus"), std::nullopt);
    EXPECT_EQ(help("traces"), std::nullopt);
}
}  // namespace
}  // namespace rocprofsys::rocsys::testing
