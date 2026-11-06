// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_benchmark_cli.h"

#include <gtest/gtest.h>

namespace rocjitsu::tools {
namespace {

TEST(DecodeBenchmarkCli, ProductionRegistrationAndCustomTypedFlags) {
  cli::Command command{"benchmark"};
  DecodeBenchmarkFlags flags;
  util::StringDiagnostic diagnostic;
  ASSERT_TRUE(register_decode_benchmark_flags(command, flags, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(command.find_flag("-t")->parse_and_set("gfx950", diagnostic.emitter()).succeeded());
  EXPECT_EQ(flags.target->value(), ROCJITSU_CODE_TARGET_GFX950);
  ASSERT_TRUE(
      flags.target->set_value(ROCJITSU_CODE_TARGET_GFX942, diagnostic.emitter()).succeeded());
  EXPECT_EQ(command.find_flag("--target")->format_value(), "gfx942");
  EXPECT_TRUE(flags.target->set_value(ROCJITSU_CODE_TARGET_INVALID, diagnostic.emitter()).failed());
  EXPECT_TRUE(flags.iterations->set_value(0, diagnostic.emitter()).failed());
  EXPECT_TRUE(command.find_flag("--iterations")->parse_and_set("0", diagnostic.emitter()).failed());
  ASSERT_TRUE(flags.target->parse_and_set("auto", diagnostic.emitter()).succeeded());
  EXPECT_FALSE(flags.target->value().has_value());
}

TEST(DecodeBenchmarkCli, ExistingAndExtendedSpellings) {
  util::StringDiagnostic diagnostic;
  cli::Command command{"benchmark"};
  DecodeBenchmarkFlags flags;
  ASSERT_TRUE(register_decode_benchmark_flags(command, flags, diagnostic.emitter()).succeeded());
  const std::vector<std::string_view> args{"input.co", "--target",        "gfx950", "--iterations",
                                           "3",        "--invalid-limit", "0"};
  auto parsed = cli::parse(command, args, diagnostic.emitter());
  ASSERT_TRUE(parsed.succeeded()) << diagnostic.message();
  EXPECT_EQ(parsed.value().positional.front(), "input.co");
  EXPECT_EQ(flags.target->value(), ROCJITSU_CODE_TARGET_GFX950);
  EXPECT_EQ(flags.iterations->value(), 3u);
  EXPECT_EQ(flags.invalid_limit->value(), 0u);
  const std::vector<std::string_view> aliases{"-t=gfx942", "--iterations=1", "-n=2", "--",
                                              "--input.co"};
  parsed = cli::parse(command, aliases, diagnostic.emitter());
  ASSERT_TRUE(parsed.succeeded()) << diagnostic.message();
  EXPECT_EQ(parsed.value().positional.front(), "--input.co");
  EXPECT_EQ(flags.target->value(), ROCJITSU_CODE_TARGET_GFX942);
  EXPECT_EQ(flags.iterations->value(), 2u);
}

TEST(DecodeBenchmarkCli, BadInputAndHelpBehavior) {
  util::StringDiagnostic diagnostic;
  for (const std::vector<std::string_view> &args :
       {std::vector<std::string_view>{"--iterations=0", "a"},
        {"--target=unknown", "a"},
        {"--unknown"},
        {"--target"},
        {"--help", "--unknown"}}) {
    cli::Command command{"benchmark"};
    DecodeBenchmarkFlags flags;
    ASSERT_TRUE(register_decode_benchmark_flags(command, flags, diagnostic.emitter()).succeeded());
    EXPECT_TRUE(cli::parse(command, args, diagnostic.emitter()).failed());
  }
  for (const std::vector<std::string_view> &args :
       {std::vector<std::string_view>{"--help"}, {"input.co", "--help"}, {"-h"}}) {
    cli::Command command{"benchmark"};
    DecodeBenchmarkFlags flags;
    ASSERT_TRUE(register_decode_benchmark_flags(command, flags, diagnostic.emitter()).succeeded());
    auto parsed = cli::parse(command, args, diagnostic.emitter());
    ASSERT_TRUE(parsed.succeeded()) << diagnostic.message();
    EXPECT_TRUE(flags.help->value());
    const std::string help = cli::format_help(command);
    EXPECT_NE(help.find("--iterations, -n VALUE"), std::string::npos);
    EXPECT_NE(help.find("Target: auto"), std::string::npos);
  }
}

} // namespace
} // namespace rocjitsu::tools
