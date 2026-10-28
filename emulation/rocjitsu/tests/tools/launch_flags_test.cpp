// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "launch_flags.h"
#include <gtest/gtest.h>

namespace rocjitsu::cli {
TEST(LaunchFlags, OmissionExplicitZeroAndRepeatedOverrides) {
  Command command{"rocjitsu", PositionalMode::AfterSeparator};
  LaunchFlags flags;
  util::StringDiagnostic diagnostic;
  ASSERT_TRUE(flags.register_flags(command, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(parse(command, {}, diagnostic.emitter()).succeeded());
  EXPECT_FALSE(flags.cpu_budget->value());
  EXPECT_FALSE(flags.wait->value());
  const std::vector<std::string_view> args{"--node=node-0",
                                           "--vfio-socket",
                                           "/tmp/gpu0.sock",
                                           "--vfio-socket=/tmp/gpu1.sock",
                                           "--cpu-thread-budget=0",
                                           "--wait-checking=off",
                                           "--wait-checking=on",
                                           "--",
                                           "application",
                                           "--wait-checking=all",
                                           ""};
  auto parsed = parse(command, args, diagnostic.emitter());
  ASSERT_TRUE(parsed.succeeded()) << diagnostic.message();
  EXPECT_EQ(flags.node->value(), "node-0");
  EXPECT_EQ(flags.vfio_socket->values(),
            (std::vector<std::string>{"/tmp/gpu0.sock", "/tmp/gpu1.sock"}));
  EXPECT_EQ(flags.cpu_budget->value(), 0u);
  EXPECT_EQ(flags.wait->value(), config::WaitChecking::On);
  EXPECT_EQ(parsed.value().positional,
            (std::vector<std::string>{"application", "--wait-checking=all", ""}));
}

TEST(LaunchFlags, GenericAndTypedUpdatesValidateTheSameStorage) {
  Command command{"rocjitsu"};
  LaunchFlags flags;
  util::StringDiagnostic diagnostic;
  ASSERT_TRUE(flags.register_flags(command, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(
      command.find_flag("--wait-checking")->parse_and_set("all", diagnostic.emitter()).succeeded());
  EXPECT_EQ(flags.wait->value(), config::WaitChecking::All);
  EXPECT_TRUE(
      flags.wait->set_value(static_cast<config::WaitChecking>(99), diagnostic.emitter()).failed());
  ASSERT_TRUE(flags.wait->set_value(std::nullopt, diagnostic.emitter()).succeeded());
  EXPECT_FALSE(flags.wait->value());
  EXPECT_TRUE(flags.cpu_budget->parse_and_set("4294967296", diagnostic.emitter()).failed());
  EXPECT_TRUE(flags.cpu_budget->parse_and_set("-1", diagnostic.emitter()).failed());
  ASSERT_TRUE(flags.cpu_budget->parse_and_set("4294967295", diagnostic.emitter()).succeeded());
  EXPECT_EQ(flags.cpu_budget->value(), UINT32_MAX);
  ASSERT_TRUE(flags.cpu_budget->set_value(std::nullopt, diagnostic.emitter()).succeeded());
  EXPECT_FALSE(flags.cpu_budget->value());
  EXPECT_TRUE(flags.vfio_ready_fd->parse_and_set("2147483648", diagnostic.emitter()).failed());
  EXPECT_TRUE(flags.vfio_ready_fd->set_value(UINT32_MAX, diagnostic.emitter()).failed());
  EXPECT_TRUE(flags.config->set_value("", diagnostic.emitter()).failed());
  EXPECT_TRUE(flags.node->set_value("", diagnostic.emitter()).failed());
  EXPECT_TRUE(flags.vfio_socket->parse_and_set("", diagnostic.emitter()).failed());
}

TEST(LaunchFlags, IndependentInvocationsAndHelpMetadata) {
  Command command{"rocjitsu"};
  LaunchFlags flags;
  LaunchFlags other;
  util::StringDiagnostic diagnostic;
  ASSERT_TRUE(flags.register_flags(command, diagnostic.emitter()).succeeded());
  ASSERT_TRUE(flags.wait->parse_and_set("off", diagnostic.emitter()).succeeded());
  EXPECT_FALSE(other.wait->value());
  const auto help = format_help(command);
  EXPECT_NE(help.find("Simulation configuration:"), std::string::npos);
  EXPECT_NE(help.find("--vfio-ready-fd VALUE"), std::string::npos);
  EXPECT_EQ(command.find_flag("-v"), flags.version);
}

} // namespace rocjitsu::cli
