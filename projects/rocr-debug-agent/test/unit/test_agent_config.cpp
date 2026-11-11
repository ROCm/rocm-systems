// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "agent_config.h"

#include <gtest/gtest.h>

#include <variant>

using namespace amd::debug_agent;

/* -h only sets help_requested and stops parsing; printing the usage and
   exiting are left to the caller.  */
TEST (AgentConfigTest, HelpRequested)
{
  auto result = parse_debug_agent_options ("-h");
  ASSERT_TRUE (std::holds_alternative<debug_agent_options_t> (result));
  EXPECT_TRUE (std::get<debug_agent_options_t> (result).help_requested);
}
