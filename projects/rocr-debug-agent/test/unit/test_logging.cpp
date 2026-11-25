// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "logging.h"

#include <gtest/gtest.h>

using namespace amd::debug_agent;

/* log_level is a global, so this only holds while no earlier test has left it
   changed.  */
TEST (LoggingTest, DefaultLogLevelIsWarning)
{
  EXPECT_EQ (log_level, log_level_t::warning);
}
