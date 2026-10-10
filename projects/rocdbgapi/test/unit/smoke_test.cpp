// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

#include <gtest/gtest.h>

/* Toolchain smoke test.  Verifies that GoogleTest links and runs on
   this platform; no rocdbgapi code is exercised here.  Real coverage
   begins in the per-module unit test commits. */

TEST (Smoke, Arithmetic)
{
  EXPECT_EQ (1 + 1, 2);
}

TEST (Smoke, GMockAvailable)
{
  /* This compiles only if GMock headers are wired up correctly. */
  testing::Test *self = this;
  EXPECT_NE (self, nullptr);
}

