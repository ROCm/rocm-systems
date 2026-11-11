// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

/* Toolchain smoke test.  Verifies that GoogleTest links and runs on
   this platform. No rocdbgapi code is exercised here.  Real coverage
   begins in the per-module unit test commits. */

TEST (Smoke, Arithmetic)
{
  EXPECT_EQ (1 + 1, 2);
}

TEST (Smoke, GMockAvailable)
{
  /* This compiles and passes only if GMock headers and matchers are
     wired up correctly. */
  EXPECT_THAT (1 + 1, testing::Eq (2));
}

