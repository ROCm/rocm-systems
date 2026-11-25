// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

#include "utils.h"

#include <gtest/gtest.h>

/* Verifies that amd-dbgapi-internal exposes internal symbols.
   utils::human_readable_size is defined in src/utils.cpp and is not in
   the exported API (src/exportmap.in), so this test links only against
   the internal archive and would fail to link against the public
   libamd-dbgapi.so.  */

TEST (InternalSymbol, HumanReadableSize)
{
  using namespace amd::dbgapi;

  EXPECT_EQ (utils::human_readable_size (512), "512");
  EXPECT_EQ (utils::human_readable_size (2 * utils::KiB), "2.0K");
  EXPECT_EQ (utils::human_readable_size (3 * utils::MiB), "3.0M");
}
