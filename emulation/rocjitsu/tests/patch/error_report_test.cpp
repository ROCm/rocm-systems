// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/error_report.h"

#include <gtest/gtest.h>

#include <string>

namespace rocjitsu {
namespace {

TEST(ErrorReport, NullPointerIsNoOp) {
  // Patch-layer convention: callers pass nullptr when they don't care about
  // the diagnostic. report() must tolerate that without crashing.
  report(nullptr, "ignored");
}

TEST(ErrorReport, AssignsMessageWhenPointerIsNonNull) {
  std::string err = "previous";
  report(&err, "hello");
  EXPECT_EQ(err, "hello");
}

TEST(ErrorReport, OverwritesPriorContents) {
  // Successive reports replace, not append. Pins the assignment semantics so
  // a future refactor doesn't silently turn this into a += or a no-op-if-set.
  std::string err;
  report(&err, "first");
  report(&err, "second");
  EXPECT_EQ(err, "second");
}

} // namespace
} // namespace rocjitsu
