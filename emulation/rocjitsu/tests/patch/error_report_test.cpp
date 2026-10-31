// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/error_report.h"

#include <gtest/gtest.h>

#include "util/diagnostic.h"

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

TEST(ErrorReport, OptionalStringSinkReceivesBorrowedEmitterDiagnostics) {
  std::string message = "previous";
  OptionalStringDiagnosticSink record(&message);
  const util::DiagnosticEmitter error(record);
  error.emit() << "first";
  EXPECT_EQ(message, "first");
  error.emit() << "second";
  EXPECT_EQ(message, "second");

  OptionalStringDiagnosticSink discard(nullptr);
  const util::DiagnosticEmitter ignored(discard);
  ignored.emit() << "ignored";
}

} // namespace
} // namespace rocjitsu
