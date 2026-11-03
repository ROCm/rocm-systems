// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instrumentor_test_access.h
/// @brief Test-only access to the Instrumentor's patch-gate skip.

#pragma once

#include "rocjitsu/code/patch/instrumentor.h"

namespace rocjitsu {

// Both patch entry points run the patch gates, which refuse points that need
// runtime support no runtime provides yet, such as the kernarg wrapper the entry
// prologue loads through. skip_patch_gates() lets a test exercise the gated
// path. A simulator test may supply that support itself (the harness hands the
// CP the wrapper); a unit test may only inspect the planned prologue and never
// dispatch it. Skipping supplies nothing the gates check for, and the planner's
// own fail-closed checks still run after them.
//
// Must live in namespace rocjitsu (not an anonymous namespace) so it names the
// same type as the friend declaration in Instrumentor.
struct InstrumentorTestAccess {
  static void skip_patch_gates(Instrumentor &instr) { instr.skip_patch_gates_ = true; }
};

} // namespace rocjitsu
