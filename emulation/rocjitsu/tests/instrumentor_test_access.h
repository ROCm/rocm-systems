// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instrumentor_test_access.h
/// @brief Test-only access to the Instrumentor's patch-gate skip.

#pragma once

#include "rocjitsu/code/patch/instrumentor.h"

namespace rocjitsu {

// Both patch entry points run the patch gates. One refuses points that need
// runtime support no runtime provides yet, such as the kernarg wrapper the entry
// prologue loads through. The others refuse kernels the Instrumentor does not
// support: ones liveness cannot follow for a probe call (a call, an unresolved
// edge, a relative register access, an RDNA4 image instruction) and ones that
// preload kernel arguments past a second firmware entry.
// skip_patch_gates() lets a test exercise the gated path. A simulator test may
// supply the runtime support itself (the harness hands the CP the wrapper); a
// unit test may only inspect the plan and never dispatch it. Skipping supplies
// nothing the gates check for and does not make an unsupported kernel's
// liveness sound; the planner's own fail-closed checks still run after them.
//
// Must live in namespace rocjitsu (not an anonymous namespace) so it names the
// same type as the friend declaration in Instrumentor.
struct InstrumentorTestAccess {
  static void skip_patch_gates(Instrumentor &instr) { instr.skip_patch_gates_ = true; }
};

} // namespace rocjitsu
