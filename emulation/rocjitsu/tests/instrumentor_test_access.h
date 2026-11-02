// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instrumentor_test_access.h
/// @brief Test-only access to the Instrumentor's patch-gate skip.

#pragma once

#include "rocjitsu/code/patch/instrumentor.h"

namespace rocjitsu {

// Both patch entry points require an explicit runtime binding for entry storage.
// skip_patch_gates() lets a test exercise that path without declaring a binding.
// A simulator test may supply its own wrapper directly to the CP; a unit test may
// only inspect the planned prologue. Skipping does not construct a wrapper, and
// the planner's own fail-closed checks still run afterward.
//
// Must live in namespace rocjitsu (not an anonymous namespace) so it names the
// same type as the friend declaration in Instrumentor.
struct InstrumentorTestAccess {
  static void skip_patch_gates(Instrumentor &instr) { instr.skip_patch_gates_ = true; }
};

} // namespace rocjitsu
