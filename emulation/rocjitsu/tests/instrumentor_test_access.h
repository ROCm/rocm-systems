// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instrumentor_test_access.h
/// @brief Test-only access to the Instrumentor's patch-gate skip.

#pragma once

#include "rocjitsu/code/patch/instrumentor.h"

namespace rocjitsu {

// Both patch entry points run the patch gates. A test that supplies what a gate
// requires, such as the kernarg wrapper the entry prologue loads through, skips
// them here. Must live in namespace rocjitsu (not an anonymous namespace) so it
// names the same type as the friend declaration in Instrumentor.
struct InstrumentorTestAccess {
  static void skip_patch_gates(Instrumentor &instr) { instr.skip_patch_gates_ = true; }
};

} // namespace rocjitsu
