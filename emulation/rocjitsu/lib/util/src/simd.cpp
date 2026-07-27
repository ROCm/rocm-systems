// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/simd.h"

#include "util/simd_test_hooks.h"

#include <cstdlib>
#include <string_view>

namespace util {
namespace {

bool init_force_scalar() {
  const char *e = std::getenv("RJ_FORCE_SCALAR");
  if (e == nullptr) {
    return false;
  }
  const std::string_view v(e);
  return !v.empty() && v != "0";
}

/// Process-wide force-scalar gate, initialized ONCE from the `RJ_FORCE_SCALAR`
/// env var at startup (dynamic init) and read with a plain load thereafter --
/// no per-call guard byte. Mutable so the test-only seam below can override it
/// in-process; production code never writes it. Safe as a dynamic-init global
/// because force_scalar() is only ever read at runtime instruction-execute,
/// never during another TU's static construction.
bool g_force_scalar = init_force_scalar();

} // namespace

bool force_scalar() { return g_force_scalar; }

void set_force_scalar_for_testing(bool v) { g_force_scalar = v; }

} // namespace util
