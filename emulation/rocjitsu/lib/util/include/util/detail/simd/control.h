// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_SIMD_CONTROL_H_
#define UTIL_DETAIL_SIMD_CONTROL_H_

#include <cstdlib>
#include <string_view>

namespace util {

namespace detail {

inline bool init_force_scalar() {
  const char *e = std::getenv("RJ_FORCE_SCALAR");
  if (e == nullptr) {
    return false;
  }
  const std::string_view v(e);
  return !v.empty() && v != "0";
}

/// Force-scalar gate, initialized ONCE from the `RJ_FORCE_SCALAR` env var at
/// image load (dynamic init) and read with a plain load thereafter -- no
/// per-call guard byte. An `inline` variable so the definition lives in this
/// header and `util` stays header-only. Mutable so the test-only seam
/// (util/simd_test_hooks.h) can override it; production code never writes it.
///
/// THERE IS ONE INSTANCE PER LINKED MODULE, NOT ONE PER PROCESS. rocjitsu builds
/// with hidden visibility (`CMAKE_CXX_VISIBILITY_PRESET hidden` in the top-level
/// CMakeLists.txt, plus `-fvisibility=hidden` in cmake/rj_add_object_library.cmake),
/// so this variable gets local binding in every module that links it: it is
/// never a dynamic symbol, and librocjitsu.so, librocjitsu_hooks.so, each plugin
/// module, the hotswap DSOs and every test executable carry a private copy.
/// Env-var control is unaffected -- each copy parses `RJ_FORCE_SCALAR`
/// independently at its own load -- but a write through the test seam reaches
/// ONLY the copy in the caller's module. Making the gate genuinely process-wide
/// would require a deliberate default-visibility attribute here (or a single
/// state owner all modules call into); that is an explicit decision, not
/// something to acquire by accident.
///
/// Safe as a dynamic-init global because force_scalar() is only ever read at
/// runtime instruction-execute, never during another TU's static construction.
inline bool g_force_scalar = init_force_scalar();

} // namespace detail

/// Switch that callers check before taking the SIMD fast path. Read ONCE from
/// the `RJ_FORCE_SCALAR` env var at image load (unset/empty/"0" => false, any
/// other value => true). Production code never writes it; a test-only seam
/// (util/simd_test_hooks.h) may override it so a single test can drive both the
/// scalar and SIMD execute paths and compare.
///
/// There is one instance per linked module, not one per process -- see
/// detail::g_force_scalar for why. `RJ_FORCE_SCALAR` therefore applies uniformly
/// (each module parses it at its own load), while a test-seam override applies
/// only within the caller's module. e2e runs force the scalar codepath by
/// setting the env var before launch, without recompiling.
inline bool force_scalar() { return detail::g_force_scalar; }

} // namespace util

#endif // UTIL_DETAIL_SIMD_CONTROL_H_
