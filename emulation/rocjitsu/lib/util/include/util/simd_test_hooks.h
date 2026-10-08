// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_UTIL_SIMD_TEST_HOOKS_H_
#define ROCJITSU_UTIL_SIMD_TEST_HOOKS_H_

#include "util/simd.h"

namespace util {

/// Test-only: override the force-scalar gate so a single test can drive both
/// the scalar and SIMD execute paths and compare results. Production code never
/// includes this header.
///
/// Reaches ONLY the gate copy in the caller's own module. The gate has local
/// binding in every module that links it (see util::detail::g_force_scalar), so
/// a test executable calling this does NOT change the gate observed by code
/// running inside librocjitsu.so or librocjitsu_hooks.so -- that code keeps
/// whatever `RJ_FORCE_SCALAR` gave its own copy at load.
///
/// This also does not reach a module the process dlopens. A plugin
/// `librocjitsu_plugin_*.so` or a hotswap hook, loaded before or after the
/// override, still reports its own value, so a test that flips the gate here and
/// expects a loaded module to follow is asserting nothing. That boundary is
/// pinned by UtilSimd.ForceScalarOverrideDoesNotReachADlopenedModule.
///
/// Use this only to drive execute paths linked into the calling module; to force
/// the scalar path everywhere, set `RJ_FORCE_SCALAR` before launch instead.
inline void set_force_scalar_for_testing(bool v) { detail::g_force_scalar = v; }

} // namespace util

#endif // ROCJITSU_UTIL_SIMD_TEST_HOOKS_H_
