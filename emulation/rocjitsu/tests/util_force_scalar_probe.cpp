// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file util_force_scalar_probe.cpp
/// @brief Loadable module that reports its own copy of the force-scalar gate.
///
/// @details The gate is an inline variable with hidden visibility, so every
/// module that links it carries a private instance. Nothing in-process can
/// observe that boundary, which is why it needs a separate module: the test
/// dlopens this one and compares what it reports against the host's own gate.

#include "util/simd.h"

extern "C" __attribute__((visibility("default"))) bool rj_probe_force_scalar() {
  return util::force_scalar();
}
