/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <string>

namespace hrr_cap {
namespace metadata {

// Collects best-effort capture environment metadata for the HRR manifest.
// The collector is safe to call from hip_capture_init(): it reads HIP runtime
// constants and initialized internal device state rather than calling public HIP
// APIs that would re-enter hip::init().
std::string collect_json();

}  // namespace metadata
}  // namespace hrr_cap
