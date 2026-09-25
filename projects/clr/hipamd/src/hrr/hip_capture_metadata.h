/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <string>

namespace hrr_cap {
namespace metadata {

// True when the kernel started this process in secure-execution mode
// (set-user-ID, set-group-ID or file capabilities). Always false off Linux.
bool secure_exec();

// Collects best-effort capture environment metadata for the HRR manifest.
// The collector is safe to call from hip_capture_init(): it reads HIP runtime
// constants and initialized internal device state rather than calling public HIP
// APIs that would re-enter hip::init().
std::string collect_json();

}  // namespace metadata
}  // namespace hrr_cap
