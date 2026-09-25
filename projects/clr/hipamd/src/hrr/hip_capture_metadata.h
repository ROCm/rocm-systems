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

// Escapes `s` for use inside a JSON string literal (quotes not included):
// backslash, double quote, the short control escapes and any other byte below
// 0x20 as \u00XX. A null `s` yields "".
std::string json_escape(const char* s);

}  // namespace metadata
}  // namespace hrr_cap
