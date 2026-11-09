/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <string>

namespace hrr_cap {
namespace metadata {

// True when the kernel started this process in secure-execution mode (set-user-ID,
// set-group-ID, file capabilities or an LSM transition). Always false off Linux.
bool secure_exec();

std::string collect_json();

// Escapes `s` for a JSON string literal, quotes not included: backslash, double quote,
// the short control escapes and any other byte below 0x20 as a four-digit hex escape.
std::string json_escape(const char* s);

}  // namespace metadata
}  // namespace hrr_cap
