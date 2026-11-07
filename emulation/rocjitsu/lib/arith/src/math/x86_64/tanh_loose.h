// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Loose TANH whole-wave numeric backends; no modifiers or destination publication.
#pragma once
#include <stdint.h>
namespace rocjitsu::arith {
// Evaluate 32 prepared FP32 inputs; requires the named ISA. All float values are supported.
void tanh_loose_x86_64_v3(uint32_t *d, const uint32_t *a);
void tanh_loose_x86_64_v4(uint32_t *d, const uint32_t *a);
} // namespace rocjitsu::arith
