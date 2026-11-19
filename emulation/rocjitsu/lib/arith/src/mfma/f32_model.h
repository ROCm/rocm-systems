// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32-accumulating MFMA policies shared by baseline and ISA-specific backends.
#pragma once

namespace rocjitsu::arith {
// This shared type is ISA-independent; arithmetic helpers stay in their ISA namespaces.
enum class MfmaF32Model { Default, Gfx908, Gfx90a };
} // namespace rocjitsu::arith
