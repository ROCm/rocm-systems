// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Operand-placement choices shared by register and literal FMA implementations.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

namespace rocjitsu::arith {

// Placement of the scalar literal in a fused multiply-add.
enum class FmaOperands { Registers, MultiplyLiteral, AddLiteral };

} // namespace rocjitsu::arith
