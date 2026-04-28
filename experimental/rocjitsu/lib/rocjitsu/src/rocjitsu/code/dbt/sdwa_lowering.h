// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sdwa_lowering.h
/// @brief CDNA SDWA alternate-encoding lowering for DBT.

#pragma once

#include <cstdint>
#include <vector>

namespace rocjitsu {

class Instruction;
class RegisterLiveness;

/// @brief Whether the instruction is a CDNA4 VOP2 SDWA alternate encoding.
[[nodiscard]] bool is_cdna4_vop2_sdwa_form(const Instruction &inst);

/// @brief Lower supported CDNA4 VOP2 SDWA forms to ordinary RDNA4 VALU.
///
/// @details The decoded Instruction only owns the base VOP2 word. The caller
/// must pass the SDWA extension dword from the original .text bytes.
[[nodiscard]] std::vector<uint32_t>
lower_cdna4_vop2_sdwa_to_rdna4(const Instruction &inst, uint64_t offset,
                               const RegisterLiveness &liveness, uint16_t dst_opcode,
                               uint32_t ext_word);

} // namespace rocjitsu
