// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <format>
#include <string>

namespace rocjitsu::test {

// s_rfe_restore_b64 s[0:1], s0 decodes on CDNA3/4 but has no execution implementation.
inline constexpr uint32_t kUnimplementedInstruction = 0x95800000u;
inline constexpr char kUnimplementedInstructionMnemonic[] = "s_rfe_restore_b64";
inline constexpr uint32_t kUndecodableInstruction = 0xbfff0000u;
// Formatted like the instruction words in the compute unit's decode diagnostic.
inline const std::string kUndecodableInstructionText =
    std::format("{:#x}", kUndecodableInstruction);

inline const std::string kUnimplementedInstructionDiagnostic =
    std::format("CU .*could not execute {}.*pc=.*pid=.*qid=.*dispatch=.*unimplemented instruction",
                kUnimplementedInstructionMnemonic);
inline const std::string kDecodeFailureDiagnostic = std::format(
    "CU .*could not decode.*pc=.*pid=.*qid=.*dispatch=.*{}", kUndecodableInstructionText);

} // namespace rocjitsu::test
