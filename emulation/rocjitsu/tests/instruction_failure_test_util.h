// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <format>
#include <string>

namespace rocjitsu::test {

// The CDNA3/4 decoder recognizes DS_ADD_U32. Executing its unsupported GDS mode
// exercises the execution-failure path independently of legal opcode coverage.
inline constexpr std::array<uint32_t, 2> kUnimplementedInstruction{0xd8010000u, 0};
inline constexpr char kUnimplementedInstructionMnemonic[] = "ds_add_u32";
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
