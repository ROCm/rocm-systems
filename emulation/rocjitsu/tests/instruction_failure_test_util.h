// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <format>
#include <string>

namespace rocjitsu::test {

// ds_add_u32 v0, v0 gds. CDNA4 reserves the DS GDS bit and the simulator does
// not model GDS on CDNA3 either, so this decodes but never executes. A reserved
// encoding stays unimplemented; an ordinary unimplemented instruction does not
// once someone implements it.
inline constexpr std::array<uint32_t, 2> kUnimplementedInstruction = {0xd8010000u, 0x00000000u};
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
