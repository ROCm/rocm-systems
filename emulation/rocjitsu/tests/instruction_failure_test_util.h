// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <format>
#include <string>

namespace rocjitsu::test {

// ds_add_u32 v0, v0 gds. It decodes, and the generated DS body rejects gds=1
// with UnimplementedInstruction before any side effect (_gen_ds_gds_guard();
// test_ds_mode_rejection_precedes_execution pins it for CDNA4 DS_ADD_U32).
// Callers assert that diagnostic, so they depend on that guard, not merely on
// CDNA4 reserving the GDS bit.
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
