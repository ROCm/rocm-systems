// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_gfx9_target_ops.cpp
/// @brief GFX9 CDNA3/CDNA4 recipes shared by MOI engines.

#include "rocjitsu/code/patch/consan/consan_moi_target_ops.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"

namespace rocjitsu::consan_detail {

bool append_restore_moi_scc_from_route_key(std::vector<uint32_t> &words,
                                           const MoiEncodedSccRestoreRequest &request,
                                           const ConSanTargetProfile &target) {
  uint16_t bit_test_opcode = 0u;
  switch (target.encoding_family) {
  case ConSanEncodingFamily::Gfx9Cdna3:
    bit_test_opcode = cdna3::kSBitcmp1B32Sopc;
    break;
  case ConSanEncodingFamily::Gfx9Cdna4:
    bit_test_opcode = cdna4::kSBitcmp1B32Sopc;
    break;
  case ConSanEncodingFamily::Gfx11:
  case ConSanEncodingFamily::Gfx12:
    return false;
  }
  const auto normalize =
      instrumentation::build_s_cselect_b32(request.encoded_sgpr, scalar_positive_inline_u32(1),
                                           scalar_positive_inline_u32(0), target.arch);
  if (!normalize)
    return false;
  words.push_back(build_sopc_encoding(target.arch, bit_test_opcode, request.encoded_sgpr,
                                      scalar_positive_inline_u32(0)));
  if (request.normalize_encoded_sgpr)
    words.push_back(*normalize);
  return true;
}

} // namespace rocjitsu::consan_detail
