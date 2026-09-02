// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_validation_gfx9_cdna_target_ops.cpp
/// @brief Independent gfx9 CDNA descriptor-resource proof.

#include "rocjitsu/code/patch/consan/consan_validation_target_ops.h"

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

namespace rocjitsu::consan_validation_target_detail {

namespace kd = rocr::llvm::amdhsa;

ConSanDescriptorResourceDeltaValidation
validate_gfx9_cdna_descriptor_resource_delta(const ConSanTargetProfile &target,
                                             const ConSanDescriptorResourceDeltaInput &input) {
  const uint32_t original_encoded_accum_offset =
      AMDHSA_BITS_GET(input.original_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET);
  const uint32_t replacement_encoded_accum_offset =
      AMDHSA_BITS_GET(input.replacement_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET);

  bool valid = true;
  if (replacement_encoded_accum_offset != original_encoded_accum_offset) {
    const auto unified_vgpr_count = [&](uint32_t rsrc1) {
      const uint32_t granulated =
          AMDHSA_BITS_GET(rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
      return (granulated + 1u) * target.vgpr_allocation_granularity_wave64;
    };
    const uint32_t original_accvgpr_base =
        (original_encoded_accum_offset + 1u) * target.accumulator_offset_granularity;
    const uint32_t replacement_accvgpr_base =
        (replacement_encoded_accum_offset + 1u) * target.accumulator_offset_granularity;
    const uint32_t original_unified_vgprs = unified_vgpr_count(input.original_rsrc1);
    const uint32_t replacement_unified_vgprs = unified_vgpr_count(input.replacement_rsrc1);
    valid = input.allow_resource_delta && original_encoded_accum_offset != 0u &&
            replacement_encoded_accum_offset != 0u &&
            (original_unified_vgprs <= original_accvgpr_base ||
             input.original_accumulator_bank_proven_empty) &&
            replacement_unified_vgprs == replacement_accvgpr_base &&
            replacement_accvgpr_base >= input.required_vgpr_count;
  }

  uint32_t normalized_rsrc3 = input.replacement_rsrc3;
  if (input.normalize_resource_delta) {
    AMDHSA_BITS_SET(normalized_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                    original_encoded_accum_offset);
  }
  return {.valid = valid, .normalized_rsrc3 = normalized_rsrc3};
}

} // namespace rocjitsu::consan_validation_target_detail
