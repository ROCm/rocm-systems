// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_descriptor.h
/// @brief Shared ConSan resource facts decoded from AMDHSA kernel descriptors.

#pragma once

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/isa/register_set.h"

#include <algorithm>
#include <cstdint>

namespace rocjitsu {

/// Return the descriptor encoding granularity selected by the target's actual
/// wave mode. This is an allocation-field fact, not an occupancy granule.
[[nodiscard]] inline uint32_t
descriptor_vgpr_granularity(const rocr::llvm::amdhsa::kernel_descriptor_t &descriptor,
                            rj_code_arch_t arch) {
  return descriptor_vgpr_granularity_for_wavefront(arch, kernel_wavefront_size(arch, descriptor));
}

/// Decode the complete unified VGPR allocation visible to ordinary vector and,
/// on descriptor-partitioned CDNA targets, accumulator registers. Malformed or
/// unsupported encodings are bounded by RocJitsu's addressable register set.
[[nodiscard]] inline uint16_t
descriptor_vgpr_allocation_count(const rocr::llvm::amdhsa::kernel_descriptor_t &descriptor,
                                 rj_code_arch_t arch) {
  const uint32_t granulated =
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                      rocr::llvm::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  const uint32_t allocated = (granulated + 1u) * descriptor_vgpr_granularity(descriptor, arch);
  return static_cast<uint16_t>(std::min<uint32_t>(allocated, REGISTER_SET_MAX_VGPRS));
}

/// Return the ordinary-VGPR prefix that ConSan may use for scratch state.
/// CDNA2-4 split their unified descriptor allocation at ACCUM_OFFSET; a
/// nonzero boundary below the allocation therefore excludes live AccVGPR
/// storage. Other targets expose the complete decoded allocation.
[[nodiscard]] inline uint16_t
descriptor_ordinary_vgpr_allocation_count(const rocr::llvm::amdhsa::kernel_descriptor_t &descriptor,
                                          rj_code_arch_t arch) {
  const uint16_t unified_count = descriptor_vgpr_allocation_count(descriptor, arch);
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  if (!profile || profile->accumulator_model != ConSanAccumulatorModel::DescriptorPartitioned)
    return unified_count;

  const uint32_t encoded_accum_offset = AMDHSA_BITS_GET(
      descriptor.compute_pgm_rsrc3, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET);
  if (encoded_accum_offset == 0u)
    return unified_count;
  const uint32_t accvgpr_base =
      (encoded_accum_offset + 1u) * profile->accumulator_offset_granularity;
  return unified_count > accvgpr_base
             ? static_cast<uint16_t>(std::min<uint32_t>(accvgpr_base, REGISTER_SET_MAX_VGPRS))
             : unified_count;
}

/// Decode the ordinary SGPR allocation available to ConSan scratch planning.
/// The AMDHSA field can encode a rounded final granule beyond the addressable
/// scalar operand file, so the returned count is clamped to RocJitsu's register
/// set rather than exposing descriptor padding as usable scratch state.
[[nodiscard]] inline uint16_t
descriptor_sgpr_allocation_count(const rocr::llvm::amdhsa::kernel_descriptor_t &descriptor,
                                 rj_code_arch_t arch) {
  const uint32_t granulated =
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                      rocr::llvm::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  return static_cast<uint16_t>(std::min<uint32_t>(
      amdgpu_kernel_descriptor_sgpr_count(granulated, arch), REGISTER_SET_MAX_SGPRS));
}

} // namespace rocjitsu
