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

/// Grow a descriptor so that @p required_ordinary_count ordinary SGPRs remain
/// addressable after the target's descriptor-managed special-register tail.
///
/// CDNA3 and CDNA4 place VCC, XNACK, and FLAT_SCRATCH in a six-SGPR tail at the
/// end of the descriptor allocation. A scratch register below the architectural
/// ordinary-SGPR limit is therefore usable only when the encoded allocation
/// also leaves room for that tail. RDNA and gfx1250 use fixed special-register
/// indices and need no tail. The function rejects zero, unsupported targets,
/// and requests beyond the target's ordinary operand file without modifying
/// the descriptor.
[[nodiscard]] inline bool
grow_descriptor_sgpr_allocation(rocr::llvm::amdhsa::kernel_descriptor_t &descriptor,
                                uint32_t required_ordinary_count, rj_code_arch_t arch) {
  constexpr uint32_t kCdnaAllocationTailSgprs = 6u;
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  if (!profile || required_ordinary_count == 0u ||
      required_ordinary_count > profile->ordinary_sgpr_limit)
    return false;

  const bool descriptor_has_allocation_tail =
      profile->accumulator_model == ConSanAccumulatorModel::DescriptorPartitioned;
  const uint32_t allocation_tail = descriptor_has_allocation_tail ? kCdnaAllocationTailSgprs : 0u;
  const uint32_t required_allocation = required_ordinary_count + allocation_tail;
  if (required_allocation <= descriptor_sgpr_allocation_count(descriptor, arch))
    return true;

  const uint32_t granularity = profile->sgpr_allocation_granularity;
  const uint32_t rounded_allocation =
      (required_allocation + granularity - 1u) / granularity * granularity;
  const uint32_t maximum_allocation = profile->ordinary_sgpr_limit + allocation_tail;
  const uint32_t rounded_maximum =
      (maximum_allocation + granularity - 1u) / granularity * granularity;
  if (rounded_allocation > rounded_maximum)
    return false;

  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                  rocr::llvm::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  (rounded_allocation / granularity - 1u));
  return true;
}

} // namespace rocjitsu
