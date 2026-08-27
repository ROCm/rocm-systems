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

/// The complete policy input for growing one kernel descriptor's ordinary
/// VGPR allocation.
///
/// A descriptor encodes one unified allocation, while ConSan consumers can
/// have narrower addressing limits and CDNA descriptors can divide that
/// allocation between ordinary VGPRs and AccVGPRs. Keeping all three facts in
/// one value prevents an engine from silently treating an accumulator bank as
/// ordinary scratch or growing beyond the register form it can emit.
struct ConSanDescriptorVgprGrowthRequest {
  /// One past the highest ordinary VGPR that the transformed kernel must be
  /// able to address. This is the final extent, not the number of new VGPRs.
  uint32_t required_ordinary_count = 0;

  /// Largest ordinary-VGPR extent the requesting instrumentation path can
  /// address. Bank-aware gfx1250 mechanics may use RocJitsu's full analysis
  /// range; ordinary MOI operand forms currently limit themselves to 256.
  uint32_t maximum_ordinary_count = 0;

  /// Trusted metadata proves that an encoded CDNA accumulator partition has
  /// no live AccVGPR values and can therefore move. A boundary above the
  /// current unified allocation is intrinsically empty and needs no proof.
  bool accumulator_bank_is_proven_empty = false;
};

/// Grow the descriptor's ordinary-VGPR extent without reclassifying live
/// accumulator storage or exceeding either descriptor or caller limits.
///
/// On descriptor-partitioned CDNA, growth may fill an empty gap below
/// ACCUM_OFFSET. It may move the boundary only when the old allocation proves
/// the bank empty or the request carries independent trusted proof. Other
/// targets update only the unified allocation field. Failure leaves both
/// descriptor fields unchanged.
[[nodiscard]] inline bool
grow_descriptor_vgpr_allocation(rocr::llvm::amdhsa::kernel_descriptor_t &descriptor,
                                const ConSanDescriptorVgprGrowthRequest &request,
                                rj_code_arch_t arch) {
  constexpr uint32_t kDescriptorAllocationGranules = 64u;
  constexpr uint32_t kDirectOrdinaryVgprLimit = 256u;

  const ConSanTargetProfile *profile = consan_target_profile(arch);
  const uint32_t granularity = descriptor_vgpr_granularity(descriptor, arch);
  if (!profile || granularity == 0u || request.required_ordinary_count == 0u ||
      request.maximum_ordinary_count == 0u)
    return false;

  const uint32_t target_ordinary_limit = profile->has_selectable_vgpr_bank
                                             ? static_cast<uint32_t>(REGISTER_SET_MAX_VGPRS)
                                             : kDirectOrdinaryVgprLimit;
  const uint32_t maximum_ordinary_count =
      std::min({request.maximum_ordinary_count, target_ordinary_limit,
                kDescriptorAllocationGranules * granularity});
  if (request.required_ordinary_count > maximum_ordinary_count)
    return false;

  const uint32_t ordinary_count = descriptor_ordinary_vgpr_allocation_count(descriptor, arch);
  if (request.required_ordinary_count <= ordinary_count)
    return true;

  const uint32_t rounded_required =
      (request.required_ordinary_count + granularity - 1u) / granularity * granularity;
  if (rounded_required > maximum_ordinary_count)
    return false;

  const uint32_t unified_count = descriptor_vgpr_allocation_count(descriptor, arch);
  if (profile->accumulator_model == ConSanAccumulatorModel::DescriptorPartitioned) {
    const uint32_t encoded_accum_offset = AMDHSA_BITS_GET(
        descriptor.compute_pgm_rsrc3, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET);
    if (encoded_accum_offset != 0u) {
      const uint32_t accumulator_granularity = profile->accumulator_offset_granularity;
      const uint32_t accvgpr_base = (encoded_accum_offset + 1u) * accumulator_granularity;
      if (rounded_required <= accvgpr_base) {
        AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                        rocr::llvm::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                        (rounded_required / granularity - 1u));
        return true;
      }

      const bool accumulator_bank_is_empty =
          unified_count <= accvgpr_base || request.accumulator_bank_is_proven_empty;
      if (!accumulator_bank_is_empty)
        return false;

      const uint32_t new_unified_count = std::max(unified_count, rounded_required);
      if (new_unified_count > maximum_ordinary_count ||
          new_unified_count % accumulator_granularity != 0u ||
          new_unified_count / accumulator_granularity > kDescriptorAllocationGranules)
        return false;
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3,
                      rocr::llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                      (new_unified_count / accumulator_granularity - 1u));
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                      rocr::llvm::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                      (new_unified_count / granularity - 1u));
      return true;
    }
  }

  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                  rocr::llvm::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  (rounded_required / granularity - 1u));
  return true;
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
