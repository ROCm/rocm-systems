// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernel_descriptor_translator.h
/// @brief DBT layer for translating AMDHSA kernel descriptor ABI/resource fields.

#pragma once

#include "rocjitsu/code/dbt/translation_diagnostic.h"
#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu {

/// @brief Additional descriptor resource requirements from instruction lowering.
struct KernelDescriptorTranslationOptions {
  uint32_t minimum_vgprs = 0;
  uint32_t minimum_sgprs = 0;
  uint32_t private_segment_fixed_size_addend = 0;
  uint32_t group_segment_fixed_size_addend = 0;
};

/// @brief Per-kernel descriptor/resource/ABI translation plan.
///
/// The descriptor translator computes this from the source kernel descriptor.
/// It does not mutate descriptor bytes. BinaryTranslator uses the plan for
/// instruction-level decisions, and CodeObjectPatcher applies the plan to the
/// ELF/kernel descriptor bytes.
struct KdTranslation {
  uint64_t descriptor_file_offset = 0;
  /// @brief Original .text-relative kernel entry decoded from the source descriptor.
  uint64_t entry_text_offset = 0;

  /// @brief New .text-relative kernel entry after DBT rewrites .text.
  ///
  /// @details This is the final launch address written into
  /// KERNEL_CODE_ENTRY_BYTE_OFFSET. It may point at descriptor ABI prologue code
  /// emitted before the relocated original kernel body.
  uint64_t target_entry_text_offset = 0;

  /// @brief New .text-relative offset of the original kernel entry block.
  ///
  /// @details Descriptor ABI prologues may make @c target_entry_text_offset
  /// point before this address. Branches and diagnostics that refer to the
  /// relocated original guest entry should use this field.
  uint64_t target_body_entry_text_offset = 0;

  /// @brief Ordinary architectural VGPRs required by translated code.
  ///
  /// @details This excludes any target AccVGPR window. Liveness analysis,
  /// semantic scratch allocation, and descriptor growth requests should use
  /// this field when they mean normal v0..vN VGPR demand.
  uint32_t target_vgpr_count = 0;

  /// @brief Unified VGPR allocation encoded in COMPUTE_PGM_RSRC1.
  ///
  /// @details On targets with AccVGPRs this may be larger than
  /// @c target_vgpr_count because the descriptor allocation must cover both
  /// ordinary VGPRs and the AccVGPR window selected by ACCUM_OFFSET. Use this
  /// field only for descriptor encoding/resource-limit checks, not ordinary
  /// VGPR liveness.
  uint32_t target_vgpr_allocation_count = 0;

  /// @brief Encoded target COMPUTE_PGM_RSRC1.GRANULATED_WORKITEM_VGPR_COUNT.
  uint32_t target_vgpr_granulated = 0;

  /// @brief First unified VGPR index reserved for target AccVGPRs.
  ///
  /// @details For CDNA-to-CDNA translations semantic scratch may force this
  /// base upward so ordinary temporary VGPRs do not alias a0, a1, ...
  uint32_t target_accvgpr_base = 0;

  /// @brief Guest AccVGPR base decoded from the source descriptor.
  uint32_t accvgpr_base = 0;

  /// @brief Number of target AccVGPRs preserved as a real AccVGPR window.
  uint32_t target_agpr_count = 0;

  /// @brief Future spill tier: number of VGPRs to virtualize through LDS.
  uint32_t vgpr_spill_to_lds_count = 0;

  /// @brief Future spill tier: number of VGPRs to virtualize through scratch memory.
  uint32_t vgpr_spill_to_scratch_count = 0;

  uint32_t target_sgpr_count = 0;
  uint32_t target_sgpr_granulated = 0;
  uint32_t sgpr_spill_count = 0;

  uint32_t target_lds_size = 0;
  uint32_t lds_spill_zone_bytes = 0;
  uint32_t lds_overflow_size = 0;
  bool needs_lds_overflow_buf = false;

  uint32_t target_private_size = 0;

  uint8_t target_wave_size = 64;
  bool force_wave64 = false;

  uint8_t target_user_sgpr_count = 0;
  bool needs_flat_scratch_init_sgpr = false;
  std::vector<uint32_t> user_sgpr_shuffle;

  /// All kernel-entry instructions required by descriptor ABI translation.
  /// BinaryTranslator places these words in the kernel-local .text cave and
  /// records the final descriptor entry offset.
  std::vector<uint32_t> prologue_words;

  uint8_t guest_wavefront_size = 64;
  uint8_t host_wavefront_size = 64;

  /// @brief Ordinary guest VGPR floor decoded from the source descriptor.
  ///
  /// @details On CDNA, COMPUTE_PGM_RSRC1 describes the unified VGPR allocation
  /// endpoint and ACCUM_OFFSET carves the trailing AccVGPR window out of that
  /// allocation. Liveness and semantic scratch allocation need only the
  /// ordinary portion so they can reuse registers that are dead at a lowering
  /// site without forcing unnecessary descriptor growth.
  uint32_t guest_vgpr_count = 0;

  /// @brief Source descriptor unified VGPR allocation after rounding.
  ///
  /// @details This is decoded from COMPUTE_PGM_RSRC1 and may include the
  /// AccVGPR window on CDNA. Use it for descriptor encoding/reporting; use
  /// @c guest_vgpr_count when asking how many ordinary VGPRs the guest used.
  uint32_t guest_vgpr_allocation_count = 0;

  /// @brief Conservative source AccVGPR window size derived from rounded count and ACCUM_OFFSET.
  uint32_t guest_agpr_count = 0;

  /// @brief Ordinary host VGPR count after translation requirements are applied.
  uint32_t host_vgpr_count = 0;

  /// @brief Host descriptor's unified VGPR allocation count.
  ///
  /// @details Mirrors @c target_vgpr_allocation_count for diagnostics/reporting
  /// and includes any preserved target AccVGPR window.
  uint32_t host_vgpr_allocation_count = 0;

  /// @brief Ordinary guest SGPR count decoded from the source descriptor.
  uint32_t guest_sgpr_count = 0;

  /// @brief Ordinary host SGPR count after translation requirements are applied.
  uint32_t host_sgpr_count = 0;

  uint32_t source_occupancy = 0;
  uint32_t target_occupancy = 0;

  bool supported = true;
  std::vector<TranslationDiagnostic> diagnostics;
};

/// @brief Compute descriptor patches and semantic metadata for one DBT pair.
class KernelDescriptorTranslator {
public:
  KernelDescriptorTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch);

  [[nodiscard]] std::vector<KdTranslation>
  translate_image(std::span<const uint8_t> image, uint64_t text_offset, uint64_t text_size,
                  const KernelDescriptorTranslationOptions &options) const;

  /// @brief Recompute one already-discovered kernel descriptor.
  ///
  /// @details BinaryTranslator uses this after instruction lowering discovers
  /// additional per-kernel SGPR/VGPR requirements. The descriptor's file offset
  /// and entry offset come from the original image-wide descriptor discovery, so
  /// this avoids rescanning or recomputing unrelated kernel descriptors.
  /// @returns A translated descriptor plan, or std::nullopt if @p descriptor_file_offset
  /// does not point at a complete AMDHSA kernel descriptor in @p image.
  [[nodiscard]] std::optional<KdTranslation>
  translate_descriptor(std::span<const uint8_t> image, uint64_t descriptor_file_offset,
                       uint64_t entry_text_offset,
                       const KernelDescriptorTranslationOptions &options) const;

private:
  rj_code_arch_t guest_arch_;
  rj_code_arch_t host_arch_;
};

} // namespace rocjitsu
