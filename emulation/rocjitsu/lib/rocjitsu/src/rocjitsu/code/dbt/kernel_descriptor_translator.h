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
#include <string_view>
#include <vector>

namespace rocjitsu {

struct KernelDescriptorResourceOverride {
  uint64_t entry_text_offset = 0;
  uint32_t minimum_vgprs = 0;
  uint32_t target_vgpr_count_override = 0;
  uint32_t minimum_sgprs = 0;
  uint32_t group_segment_fixed_size_addend = 0;
  uint32_t private_segment_fixed_size_addend = 0;
};

/// @brief Additional descriptor resource requirements from instruction lowering.
struct KernelDescriptorTranslationOptions {
  uint32_t minimum_vgprs = 0;
  uint32_t target_vgpr_count_override = 0;
  uint32_t minimum_sgprs = 0;
  uint32_t private_segment_fixed_size_addend = 0;
  uint32_t group_segment_fixed_size_addend = 0;
  std::span<const KernelDescriptorResourceOverride> kernel_overrides;
};

/// @brief Per-kernel descriptor/resource/ABI translation plan.
///
/// The descriptor translator computes this from the source kernel descriptor.
/// It does not mutate descriptor bytes. BinaryTranslator uses the plan for
/// instruction-level decisions, and CodeObjectPatcher applies the plan to the
/// ELF/kernel descriptor bytes.
struct KdTranslation {
  uint64_t descriptor_file_offset = 0;
  /// @brief Kernel symbol name without the AMDHSA ".kd" descriptor suffix.
  std::string kernel_name;
  /// @brief Original .text-relative kernel entry decoded from the source descriptor.
  uint64_t entry_text_offset = 0;
  std::string symbol_name;

  /// @brief True when the source descriptor requests CP kernarg preloading.
  ///
  /// @details AMDHSA uses kernarg_preload_spec_length != 0 to request that
  /// compatible CP firmware copy dwords from the kernarg segment into User SGPRs
  /// before entering the kernel. When this is enabled, compatible firmware starts
  /// execution at KERNEL_CODE_ENTRY_BYTE_OFFSET + 256 while older/incompatible
  /// firmware starts at KERNEL_CODE_ENTRY_BYTE_OFFSET. DBT must therefore treat
  /// both source addresses as legal hardware entries for this kernel.
  bool has_kernarg_preload = false;

  /// @brief Source .text-relative entry used by compatible kernarg-preload firmware.
  uint64_t kernarg_preload_entry_text_offset = 0;

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

  /// @brief True when this plan describes a non-symbolized sidecar descriptor.
  ///
  /// @details Sidecar descriptors are materialized in a loaded ELF segment and
  /// referenced only by rocjitsu metadata; they are not exported as additional
  /// AMDHSA kernel symbols. Virtual LDS is the first producer, but the patching
  /// contract is deliberately descriptor-generic so DBI instrumentation can add
  /// its own sidecar descriptors later.
  bool sidecar_descriptor = false;

  uint32_t target_sgpr_count = 0;
  uint32_t target_sgpr_granulated = 0;
  uint32_t sgpr_spill_count = 0;
  int16_t rdna4_grid_x_sgpr = -1;
  int16_t rdna4_grid_yz_sgpr = -1;

  /// @brief Hardware LDS bytes encoded in GROUP_SEGMENT_FIXED_SIZE.
  uint32_t target_lds_size = 0;
  uint32_t lds_spill_zone_base = 0;
  uint32_t lds_spill_zone_bytes = 0;
  /// @brief Static LDS bytes that must be backed by a virtual LDS buffer.
  uint32_t lds_overflow_size = 0;
  /// @brief True when the dispatch path must provide a virtual LDS buffer.
  bool needs_lds_overflow_buf = false;
  /// @brief Concrete scratch choices made while lowering the virtual-LDS body.
  ///
  /// @details This state is deliberately separate from descriptor-derived ABI
  /// fields because descriptor recomputation must preserve it byte-for-byte.
  VirtualLdsLoweringState virtual_lds_lowering;
  /// @brief Source kernarg bytes copied into the wrapper prefix.
  ///
  /// @details The copied prefix preserves original kernarg offsets while DBT
  /// prologue code consumes extension payloads from the same wrapper. The size
  /// is the descriptor kernarg byte count widened only by descriptor-declared
  /// kernarg preloads; DBT deliberately does not infer additional live bytes by
  /// scanning guest code.
  uint32_t kernarg_size = 0;
  /// @brief Kernarg segment size required by the translated target descriptor.
  ///
  /// @details Normal kernels preserve the source descriptor size. Virtual-LDS
  /// kernels advertise the rocjitsu wrapper size: copied original prefix,
  /// original-pointer field, and extension payloads.
  uint32_t target_kernarg_size = 0;
  /// @brief Wrapper byte offset of the saved original kernarg pointer.
  uint32_t kernarg_wrapper_original_pointer_offset = 0;
  /// @brief Wrapper byte offset of the virtual-LDS runtime state payload.
  uint32_t lds_overflow_kernarg_pointer_offset = 0;
  /// @brief True if the source descriptor exposes a kernarg segment pointer SGPR.
  bool source_has_kernarg_segment_ptr = false;
  /// @brief True if the target descriptor exposes a kernarg segment pointer SGPR.
  bool has_kernarg_segment_ptr = false;
  /// @brief First target SGPR of the descriptor-selected kernarg segment pointer pair.
  uint16_t kernarg_segment_ptr_sgpr = 0;
  /// @brief True if the source descriptor exposes a dispatch-packet pointer SGPR.
  bool has_dispatch_ptr = false;
  /// @brief First SGPR of the descriptor-selected dispatch-packet pointer pair.
  uint16_t dispatch_ptr_sgpr = 0;
  /// @brief Descriptor-selected workgroup-id SGPRs, or -1 when a dimension is disabled.
  int16_t workgroup_id_sgpr_x = -1;
  int16_t workgroup_id_sgpr_y = -1;
  int16_t workgroup_id_sgpr_z = -1;
  /// @brief Target workgroup-id SGPRs seen before wrapper prologue ABI repair.
  int16_t lds_overflow_workgroup_id_sgpr_x = -1;
  int16_t lds_overflow_workgroup_id_sgpr_y = -1;
  int16_t lds_overflow_workgroup_id_sgpr_z = -1;

  uint32_t target_private_size = 0;
  uint32_t private_spill_zone_base = 0;
  uint32_t private_spill_zone_bytes = 0;

  uint8_t target_wave_size = 64;
  bool force_wave64 = false;

  uint8_t target_user_sgpr_count = 0;
  uint32_t target_abi_sgpr_count = 0;
  uint32_t target_source_sgpr_count = 0;
  bool needs_flat_scratch_init_sgpr = false;
  std::vector<uint32_t> user_sgpr_shuffle;

  /// All kernel-entry instructions required by descriptor ABI translation.
  /// BinaryTranslator places these words in the kernel-local .text cave and
  /// records the final descriptor entry offset.
  std::vector<uint32_t> prologue_words;

  /// @brief Clear the target descriptor's source kernarg-preload request.
  ///
  /// @details GFX1250 preload is rebuilt explicitly in @c prologue_words for
  /// RDNA4. Leaving the source descriptor bit set lets the runtime/CP treat the
  /// redirected entry as a preload-special entry instead of the DBT prologue.
  bool clears_kernarg_preload = false;

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
  /// @brief True when DBT preserved this source kernel instead of translating it.
  bool skipped = false;
  std::vector<TranslationDiagnostic> diagnostics;

  /// @brief Replace target resource/ABI state with a minimal skipped-kernel stub plan.
  void configure_skipped_stub();
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
  [[nodiscard]] std::optional<KdTranslation> translate_descriptor(
      std::span<const uint8_t> image, uint64_t descriptor_file_offset, uint64_t entry_text_offset,
      const KernelDescriptorTranslationOptions &options, std::string_view symbol_name = {}) const;

private:
  rj_code_arch_t guest_arch_;
  rj_code_arch_t host_arch_;
};

} // namespace rocjitsu
