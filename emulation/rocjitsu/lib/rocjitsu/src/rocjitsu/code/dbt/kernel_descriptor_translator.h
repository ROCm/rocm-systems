// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernel_descriptor_translator.h
/// @brief DBT layer for translating AMDHSA kernel descriptor ABI/resource fields.

#pragma once

#include "rocjitsu/code/dbt/translation_diagnostic.h"
#include "rocjitsu/code/rj_code.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

/// @brief COMPUTE_PGM_RSRC3 layout families, per the compute_pgm_rsrc3 tables
/// in llvm/docs/AMDGPUUsage.rst.
///
/// @details RSRC3 is not one register. The families disagree on bits the others
/// reserve, so a source word may only be carried to a target in the same family:
///
/// - Gfx9Accum: ACCUM_OFFSET in bits 5:0.
/// - Gfx10: SHARED_VGPR_COUNT in bits 3:0; bits 11:4 and bit 31 reserved. There
///   is no INST_PREF_SIZE here at all -- it is a GFX11+ field.
/// - Gfx11: SHARED_VGPR_COUNT in bits 3:0, a 6-bit INST_PREF_SIZE at 9:4,
///   TRAP_ON_START and TRAP_ON_END at bits 10 and 11, IMAGE_OP at bit 31.
/// - Gfx120: bits 3:0 reserved, INST_PREF_SIZE widened to 8 bits at 11:4
///   (absorbing both trap bits), GLG_EN at 13; bits 21:14 reserved.
/// - Gfx125: as Gfx120, but bits 21:14 carry NAMED_BAR_CNT,
///   ENABLE_DYNAMIC_VGPR, TCP_SPLIT and ENABLE_DIDT_THROTTLE.
enum class Rsrc3Layout : uint8_t { Incompatible, Gfx9Accum, Gfx10, Gfx11, Gfx120, Gfx125 };

/// @brief RSRC3 layout family for an architecture.
[[nodiscard]] Rsrc3Layout rsrc3_layout(rj_code_arch_t arch);

/// @brief True when the source RSRC3 word can be carried to the target verbatim.
[[nodiscard]] bool rsrc3_carries_verbatim(rj_code_arch_t source_arch, rj_code_arch_t target_arch);

/// @brief True for the layouts that encode SHARED_VGPR_COUNT in bits 3:0.
[[nodiscard]] bool rsrc3_layout_has_shared_vgpr_count(Rsrc3Layout layout);

/// @brief Architectural VGPRs a wave64 shared-VGPR request removes from the
/// descriptor's 256-register budget.
///
/// @details LLVM constrains a wave64 descriptor to
/// (compute_pgm_rsrc1.vgprs + 1) * 4 + shared_vgpr_count * 8 <= 256.
[[nodiscard]] constexpr uint32_t rsrc3_shared_vgpr_reserved_registers(uint32_t shared_vgpr_count) {
  return shared_vgpr_count * 8u;
}

/// @brief Architecture-neutral descriptor resource requests.
struct KernelDescriptorResourceOptions {
  /// @brief Minimum ordinary VGPR count requested by instruction lowering.
  uint32_t minimum_vgprs = 0;
  /// @brief Minimum SGPR count requested by instruction lowering.
  uint32_t minimum_sgprs = 0;
  /// @brief Additional per-lane private bytes requested by DBT or DBI.
  uint32_t private_segment_fixed_size_addend = 0;
  /// @brief Additional workgroup LDS bytes requested by DBT or DBI.
  uint32_t group_segment_fixed_size_addend = 0;
};

/// @brief Descriptor policy owned by the virtual-LDS feature.
struct VirtualLdsDescriptorOptions {
  /// @brief Encode the target descriptor with zero hardware LDS and require a
  /// virtual LDS backing buffer for the kernel body.
  ///
  /// @details This is an explicit per-descriptor mode because the runtime must
  /// only select the virtualized copy when the dispatch's total LDS demand
  /// cannot fit on the host GPU. The descriptor translator records the source
  /// static byte demand in KdTranslation; runtime dispatch compares it with the
  /// AQL packet's total group-segment size and uses hardware LDS whenever that
  /// request still fits.
  bool virtualize_lds = false;

  /// @brief Permit a normal descriptor to retain an LDS size above the host
  /// hardware limit.
  ///
  /// @details This is only valid when the caller is also emitting a virtual-LDS
  /// sidecar descriptor for the same kernel. The normal descriptor is kept so
  /// launches that fit on the host can still use hardware LDS; runtime dispatch
  /// rewriting must select the virtual descriptor only when the packet's total
  /// LDS demand exceeds the host limit.
  bool allow_oversized_lds = false;
};

/// @brief Complete input to descriptor translation.
///
/// @details The component bases let generic descriptor planning consume only
/// resource requests, without depending on a particular semantic feature.
struct KernelDescriptorTranslationOptions : KernelDescriptorResourceOptions,
                                            VirtualLdsDescriptorOptions {};

/// @brief Mutable code-lowering state attached to a virtual-LDS descriptor plan.
///
/// @details KernelDescriptorTranslator computes descriptor ABI/resource fields,
/// while instruction lowering chooses concrete scratch registers and spill slots.
/// Keeping the latter in one value makes descriptor recomputation and failed-
/// kernel rollback transfer that state atomically instead of copying a growing
/// list of unrelated fields.
struct VirtualLdsLoweringState {
  uint16_t base_sgpr = 0;
  uint16_t prologue_temp_sgpr = 0;
  bool base_sgpr_spill_per_use = false;
  bool base_pointer_spilled = false;
  uint32_t base_pointer_spill_offset = 0;
  bool entry_temp_vgprs_valid = false;
  uint8_t entry_temp_vgpr_lo = 0;
  uint8_t entry_temp_vgpr_hi = 0;
};

/// @brief Immutable source identity and entry facts for one kernel descriptor.
struct KernelDescriptorFacts {
  /// @brief File offset of the original AMDHSA kernel descriptor.
  ///
  /// @note This offset is only valid against the pre-translation source image.
  /// After .text is replaced/grown, a descriptor section that follows .text is
  /// shifted and this offset no longer points at the descriptor. Consumers that
  /// run post-growth (e.g. sidecar materialization) must use
  /// @ref source_descriptor_bytes instead of re-reading at this offset.
  uint64_t descriptor_file_offset = 0;

  /// @brief Snapshot of the 64 source AMDHSA kernel-descriptor bytes.
  ///
  /// @details Captured at translation time while @ref descriptor_file_offset is
  /// still valid, so sidecar descriptors materialized after .text growth copy
  /// their template from here rather than from a stale, now-shifted file offset.
  std::array<uint8_t, 64> source_descriptor_bytes{};

  /// @brief Kernel symbol name without the AMDHSA ".kd" descriptor suffix.
  std::string kernel_name;

  /// @brief Original .text-relative kernel entry decoded from the source descriptor.
  uint64_t entry_text_offset = 0;

  /// @brief True when legacy preload firmware may enter 256 bytes after the descriptor entry.
  ///
  /// @details CDNA3/CDNA4 use the legacy firmware compatibility scheme in which
  /// a non-zero kernarg_preload_spec_length makes compatible firmware add 256
  /// bytes to KERNEL_CODE_ENTRY_BYTE_OFFSET. GFX1250 also supports kernarg
  /// preloading, but its descriptor entry is the single hardware entry and must
  /// not be treated as the start of this compatibility window.
  bool has_kernarg_preload_firmware_skip = false;

  /// @brief Source .text-relative entry used by compatible kernarg-preload firmware.
  uint64_t kernarg_preload_firmware_entry_text_offset = 0;

  /// @brief Architecture the source descriptor was produced for.
  ///
  /// @details COMPUTE_PGM_RSRC3 has no single layout, so the patcher needs the
  /// source architecture to decide whether the source word can be carried over
  /// verbatim or must be rebuilt for the target. See @ref Rsrc3Layout for the
  /// per-family field maps; in particular GFX10 encodes SHARED_VGPR_COUNT but
  /// reserves bits 11:4, so INST_PREF_SIZE cannot be read from a GFX10 source --
  /// that field starts at GFX11.
  rj_code_arch_t source_arch = ROCJITSU_CODE_ARCH_INVALID;
};

/// @brief Target descriptor resources plus source/target accounting facts.
///
/// @details This plan is feature-neutral: semantic lowerings and DBI can request
/// resources without depending on entry-ABI or virtual-LDS implementation state.
struct KernelResourcePlan {
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

  /// @brief Ordinary target SGPR count required by translated code.
  uint32_t target_sgpr_count = 0;

  /// @brief Encoded target SGPR allocation granule.
  uint32_t target_sgpr_granulated = 0;

  /// @brief Hardware LDS bytes encoded in GROUP_SEGMENT_FIXED_SIZE.
  uint32_t target_lds_size = 0;

  /// @brief Per-lane private segment size encoded in the target descriptor.
  uint32_t target_private_size = 0;

  /// @brief Whether the source kernel uses a runtime-managed dynamic call stack.
  bool uses_dynamic_stack = false;

  /// @brief Target descriptor wavefront size.
  uint8_t target_wave_size = 64;

  /// @brief True when ABI translation explicitly forces wave64 execution.
  bool force_wave64 = false;

  /// @brief Guest and host wave sizes used for resource accounting.
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

  /// @brief Ordinary host VGPR count after lowering requirements are applied.
  uint32_t host_vgpr_count = 0;

  /// @brief Host descriptor's unified VGPR allocation count.
  ///
  /// @details Mirrors @c target_vgpr_allocation_count for diagnostics/reporting
  /// and includes any preserved target AccVGPR window.
  uint32_t host_vgpr_allocation_count = 0;

  /// @brief Ordinary source and target SGPR counts.
  uint32_t guest_sgpr_count = 0;
  uint32_t host_sgpr_count = 0;

  /// @brief Estimated source and target occupancy after resource translation.
  uint32_t source_occupancy = 0;
  uint32_t target_occupancy = 0;
};

/// @brief Kernel entry placement and target AMDHSA entry-ABI plan.
struct KernelEntryPlan {
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

  /// @brief Target and source initialized user-SGPR counts.
  uint8_t target_user_sgpr_count = 0;
  uint8_t source_user_sgpr_count = 0;

  /// @brief First source-visible SGPR restored after adding a target kernarg pointer.
  uint16_t user_sgpr_repair_start = 0;

  /// @brief Number of initialized SGPR dwords shifted down before guest code.
  uint16_t user_sgpr_repair_count = 0;

  /// @brief True when target entry ABI requires flat-scratch initialization.
  bool needs_flat_scratch_init_sgpr = false;

  /// @brief Encoded SGPR moves used to repair source-visible entry state.
  std::vector<uint32_t> user_sgpr_shuffle;

  /// @brief All kernel-entry instructions required by descriptor ABI translation.
  ///
  /// @details BinaryTranslator places these words in the kernel-local .text cave
  /// and records the final descriptor entry offset.
  std::vector<uint32_t> prologue_words;
};

/// @brief Optional sidecar and runtime ABI plan for virtual LDS.
struct VirtualLdsPlan {
  /// @brief True for a non-symbolized descriptor referenced through metadata.
  ///
  /// @details Sidecar descriptors are materialized in a loaded ELF segment and
  /// referenced only by rocjitsu metadata; they are not exported as additional
  /// AMDHSA kernel symbols. Virtual LDS is the first producer, but the patching
  /// contract is deliberately descriptor-generic so DBI instrumentation can add
  /// its own sidecar descriptors later.
  bool sidecar_descriptor = false;

  /// @brief Static LDS bytes that must be backed by a virtual LDS buffer.
  uint32_t lds_overflow_size = 0;

  /// @brief True when the dispatch path must provide a virtual LDS buffer.
  bool needs_lds_overflow_buf = false;

  /// @brief True on a normal (hardware-LDS) descriptor whose static LDS exceeds
  /// the host per-workgroup limit, so it is only launchable through its virtual
  /// sidecar variant.
  ///
  /// @details Such a normal descriptor is translated with allow_oversized_lds so
  /// the sidecar can be built, but its advertised LDS would fault on the host if
  /// the launch ever fell back to it. When the sidecar variant fails and is
  /// skipped, the normal descriptor must be stubbed too rather than left as a
  /// dispatchable-but-doomed launch target.
  bool static_lds_requires_sidecar = false;

  /// @brief Concrete scratch choices made while lowering the virtual-LDS body.
  ///
  /// @details This state is deliberately separate from descriptor-derived ABI
  /// fields because descriptor recomputation must preserve it byte-for-byte.
  VirtualLdsLoweringState virtual_lds_lowering;

  /// @brief Wrapper byte offset of the virtual-LDS runtime state payload.
  uint32_t lds_overflow_kernarg_pointer_offset = 0;

  /// @brief Workgroup-id SGPRs before wrapper-prologue ABI repair.
  int16_t lds_overflow_workgroup_id_sgpr_x = -1;
  int16_t lds_overflow_workgroup_id_sgpr_y = -1;
  int16_t lds_overflow_workgroup_id_sgpr_z = -1;
};

/// @brief Composed per-kernel descriptor/resource/ABI translation plan.
///
/// @details The component plans are independently reusable by descriptor
/// translation, semantic DBT, patching, and future DBI entry instrumentation.
/// Inheritance retains the established field syntax while call sites migrate to
/// narrower component types.
struct KdTranslation : KernelDescriptorFacts, KernelResourcePlan, KernelEntryPlan, VirtualLdsPlan {
  /// @brief False when resource or ABI translation cannot be represented safely.
  bool supported = true;

  /// @brief True when DBT emitted a target `s_endpgm` stub instead of translating this kernel.
  bool skipped = false;

  /// @brief Structured warnings and errors produced while planning this descriptor.
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
                  const KernelDescriptorTranslationOptions &options,
                  std::optional<size_t> text_section_index = std::nullopt) const;

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
