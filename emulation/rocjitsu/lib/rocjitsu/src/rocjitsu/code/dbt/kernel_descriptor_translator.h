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
#include <string>
#include <vector>

namespace rocjitsu {

/// @brief Additional descriptor resource requirements from instruction lowering.
struct KernelDescriptorTranslationOptions {
  uint32_t minimum_vgprs = 0;
  uint32_t minimum_sgprs = 0;
  uint32_t private_segment_fixed_size_addend = 0;
  uint32_t group_segment_fixed_size_addend = 0;
  /// @brief Minimum source kernarg byte range that must be preserved.
  ///
  /// @details Some kernels, notably Tensile UserArgs kernels, advertise the
  /// ABI-visible fixed kernarg header in the descriptor while scalar prologue
  /// code also reads inline argument records beyond that byte count. When DBT
  /// appends virtual-LDS state, the runtime redirects the dispatch packet to a
  /// copied kernarg buffer. The copy size must therefore cover every statically
  /// proven kernarg read, otherwise the appended state can overlap live inline
  /// arguments and change kernel control flow.
  uint32_t minimum_kernarg_preserve_size = 0;
  /// @brief Encode the target descriptor with zero hardware LDS and require a
  /// virtual LDS backing buffer for the kernel body.
  ///
  /// @details This is an explicit per-descriptor mode because the runtime must
  /// only select the virtualized copy when the dispatch's static plus dynamic
  /// LDS demand cannot fit on the host GPU. The descriptor translator records
  /// the static byte demand in KdTranslation; runtime dispatch adds the packet's
  /// dynamic group-segment size when allocating the backing buffer.
  bool virtualize_lds = false;
  /// @brief Permit a normal descriptor to retain an LDS size above the host
  /// hardware limit.
  ///
  /// @details This is only valid when the caller is also emitting a virtual-LDS
  /// sidecar descriptor for the same kernel. The normal descriptor is kept so
  /// launches that fit on the host can still use hardware LDS; runtime dispatch
  /// rewriting must select the virtual descriptor whenever static plus dynamic
  /// LDS exceeds the host limit.
  bool allow_oversized_lds = false;
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

  /// @brief True when this plan describes the sidecar virtual-LDS descriptor.
  ///
  /// @details The original symbol descriptor remains the normal hardware-LDS
  /// path. Sidecar descriptors are materialized in a loaded ELF segment and
  /// referenced only by rocjitsu's dispatch metadata; they are not exported as
  /// additional AMDHSA kernel symbols.
  bool virtual_lds_variant = false;

  uint32_t target_sgpr_count = 0;
  uint32_t target_sgpr_granulated = 0;
  uint32_t sgpr_spill_count = 0;

  /// @brief Hardware LDS bytes encoded in GROUP_SEGMENT_FIXED_SIZE.
  uint32_t target_lds_size = 0;
  /// @brief Future in-hardware LDS bytes reserved for register spill zones.
  uint32_t lds_spill_zone_bytes = 0;
  /// @brief Static LDS bytes that must be backed by a virtual LDS buffer.
  uint32_t lds_overflow_size = 0;
  /// @brief True when the dispatch path must provide a virtual LDS buffer.
  bool needs_lds_overflow_buf = false;
  /// @brief SGPR pair reserved for the virtual LDS backing-buffer base address.
  uint16_t lds_overflow_base_sgpr = 0;
  /// @brief Temporary SGPR pair used by the entry prologue's workgroup offset math.
  ///
  /// @details The virtual-LDS prologue adjusts the backing pointer by the
  /// current workgroup's per-dispatch byte stride before the translated body
  /// runs. This scratch SGPR is descriptor-backed when there is room, otherwise
  /// it borrows an ordinary high SGPR at entry before guest code has defined it.
  uint16_t lds_overflow_prologue_temp_sgpr = 0;
  /// @brief True when @ref lds_overflow_base_sgpr is preserved around each DS use.
  bool lds_overflow_base_sgpr_spill_per_use = false;
  /// @brief True when the backing pointer itself is saved in private scratch.
  ///
  /// @details The dispatch/kernarg pointer SGPR pair is guest-owned after
  /// entry. When the virtual-LDS base SGPR pair is borrowed per use, DBT must
  /// consume the runtime pointer before the body can clobber that SGPR pair and
  /// reload the saved backing pointer from private scratch at each lowered LDS
  /// operation.
  bool lds_overflow_base_pointer_spilled = false;
  /// @brief Per-lane private scratch offset of the saved 64-bit backing pointer.
  uint32_t lds_overflow_base_pointer_spill_offset = 0;
  /// @brief True when spill-per-use entry prologue VGPR temps have been chosen.
  ///
  /// @details The descriptor may be recomputed after body lowering requests
  /// more registers or private scratch. Spill-per-use virtual LDS prologues
  /// are emitted before that recompute, so the temporary VGPRs used to save
  /// the runtime backing pointer must remain byte-for-byte stable across the
  /// recomputed descriptor.
  bool lds_overflow_entry_temp_vgprs_valid = false;
  /// @brief First VGPR temp used by the spill-per-use entry prologue.
  uint8_t lds_overflow_entry_temp_vgpr_lo = 0;
  /// @brief Second VGPR temp used by the spill-per-use entry prologue.
  uint8_t lds_overflow_entry_temp_vgpr_hi = 0;
  /// @brief Source kernarg bytes the runtime must preserve before appended data.
  ///
  /// @details Some assembly code objects leave the descriptor kernarg_size field
  /// at zero while using AMDHSA kernarg preloading. For virtual LDS, dispatch
  /// rewriting must still copy the preloaded byte range into the extended
  /// kernarg buffer before writing the backing-pointer tail.
  uint32_t kernarg_size = 0;
  /// @brief Kernarg segment size required by the translated target descriptor.
  ///
  /// @details Normal kernels preserve @ref kernarg_size. Virtual-LDS kernels
  /// append a rocjitsu runtime-state block after the original kernarg payload
  /// and load it in the entry prologue, so their target descriptor advertises
  /// the extended byte count.
  uint32_t target_kernarg_size = 0;
  /// @brief Byte offset of the virtual-LDS runtime state.
  ///
  /// @details This is a kernarg-segment offset for the normal extended-kernarg
  /// ABI. When @ref lds_overflow_pointer_in_dispatch_packet is true, it is
  /// instead an `hsa_kernel_dispatch_packet_t` byte offset that points at a
  /// GPU-visible rocjitsu runtime-state block.
  uint32_t lds_overflow_kernarg_pointer_offset = 0;
  /// @brief True when the backing pointer is passed through the AQL dispatch packet.
  bool lds_overflow_pointer_in_dispatch_packet = false;
  /// @brief True if the source descriptor exposes a kernarg segment pointer SGPR.
  bool has_kernarg_segment_ptr = false;
  /// @brief First SGPR of the descriptor-selected kernarg segment pointer pair.
  uint16_t kernarg_segment_ptr_sgpr = 0;
  /// @brief True if the source descriptor exposes a dispatch-packet pointer SGPR.
  bool has_dispatch_ptr = false;
  /// @brief First SGPR of the descriptor-selected dispatch-packet pointer pair.
  uint16_t dispatch_ptr_sgpr = 0;
  /// @brief Descriptor-selected workgroup-id SGPRs, or -1 when a dimension is disabled.
  int16_t workgroup_id_sgpr_x = -1;
  int16_t workgroup_id_sgpr_y = -1;
  int16_t workgroup_id_sgpr_z = -1;

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
  /// @brief True when DBT preserved this source kernel instead of translating it.
  bool skipped = false;
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
