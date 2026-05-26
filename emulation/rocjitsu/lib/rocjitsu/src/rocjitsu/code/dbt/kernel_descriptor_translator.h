// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernel_descriptor_translator.h
/// @brief DBT layer for translating AMDHSA kernel descriptor ABI/resource fields.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
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
};

/// @brief Per-kernel descriptor/resource/ABI translation plan.
///
/// The descriptor translator computes this from the source kernel descriptor.
/// It does not mutate descriptor bytes. BinaryTranslator uses the plan for
/// instruction-level decisions, and CodeObjectPatcher applies the plan to the
/// ELF/kernel descriptor bytes.
struct KdTranslation {
  uint64_t descriptor_file_offset = 0;
  uint64_t entry_text_offset = 0;

  uint32_t target_vgpr_count = 0;
  uint32_t target_vgpr_granulated = 0;
  uint32_t accvgpr_base = 0;
  uint32_t vgpr_spill_to_lds_count = 0;
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
  /// CodeObjectPatcher places these words in a cave and redirects the KD entry.
  std::vector<uint32_t> prologue_words;

  uint8_t guest_wavefront_size = 64;
  uint8_t host_wavefront_size = 64;

  uint32_t guest_vgpr_count = 0;
  uint32_t host_vgpr_count = 0;
  uint32_t guest_sgpr_count = 0;
  uint32_t host_sgpr_count = 0;

  uint32_t source_occupancy = 0;
  uint32_t target_occupancy = 0;

  bool supported = true;
  std::vector<std::string> warnings;
};

/// @brief Compute descriptor patches and semantic metadata for one DBT pair.
class KernelDescriptorTranslator {
public:
  KernelDescriptorTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch);

  [[nodiscard]] std::vector<KdTranslation>
  translate_image(std::span<const uint8_t> image, uint64_t text_offset, uint64_t text_size,
                  const KernelDescriptorTranslationOptions &options) const;

private:
  rj_code_arch_t guest_arch_;
  rj_code_arch_t host_arch_;
};

} // namespace rocjitsu
