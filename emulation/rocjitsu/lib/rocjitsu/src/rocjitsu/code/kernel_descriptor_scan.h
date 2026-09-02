// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernel_descriptor_scan.h
/// @brief Shared AMDHSA kernel-descriptor discovery used by DBT and DBI.

#pragma once

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/rj_code.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

/// @brief One AMDHSA kernel descriptor located in an ELF image.
struct KernelDescriptorInfo {
  uint64_t descriptor_file_offset = 0; ///< File offset of the 64-byte descriptor (pre-growth).
  std::string kernel_name;             ///< Symbol name minus the ".kd" suffix.
  uint64_t entry_text_offset = 0;      ///< .text-relative kernel entry.
  rocr::llvm::amdhsa::kernel_descriptor_t descriptor{}; ///< Raw descriptor bytes.
};

/// Result of matching exact kernel names against an AMDHSA ELF symbol index.
enum class KernelNameIndexMatch {
  Matched,
  NoMatch,
  Indeterminate,
};

/// @brief Match any exact kernel name without decoding executable contents.
///
/// @details Walks only the ELF section headers, symbol tables, and linked
/// string tables. `NoMatch` is returned only after every advertised symbol
/// table was read safely; malformed or unsupported indexing returns
/// `Indeterminate` so a caller cannot mistake parser uncertainty for proof
/// that a requested kernel is absent. Names omit the AMDHSA `.kd` suffix.
/// Work and retained memory are independent of executable instruction count.
[[nodiscard]] KernelNameIndexMatch
match_kernel_name_index(std::span<const uint8_t> image,
                        std::span<const std::string> exact_kernel_names);

/// @brief Locate every ".kd" descriptor whose entry lands in .text.
///
/// @details Walks .symtab/.dynsym, decodes each descriptor's file offset and
/// .text-relative entry, drops entries outside .text, and dedups by file offset.
/// The single discovery routine shared by DBT translation and DBI; operates on the
/// raw, pre-growth image.
[[nodiscard]] std::vector<KernelDescriptorInfo>
scan_kernel_descriptors(std::span<const uint8_t> image, uint64_t text_offset, uint64_t text_size,
                        std::optional<size_t> text_section_index = std::nullopt);

/// @brief Wavefront size (32 or 64) the launch hardware interprets for @p desc.
///
/// @details CDNA is Wave64; gfx1250 is Wave32-only; RDNA opts into Wave32 via the
/// descriptor's ENABLE_WAVEFRONT_SIZE32 (a clear bit means Wave64). Shared by DBT
/// resource accounting and DBI descriptor decoding.
[[nodiscard]] uint8_t kernel_wavefront_size(rj_code_arch_t arch,
                                            const rocr::llvm::amdhsa::kernel_descriptor_t &desc);

/// @brief AMDHSA descriptor encoding granule for GRANULATED_WORKITEM_VGPR_COUNT.
///
/// @details This is the descriptor-encoding granularity (kernel VGPR count =
/// (granulated + 1) * granule), not the physical VGPR allocation block. CDNA1 uses
/// 4, other CDNA 8, gfx1250 16, and RDNA is wave-size dependent: 8 for Wave32, 4
/// for Wave64. Shared by DBT resource accounting and DBI descriptor decoding.
[[nodiscard]] uint32_t descriptor_vgpr_granularity_for_wavefront(rj_code_arch_t arch,
                                                                 uint32_t wavefront_size);

} // namespace rocjitsu
