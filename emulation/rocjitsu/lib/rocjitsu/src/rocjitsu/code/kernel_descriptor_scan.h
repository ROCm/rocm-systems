// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernel_descriptor_scan.h
/// @brief Shared AMDHSA kernel-descriptor discovery used by DBT and DBI.

#ifndef ROCJITSU_CODE_KERNEL_DESCRIPTOR_SCAN_H_
#define ROCJITSU_CODE_KERNEL_DESCRIPTOR_SCAN_H_

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <cstdint>
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

/// @brief Locate every ".kd" descriptor whose entry lands in .text.
///
/// @details Walks .symtab/.dynsym, decodes each descriptor's file offset and
/// .text-relative entry, drops entries outside .text, and dedups by file offset.
/// The single discovery routine shared by DBT translation and DBI; operates on the
/// raw, pre-growth image.
[[nodiscard]] std::vector<KernelDescriptorInfo>
scan_kernel_descriptors(std::span<const uint8_t> image, uint64_t text_offset, uint64_t text_size);

} // namespace rocjitsu

#endif // ROCJITSU_CODE_KERNEL_DESCRIPTOR_SCAN_H_
