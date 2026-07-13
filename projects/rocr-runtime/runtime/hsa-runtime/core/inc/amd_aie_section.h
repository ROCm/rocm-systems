/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_CORE_INC_AMD_AIE_SECTION_H_
#define HSA_RUNTIME_CORE_INC_AMD_AIE_SECTION_H_

#include <cstdint>

namespace rocr {
namespace AMD {

/// @brief Section magic: 'A','I','E','K' little-endian.
constexpr uint32_t kAieSectionMagic = 0x4B454941u;
/// @brief Section format major version; a mismatch is rejected.
constexpr uint16_t kAieSectionVersionMajor = 1;
/// @brief Section format minor version; bumped for additive-only changes.
constexpr uint16_t kAieSectionVersionMinor = 0;

/// @brief Header of the AIE hsaco section.
///
/// All @c *_offset fields are section-relative (bytes from the section start).
struct aie_section_header {
  /// @brief Must equal kAieSectionMagic.
  uint32_t magic;
  /// @brief Must equal kAieSectionVersionMajor.
  uint16_t version_major;
  /// @brief Minor version; additive-only.
  uint16_t version_minor;
  /// @brief Offset from section base to the kernel table.
  uint32_t header_size;
  /// @brief Number of kernel table entries.
  uint32_t kernel_count;
  /// @brief Stride in bytes between kernel entries.
  uint32_t kernel_entry_size;
  /// @brief Section-relative offset of the string table.
  uint32_t string_table_offset;
  /// @brief Size of the string table in bytes.
  uint32_t string_table_size;
  /// @brief Blob pool spans [blob_pool_offset, section_end).
  uint32_t blob_pool_offset;
  /// @brief Reserved; must be 0.
  uint32_t reserved[4];
};

/// @brief One kernel entry in the AIE section's kernel table.
struct aie_kernel_entry {
  /// @brief Kernel name offset, relative to string_table_offset; NUL-terminated.
  uint32_t name_offset;
  /// @brief Section-relative offset of the instruction blob; required.
  uint32_t insts_offset;
  /// @brief Instruction blob size in bytes; required, > 0.
  uint32_t insts_size;
  /// @brief Section-relative offset of the PDI blob; 0 if no PDI.
  uint32_t pdi_offset;
  /// @brief PDI blob size in bytes; 0 if no PDI.
  uint32_t pdi_size;
  /// @brief Kernel argument buffer size in bytes.
  uint32_t kernarg_size;
  /// @brief Number of NPU columns the kernel uses.
  uint32_t num_cols;
  /// @brief Reserved; must be 0.
  uint32_t reserved[4];
};

/// @brief Internal, host-side kernel descriptor.
///
/// The @c kernel_object handle returned by HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT
/// is a pointer to one of these. Owned by the loaded code object and freed at
/// executable destroy.
struct AieKernelDescriptor {
  /// @brief Descriptor version; set to kAieKernelDescriptorVersion.
  uint32_t version;
  /// @brief Reserved; must be 0.
  uint32_t reserved0;
  /// @brief Device address of the instruction blob (an XDNA BO).
  uint64_t insts_dev_addr;
  /// @brief Instruction blob size in bytes.
  uint64_t insts_size;
  /// @brief Device address of the PDI blob (an XDNA BO), or 0.
  uint64_t pdi_dev_addr;
  /// @brief PDI blob size in bytes; 0 if no PDI.
  uint64_t pdi_size;
  /// @brief Kernel argument buffer size in bytes.
  uint32_t kernarg_size;
  /// @brief Number of NPU columns the kernel uses.
  uint32_t num_cols;
};

/// @brief Current AieKernelDescriptor::version value.
constexpr uint32_t kAieKernelDescriptorVersion = 1;

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_AIE_SECTION_H_
