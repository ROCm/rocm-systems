/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_CORE_INC_AMD_AIE_SECTION_H_
#define HSA_RUNTIME_CORE_INC_AMD_AIE_SECTION_H_

#include <cstdint>
#include <vector>

#include "core/inc/amd_aie_elf.h"

namespace rocr {
namespace AMD {

/// @brief Section magic: 'A','I','E','K' little-endian.
constexpr uint32_t kAieSectionMagic = 0x4B454941u;
/// @brief Section format major version; a mismatch is rejected.
constexpr uint16_t kAieSectionVersionMajor = 1;
/// @brief Section format minor version; bumped for additive-only changes.
constexpr uint16_t kAieSectionVersionMinor = 0;

/// @brief Payload kind of an @ref aie_kernel_entry.
///
/// The first @ref Count values are stored on disk: they are fixed and must never be
/// reused. The loader rejects anything not below @ref Count, which is what keeps a
/// future kind from being mis-dispatched by a build that predates it.
enum class AieKernelKind : uint32_t {
  /// @brief PDI plus a standalone instruction sequence; @c insts_* and @c pdi_* both used.
  PdiInsts = 0,
  /// @brief Nested full ELF; @c insts_offset / @c insts_size locate the ELF, @c pdi_* are 0.
  FullElf = 1,

  /// @brief Number of on-disk kinds, and the bound the loader validates against.
  Count,
  /// @brief No hardware context has been built for any kind yet. Queue state only --
  /// never written to disk, which is why it can share a value with @ref Count.
  Undecided = Count,
};

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
  /// @brief Payload kind; see @ref AieKernelKind.
  AieKernelKind kind;
  /// @brief Reserved; must be 0.
  uint32_t reserved[3];
};

/// @brief Internal, host-side kernel descriptor.
///
/// The @c kernel_object handle returned by HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT
/// is a pointer to one of these. Owned by the loaded code object and freed at
/// executable destroy.
struct AieKernelDescriptor {
  /// @brief Descriptor version; set to kAieKernelDescriptorVersion.
  uint32_t version;
  /// @brief Payload kind this descriptor was built from.
  AieKernelKind kind;
  /// @brief Reserved; must be 0.
  uint32_t reserved0;
  /// @brief Host virtual address of the instruction blob's XDNA BO (mmap'd to
  /// device); used directly as the device instruction address at submit.
  void* insts_bo_va;
  /// @brief Instruction blob size in bytes.
  uint64_t insts_size;
  /// @brief XDNA BO handle of the instruction blob, resolved once at load. The
  /// blob is immutable, so the handle is stable for the object's lifetime.
  uint32_t insts_bo_handle;
  /// @brief XDNA BO handle of the PDI blob, resolved once at load; unused when
  /// @ref pdi_size is 0 (no PDI).
  uint32_t pdi_bo_handle;
  /// @brief PDI blob size in bytes; 0 if no PDI.
  uint64_t pdi_size;
  /// @brief Kernel argument buffer size in bytes.
  uint32_t kernarg_size;
  /// @brief Number of NPU columns the kernel uses.
  uint32_t num_cols;
  /// @brief Pristine control code, in host memory. FullElf only; null for PdiInsts.
  ///
  /// The NPU never fetches this -- it is only ever a memcpy source for the per-dispatch
  /// buffer the driver allocates -- so it needs neither device memory nor alignment.
  void* ctrl_code;
  /// @brief Size of @ref ctrl_code in bytes; 0 for PdiInsts.
  uint64_t ctrl_code_size;
  /// @brief Number of arguments the control code references. FullElf only.
  uint32_t num_args;
  /// @brief Argument patch sites, flattened; sites for argument @c i are
  /// [arg_site_offset[i], arg_site_offset[i + 1]).
  std::vector<aie_elf::PatchSite> arg_sites;
  /// @brief Index into @ref arg_sites per argument; size is @ref num_args + 1.
  std::vector<uint32_t> arg_site_offset;
};

/// @brief Current AieKernelDescriptor::version value.
constexpr uint32_t kAieKernelDescriptorVersion = 1;

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_AIE_SECTION_H_
