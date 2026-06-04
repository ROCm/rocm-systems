// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_CODE_AMDGPU_ELF_READER_H_
#define ROCJITSU_CODE_AMDGPU_ELF_READER_H_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "rocjitsu/code/rj_code.h"

namespace rocjitsu {

struct AmdGpuKernelSite {
  std::string kernel_name;
  uint64_t descriptor_file_offset = 0;
  uint64_t entry_file_offset = 0;
};

bool is_supported_amdgpu_elf(std::span<const uint8_t> image);

std::vector<AmdGpuKernelSite> discover_amdgpu_kernel_sites(std::span<const uint8_t> image);

/// @brief Build a kernel-entry AFL device-counter probe.
///
/// The probe treats @p state_pointer as the base address of the device counter
/// array allocated by the RocFuzz AFL runtime. It writes a nonzero value to
/// counter slot 0 when a patched kernel entry executes before branching back to
/// the original kernel entry.
/// @p state_pointer is not AFL's host shared-memory bitmap; it is the
/// GPU-visible staging buffer that persistent_end later merges into AFL's map.
///
/// This is intentionally a fixed entry-only probe: for the minimal vector-add
/// smoke, one inserted prologue proves that raw AMDGPU ELF patching can feed
/// device-side coverage back into AFL without pulling in the full edge planner.
///
/// @param state_pointer Device virtual address of the AFL device-counter state.
/// @param arch Target ISA used for wait and memory instruction variants.
/// @returns Encoded 32-bit AMDGPU instruction words for the entry probe.
std::vector<uint32_t>
build_amdgpu_entry_counter_probe_words(uint64_t state_pointer,
                                       rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4);

/// @brief Rewrite raw AMDGPU ELF kernel descriptors to run an entry probe.
///
/// For each discovered `.kd` symbol in a supported raw AMDGPU ELF image, this
/// appends an entry-counter prologue into a code cave and redirects the kernel
/// descriptor entry to that prologue. Unsupported or malformed input fails open
/// by returning the original bytes unchanged.
///
/// @param image Raw AMDGPU ELF bytes.
/// @param state_pointer Device pointer to the AFL counter buffer.
/// @returns Rewritten ELF bytes when patching succeeds, or a copy of @p image.
std::vector<uint8_t> patch_amdgpu_elf_kernel_entries(std::span<const uint8_t> image,
                                                     uint64_t state_pointer);

} // namespace rocjitsu

#endif // ROCJITSU_CODE_AMDGPU_ELF_READER_H_
