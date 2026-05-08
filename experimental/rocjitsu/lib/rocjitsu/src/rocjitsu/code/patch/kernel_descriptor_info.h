// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernel_descriptor_info.h
/// @brief Kernel-descriptor-derived metadata used by patching and analysis.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu {

/// @brief Information about a kernel's workgroup_id SGPR layout.
struct KernelWorkGroupIdInfo {
  uint64_t entry_text_offset; ///< Kernel entry offset relative to .text start.
  int8_t sgpr_wg_id_x;        ///< SGPR index for workgroup_id_x, or -1 if not used.
  int8_t sgpr_wg_id_y;        ///< SGPR index for workgroup_id_y, or -1 if not used.
  int8_t sgpr_wg_id_z;        ///< SGPR index for workgroup_id_z, or -1 if not used.
};

/// @brief Kernel descriptor metadata normalized for consumers.
struct KernelDescriptorInfo {
  uint64_t entry_text_offset = 0;
  std::optional<uint8_t> wavefront_size;
  KernelWorkGroupIdInfo workgroup_id{0, -1, -1, -1};
};

/// @brief Parse all kernel descriptors associated with the text section.
[[nodiscard]] std::vector<KernelDescriptorInfo>
collect_kernel_descriptor_info(std::span<const uint8_t> image, uint64_t text_offset,
                               uint64_t text_size);

/// @brief Wavefront size to use for one kernel translation scope.
[[nodiscard]] uint8_t kernel_wavefront_size(rj_code_arch_t guest_arch,
                                            const KernelDescriptorInfo *kernel);

} // namespace rocjitsu
