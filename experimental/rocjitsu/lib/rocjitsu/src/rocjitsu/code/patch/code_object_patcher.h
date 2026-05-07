// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/kernel_descriptor_info.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "rocjitsu/code/rj_code.h"

namespace rocjitsu {

class AmdGpuCodeObject;

class CodeObjectPatcher {
public:
  explicit CodeObjectPatcher(const AmdGpuCodeObject &obj);

  std::span<uint8_t> text_bytes();
  std::span<const uint8_t> text_bytes() const;

  void overwrite_text(std::span<const uint8_t> new_text);

  void update_elf_flags(uint32_t new_flags);

  /// @brief Patch kernel descriptors for target Wave64 mode.
  /// @param target_arch Target architecture whose descriptor packing rules apply.
  /// Clears ENABLE_WAVEFRONT_SIZE32 (bit 10 of kernel_code_properties).
  void patch_kernel_descriptors_for_wave64(rj_code_arch_t target_arch);

  void append_cave_body(std::span<const uint32_t> words);

  uint64_t cave_body_size() const { return cave_body_.size(); }

  /// @brief Set the byte offset within .text where the cave body will be placed.
  /// This must be called before any apply_semantic() calls so branch offsets
  /// are computed correctly.
  void set_cave_start(uint64_t offset) { cave_start_ = offset; }

  /// @brief Get the cave start offset within .text.
  uint64_t cave_start() const { return cave_start_; }

  std::span<const uint8_t> cave_body() const { return cave_body_; }

  /// @brief Get parsed kernel descriptor metadata for the current image.
  std::vector<KernelDescriptorInfo> kernel_descriptor_info() const;

  std::vector<uint8_t> emit() const;

private:
  std::vector<uint8_t> image_;
  uint64_t text_offset_;
  uint64_t text_size_;
  std::vector<uint8_t> cave_body_;
  uint64_t cave_start_ = 0;
};

} // namespace rocjitsu
