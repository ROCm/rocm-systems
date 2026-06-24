// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file amdgpu_code_object.h
/// @brief AMD GPU HSA ELF code object representation.

#ifndef ROCJITSU_CODE_AMDGPU_CODE_OBJECT_H_
#define ROCJITSU_CODE_AMDGPU_CODE_OBJECT_H_

#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

/// @brief Represents a single AMD GPU HSA ELF code object.
///
/// A code object is a device ELF containing GPU machine code (.text sections),
/// read-only data (.rodata), and metadata. It may be loaded from a standalone
/// file or extracted from a HIP fat binary by Executable.
///
/// @see [AMDGPU Backend](https://llvm.org/docs/AMDGPUUsage.html)
class AmdGpuCodeObject : public CodeObject {
public:
  AmdGpuCodeObject() = default;
  AmdGpuCodeObject(const AmdGpuCodeObject &) = delete;
  AmdGpuCodeObject &operator=(const AmdGpuCodeObject &) = delete;
  AmdGpuCodeObject(AmdGpuCodeObject &&) noexcept;
  AmdGpuCodeObject &operator=(AmdGpuCodeObject &&) = delete;
  ~AmdGpuCodeObject();

  /// @brief Load a standalone device ELF from @p elf_path.
  /// @param[in] elf_path Path to a standalone device ELF file.
  explicit AmdGpuCodeObject(const std::string &elf_path);

  /// @brief Construct from raw ELF bytes in memory.
  /// @param[in] elf_bytes Pointer to the ELF image.
  /// @param[in] elf_size Size of the ELF image in bytes.
  AmdGpuCodeObject(const uint8_t *elf_bytes, size_t elf_size);

  /// @brief Construct from an embedded ELF image copied out of a HIP fat binary.
  /// @param[in] size Size of the embedded ELF in bytes.
  /// @param[in] elf_bytes Pointer to the embedded ELF image.
  /// @param[in] offload_kind Clang offload bundle kind string (e.g. "hip", "hipv4").
  /// @param[in] target_triple GPU target triple string (e.g. "gfx942", "gfx950", "gfx1250").
  AmdGpuCodeObject(const uint8_t *elf_bytes, size_t elf_size, std::string offload_kind,
                   std::string target_triple);

  /// @brief The target ID for this code object (e.g. ROCJITSU_CODE_TARGET_GFX942).
  /// @returns Target ID enum value.
  rj_code_target_id_t target_id() const { return target_id_; }

  /// @brief The GPU target triple string (e.g. "gfx942").
  /// @returns Reference to the target triple string.
  const std::string &target_triple() const { return target_triple_; }

  uint64_t kernel_descriptor_offset(const std::string &kernel_name) const override;

private:
  void load_sections();

  rj_code_target_id_t target_id_ = ROCJITSU_CODE_TARGET_INVALID;
  std::string offload_kind_;
  std::string target_triple_;
  std::unordered_map<std::string, uint64_t> kd_offsets_; ///< kernel_name -> .kd symbol offset
};

} // namespace rocjitsu

#endif // ROCJITSU_CODE_AMDGPU_CODE_OBJECT_H_
