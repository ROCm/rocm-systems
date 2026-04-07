// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file executable.h
/// @brief Loader for x86 HIP fat binaries and standalone device ELFs.

#ifndef ROCJITSU_CODE_EXECUTABLE_H_
#define ROCJITSU_CODE_EXECUTABLE_H_

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/rj_code.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

/// @brief Loader for executables (x86 HIP fat binaries or standalone device ELFs).
///
/// Derives from CodeObject since the executable is itself an ELF file with a
/// header and sections. For x86 HIP fat binaries, the inherited members
/// (image_, header_, sections_) describe the host ELF, while embedded device
/// code objects are extracted from the `.hip_fatbin` section's Clang offload
/// bundles. For standalone device ELFs, a single child code object is created.
///
/// @see [Clang Offload Bundler](https://clang.llvm.org/docs/ClangOffloadBundler.html)
/// @see [AMDGPU Backend](https://llvm.org/docs/AMDGPUUsage.html)
class Executable : public CodeObject {
public:
  Executable() = default;
  Executable(const Executable &) = delete;
  Executable &operator=(const Executable &) = delete;
  Executable(Executable &&) = delete;
  Executable &operator=(Executable &&) = delete;
  ~Executable() override;

  /// @brief Load an executable from the file at @p path.
  /// @param[in] path Path to an x86 HIP fat binary or standalone device ELF.
  explicit Executable(const std::string &path);

  /// @brief Number of code objects for the given target.
  /// @param[in] target Target ID to query.
  /// @returns Number of code objects matching @p target, or 0 if none.
  uint32_t num_code_objects(rj_code_target_id_t target) const;

  /// @brief Access a code object by target and index.
  /// @param[in] target Target ID to look up.
  /// @param[in] index Index within the set of code objects for @p target.
  /// @returns Pointer to the code object, or nullptr if @p target or @p index is out of range.
  AmdGpuCodeObject *code_object(rj_code_target_id_t target, uint32_t index);
  /// @copydoc code_object(rj_code_target_id_t, uint32_t)
  const AmdGpuCodeObject *code_object(rj_code_target_id_t target, uint32_t index) const;

private:
  static constexpr char CLANG_OFFLOAD_MAGIC_STR[] = {'_', '_', 'C', 'L', 'A', 'N', 'G', '_',
                                                     'O', 'F', 'F', 'L', 'O', 'A', 'D', '_',
                                                     'B', 'U', 'N', 'D', 'L', 'E', '_', '_'};
  static constexpr std::size_t CLANG_OFFLOAD_MAGIC_STR_SIZE = sizeof(CLANG_OFFLOAD_MAGIC_STR);

  struct ClangOffloadBundleInfo {
    uint64_t offset;
    uint64_t size;
    uint64_t bundle_entry_id_size;
    std::string bundle_entry_id;
  };

  struct ClangOffloadBundleHeader {
    char magic[CLANG_OFFLOAD_MAGIC_STR_SIZE];
    uint64_t num_code_objs;
  };

  void load_device_elf(const std::string &path);
  void load_fat_binary(std::ifstream &elf_file);
  void load_hip_fatbin(const Section &fatbin_section, std::ifstream &elf_file);
  void register_code_object(std::unique_ptr<AmdGpuCodeObject> co);

  std::vector<std::unique_ptr<AmdGpuCodeObject>> owned_code_objs_;
  std::unordered_map<rj_code_target_id_t, std::vector<AmdGpuCodeObject *>> code_objs_by_target_;
};

} // namespace rocjitsu

#endif // ROCJITSU_CODE_EXECUTABLE_H_
