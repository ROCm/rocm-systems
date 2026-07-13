/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_CORE_INC_AMD_AIE_CODE_HPP_
#define HSA_RUNTIME_CORE_INC_AMD_AIE_CODE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocr {
namespace amd { namespace elf { class Image; } }
namespace AMD {

/// @brief Parsed metadata for one AIE kernel; blob pointers alias the ELF buffer.
struct AieKernelInfo {
  /// @brief Kernel name.
  std::string name;
  /// @brief Instruction blob in the ELF buffer; non-nullptr after parse.
  const uint8_t* insts_data = nullptr;
  /// @brief Instruction blob size in bytes; > 0.
  uint64_t insts_size = 0;
  /// @brief PDI blob in the ELF buffer; nullptr if no PDI (full-ELF).
  const uint8_t* pdi_data = nullptr;
  /// @brief PDI blob size in bytes; 0 if no PDI.
  uint64_t pdi_size = 0;
  /// @brief Kernel argument buffer size in bytes.
  uint32_t kernarg_size = 0;
  /// @brief Number of NPU columns the kernel uses.
  uint32_t num_cols = 0;
};

/// @brief Parses an AIE hsaco section and exposes its per-kernel metadata.
///
/// Blob pointers in the returned @ref AieKernelInfo alias the caller's ELF buffer, so
/// that buffer must outlive this object.
class AieCode {
 public:
  /// @brief Parses @p data; returns nullptr if it is not a valid AIE code object.
  static std::unique_ptr<AieCode> Create(const void* data, size_t size);

  /// @brief Returns true if @p data is an ELF containing an AIE section.
  static bool IsAieCodeObject(const void* data, size_t size);

  /// @brief Returns the arch section name.
  const std::string& GetArchSectionName() const { return arch_section_name_; }

  /// @brief Returns the names of all kernels in the object.
  std::vector<std::string> GetKernelNames() const;

  /// @brief Returns the kernel with @p name, or nullptr if absent.
  const AieKernelInfo* GetKernel(const std::string& name) const;

 private:
  AieCode() = default;
  bool Parse();

  /// @brief Parsed ELF view over the caller's buffer.
  std::unique_ptr<amd::elf::Image> elf_;
  /// @brief Start of the caller's ELF buffer.
  const uint8_t* elf_base_ = nullptr;
  /// @brief Size of the caller's ELF buffer.
  size_t elf_size_ = 0;
  /// @brief Start of the arch section in the ELF buffer.
  const uint8_t* section_base_ = nullptr;
  /// @brief Size of the arch section in bytes.
  uint64_t section_size_ = 0;
  /// @brief Architecture.
  std::string arch_section_name_;
  /// @brief Parsed kernels keyed by name.
  std::unordered_map<std::string, AieKernelInfo> kernels_;
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_AIE_CODE_HPP_
