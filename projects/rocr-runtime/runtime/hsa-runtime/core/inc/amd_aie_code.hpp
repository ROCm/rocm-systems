/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_CORE_INC_AMD_AIE_CODE_HPP_
#define HSA_RUNTIME_CORE_INC_AMD_AIE_CODE_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rocr {
namespace amd { namespace elf { class Image; } }
namespace AMD {

/// @brief Parsed metadata for one AIE kernel; blob pointers alias the ELF buffer.
struct AieKernelInfo {
  std::string name;  ///< Kernel name.
  const uint8_t* insts_data =
      nullptr;                        ///< Instruction blob in the ELF buffer; non-null after parse.
  uint64_t insts_size = 0;            ///< Instruction blob size in bytes; > 0.
  const uint8_t* pdi_data = nullptr;  ///< PDI blob in the ELF buffer; null if no PDI.
  uint64_t pdi_size = 0;              ///< PDI blob size in bytes; 0 if no PDI.
  uint32_t kernarg_size = 0;          ///< Kernel argument buffer size in bytes.
  uint32_t num_cols = 0;              ///< Number of NPU columns the kernel uses.
};

/// @brief Parses an aie2/aie2p hsaco section and exposes its per-kernel metadata.
///
/// Blob pointers in the returned AieKernelInfo alias the caller's ELF buffer, so
/// that buffer must outlive this object.
class AieCode {
 public:
  /// @brief Parses @p data; returns null if it is not a valid AIE code object.
  static std::unique_ptr<AieCode> Create(const void* data, size_t size);
  /// @brief Returns true if @p data is an ELF containing an aie2/aie2p section.
  static bool IsAieCodeObject(const void* data, size_t size);

  /// @brief Returns the arch section name ("aie2" or "aie2p").
  const std::string& GetArchSectionName() const { return arch_section_name_; }
  /// @brief Returns the names of all kernels in the object.
  std::vector<std::string> GetKernelNames() const;
  /// @brief Returns the kernel with @p name, or null if absent.
  const AieKernelInfo* GetKernel(const std::string& name) const;
  /// @brief Returns the number of kernels in the object.
  size_t GetKernelCount() const { return kernels_.size(); }

 private:
  AieCode() = default;
  bool Parse();

  std::unique_ptr<amd::elf::Image> elf_;          ///< Parsed ELF view over the caller's buffer.
  const uint8_t* elf_base_ = nullptr;             ///< Start of the caller's ELF buffer.
  size_t elf_size_ = 0;                           ///< Size of the caller's ELF buffer.
  const uint8_t* section_base_ = nullptr;         ///< Start of the arch section in the ELF buffer.
  uint64_t section_size_ = 0;                     ///< Size of the arch section in bytes.
  std::string arch_section_name_;                 ///< "aie2" or "aie2p".
  std::map<std::string, AieKernelInfo> kernels_;  ///< Parsed kernels keyed by name.
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_AIE_CODE_HPP_
