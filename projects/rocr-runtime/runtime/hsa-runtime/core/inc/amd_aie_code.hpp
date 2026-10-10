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
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/inc/amd_aie_section.h"

namespace rocr {
namespace amd {
namespace elf {
class Image;
class Section;
}  // namespace elf
}  // namespace amd
namespace AMD {

/// @brief Parsed metadata for one AIE kernel; blob pointers alias the ELF buffer.
struct AieKernelInfo {
  /// @brief Kernel name.
  std::string name;
  /// @brief Instruction blob (PdiInsts) or nested full ELF (FullElf) in the ELF buffer;
  /// non-nullptr after parse.
  const uint8_t* insts_data = nullptr;
  /// @brief Size in bytes of the blob at @ref insts_data; > 0.
  uint64_t insts_size = 0;
  /// @brief PDI blob in the ELF buffer; nullptr if no PDI (full-ELF).
  const uint8_t* pdi_data = nullptr;
  /// @brief PDI blob size in bytes; 0 if no PDI.
  uint64_t pdi_size = 0;
  /// @brief Kernel argument buffer size in bytes.
  uint32_t kernarg_size = 0;
  /// @brief Number of columns of the NPU partition the kernel was compiled for (the partition
  /// width, not the columns the design occupies).
  uint32_t num_cols = 0;
  /// @brief Payload kind; see @ref AieKernelKind.
  AieKernelKind kind = AieKernelKind::PdiInsts;
};

/// @brief Parses an AIE hsaco section and exposes its per-kernel metadata.
///
/// Blob pointers in the returned @ref AieKernelInfo alias the caller's ELF buffer, so
/// that buffer must outlive this object.
class AieCode {
 public:
  /// @brief Parses the AIE section for @p arch in @p data.
  ///
  /// @param [in] data Pointer to the hsaco bytes; must outlive the returned object.
  /// @param [in] size Size of @p data in bytes.
  /// @param [in] arch Arch name of the agent, which is also the name of the section to parse.
  /// @param [out] out The parsed object; only set on success.
  /// @retval HSA_STATUS_SUCCESS The section for @p arch was found and is well formed.
  /// @retval HSA_STATUS_ERROR_INVALID_ARGUMENT @p out is nullptr.
  /// @retval HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS @p data has no AIE section for @p arch.
  /// @retval HSA_STATUS_ERROR_INVALID_CODE_OBJECT @p data is empty or not an ELF64, @p arch is
  /// empty, or the AIE section for @p arch is malformed.
  static hsa_status_t Create(const void* data, size_t size, std::string_view arch,
                             std::unique_ptr<AieCode>* out);

  /// @brief Returns true if @p data is an ELF containing an AIE section for any arch.
  static bool IsAieCodeObject(const void* data, size_t size);

  /// @brief Returns the arch section name.
  const std::string& GetArchSectionName() const { return arch_section_name_; }

  /// @brief Returns the names of all kernels in the object.
  std::vector<std::string> GetKernelNames() const;

  /// @brief Returns the kernel with @p name, or nullptr if absent.
  const AieKernelInfo* GetKernel(const std::string& name) const;

 private:
  AieCode() = default;
  /// @brief Parses the AIE section @p sec, whose header has already been bounds-checked.
  bool Parse(amd::elf::Section* sec);

  /// @brief ELF view over the caller's buffer; does not copy or own the bytes.
  std::unique_ptr<amd::elf::Image> elf_;
  /// @brief Arch name, which is also the name of the parsed section.
  std::string arch_section_name_;
  /// @brief Parsed kernels keyed by name.
  std::unordered_map<std::string, AieKernelInfo> kernels_;
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_AIE_CODE_HPP_
