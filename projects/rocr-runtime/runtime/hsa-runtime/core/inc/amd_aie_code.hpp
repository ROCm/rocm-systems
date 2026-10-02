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
namespace amd { namespace elf { class Image; class Section; } }
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
  ///
  /// Carried from the on-disk entry even though nothing consumes it yet: it is the partition
  /// geometry a payload declares, which is what decides whether two payloads may share a hardware
  /// context. Dropping it here would mean re-parsing the section to get it back.
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
  /// @retval HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS @p data has no AIE section for @p arch.
  /// @retval HSA_STATUS_ERROR_INVALID_CODE_OBJECT @p data is not an ELF64, or the AIE section for
  /// @p arch is malformed.
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

  /// @brief Parsed ELF view over the caller's buffer; owns the base/size (data()/size()).
  std::unique_ptr<amd::elf::Image> elf_;
  /// @brief Architecture.
  std::string arch_section_name_;
  /// @brief Parsed kernels keyed by name.
  std::unordered_map<std::string, AieKernelInfo> kernels_;
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_AIE_CODE_HPP_
