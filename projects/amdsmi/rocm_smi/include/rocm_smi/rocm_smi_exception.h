// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_EXCEPTION_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_EXCEPTION_H_

#include <exception>
#include <string>

#include "rocm_smi/rocm_smi.h"

#define THROW_IF_NULLPTR_DEREF(PTR)                                         \
  assert((PTR) != nullptr);                                                 \
  if ((PTR) == nullptr) {                                                   \
    throw amd::smi::rsmi_exception(RSMI_STATUS_INVALID_ARGS, __FUNCTION__); \
  }

namespace amd::smi {

/// @brief Exception type which carries an error code to return to the user.
class rsmi_exception : public std::exception {
 public:
  rsmi_exception(rsmi_status_t error, const std::string description)
      : err_(error), desc_(description) {}
  rsmi_status_t error_code() const noexcept { return err_; }
  const char* what() const noexcept override { return desc_.c_str(); }

 private:
  rsmi_status_t err_;
  std::string desc_;
};

}  // namespace amd::smi

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_EXCEPTION_H_
