/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_EXCEPTION_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_EXCEPTION_H_

#include <exception>
#include <string>

#include "rocm_smi/rocm_smi.h"

#define THROW_IF_NULLPTR_DEREF(PTR) \
  assert((PTR) != nullptr); \
  if ((PTR) == nullptr) { \
    throw amd::smi::rsmi_exception(RSMI_STATUS_INVALID_ARGS, __FUNCTION__); \
  }

namespace amd::smi {

/// @brief Exception type which carries an error code to return to the user.
class rsmi_exception : public std::exception {
 public:
  rsmi_exception(rsmi_status_t error, const std::string description) :
                                            err_(error), desc_(description) {}
  rsmi_status_t error_code() const noexcept { return err_; }
  const char* what() const noexcept override { return desc_.c_str(); }

 private:
  rsmi_status_t err_;
  std::string desc_;
};

} // namespace amd::smi

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_EXCEPTION_H_

