/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */
#ifndef SRC_UTILS_EXCEPTION_H_
#define SRC_UTILS_EXCEPTION_H_

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "helper.h"
#include "rocprofiler.h"

// TODO(aelwazir): namespace rocprofiler
namespace rocprofiler {

class Exception : public std::runtime_error {
 public:
  Exception() = delete;

  explicit Exception(rocprofiler_status_t status, const char* what_arg = "")
      : std::runtime_error(std::string(rocprofiler_error_str(status)) + " " + what_arg),
        status_(status) {}
  rocprofiler_status_t status() const noexcept { return status_; }

  explicit Exception(const std::string& message) : std::runtime_error(message) {
    message_ = message;
  }
  const char* what() const noexcept override { return message_.c_str(); }

 protected:
  rocprofiler_status_t status_;
  std::string message_;
};

};  // namespace rocprofiler

// TODO(aelwazir): throw instead of the macros

#endif  // SRC_UTILS_EXCEPTION_H_
