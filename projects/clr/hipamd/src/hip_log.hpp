/* Copyright (c) 2015 - 2025 Advanced Micro Devices, Inc.

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
#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>
#include "vdi_common.hpp"

namespace hip {

class LoggingInfo {

public:
  void init();

  // Singleton instance
  static LoggingInfo& instance() {
    if (lginfo_ == nullptr) {
      // __hipRegisterFatBinary() will call this when app starts, thus
      // there is no multiple entry issue here.
      lginfo_ = new LoggingInfo();
    }
    return *lginfo_;
  }

  size_t log_level_;
  size_t log_size_;
  size_t log_mask_;

  amd::Monitor lg_lock_{true};

private:
  // Singleton object
  static LoggingInfo* lginfo_;
  LoggingInfo() {}
  ~LoggingInfo() {}
};
} // namespace::hip_impl
