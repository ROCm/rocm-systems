/* Copyright (c) 2009 - 2021 Advanced Micro Devices, Inc.

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

#include <memory>
#include "rocprogram.hpp"
#include "top.hpp"
#include "rocprintf.hpp"

namespace amd::roc {

class Kernel : public device::Kernel {
 public:
  Kernel(std::string name, Program* prog) : device::Kernel(prog->device(), name, *prog) {}

  virtual ~Kernel() {
    if (program() != nullptr) {
      // Add kernel to the map of all kernels on the device
      program()->rocDevice().RemoveKernel(*this);
    }
  }

  //! Initializes the metadata required for this kernel
  virtual bool init() final;

  //! Setup after code object loading
  bool postLoad();

  const Program* program() const { return static_cast<const Program*>(&prog_); }

  //! Pull demangled name, used only for logging
  const std::string& getDemangledName() {
    if (demangled_name_.empty()) {
      initDemangledName();
    }
    return demangled_name_;
  }

 private:
  void initDemangledName() {
    if (demangled_name_.empty()) {
      amd::Os::CxaDemangle(name(), &demangled_name_);
    }
  }

  std::string demangled_name_;  //!< Cache demangled name
};

}  // namespace amd::roc

