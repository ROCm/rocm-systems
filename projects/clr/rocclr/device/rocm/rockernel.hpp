/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <memory>
#include "rocprogram.hpp"
#include "top.hpp"
#include "rocprintf.hpp"

namespace amd::roc {

class Kernel : public device::Kernel {
 public:
  Kernel(const std::string& name, Program* prog) : device::Kernel(prog->device(), name, *prog) {}

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
    std::call_once(demangle_once_, [this] {
      amd::Os::CxaDemangle(name(), &demangled_name_);
    });
    return demangled_name_;
  }

  std::string demangled_name_;  //!< Cache demangled name
  std::once_flag demangle_once_;
};

}  // namespace amd::roc

