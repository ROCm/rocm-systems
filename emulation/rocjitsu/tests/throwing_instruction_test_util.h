// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"

#include <stdexcept>

namespace rocjitsu::test {

// Exception-propagation tests must inject a host exception explicitly: an
// unimplemented guest instruction is an ordinary simulator failure.
class ThrowingInstructionPlugin final : public ExecutionPlugin {
public:
  bool observes_scalar_register_reads() const override { return true; }
  bool supports_async_instructions() const override { return false; }

  ThrowingInstructionPlugin() : ExecutionPlugin("throwing_instruction") {}

  // Throw from inside execution so the CU's instruction cleanup also runs.
  void onAmdgpuReadScalarRegister(const amdgpu::Wavefront *, RegisterRef) override {
    throw std::runtime_error("injected instruction callback failure");
  }
};

} // namespace rocjitsu::test
