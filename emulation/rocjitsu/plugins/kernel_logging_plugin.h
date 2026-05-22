// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "plugins/execution_plugin.h"

#include <cstdint>
#include <cstdio>
#include <unordered_set>

namespace rocjitsu {
namespace amdgpu {

/// @brief Logging plugin that logs kernel dispatches and detects mfma usage.
class KernelLoggingPlugin : public ExecutionPlugin {
public:
  KernelLoggingPlugin();
  ~KernelLoggingPlugin() override;

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override;
  void afterAmdgpuExecuteInstruction(uint64_t pc, const Instruction &inst, Wavefront &wf) override;

private:
  int dispatch_count_ = 0;
  std::unordered_set<uint32_t> mfma_printed_;

  template <typename... Args> void log(const char *fmt, Args... args) {
    fprintf(stderr, fmt, args...);
  }
};

} // namespace amdgpu
} // namespace rocjitsu
