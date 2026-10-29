// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <variant>

namespace rocjitsu::amdgpu {

/// A live dispatch or wave state that the context producer cannot currently save.
struct Unsupported {
  bool operator==(const Unsupported &) const = default;
};

/// Distinguish unsupported live state from failures accessing the save area.
using QueueContextSaveOutcome = std::variant<VmAccessOutcome, Unsupported>;

} // namespace rocjitsu::amdgpu
