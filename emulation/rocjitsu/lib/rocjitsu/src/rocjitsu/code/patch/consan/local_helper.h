// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/consan/contract.h"
#include "rocjitsu/code/scoped_cfg_edges.h"

#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::consan {
struct InputError;

/// One direct call, a straight-line local callee, and its validated continuation.
struct LocalHelper {
  BasicBlock *caller = nullptr;
  BasicBlock *callee = nullptr;
  BasicBlock *continuation = nullptr;
  const Instruction *terminal_access = nullptr;
  std::optional<AccessKind> api;
  std::vector<ScopedCfgEdge> edges;

  std::span<BasicBlock *const> predecessors(const BasicBlock &block) const;
  std::span<BasicBlock *const> successors(const BasicBlock &block) const;
  bool is_transfer(const Instruction &instruction) const;
};

FailureOr<LocalHelper> qualify_local_helper(const AmdGpuCodeObject &object,
                                            std::span<BasicBlock *const> blocks, uint32_t wave_size,
                                            InputError &error);
} // namespace rocjitsu::consan
