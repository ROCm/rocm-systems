// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file disasm_cache.h
/// @brief PC-to-disassembly cache shared by plugins that quote instructions.
///
/// Plugins that name an instruction in their output need its text long after
/// the instruction executed. Decoding lazily at report time would need the
/// compute unit's decoder, so instead every plugin that quotes instructions
/// records the disassembly the first time it sees a PC.

#pragma once

#include "rocjitsu/isa/instruction.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace rocjitsu::plugins {

/// @brief PC-to-disassembly cache, safe to share across partition threads.
///
/// Keyed by absolute PC rather than by dispatch-relative offset, because a
/// dispatch does not necessarily execute one compact, monotonic text range;
/// helper and trampoline code can sit far from the first PC observed.
///
/// Sharing one cache across the wavefronts of a dispatch is deliberate: they
/// execute the same kernel code, and per-wavefront caches thrashed under
/// round-robin scheduling.
class DisasmCache {
public:
  /// Records the disassembly of @p inst the first time @p pc is seen.
  void record(uint64_t pc, const Instruction &inst) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (entries_.contains(pc))
      return;
    entries_.emplace(pc, inst.disassemble());
  }

  /// Disassembly recorded for @p pc, or an empty string if none was.
  std::string lookup(uint64_t pc) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(pc);
    return it != entries_.end() ? it->second : std::string{};
  }

  std::unordered_map<uint64_t, std::string> to_map() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
  }

private:
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, std::string> entries_;
};

} // namespace rocjitsu::plugins
