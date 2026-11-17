// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

namespace rocjitsu::hooks {

/// One chain-safe function-pointer replacement in a process HSA API table.
///
/// A tool must call the function that occupied a table slot when it loaded and
/// restore that function only if its own replacement still occupies the slot.
/// This small owner makes those two chaining rules one invariant shared by the
/// RocJitsu HSA tools rather than open-coded save/patch/compare/restore state.
template <typename FunctionPointer> class HsaApiFunctionPatch {
  static_assert(std::is_pointer_v<FunctionPointer>);

public:
  void capture(FunctionPointer *slot) {
    slot_ = slot;
    original_ = slot == nullptr ? nullptr : *slot;
    replacement_ = nullptr;
  }

  void install(FunctionPointer replacement) {
    replacement_ = replacement;
    if (slot_ != nullptr && replacement_ != nullptr)
      *slot_ = replacement_;
  }

  void restore() {
    if (slot_ != nullptr && replacement_ != nullptr && *slot_ == replacement_)
      *slot_ = original_;
    replacement_ = nullptr;
  }

  void clear() {
    slot_ = nullptr;
    original_ = nullptr;
    replacement_ = nullptr;
  }

  [[nodiscard]] FunctionPointer original() const { return original_; }

private:
  FunctionPointer *slot_ = nullptr;
  FunctionPointer original_ = nullptr;
  FunctionPointer replacement_ = nullptr;
};

} // namespace rocjitsu::hooks
