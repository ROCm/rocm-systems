// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consumer_cursor_journal.h
/// @brief Retry-safe consumer cursor loading and publication.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

/// @brief Journals a queue consumer cursor and its publication snapshot.
/// @details Retirement makes the new cursor locally visible immediately. The
/// exact GpuVmAccess snapshot supplied at retirement is retained until the
/// corresponding atomic store completes, so retries cannot publish through a
/// replacement address-space root.
class ConsumerCursorJournal {
public:
  explicit ConsumerCursorJournal(uint64_t address,
                                 std::optional<uint64_t> initial_cursor = std::nullopt,
                                 uint32_t width = sizeof(uint64_t));

  [[nodiscard]] VmAccessOutcome initialize(const GpuVmAccess &access);
  /// @pre No earlier retirement is awaiting publication.
  void retire(uint64_t cursor, GpuVmAccess access);
  [[nodiscard]] VmAccessOutcome publish();
  void reset(std::optional<uint64_t> initial_cursor = std::nullopt);

  [[nodiscard]] uint64_t cursor() const { return cursor_; }
  [[nodiscard]] bool initialized() const { return initialized_; }
  [[nodiscard]] bool publication_pending() const { return publication_access_.has_value(); }
  [[nodiscard]] uint64_t address() const { return address_; }
  [[nodiscard]] uint32_t width() const { return width_; }

private:
  uint64_t address_ = 0;
  uint32_t width_ = sizeof(uint64_t);
  uint64_t cursor_ = 0;
  bool initialized_ = false;
  std::optional<GpuVmAccess> publication_access_;
};

} // namespace rocjitsu::amdgpu
