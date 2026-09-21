// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MEMORY_ISSUE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MEMORY_ISSUE_H_

#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rocjitsu::amdgpu {

/// @brief Completion ordering associated with an AMDGPU memory issue.
/// @details Counter membership and completion ordering are separate: operations
/// can share a wait counter without completing in one usable FIFO order.
enum class MemoryCompletionClass : uint8_t {
  UNCLASSIFIED,
  VMEM,
  LDS,
  GDS,
  ASYNC_LOAD,
  ASYNC_STORE,
  UNORDERED,
};

/// @brief One counter increment and the FIFO class used to prove its progress.
/// @details This is byte-packed so three obligations plus EXEC policy fit in
/// Instruction's existing padding. Consumers should use the accessors rather
/// than depend on the encoding.
class MemoryCounterObligation {
public:
  static constexpr uint8_t MAX_COUNTER_INCREMENT = 2;

  constexpr MemoryCounterObligation() = default;
  constexpr MemoryCounterObligation(WaitCounterType wait_counter_type,
                                    MemoryCompletionClass completion_class,
                                    uint8_t counter_increment = 1)
      : encoded_(encode(wait_counter_type, completion_class, counter_increment)) {}

  [[nodiscard]] constexpr bool valid() const {
    return completion_class() != MemoryCompletionClass::UNCLASSIFIED;
  }
  [[nodiscard]] constexpr WaitCounterType wait_counter_type() const {
    return static_cast<WaitCounterType>(encoded_ & 0x0f);
  }
  [[nodiscard]] constexpr MemoryCompletionClass completion_class() const {
    return static_cast<MemoryCompletionClass>((encoded_ >> 4) & 0x07);
  }
  [[nodiscard]] constexpr uint8_t counter_increment() const { return 1 + (encoded_ >> 7); }

private:
  static constexpr uint8_t encode(WaitCounterType wait_counter_type,
                                  MemoryCompletionClass completion_class,
                                  uint8_t counter_increment) {
    assert(counter_increment >= 1 && counter_increment <= MAX_COUNTER_INCREMENT);
    assert(completion_class != MemoryCompletionClass::UNCLASSIFIED);
    assert(static_cast<uint8_t>(wait_counter_type) < 16);
    return static_cast<uint8_t>(((counter_increment - 1) << 7) |
                                (static_cast<uint8_t>(completion_class) << 4) |
                                static_cast<uint8_t>(wait_counter_type));
  }

  static_assert(static_cast<uint8_t>(WaitCounterType::ASYNCCNT) < 16);
  static_assert(static_cast<uint8_t>(MemoryCompletionClass::UNORDERED) < 8);
  uint8_t encoded_ = 0;
};

/// @brief Typed description of an AMDGPU instruction's memory-issue semantics.
/// @details Most instructions contribute to one wait-counter domain. Generic
/// FLAT and pre-GFX12 stores can contribute to multiple simultaneous domains.
/// Each obligation carries its own completion-order class because operations
/// sharing one counter need not form one FIFO, and one instruction's different
/// counter domains can have different ordering guarantees. The memory route
/// does not change these obligations. exec_masked distinguishes ordinary
/// vector memory operations from scalar memory and the few vector operations
/// that execute independently of EXEC.
///
/// This descriptor covers instructions modeled through rocJITsu's scalar,
/// vector, and local memory pipelines. It is not a complete inventory of every
/// hardware event counted by wait instructions. Counter-producing operations
/// outside those pipelines, such as messages and timestamp queries, require
/// separate accounting by consumers that model total counter occupancy.
struct MemoryIssueInfo {
  static constexpr size_t MAX_COUNTER_OBLIGATIONS = 3;

  [[nodiscard]] std::span<const MemoryCounterObligation> counter_obligations() const {
    return {counter_obligations_.data(), num_counter_obligations_};
  }
  [[nodiscard]] bool empty() const { return num_counter_obligations_ == 0; }

  std::array<MemoryCounterObligation, MAX_COUNTER_OBLIGATIONS> counter_obligations_{};
  uint8_t num_counter_obligations_ = 0;
  bool exec_masked = true;
};

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MEMORY_ISSUE_H_
