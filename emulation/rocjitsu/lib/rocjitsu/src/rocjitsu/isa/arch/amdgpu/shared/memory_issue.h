// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MEMORY_ISSUE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MEMORY_ISSUE_H_

#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter.h"

#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

/// @brief Completion ordering associated with an AMDGPU memory issue.
/// @details Counter membership and completion ordering are separate: operations
/// can share a wait counter without completing in one usable FIFO order.
enum class MemoryCompletionClass : uint8_t {
  UNCLASSIFIED,
  VMEM,
  LDS,
  UNORDERED,
};

/// @brief Typed description of an AMDGPU instruction's memory-issue semantics.
/// @details Most instructions contribute to one wait-counter domain. Generic
/// FLAT instructions contribute to two simultaneous domains because hardware
/// issues complementary vector-memory and LDS portions. The memory route does
/// not change these obligations. exec_masked distinguishes ordinary vector
/// memory operations from scalar memory and the few vector operations that
/// execute independently of EXEC.
///
/// This descriptor covers instructions modeled through rocJITsu's scalar,
/// vector, and local memory pipelines. It is not a complete inventory of every
/// hardware event counted by wait instructions. Counter-producing operations
/// outside those pipelines, such as messages and timestamp queries, require
/// separate accounting by consumers that model total counter occupancy.
struct MemoryIssueInfo {
  WaitCounterType wait_counter_type = WaitCounterType::VMCNT;
  MemoryCompletionClass completion_class = MemoryCompletionClass::UNCLASSIFIED;
  std::optional<WaitCounterType> additional_wait_counter_type;
  bool exec_masked = true;
};

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MEMORY_ISSUE_H_
