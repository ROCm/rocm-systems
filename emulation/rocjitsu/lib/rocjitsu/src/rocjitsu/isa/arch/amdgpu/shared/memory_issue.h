// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MEMORY_ISSUE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MEMORY_ISSUE_H_

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rocjitsu::amdgpu {

/// RDNA and gfx1250 allocate a vector-memory and an LDS counter position
/// for every generic FLAT, even when all requesting lanes resolve to one space.
/// CDNA1–4 use address-domain participation as an unverified model. CDNA3/4
/// ISA section 10.2 describes joint VM and LGKM completion.
[[nodiscard]] constexpr bool flat_issues_both_counter_domains(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
}

/// With the GFX10/GFX11 wait counters, each generic FLAT result needs both
/// domains. Split-counter targets can establish completion per lane and space.
[[nodiscard]] constexpr bool flat_requires_both_counters(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5;
}

/// Active generic FLAT issues both VM/VS and LGKM on legacy targets.
[[nodiscard]] constexpr bool legacy_flat_issues_both_counter_domains(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
         arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
}

/// The empty-EXEC rule has separate target evidence from active FLAT requests.
[[nodiscard]] constexpr bool legacy_flat_issues_with_empty_exec(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
}

/// GFX11/12 charge both domains for an issued FLAT. GFX11 results need both
/// counter acknowledgments; split-counter GFX12 results follow their route.
[[nodiscard]] constexpr bool flat_issues_both_routed_domains(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
}

/// A zero-EXEC FLAT can be skipped only while the target-specific physical
/// issue queues are idle. GFX11 checks VM/VS; RDNA4 checks VM/VS, DS, and
/// SAMPLE; CDNA5 checks VM/VS, DS, ASYNC, and TENSOR. The last argument is
/// SAMPLE on RDNA4 or ASYNC/TENSOR on CDNA5. A modeled BVH result alone does
/// not establish physical queue occupancy in the simulator.
[[nodiscard]] constexpr bool flat_issues_with_empty_exec(rj_code_arch_t arch, bool vmem_pending,
                                                         bool ds_pending,
                                                         bool other_issue_queue_pending) {
  if (legacy_flat_issues_with_empty_exec(arch))
    return true;
  if (!flat_issues_both_routed_domains(arch))
    return false;
  if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5)
    return vmem_pending;
  return vmem_pending || ds_pending || other_issue_queue_pending;
}

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
  /// Tensor DMA transfers, which complete on TENSORCNT in issue order. Tensor
  /// DMA instructions carry no MemoryIssueInfo, so no instruction metadata uses
  /// this class yet.
  TENSOR,
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

  static_assert(static_cast<uint8_t>(WaitCounterType::BVHCNT) < 16);
  static_assert(static_cast<uint8_t>(MemoryCompletionClass::UNORDERED) < 8);
  uint8_t encoded_ = 0;
};

/// @brief Typed description of an AMDGPU instruction's memory-issue semantics.
/// @details Most instructions contribute to one wait-counter domain. Generic
/// FLAT can contribute to both VMEM and DS domains; VMEM stores and atomics do
/// not contribute an EXP obligation on supported targets.
/// Each obligation carries its own completion-order class because operations
/// sharing one counter need not form one FIFO, and one instruction's different
/// counter domains can have different ordering guarantees. Decoded FLAT metadata
/// lists possible domains. Address routing and the target determine the issued credits.
/// One race-detector event requires every attached obligation for completion,
/// including destination-register and memory dependencies. A mixed-route FLAT
/// splits into per-route events via for_flat_memory_domains(), so a wait can
/// release only the lanes on its completed route. The gfx1250 race detector
/// also records counter slots for unused image domains. This API does not
/// represent a separate source-register lifetime. The same obligations drive
/// emulated counter increments and are exposed to execution plugins.
/// exec_masked distinguishes ordinary vector memory operations from scalar
/// memory and the few vector operations that execute independently of EXEC.
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

  /// Select physical issue credits independently of the return route of each lane.
  [[nodiscard]] MemoryIssueInfo for_flat_issue(rj_code_arch_t arch, uint64_t requests,
                                               uint64_t shared, bool empty_exec,
                                               bool issue_empty = false,
                                               bool returned_without_request = false) const {
    if ((legacy_flat_issues_both_counter_domains(arch) &&
         (requests || returned_without_request)) ||
        (legacy_flat_issues_with_empty_exec(arch) && empty_exec) ||
        (flat_issues_both_routed_domains(arch) && (requests || (empty_exec && issue_empty))))
      return *this;
    return for_flat_memory_domains((requests & ~shared) != 0, (requests & shared) != 0);
  }

  /// Select FLAT's VMEM and/or DS obligations for the resolved request domains.
  /// Preserve each selected obligation's counter increment and completion class,
  /// exec_masked, and any obligations outside those domains. This only filters
  /// completion requirements; it does not track a separate source-register lifetime.
  [[nodiscard]] MemoryIssueInfo for_flat_memory_domains(bool vmem, bool lds) const {
    MemoryIssueInfo result;
    result.exec_masked = exec_masked;
    result.counts_with_empty_exec = counts_with_empty_exec;
    for (const auto obligation : counter_obligations()) {
      switch (obligation.wait_counter_type()) {
      case WaitCounterType::VMCNT:
      case WaitCounterType::VSCNT:
      case WaitCounterType::LOADCNT:
      case WaitCounterType::STORECNT:
        if (!vmem)
          continue;
        break;
      case WaitCounterType::LGKMCNT:
      case WaitCounterType::DSCNT:
        if (!lds)
          continue;
        break;
      default:
        break;
      }
      result.counter_obligations_[result.num_counter_obligations_++] = obligation;
    }
    return result;
  }

  std::array<MemoryCounterObligation, MAX_COUNTER_OBLIGATIONS> counter_obligations_{};
  uint8_t num_counter_obligations_ = 0;
  bool exec_masked = true;
  /// Count this instruction even when its EXEC-controlled data lanes are empty.
  bool counts_with_empty_exec = false;
};

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MEMORY_ISSUE_H_
