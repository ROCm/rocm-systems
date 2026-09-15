// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file memory_pipeline.h
/// @brief Memory pipelines for scalar, global, and local memory operations.

#ifndef ROCJITSU_VM_AMDGPU_MEMORY_PIPELINE_H_
#define ROCJITSU_VM_AMDGPU_MEMORY_PIPELINE_H_

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <queue>
#include <utility>

namespace rocjitsu {
namespace amdgpu {

class L1ScalarCache;
class L1VectorCache;
class L2Cache;
class Lds;

enum class [[nodiscard]] MemoryAccessCompletion {
  Complete,
  Deferred,
};

using MemoryAccessDeferredCompletion = std::function<void()>;

/// @brief Base class for a memory pipeline stage (scalar, global, or local).
///
/// @details Models the memory access pipeline as two FIFO queues:
/// - issued_: instructions that need initiate_access() called
/// - returned_: instructions whose memory response has arrived, awaiting
///   register writeback via complete_access().
///
/// In functional mode, L1/L2/HBM accesses are synchronous and produce an
/// immediate response, so instructions move from issued_ to returned_
/// in a single tick() and then complete on the next tick().
class MemoryPipeline {
public:
  explicit MemoryPipeline(WaitCounterType type) : counter_type_(type) {}
  virtual ~MemoryPipeline() = default;

  struct PipelineEntry {
    Instruction *inst;
    Wavefront *wf;
  };

  /// @brief Issue a memory instruction to this pipeline.
  ///
  /// In functional mode, memory accesses normally complete synchronously:
  /// the load or store is initiated and completed within this call, and the
  /// wait-counter obligations are released only after complete_access()
  /// finishes all writeback work. A timing backend may return Deferred and
  /// release the counters later through finish_completed_access().
  void issue(Instruction *inst, Wavefront &wf) {
    std::array<WaitCounterType, MemoryIssueInfo::MAX_COUNTER_OBLIGATIONS> issue_counters{};
    uint8_t num_issue_counters = 0;
    const auto *issue = inst->amdgpu_memory_issue_info();
    if (issue) {
      for (const auto obligation : issue->counter_obligations())
        issue_counters[num_issue_counters++] = obligation.wait_counter_type();
    }
    if (!issue) {
      WaitCounterType issue_counter = counter_type_;
      auto *state = inst->data();
      assert(state && "memory instruction has no dynamic state");
      switch (state->tag()) {
      case SCALAR_MEM:
        issue_counter = inst->data_as<ScalarMemState>()->wait_counter_type;
        break;
      case GLOBAL_MEM:
      case LOCAL_MEM:
        issue_counter = inst->data_as<VectorMemState>()->wait_counter_type;
        break;
      default:
        break;
      }
      issue_counters[num_issue_counters++] = issue_counter;
    }
    for (uint8_t i = 0; i < num_issue_counters; ++i)
      wf.wait_counters().increment(issue_counters[i]);
    initiate_access(*inst, wf);
    // The wait counters pin wf/inst ownership until this callback releases them.
    // ComputeUnitCore retires ENDING wavefronts only after wait_counters().empty(),
    // so a deferred backend must invoke this exactly once while that counter is held.
    MemoryAccessDeferredCompletion deferred_completion = [this, inst, &wf, issue_counters,
                                                          num_issue_counters]() {
      finish_completed_access(inst, wf, issue_counters, num_issue_counters);
    };
    MemoryAccessCompletion completion = complete_access(*inst, wf, std::move(deferred_completion));
    if (completion == MemoryAccessCompletion::Complete)
      finish_completed_access(inst, wf, issue_counters, num_issue_counters);
  }

  /// @brief Advance the pipeline by one cycle (no-op in functional mode).
  void tick() {}

  bool empty() const { return true; }

  WaitCounterType counter_type() const { return counter_type_; }

protected:
  virtual void initiate_access(Instruction &inst, Wavefront &wf) = 0;
  /// Return Complete and do not call complete, or return Deferred and call it
  /// exactly once after the memory response is ready for architectural writeback.
  virtual MemoryAccessCompletion complete_access(Instruction &inst, Wavefront &wf,
                                                 MemoryAccessDeferredCompletion complete) = 0;

  void finish_completed_access(
      Instruction *inst, Wavefront &wf,
      const std::array<WaitCounterType, MemoryIssueInfo::MAX_COUNTER_OBLIGATIONS> &counters,
      uint8_t num_counters) {
    for (uint8_t i = 0; i < num_counters; ++i)
      wf.release_wait_counter(counters[i]);
    delete inst;
  }

  WaitCounterType counter_type_;
  std::queue<PipelineEntry> issued_;
  std::queue<PipelineEntry> returned_;
};

/// @brief Scalar memory pipeline for SMEM instructions.
///
/// Routes all scalar loads and stores through the L1 Scalar Cache (K$).
/// Dirty lines are written back to L2 on eviction or via s_dcache_wb.
class ScalarMemPipeline : public MemoryPipeline {
public:
  /// @param l1 L1 Scalar Cache (K$), not owned.
  explicit ScalarMemPipeline(L1ScalarCache *l1)
      : MemoryPipeline(WaitCounterType::LGKMCNT), l1_(l1) {}

protected:
  void initiate_access(Instruction &inst, Wavefront &wf) override;
  MemoryAccessCompletion complete_access(Instruction &inst, Wavefront &wf,
                                         MemoryAccessDeferredCompletion complete) override;

private:
  L1ScalarCache *l1_;
};

/// @brief Global memory pipeline (V$ → L2 → HBM).
class GlobalMemPipeline : public MemoryPipeline {
public:
  GlobalMemPipeline(L1VectorCache *l1, L2Cache *l2)
      : MemoryPipeline(WaitCounterType::VMCNT), l1_(l1), l2_(l2) {}

  void set_l2(L2Cache *l2) { l2_ = l2; }

protected:
  void initiate_access(Instruction &inst, Wavefront &wf) override;
  MemoryAccessCompletion complete_access(Instruction &inst, Wavefront &wf,
                                         MemoryAccessDeferredCompletion complete) override;

private:
  L1VectorCache *l1_;
  L2Cache *l2_;
};

/// @brief Local memory pipeline (LDS).
class LocalMemPipeline : public MemoryPipeline {
public:
  LocalMemPipeline() : MemoryPipeline(WaitCounterType::LGKMCNT) {}

protected:
  void initiate_access(Instruction &inst, Wavefront &wf) override;
  MemoryAccessCompletion complete_access(Instruction &inst, Wavefront &wf,
                                         MemoryAccessDeferredCompletion complete) override;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_MEMORY_PIPELINE_H_
