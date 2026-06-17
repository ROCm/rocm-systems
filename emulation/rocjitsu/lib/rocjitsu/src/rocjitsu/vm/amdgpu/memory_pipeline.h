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

#include <cstdint>
#include <queue>

namespace rocjitsu {
namespace amdgpu {

class L1ScalarCache;
class L1VectorCache;
class L2Cache;
class Lds;

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
  /// In functional mode, memory accesses complete synchronously: the load
  /// or store is initiated and completed within this call, and the wait
  /// counter is incremented then immediately decremented so that a
  /// subsequent s_waitcnt sees the operation as already done.  This
  /// eliminates the two-tick deferred pipeline that previously allowed
  /// ALU instructions to read destination VGPRs between issue and
  /// writeback — a hazard that real hardware prevents via scoreboarding.
  void issue(Instruction *inst, Wavefront &wf) {
    WaitCounterType issue_counter = counter_type_;
    if (auto *state = inst->data()) {
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
    }
    wf.wait_counters().increment(issue_counter);
    initiate_access(*inst, wf);
    complete_access(*inst, wf);
    wf.wait_counters().decrement(issue_counter);
    if (wf.state() == WfState::WAITCNT && wf.wait_satisfied())
      wf.set_state(WfState::RUNNING);
    if (wf.state() == WfState::ENDING && wf.wait_counters().empty())
      wf.halt();
    delete inst;
  }

  /// @brief Advance the pipeline by one cycle (no-op in functional mode).
  void tick() {}

  bool empty() const { return true; }

  WaitCounterType counter_type() const { return counter_type_; }

protected:
  virtual void initiate_access(Instruction &inst, Wavefront &wf) = 0;
  virtual void complete_access(Instruction &inst, Wavefront &wf) = 0;

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
  void complete_access(Instruction &inst, Wavefront &wf) override;

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
  void complete_access(Instruction &inst, Wavefront &wf) override;

private:
  L1VectorCache *l1_;
  L2Cache *l2_;
};

/// @brief Local memory pipeline (LDS).
class LocalMemPipeline : public MemoryPipeline {
public:
  explicit LocalMemPipeline(Lds *lds) : MemoryPipeline(WaitCounterType::LGKMCNT), lds_(lds) {}

protected:
  void initiate_access(Instruction &inst, Wavefront &wf) override;
  void complete_access(Instruction &inst, Wavefront &wf) override;

private:
  Lds *lds_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_MEMORY_PIPELINE_H_
