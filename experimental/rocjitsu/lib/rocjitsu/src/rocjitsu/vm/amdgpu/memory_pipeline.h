// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file memory_pipeline.h
/// @brief Memory pipelines for scalar, global, and local memory operations.

#ifndef ROCJITSU_VM_AMDGPU_MEMORY_PIPELINE_H_
#define ROCJITSU_VM_AMDGPU_MEMORY_PIPELINE_H_

#include "rocjitsu/isa/instruction.h"
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
  /// @param inst Raw instruction pointer (caller transfers ownership).
  /// @param wf   The issuing wavefront.
  void issue(Instruction *inst, Wavefront &wf) {
    wf.wait_counters().increment(counter_type_);
    issued_.push({inst, &wf});
  }

  /// @brief Advance the pipeline by one cycle.
  void tick() {
    // Process returned instructions: decrement counters and complete.
    while (!returned_.empty()) {
      auto entry = returned_.front();
      returned_.pop();
      entry.wf->wait_counters().decrement(counter_type_);
      if (entry.wf->state() == WfState::WAITCNT && entry.wf->wait_satisfied())
        entry.wf->set_state(WfState::RUNNING);
      // If the wavefront is in ENDING state (s_endpgm executed but draining),
      // halt it once all outstanding memory ops are complete.
      if (entry.wf->state() == WfState::ENDING && entry.wf->wait_counters().empty())
        entry.wf->halt();
      complete_access(*entry.inst, *entry.wf);
      delete entry.inst;
    }

    // Initiate access for newly issued instructions.
    while (!issued_.empty()) {
      auto entry = issued_.front();
      issued_.pop();
      initiate_access(*entry.inst, *entry.wf);
      returned_.push(entry);
    }
  }

  bool empty() const { return issued_.empty() && returned_.empty(); }

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
