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

namespace rocjitsu {
namespace amdgpu {

class L1ScalarCache;
class L1VectorCache;
class L2Cache;
class Lds;

/// @brief Base class for a memory pipeline stage (scalar, global, or local).
///
/// @details Functional memory accesses complete synchronously during issue:
/// initiate the access, perform architectural writeback, then release ownership.
class MemoryPipeline {
public:
  explicit MemoryPipeline(WaitCounterType type) : counter_type_(type) {}

protected:
  /// @brief Issue a memory instruction through its concrete pipeline.
  ///
  /// Memory accesses complete synchronously: initiate the access, perform
  /// architectural writeback, then release the wait counter and instruction.
  template <typename Pipeline>
  void issue_impl(Pipeline &pipeline, Instruction *inst, Wavefront &wf) {
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
    pipeline.initiate_access(*inst, wf);
    pipeline.complete_access(*inst, wf);
    finish_completed_access(inst, wf, issue_counter);
  }

public:
  WaitCounterType counter_type() const { return counter_type_; }

protected:
  void finish_completed_access(Instruction *inst, Wavefront &wf, WaitCounterType counter) {
    wf.release_wait_counter(counter);
    delete inst;
  }

  WaitCounterType counter_type_;
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

  void issue(Instruction *inst, Wavefront &wf) { issue_impl(*this, inst, wf); }

protected:
  friend class MemoryPipeline;

  void initiate_access(Instruction &inst, Wavefront &wf);
  void complete_access(Instruction &inst, Wavefront &wf);

private:
  L1ScalarCache *l1_;
};

/// @brief Global memory pipeline (V$ → L2 → HBM).
class GlobalMemPipeline : public MemoryPipeline {
public:
  GlobalMemPipeline(L1VectorCache *l1, L2Cache *l2)
      : MemoryPipeline(WaitCounterType::VMCNT), l1_(l1), l2_(l2) {}

  void set_l2(L2Cache *l2) { l2_ = l2; }

  void issue(Instruction *inst, Wavefront &wf) { issue_impl(*this, inst, wf); }

protected:
  friend class MemoryPipeline;

  void initiate_access(Instruction &inst, Wavefront &wf);
  void complete_access(Instruction &inst, Wavefront &wf);

private:
  L1VectorCache *l1_;
  L2Cache *l2_;
};

/// @brief Local memory pipeline (LDS).
class LocalMemPipeline : public MemoryPipeline {
public:
  LocalMemPipeline() : MemoryPipeline(WaitCounterType::LGKMCNT) {}

  void issue(Instruction *inst, Wavefront &wf) { issue_impl(*this, inst, wf); }

protected:
  friend class MemoryPipeline;

  void initiate_access(Instruction &inst, Wavefront &wf);
  void complete_access(Instruction &inst, Wavefront &wf);
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_MEMORY_PIPELINE_H_
