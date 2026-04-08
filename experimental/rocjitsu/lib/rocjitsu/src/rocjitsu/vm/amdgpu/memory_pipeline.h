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
#include <memory>
#include <queue>

namespace rocjitsu {
namespace amdgpu {

class L1ScalarCache;
class L1VectorCache;
class L2Cache;
class Lds;

/// @brief A queued entry in a memory pipeline, owning the instruction.
struct PipelineEntry {
  std::unique_ptr<Instruction> inst; ///< Owns the in-flight memory instruction.
  Wavefront *wf;                     ///< The issuing wavefront.
};

/// @brief Abstract base for memory pipelines that track in-flight operations.
///
/// @details Three concrete subclasses are used per Compute Unit: ScalarMemPipeline
/// (SMEM), GlobalMemPipeline (FLAT/MUBUF/MTBUF + atomics), and LocalMemPipeline
/// (DS). Each subclass knows which DynamicInstState subtype to cast to and which
/// cache controller method to call.
///
/// The pipeline has two internal queues:
/// - issued_: instructions that need initiate_access() called
/// - returned_: instructions whose memory response has arrived, awaiting
///   complete_access() and counter decrement
///
/// In the current synchronous implementation, initiate_access() gets an
/// immediate response, so instructions move from issued_ to returned_
/// within the same tick().
class MemoryPipeline {
public:
  /// @brief Construct a pipeline that tracks the given counter type.
  /// @param counter_type The wait counter type this pipeline manages.
  MemoryPipeline(WaitCounterType counter_type) : counter_type_(counter_type) {}
  virtual ~MemoryPipeline() = default;

  /// @brief Issue a memory instruction into the pipeline.
  ///
  /// Increments the wavefront's wait counter and enqueues the instruction.
  /// Caller transfers ownership of the instruction.
  /// @param inst The memory instruction (ownership transferred).
  /// @param wf The issuing wavefront.
  void issue(std::unique_ptr<Instruction> inst, Wavefront &wf) {
    wf.wait_counters().increment(counter_type_);
    issued_.push({std::move(inst), &wf});
  }

  /// @brief Advance the pipeline by one cycle.
  ///
  /// Drains ALL returned instructions (complete_access + counter decrement),
  /// then initiates ALL pending issued instructions. This eliminates the
  /// artificial 2-tick minimum latency of the previous one-at-a-time approach.
  void tick() {
    while (!returned_.empty()) {
      auto &entry = returned_.front();
      complete_access(*entry.inst, *entry.wf);
      entry.wf->wait_counters().decrement(counter_type_);
      if (entry.wf->state() == WfState::WAITCNT && entry.wf->wait_satisfied())
        entry.wf->set_state(WfState::RUNNING);
      returned_.pop();
    }

    while (!issued_.empty()) {
      auto entry = std::move(issued_.front());
      issued_.pop();
      initiate_access(*entry.inst, *entry.wf);
      returned_.push(std::move(entry));
    }
  }

  /// @brief Check whether the pipeline has any in-flight operations.
  /// @retval true No instructions are in-flight.
  /// @retval false One or more instructions are still being processed.
  bool empty() const { return issued_.empty() && returned_.empty(); }

  /// @brief Return the counter type this pipeline manages.
  /// @returns The wait counter type.
  WaitCounterType counter_type() const { return counter_type_; }

protected:
  /// @brief Send request to cache controller. Subclass casts inst.data()
  /// to the appropriate DynamicInstState subtype.
  /// @param inst The memory instruction to initiate.
  /// @param wf The issuing wavefront.
  virtual void initiate_access(Instruction &inst, Wavefront &wf) = 0;

  /// @brief Write results back to registers after memory response.
  /// @param inst The completed memory instruction.
  /// @param wf The issuing wavefront.
  virtual void complete_access(Instruction &inst, Wavefront &wf) = 0;

private:
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

/// @brief Global memory pipeline for FLAT/MUBUF/MTBUF instructions.
///
/// Routes loads/stores through L1 Vector Cache (V$). Atomic RMW operations
/// bypass L1 and execute at L2 directly, matching real hardware behavior.
class GlobalMemPipeline : public MemoryPipeline {
public:
  /// @param l1 L1 Vector Cache (V$), not owned.
  /// @param l2 L2 cache (for atomic RMW), not owned.
  GlobalMemPipeline(L1VectorCache *l1, L2Cache *l2)
      : MemoryPipeline(WaitCounterType::VMCNT), l1_(l1), l2_(l2) {}

protected:
  void initiate_access(Instruction &inst, Wavefront &wf) override;
  void complete_access(Instruction &inst, Wavefront &wf) override;

private:
  L1VectorCache *l1_;
  L2Cache *l2_;
};

/// @brief Local memory pipeline for DS instructions.
///
/// Routes directly to the Local Data Share (LDS).
class LocalMemPipeline : public MemoryPipeline {
public:
  /// @param lds Local Data Share, not owned.
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
