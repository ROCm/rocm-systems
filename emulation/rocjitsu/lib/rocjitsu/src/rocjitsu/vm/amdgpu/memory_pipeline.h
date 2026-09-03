// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file memory_pipeline.h
/// @brief Memory pipelines for scalar, global, and local memory operations.

#ifndef ROCJITSU_VM_AMDGPU_MEMORY_PIPELINE_H_
#define ROCJITSU_VM_AMDGPU_MEMORY_PIPELINE_H_

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

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
  using FaultHandler = std::function<void(Wavefront &, VmAccessOutcome)>;

  explicit MemoryPipeline(WaitCounterType type) : counter_type_(type) {}
  virtual ~MemoryPipeline();

  struct PipelineEntry {
    Instruction *inst;
    Wavefront *wf;
    WaitCounterType counter;
    uint64_t wave_generation;
  };

  /// @brief Issue a memory instruction to this pipeline.
  ///
  /// In functional mode, memory accesses normally complete synchronously:
  /// the load or store is initiated and completed within this call, and the
  /// wait counter is released only after complete_access() finishes all
  /// writeback work.  A timing backend may return Deferred and release the
  /// counter later through finish_completed_access().
  VmAccessOutcome issue(Instruction *inst, Wavefront &wf);

  /// @brief Issue an instruction and retain it for retry on Unavailable.
  /// @details Returning Complete means the pipeline accepted ownership, whether
  /// the backing completed synchronously or the request was queued. Permanent
  /// failures are returned synchronously and leave no retained instruction.
  VmAccessOutcome issue_deferred(Instruction *inst, Wavefront &wf);

  /// @brief Retry every request that was unavailable on its preceding attempt.
  void tick();

  bool empty() const { return issued_.empty() && returned_.empty(); }

  WaitCounterType counter_type() const { return counter_type_; }
  void set_fault_handler(FaultHandler handler) { fault_handler_ = std::move(handler); }

  /// @brief Retain an operation whose first access attempt already returned Unavailable.
  void defer_unavailable(Instruction *inst, Wavefront &wf);

  /// @brief Cancel every retained operation belonging to one wave-slot occupant.
  void cancel(Wavefront &wf);

protected:
  virtual VmAccessOutcome initiate_access(Instruction &inst, Wavefront &wf) = 0;
  /// Return Complete and do not call complete, or return Deferred and call it
  /// exactly once after the memory response is ready for architectural writeback.
  virtual MemoryAccessCompletion complete_access(Instruction &inst, Wavefront &wf,
                                                 MemoryAccessDeferredCompletion complete) = 0;

  VmAccessOutcome issue_impl(Instruction *inst, Wavefront &wf, bool retain_unavailable);
  [[nodiscard]] WaitCounterType issue_counter(const Instruction &inst) const;
  void complete_entry(PipelineEntry entry);

  void finish_completed_access(Instruction *inst, Wavefront &wf, WaitCounterType counter,
                               uint64_t wave_generation) {
    if (wf.dispatch_generation() != wave_generation) {
      delete inst;
      return;
    }
    wf.release_wait_counter(counter);
    if (wf.state() == WfState::VM_RETRY)
      wf.set_state(WfState::RUNNING);
    delete inst;
  }

  WaitCounterType counter_type_;
  std::queue<PipelineEntry> issued_;
  std::queue<PipelineEntry> returned_;
  FaultHandler fault_handler_;
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
  VmAccessOutcome initiate_access(Instruction &inst, Wavefront &wf) override;
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
  VmAccessOutcome initiate_access(Instruction &inst, Wavefront &wf) override;
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
  VmAccessOutcome initiate_access(Instruction &inst, Wavefront &wf) override;
  MemoryAccessCompletion complete_access(Instruction &inst, Wavefront &wf,
                                         MemoryAccessDeferredCompletion complete) override;
};

/// @brief Retry queue for CDNA5 tensor DMA operations prepared by ISA execution.
class TensorDmaPipeline : public MemoryPipeline {
public:
  TensorDmaPipeline() : MemoryPipeline(WaitCounterType::TENSORCNT) {}

protected:
  VmAccessOutcome initiate_access(Instruction &inst, Wavefront &wf) override;
  MemoryAccessCompletion complete_access(Instruction &inst, Wavefront &wf,
                                         MemoryAccessDeferredCompletion complete) override;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_MEMORY_PIPELINE_H_
