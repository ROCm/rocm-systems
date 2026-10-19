// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file memory_pipeline.h
/// @brief Memory pipelines for scalar, global, and local memory operations.

#pragma once

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <span>
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
  using IngressHandler = std::function<MemoryIngressAdmission(uint32_t, const GpuVmAccess &,
                                                              std::span<const MemoryIngressSpan>)>;
  /// Optional cache-only translated System read. A null completion probes
  /// resident bytes; a completion carries returned bytes' stamp and origin.
  /// The handler must independently prove current alias, translation, and
  /// writer authority before returning a hit or installing a line.
  using PeerCacheReadHandler = std::function<bool(uint32_t, const GpuVmAccess &, uint64_t,
                                                  std::span<std::byte>, const VmCompletedRead *)>;
  static constexpr std::size_t MAX_WAIT_COUNTER_TOKENS =
      MemoryIssueInfo::MAX_COUNTER_OBLIGATIONS * MemoryCounterObligation::MAX_COUNTER_INCREMENT;

  struct WaitCounterTokens {
    std::array<WaitCounterType, MAX_WAIT_COUNTER_TOKENS> types{};
    uint8_t size = 0;
  };

  explicit MemoryPipeline(WaitCounterType type) : counter_type_(type) {}
  virtual ~MemoryPipeline();

  struct PipelineEntry {
    Instruction *inst;
    Wavefront *wf;
    WaitCounterTokens counters;
    uint64_t wave_generation;
  };

  /// @brief Issue a memory instruction to this pipeline.
  ///
  /// In functional mode, memory accesses normally complete synchronously:
  /// the load or store is initiated and completed within this call, and the
  /// wait-counter obligations are released only after complete_access()
  /// finishes all writeback work. A timing backend may return Deferred and
  /// release the counters later through finish_completed_access().
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
  void set_ingress_handler(IngressHandler handler) { ingress_handler_ = std::move(handler); }
  void set_peer_cache_read_handler(PeerCacheReadHandler handler) {
    peer_cache_read_handler_ = std::move(handler);
  }

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
  [[nodiscard]] WaitCounterTokens issue_counters(const Instruction &inst) const;
  void acquire_wait_counters(Wavefront &wf, const WaitCounterTokens &counters);
  void release_wait_counters(Wavefront &wf, const WaitCounterTokens &counters);
  void complete_entry(PipelineEntry entry);

  void finish_completed_access(Instruction *inst, Wavefront &wf, const WaitCounterTokens &counters,
                               uint64_t wave_generation) {
    if (wf.dispatch_generation() != wave_generation) {
      delete inst;
      return;
    }
    release_wait_counters(wf, counters);
    if (wf.state() == WfState::VM_RETRY)
      wf.set_state(WfState::RUNNING);
    delete inst;
  }

  WaitCounterType counter_type_;
  std::queue<PipelineEntry> issued_;
  std::queue<PipelineEntry> returned_;
  FaultHandler fault_handler_;
  IngressHandler ingress_handler_;
  PeerCacheReadHandler peer_cache_read_handler_;

protected:
  [[nodiscard]] MemoryIngressAdmission
  admit_ingress(uint32_t process_id, const GpuVmAccess &access,
                std::span<const MemoryIngressSpan> spans) const {
    return ingress_handler_ ? ingress_handler_(process_id, access, spans)
                            : MemoryIngressAdmission{};
  }
  [[nodiscard]] bool has_ingress_handler() const { return static_cast<bool>(ingress_handler_); }
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
  using PreparedAtomicBoundaryHandler =
      std::function<std::optional<DeviceCacheMaintenanceLease>(uint32_t vmid)>;

  GlobalMemPipeline(L1VectorCache *l1, L2Cache *l2)
      : MemoryPipeline(WaitCounterType::VMCNT), l1_(l1), l2_(l2) {}

  void set_l2(L2Cache *l2) { l2_ = l2; }
  /// @brief Use KFD's owning-GPU prepared boundary for local vector atomics.
  /// @details A retired frontend may decline VMID-zero work so the standalone
  /// boundary remains usable after KFD teardown.
  void set_prepared_atomic_boundary_handler(PreparedAtomicBoundaryHandler handler) {
    prepared_atomic_boundary_handler_ = std::move(handler);
  }

protected:
  VmAccessOutcome initiate_access(Instruction &inst, Wavefront &wf) override;
  MemoryAccessCompletion complete_access(Instruction &inst, Wavefront &wf,
                                         MemoryAccessDeferredCompletion complete) override;

private:
  VmAccessOutcome initiate_global_access(VectorMemState &state, Wavefront &wf,
                                         uint64_t request_lanes);
  L1VectorCache *l1_;
  L2Cache *l2_;
  PreparedAtomicBoundaryHandler prepared_atomic_boundary_handler_;
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
