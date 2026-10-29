// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file memory_pipeline.h
/// @brief Memory pipelines for scalar, global, and local memory operations.

#pragma once

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

class L1ScalarCache;
class L1VectorCache;
class L2Cache;
class Lds;

/// @brief Lanes whose atomic is not aligned to its access width.
/// @details Non-atomics, byte atomics and architectures other than RDNA3/3.5/4
/// and CDNA5 have none. Two-address DS atomics check both addresses.
/// SH_MEM_CONFIG.alignment_mode is not modelled: the emulator implements
/// UNALIGNED, the mode KFD programs, in which non-atomic, non-formatted VMEM
/// accepts any alignment (measured on gfx1100 and gfx1201) while atomics must
/// be payload aligned. The DWORD, DWORD_STRICT and STRICT modes are unsupported.
/// @param lanes Lanes to check; others are ignored.
/// @param address_bias Added to each lane address before checking alignment.
///        Pre-routing callers pass `lds_base - shared_aperture_base` for FLAT
///        LDS lanes, so they are checked as the workgroup LDS addresses routing
///        forms.
uint64_t misaligned_atomic_lanes(const VectorMemState &d, const Wavefront &wf, uint64_t lanes,
                                 uint64_t address_bias = 0);

/// @brief A byte range a memory instruction requests from memory.
struct MemoryRequestRange {
  uint64_t address = 0;
  uint32_t size = 0;
};

/// @brief Lanes that raise MEMVIOL and therefore perform no memory access.
/// @details Address-formation MEMVIOL (FLAT hole, private 64-bit atomic) and
/// misaligned atomics. Hardware suppresses only these lanes: gfx1100 and
/// gfx1201 commit every other lane of the instruction, and report a
/// translation fault on another lane instead of the aperture violation.
/// Shared by the global and local pipelines and by the debugger preflight,
/// which must not probe these lanes.
/// @param request_lanes Lanes that issue; see transpose_request_lane_mask.
/// @param lds_lanes Requesting lanes that address LDS.
/// @param lds_bias See misaligned_atomic_lanes' address_bias, for @p lds_lanes.
uint64_t vector_memviol_lanes(const VectorMemState &d, const Wavefront &wf, uint64_t request_lanes,
                              uint64_t lds_lanes, uint64_t lds_bias = 0);

/// @brief Instruction-level rejection checked before any scalar memory side effect.
/// @return Malformed for an invalid atomic shape, ApertureViolation for
///         address-formation MEMVIOL, otherwise Complete.
VmAccessOutcome scalar_memory_admission(const ScalarMemState &d);

/// @brief Append the global-memory byte ranges an admitted vector instruction
/// requests for @p lanes.
/// @details Matches what the global pipeline accesses: one element per lane for
/// atomics, per-element lane masks (buffer bounds, M0 block masks) and
/// swizzled scratch layouts. Callers exclude LDS lanes.
void append_vector_request_ranges(const VectorMemState &d, uint64_t lanes,
                                  std::vector<MemoryRequestRange> &ranges);

/// @brief Append the byte ranges an admitted scalar instruction requests.
/// @details Descriptor-masked dwords request nothing, so a fully masked buffer
/// access appends no range. Atomics and sub-dword loads request their element.
void append_scalar_request_ranges(const ScalarMemState &d, std::vector<MemoryRequestRange> &ranges);

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
  using FaultHandler = std::function<void(Wavefront &, VmAccessOutcome, uint64_t issue_pc)>;
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
    uint64_t issue_pc;
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

  /// @brief Retry only one queue's retained work without requesting an engine yield.
  /// @details The caller holds the CU wave-state lock and owns this queue's pause.
  void retry_for_queue(uint32_t process_id, uint32_t queue_id);

  bool empty() const { return issued_.empty() && returned_.empty(); }

  WaitCounterType counter_type() const { return counter_type_; }
  void set_fault_handler(FaultHandler handler) { fault_handler_ = std::move(handler); }

  /// @brief Retain an operation whose first access attempt already returned Unavailable.
  void defer_unavailable(Instruction *inst, Wavefront &wf);

  /// @brief Cancel one unavailable side-effect-free load and rewind its issuing PC.
  /// @details Used by debugger suspension so an unresolved load can be saved and
  /// re-issued after restore. Stores, atomics and LDS-destination loads are retained.
  bool rewind_unavailable_load(Wavefront &wf);

  /// @brief Cancel every retained operation belonging to one wave-slot occupant.
  void cancel(Wavefront &wf);

protected:
  virtual VmAccessOutcome initiate_access(Instruction &inst, Wavefront &wf) = 0;
  /// Return Complete and do not call complete, or return Deferred and call it
  /// exactly once after the memory response is ready for architectural writeback.
  virtual MemoryAccessCompletion complete_access(Instruction &inst, Wavefront &wf,
                                                 MemoryAccessDeferredCompletion complete) = 0;

  void retry_entries(bool all_queues, uint32_t process_id, uint32_t queue_id);
  VmAccessOutcome issue_impl(Instruction *inst, Wavefront &wf, bool retain_unavailable);
  [[nodiscard]] WaitCounterTokens issue_counters(const Instruction &inst) const;
  void acquire_wait_counters(Wavefront &wf, const WaitCounterTokens &counters);
  void release_wait_counters(Wavefront &wf, const WaitCounterTokens &counters);
  bool cancel_wait_check_issue(const PipelineEntry &entry);
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
  VmAccessOutcome initiate_global_access(VectorMemState &state, Wavefront &wf,
                                         uint64_t request_lanes);
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
