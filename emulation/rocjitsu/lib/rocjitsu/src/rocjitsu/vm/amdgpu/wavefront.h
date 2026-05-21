// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file wavefront.h
/// @brief AMDGPU wavefront execution state and ISA-parameterized wavefront.

#ifndef ROCJITSU_VM_AMDGPU_WAVEFRONT_H_
#define ROCJITSU_VM_AMDGPU_WAVEFRONT_H_

#include "rocjitsu/base/api.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"
#include "rocjitsu/vm/thread_context.h"

#include <cstdint>
#include <memory>
#include <string_view>

namespace rocjitsu {
namespace amdgpu {

// Forward declaration - wavefront accesses registers through its CU.
class ComputeUnitCore;

/// @brief Wavefront execution state.
enum class WfState : uint8_t {
  HALTED,  ///< Slot is currently unused and is available for dispatch.
  RUNNING, ///< In a running state and can be considered for scheduling.
  WAITCNT, ///< Stalled at a waitcnt.
  BARRIER, ///< Stalled at a barrier.
  ENDING,  ///< s_endpgm executed but outstanding memory ops are draining.
};

/// @brief Allocation slice within a register file.
struct RegAllocation {
  uint32_t base = 0;  ///< First register index in the physical file.
  uint32_t count = 0; ///< Number of registers allocated.
};

/// @brief AMDGPU wavefront execution state.
///
/// @details The wavefront does not own its register storage, the parent
/// ComputeUnitCore holds the physical SGPR and VGPR files. Callers access
/// registers via `wf.cu().read_sgpr(wf.sgpr_alloc().base + idx)` etc.
///
/// Each wavefront is permanently bound to a ComputeUnitCore and a slot index
/// (wf_id) at construction time. These persist across reset()/dispatch
/// cycles. Dynamic dispatch state (wg_id, pc, register allocations,
/// execution masks) is set when the slot is activated and reset by reset().
///
/// A slot is considered dispatched (active) when it has a nonzero register
/// allocation (sgpr_alloc_.count > 0). After clear(), the slot is idle.
///
/// wf_size and max register counts come from the ISA struct and are fixed
/// at construction. num_sgprs and num_vgprs are the per-dispatch allocation
/// sizes set from code object metadata.
///
/// Derives from ThreadContext so that instruction execute() methods can
/// static_cast the ThreadContext& parameter to Wavefront&.
class Wavefront : public ThreadContext {
public:
  ~Wavefront() override = default;

  /// @brief Return the number of lanes per wavefront.
  /// @returns Lanes per wavefront (ISA-fixed).
  uint32_t wf_size() const { return wf_size_; }

  /// @brief Return the ISA maximum SGPRs per wavefront.
  /// @returns Maximum scalar registers.
  uint32_t max_sgprs() const { return max_sgprs_; }

  /// @brief Return the ISA maximum VGPRs per wavefront.
  /// @returns Maximum vector registers.
  uint32_t max_vgprs() const { return max_vgprs_; }

  /// @brief Return the number of allocated scalar registers.
  /// @returns Per-dispatch SGPR allocation count.
  uint32_t num_sgprs() const { return num_sgprs_; }

  /// @brief Return the number of allocated vector registers.
  /// @returns Per-dispatch VGPR allocation count.
  uint32_t num_vgprs() const { return num_vgprs_; }

  /// @brief Read the raw status register value.
  /// @returns Status register as a raw uint32_t.
  virtual uint32_t status_raw() const = 0;

  /// @brief Write the raw status register value.
  /// @param val New status register value.
  virtual void set_status_raw(uint32_t val) = 0;

  /// @brief Return the wavefront slot index within the CU.
  /// @returns Permanent slot index.
  uint32_t wf_id() const { return wf_id_; }

  /// @brief Return the workgroup ID assigned at dispatch.
  /// @returns Workgroup ID.
  uint32_t wg_id() const { return wg_id_; }

  /// @brief Return the dispatch ID assigned at dispatch.
  uint32_t dispatch_id() const { return dispatch_id_; }

  /// @brief Set the dispatch ID (called by DispatchController).
  void set_dispatch_id(uint32_t id) { dispatch_id_ = id; }

  /// @brief Return the per-WG LDS base offset assigned at dispatch.
  uint32_t lds_base() const { return lds_base_; }

  /// @brief Set the per-WG LDS base offset.
  void set_lds_base(uint32_t base) { lds_base_ = base; }

  /// @brief Return the SGPR register file allocation.
  /// @returns Const reference to the SGPR allocation slice.
  const RegAllocation &sgpr_alloc() const { return sgpr_alloc_; }

  /// @brief Return the VGPR register file allocation.
  /// @returns Const reference to the VGPR allocation slice.
  const RegAllocation &vgpr_alloc() const { return vgpr_alloc_; }

  /// @brief Return the parent compute unit.
  /// @returns Reference to the owning ComputeUnitCore.
  ComputeUnitCore &cu() { return cu_; }

  /// @returns Const reference to the owning ComputeUnitCore.
  const ComputeUnitCore &cu() const { return cu_; }

  /// @brief Return the EXEC mask.
  /// @returns EXEC mask (one bit per lane, 1 = active).
  uint64_t exec() const { return exec_; }

  /// @brief Set the EXEC mask.
  /// @param val New EXEC mask value.
  void set_exec(uint64_t val) { exec_ = val; }

  /// @brief Return the vector condition code.
  /// @returns VCC register value.
  uint64_t vcc() const { return vcc_; }

  /// @brief Set the vector condition code.
  /// @param val New VCC value.
  void set_vcc(uint64_t val) { vcc_ = val; }

  /// @brief Return the M0 special register.
  /// @returns M0 register value.
  uint32_t m0() const { return m0_; }

  /// @brief Set the M0 special register.
  /// @param val New M0 value.
  void set_m0(uint32_t val) { m0_ = val; }

  /// @brief Return the per-wavefront scratch (private segment) base address.
  /// @returns Byte address in GPU memory where this wavefront's scratch starts.
  uint64_t scratch_base() const { return scratch_base_; }

  /// @brief Set the per-wavefront scratch base address.
  /// @param val Scratch base byte address (set at dispatch by CP).
  void set_scratch_base(uint64_t val) { scratch_base_ = val; }

  /// @brief Return the wait counters for outstanding memory operations.
  /// @returns Reference to the wait counters.
  WaitCounters &wait_counters() { return wait_counters_; }

  /// @returns Const reference to the wait counters.
  const WaitCounters &wait_counters() const { return wait_counters_; }

  /// @brief Set the s_waitcnt target thresholds and stall if not yet satisfied.
  ///
  /// @details Used by GFX9 (CDNA1-4), GFX10 (RDNA1/2), and GFX11 (RDNA3/3.5)
  /// for the monolithic S_WAITCNT instruction.  Sets vmcnt, lgkmcnt, expcnt
  /// thresholds and transitions to WAITCNT if any counter currently exceeds
  /// its target.
  /// @param vmcnt VM counter threshold.
  /// @param lgkmcnt LGKM counter threshold.
  /// @param expcnt Export counter threshold.
  void set_wait_target(uint8_t vmcnt, uint8_t lgkmcnt, uint8_t expcnt) {
    wait_target_.vmcnt = vmcnt;
    wait_target_.lgkmcnt = lgkmcnt;
    wait_target_.expcnt = expcnt;
    if (!wait_satisfied())
      state_ = WfState::WAITCNT;
  }

  /// @brief Set the VSCNT target (GFX10 S_WAITCNT_VSCNT).
  void set_wait_target_vscnt(uint8_t threshold) {
    wait_target_.vscnt = threshold;
    if (!wait_satisfied())
      state_ = WfState::WAITCNT;
  }

  /// @brief Set the LOADCNT target (GFX11+ S_WAITCNT_VMCNT / S_WAIT_LOADCNT).
  void set_wait_target_loadcnt(uint8_t threshold) {
    wait_target_.vmcnt = threshold;
    if (!wait_satisfied())
      state_ = WfState::WAITCNT;
  }

  /// @brief Set the STORECNT target (GFX11+ S_WAITCNT_VSCNT / S_WAIT_STORECNT).
  void set_wait_target_storecnt(uint8_t threshold) {
    wait_target_.vscnt = threshold;
    if (!wait_satisfied())
      state_ = WfState::WAITCNT;
  }

  /// @brief Set the DSCNT target (GFX11+ S_WAITCNT_LGKMCNT / S_WAIT_DSCNT).
  void set_wait_target_dscnt(uint8_t threshold) {
    wait_target_.dscnt = threshold;
    if (!wait_satisfied())
      state_ = WfState::WAITCNT;
  }

  /// @brief Set the KMCNT target (GFX11+ S_WAIT_KMCNT).
  void set_wait_target_kmcnt(uint8_t threshold) {
    wait_target_.kmcnt = threshold;
    if (!wait_satisfied())
      state_ = WfState::WAITCNT;
  }

  /// @brief Set combined STORECNT + DSCNT targets (GFX12 S_WAIT_STORECNT_DSCNT).
  void set_wait_target_storecnt_dscnt(uint8_t storecnt, uint8_t dscnt) {
    wait_target_.vscnt = storecnt;
    wait_target_.dscnt = dscnt;
    if (!wait_satisfied())
      state_ = WfState::WAITCNT;
  }

  /// @brief Set combined LOADCNT + DSCNT targets (GFX12 S_WAIT_LOADCNT_DSCNT).
  void set_wait_target_loadcnt_dscnt(uint8_t loadcnt, uint8_t dscnt) {
    wait_target_.vmcnt = loadcnt;
    wait_target_.dscnt = dscnt;
    if (!wait_satisfied())
      state_ = WfState::WAITCNT;
  }

  /// @brief Set a single split-wait counter threshold by name.
  ///
  /// Used by the generated S_WAIT_* / S_WAITCNT_* instruction execute()
  /// bodies.  Maps the instruction's semantic name to the correct target.
  void set_wait_counter(const char *counter_name, uint16_t threshold) {
    using namespace std::string_view_literals;
    std::string_view name{counter_name};
    auto t = static_cast<uint8_t>(threshold);
    if (name == "wait_loadcnt")
      set_wait_target_loadcnt(t);
    else if (name == "wait_storecnt")
      set_wait_target_storecnt(t);
    else if (name == "wait_dscnt")
      set_wait_target_dscnt(t);
    else if (name == "wait_kmcnt")
      set_wait_target_kmcnt(t);
    else if (name == "wait_expcnt") {
      wait_target_.expcnt = static_cast<uint8_t>(threshold & 0x07);
      if (!wait_satisfied())
        state_ = WfState::WAITCNT;
    } else if (name == "wait_samplecnt" || name == "wait_bvhcnt") {
      wait_target_.vmcnt = t; // map to vmcnt
      if (!wait_satisfied())
        state_ = WfState::WAITCNT;
    } else if (name == "wait_loadcnt_dscnt") {
      set_wait_target_loadcnt_dscnt(static_cast<uint8_t>((threshold >> 8) & 0x3F),
                                    static_cast<uint8_t>(threshold & 0x3F));
    } else if (name == "wait_storecnt_dscnt") {
      set_wait_target_storecnt_dscnt(static_cast<uint8_t>((threshold >> 8) & 0x3F),
                                     static_cast<uint8_t>(threshold & 0x3F));
    }
  }

  /// @brief Check whether all wait counter thresholds are satisfied.
  /// @retval true All counters are at or below their targets.
  /// @retval false One or more counters exceed their targets.
  bool wait_satisfied() const { return wait_target_.satisfied(wait_counters_); }

  /// @brief Read the Scalar Condition Code (SCC) from the status register.
  /// @retval true SCC bit is set.
  /// @retval false SCC bit is clear.
  bool read_scc() const { return status_raw() & 1u; }

  /// @brief Write the Scalar Condition Code (SCC) in the status register.
  /// @param val New SCC value.
  void write_scc(bool val) {
    uint32_t s = status_raw();
    set_status_raw(val ? (s | 1u) : (s & ~1u));
  }

  /// @brief Return the current execution state.
  /// @returns Current WfState.
  WfState state() const { return state_; }

  /// @brief Set the execution state.
  /// @param s New execution state.
  void set_state(WfState s) { state_ = s; }

  /// @brief Check whether this wavefront slot is halted.
  /// @retval true Slot is halted and available for dispatch.
  /// @retval false Slot is active (running, waiting, or at a barrier).
  bool is_halted() const { return state_ == WfState::HALTED; }

  /// @brief Halt this wavefront and notify the CU for WG completion tracking.
  /// @details Transitions to HALTED and decrements the CU's per-WG refcount.
  /// When the refcount reaches zero (all WFs in the WG halted), the CU fires
  /// notify_wg_complete to the CP. This is the sole completion detection path —
  /// driven entirely by s_endpgm → end() → halt().
  void halt();

  /// @brief End program execution. If all memory ops are drained, halts
  /// immediately. Otherwise, transitions to ENDING and lets the memory
  /// pipeline drain remaining ops before halting.
  void end() {
    if (wait_counters_.empty())
      halt();
    else
      state_ = WfState::ENDING;
  }

  /// @brief Log instruction count at end for trace/debug.
  void trace_end_summary() const;

  /// @brief Reset dynamic dispatch state so this slot can be reused.
  ///
  /// @details Resets register allocations, workgroup ID, and execution state back
  /// to defaults. Does not change permanent bindings (cu_, wf_id_) or ISA-fixed
  /// properties (wf_size_, max_sgprs_, max_vgprs_) or the status register.
  void reset() {
    pc = 0;
    wg_id_ = 0;
    dispatch_id_ = 0;
    num_sgprs_ = 0;
    num_vgprs_ = 0;
    sgpr_alloc_ = {};
    vgpr_alloc_ = {};
    exec_ = ~0ULL;
    vcc_ = 0;
    m0_ = 0;
    scratch_base_ = 0;
    wait_counters_ = {};
    wait_target_ = {};
    state_ = WfState::HALTED;
  }

protected:
  /// @brief Construct a wavefront bound to a CU slot.
  /// @param cu Parent compute unit (permanent binding).
  /// @param wf_id Slot index within the CU (permanent binding).
  /// @param wf_size Lanes per wavefront (ISA-fixed).
  /// @param max_sgprs Maximum SGPRs per wavefront (ISA-fixed).
  /// @param max_vgprs Maximum VGPRs per wavefront (ISA-fixed).
  Wavefront(ComputeUnitCore &cu, uint32_t wf_id, uint32_t wf_size, uint32_t max_sgprs,
            uint32_t max_vgprs)
      : cu_(cu), wf_id_(wf_id), wf_size_(wf_size), max_sgprs_(max_sgprs), max_vgprs_(max_vgprs) {}

  ComputeUnitCore &cu_;      ///< Parent CU (permanent, set at construction).
  uint32_t wf_id_ = 0;       ///< Slot index within the CU (permanent).
  uint32_t wg_id_ = 0;       ///< Workgroup ID (set per dispatch).
  uint32_t dispatch_id_ = 0; ///< Dispatch ID (set per dispatch, unique per dispatch).
  uint32_t lds_base_ = 0;    ///< Per-WG LDS base offset (set per dispatch).

  uint32_t wf_size_ = 0;   ///< Lanes per wavefront (ISA-fixed).
  uint32_t num_sgprs_ = 0; ///< Allocated scalar registers (set at dispatch).
  uint32_t num_vgprs_ = 0; ///< Allocated vector registers (set at dispatch).
  uint32_t max_sgprs_ = 0; ///< ISA maximum SGPRs per wavefront.
  uint32_t max_vgprs_ = 0; ///< ISA maximum VGPRs per wavefront.

  RegAllocation sgpr_alloc_; ///< Slice in CU's SGPR file.
  RegAllocation vgpr_alloc_; ///< Slice in CU's VGPR file.

private:
  uint64_t exec_ = ~0ULL;           ///< EXEC mask -- one bit per lane (1 = active).
  uint64_t vcc_ = 0;                ///< Vector condition code (per-lane comparison result).
  uint32_t m0_ = 0;                 ///< M0 special register (misc addressing).
  uint64_t scratch_base_ = 0;       ///< Per-wavefront scratch (private segment) base address.
  WfState state_ = WfState::HALTED; ///< Current execution state.
  WaitCounters wait_counters_;      ///< Outstanding memory operation counters.

public:
  uint32_t trace_inst_count_ = 0; ///< Debug: instruction count for trace.
private:
  WaitTarget wait_target_; ///< Current s_waitcnt thresholds.

  friend class ComputeUnitCore; // CU sets allocation fields during dispatch.
};

/// @brief ISA-parameterized concrete wavefront with ISA-specific status register.
///
/// @details The Isa trait provides WF_SIZE, MAX_SGPRS_PER_WF, MAX_VGPRS_PER_WF, and StatusReg.
/// Register storage lives in the parent ComputeUnit's physical register files;
/// this class only adds the ISA-specific status register type.
///
/// @tparam Isa ISA traits struct satisfying the GpuIsa concept.
template <GpuIsa Isa> class IsaWavefront final : public Wavefront {
public:
  using StatusType = typename Isa::StatusReg;

  /// @brief Construct a wavefront bound to a CU slot.
  /// @param cu Parent compute unit.
  /// @param wf_id Slot index within the CU.
  IsaWavefront(ComputeUnitCore &cu, uint32_t wf_id)
      : Wavefront(cu, wf_id, Isa::WF_SIZE, Isa::MAX_SGPRS_PER_WF, Isa::MAX_VGPRS_PER_WF) {}

  /// @brief Return the raw status register value.
  /// @returns Raw status register value.
  uint32_t status_raw() const override { return static_cast<uint32_t>(status); }

  /// @brief Set the raw status register value.
  /// @param[in] val New raw status register value.
  void set_status_raw(uint32_t val) override { status = val; }

  /// @brief ISA-specific status register (SCC, EXECZ, VCCZ, HALT, etc.).
  StatusType status{0};
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_WAVEFRONT_H_
