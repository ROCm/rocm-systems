// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file wavefront.h
/// @brief AMDGPU wavefront execution state and ISA-parameterized wavefront.

#ifndef ROCJITSU_VM_AMDGPU_WAVEFRONT_H_
#define ROCJITSU_VM_AMDGPU_WAVEFRONT_H_

#include "rocjitsu/base/api.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/instruction_compute_unit_view.h"
#include "rocjitsu/vm/amdgpu/vgpr_msb.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"
#include "rocjitsu/vm/plugins/wavefront_state.h"
#include "rocjitsu/vm/thread_context.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

// Forward declaration - wavefront accesses registers through its CU.
class ComputeUnitCore;
class Lds;

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
/// registers through Operand or RegisterAccess in instruction code and through the owning CU in VM
/// code.
///
/// Each wavefront is permanently bound to a ComputeUnitCore and a slot index
/// (wf_id) at construction time. These persist across reset()/dispatch
/// cycles. Dynamic dispatch state (wg_id, pc, register allocations,
/// execution masks) is set when the slot is activated and reset by reset().
///
/// Plugin state (plugin_states_) is NOT cleared by reset(). Plugins must
/// set their per-wavefront state in onAmdgpuWorkgroupDispatched, which
/// fires before the wavefront's first instruction.
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

  /// @brief Read the raw MODE register value.
  uint32_t mode_raw() const { return mode_raw_; }

  /// @brief Write the raw MODE register value.
  void set_mode_raw(uint32_t val) {
    mode_raw_ = val;
    vgpr_msb_mode_ = mode_layout_to_set_vgpr_msb(
        static_cast<uint8_t>((val & VGPR_MSB_MODE_MASK) >> VGPR_MSB_MODE_SHIFT));
  }

  /// @brief Return current S_SET_VGPR_MSB-format VGPR high-bank bits.
  uint8_t vgpr_msb_mode() const { return vgpr_msb_mode_; }

  /// @brief Set current S_SET_VGPR_MSB-format VGPR high-bank bits.
  void set_vgpr_msb_mode(uint8_t val) {
    vgpr_msb_mode_ = val;
    uint32_t mode_bits = static_cast<uint32_t>(set_vgpr_msb_to_mode_layout(val))
                         << VGPR_MSB_MODE_SHIFT;
    mode_raw_ = (mode_raw_ & ~VGPR_MSB_MODE_MASK) | mode_bits;
  }

  /// @brief Read the raw WAVE_SCHED_MODE register value.
  uint32_t wave_sched_mode_raw() const { return wave_sched_mode_raw_; }

  /// @brief Write the raw WAVE_SCHED_MODE register value.
  void set_wave_sched_mode_raw(uint32_t val) { wave_sched_mode_raw_ = val; }

  /// @brief Return the two-bit VGPR high-bank selector for an operand role.
  uint32_t vgpr_msb_for_role(VgprMsbRole role) const {
    switch (role) {
    case VgprMsbRole::Src0:
      return vgpr_msb_mode_ & 0x3u;
    case VgprMsbRole::Src1:
      return (vgpr_msb_mode_ >> 2) & 0x3u;
    case VgprMsbRole::Src2:
      return (vgpr_msb_mode_ >> 4) & 0x3u;
    case VgprMsbRole::Dst:
      return (vgpr_msb_mode_ >> 6) & 0x3u;
    case VgprMsbRole::None:
      return 0;
    }
    return 0;
  }

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

  /// @brief Return the AQL ring packet id (dispatch's ring index) for this wave.
  /// @details This is the queue read index at which the dispatch packet was
  /// fetched. rocm-dbgapi correlates a trapped wave to its dispatch by matching
  /// this against the queue's read/write dispatch ids (TTMP11[6:30]).
  uint32_t aql_packet_id() const { return aql_packet_id_; }

  /// @brief Set the AQL ring packet id (called at dispatch).
  void set_aql_packet_id(uint32_t id) { aql_packet_id_ = id; }

  /// @brief GPU load bias for code-object-relative function addresses.
  uint64_t code_load_bias() const { return code_load_bias_; }
  void set_code_load_bias(uint64_t bias) { code_load_bias_ = bias; }

  /// @brief Return this wave's position within its workgroup (0-based).
  uint32_t wave_in_group() const { return wave_in_group_; }

  /// @brief Set this wave's position within its workgroup (called at dispatch).
  void set_wave_in_group(uint32_t pos) { wave_in_group_ = pos; }

  /// @brief Return the KFD queue ID that launched this wave.
  uint32_t queue_id() const { return queue_id_; }

  /// @brief Set the KFD queue ID (called by the command processor at dispatch).
  /// @details Used by the debugger path to correlate a stopped wave with the
  /// queue whose context-save-restore area holds its saved state.
  void set_queue_id(uint32_t id) { queue_id_ = id; }

  /// @brief Return the owning process ID (PASID analog).
  uint32_t process_id() const { return process_id_; }

  /// @brief Set the owning process ID at dispatch time.
  void set_process_id(uint32_t id) { process_id_ = id; }

  /// @brief Return the per-WG LDS base offset assigned at dispatch.
  uint32_t lds_base() const { return lds_base_; }

  /// @brief Set the per-WG LDS base offset.
  void set_lds_base(uint32_t base) { lds_base_ = base; }

  /// @brief Return the aligned LDS allocation size for this workgroup.
  uint32_t lds_size() const { return lds_size_; }

  /// @brief Set the aligned LDS allocation size for this workgroup.
  void set_lds_size(uint32_t size) { lds_size_ = size; }

  /// @brief Return the LDS backing selected for this workgroup placement.
  ///
  /// CU-mode workgroups use their owning CU's LDS. WGP-mode workgroups can
  /// instead use a backing shared by the two sibling CUs in the WGP.
  Lds &lds();
  const Lds &lds() const;

  /// @brief Override the LDS backing for this workgroup placement.
  /// Passing nullptr restores the owning CU's LDS.
  void set_lds(Lds *lds) { lds_ = lds; }

  /// @brief Return this workgroup's rank within its cluster.
  uint32_t cluster_rank() const { return cluster_rank_; }

  /// @brief Return the number of workgroups in this workgroup's cluster.
  uint32_t cluster_size() const { return cluster_size_; }

  /// @brief Set cluster placement metadata computed by the command processor.
  void set_cluster_info(uint32_t rank, uint32_t size) {
    cluster_rank_ = rank;
    cluster_size_ = size == 0 ? 1 : size;
  }

  /// @brief Return the SGPR register file allocation.
  /// @returns Const reference to the SGPR allocation slice.
  const RegAllocation &sgpr_alloc() const { return sgpr_alloc_; }

  /// @brief Return the VGPR register file allocation.
  /// @returns Const reference to the VGPR allocation slice.
  const RegAllocation &vgpr_alloc() const { return vgpr_alloc_; }

  /// @brief Return the instruction-facing compute-unit service view.
  /// @returns Narrow view over the owning compute unit.
  InstructionComputeUnitView &cu() { return cu_view_; }

  /// @returns Const narrow view over the owning compute unit.
  const InstructionComputeUnitView &cu() const { return cu_view_; }

  /// @brief Return the EXEC mask.
  /// @returns EXEC mask (one bit per lane, 1 = active).
  uint64_t exec() const { return exec_ & lane_mask(); }

  /// @brief Return the raw architectural EXEC register pair.
  ///
  /// Wave32 instructions may use EXEC_HI as scalar scratch even though its
  /// bits do not select active lanes. Operand decoding therefore needs the
  /// unmasked register value, while vector execution should use exec().
  uint64_t exec_raw() const { return exec_; }

  /// @brief Set the active-lane portion of the EXEC register pair.
  /// @param val New EXEC mask value.
  ///
  /// Wave32 leaves EXEC_HI available as scalar scratch. Vector instructions
  /// that update the execution mask must therefore preserve the non-lane bits.
  void set_exec(uint64_t val) { exec_ = (exec_ & ~lane_mask()) | (val & lane_mask()); }

  /// @brief Set the raw architectural EXEC register pair.
  void set_exec_raw(uint64_t val) { exec_ = val; }

  /// @brief Return the VCC scalar register pair.
  /// @returns Raw VCC register value.
  uint64_t vcc() const { return vcc_; }

  /// @brief Set the VCC scalar register pair.
  /// @param val New VCC value.
  void set_vcc(uint64_t val) { vcc_ = val; }

  /// @brief Return the M0 special register.
  /// @returns M0 register value.
  uint32_t m0() const { return m0_; }

  /// @brief Set the M0 special register.
  /// @param val New M0 value.
  void set_m0(uint32_t val) { m0_ = val; }

  static constexpr uint32_t GPR_IDX_EN_BIT = 1u << 27;
  bool gpr_idx_en() const { return mode_raw_ & GPR_IDX_EN_BIT; }
  uint32_t gpr_idx_offset() const { return m0_ & 0xFF; }
  uint32_t gpr_idx_mode() const { return (m0_ >> 8) & 0xF; }

  /// @brief Return the per-wavefront scratch (private segment) base address.
  /// @returns Byte address in GPU memory where this wavefront's scratch starts.
  uint64_t scratch_base() const { return scratch_base_; }

  /// @brief Set the per-wavefront scratch base address.
  /// @param val Scratch base byte address (set at dispatch by CP).
  void set_scratch_base(uint64_t val) { scratch_base_ = val; }

  /// @brief Return the per-lane private scratch allocation size in bytes.
  uint32_t scratch_lane_size() const { return scratch_lane_size_; }

  /// @brief Set the per-lane private scratch allocation size in bytes.
  void set_scratch_lane_size(uint32_t val) { scratch_lane_size_ = val; }

  /// @brief Return this wave's scratch scoreboard id (its slot in the queue's
  /// scratch allocation). rocm-dbgapi multiplies it by COMPUTE_TMPRING_SIZE's
  /// per-wave size to locate each wave's private memory.
  uint32_t scratch_scoreboard_id() const { return scratch_scoreboard_id_; }

  /// @brief Set this wave's scratch scoreboard id (set at dispatch by the CP).
  void set_scratch_scoreboard_id(uint32_t val) { scratch_scoreboard_id_ = val; }

  uint64_t shared_aperture_base() const { return shared_aperture_base_; }
  uint64_t shared_aperture_limit() const { return shared_aperture_limit_; }
  uint64_t private_aperture_base() const { return private_aperture_base_; }
  uint64_t private_aperture_limit() const { return private_aperture_limit_; }
  void set_apertures(uint64_t sb, uint64_t sl, uint64_t pb, uint64_t pl) {
    shared_aperture_base_ = sb;
    shared_aperture_limit_ = sl;
    private_aperture_base_ = pb;
    private_aperture_limit_ = pl;
  }

  /// @brief Return the wait counters for outstanding memory operations.
  /// @returns Reference to the wait counters.
  WaitCounters &wait_counters() { return wait_counters_; }

  /// @returns Const reference to the wait counters.
  const WaitCounters &wait_counters() const { return wait_counters_; }

  /// @brief Retire one outstanding wait-counter operation and wake the wave if ready.
  void release_wait_counter(WaitCounterType type);

  /// @brief Set the s_waitcnt target thresholds and stall if not yet satisfied.
  ///
  /// @details Used by GFX9 (CDNA1-4), GFX10 (RDNA1/2), and GFX11 (RDNA3/3.5)
  /// for the monolithic S_WAITCNT instruction.  Sets vmcnt, lgkmcnt, expcnt
  /// thresholds and transitions to WAITCNT if any counter currently exceeds
  /// its target.
  /// @param vmcnt VM counter threshold.
  /// @param lgkmcnt LGKM counter threshold.
  /// @param expcnt Export counter threshold.
  const WaitTarget &wait_target() const { return wait_target_; }

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

  /// @brief Set the TENSORCNT target (GFX12.5 S_WAIT_TENSORCNT).
  void set_wait_target_tensorcnt(uint8_t threshold) {
    wait_target_.tensorcnt = threshold;
    if (!wait_satisfied())
      state_ = WfState::WAITCNT;
  }

  /// @brief Set the ASYNCCNT target (GFX12.5 S_WAIT_ASYNCCNT).
  void set_wait_target_asynccnt(uint8_t threshold) {
    wait_target_.asynccnt = threshold;
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
    else if (name == "wait_tensorcnt")
      set_wait_target_tensorcnt(t);
    else if (name == "wait_asynccnt")
      set_wait_target_asynccnt(t);
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

  // -- KFD debugger (trap) state --
  //
  // These model the wave-level state the AMD trap handler maintains for
  // rocm-dbgapi: the trap temporary registers (TTMP0-15), the trap status
  // register (TRAPSTS), and the debug halt/single-step bits.

  /// @brief Read a trap temporary register (TTMP0-15).
  uint32_t ttmp(uint32_t idx) const { return idx < 16 ? ttmp_[idx] : 0; }

  /// @brief Write a trap temporary register (TTMP0-15).
  void set_ttmp(uint32_t idx, uint32_t val) {
    if (idx < 16)
      ttmp_[idx] = val;
  }

  /// @brief Whether the wave is currently fetching from its configured TBA.
  bool in_trap_handler() const { return in_trap_handler_; }
  void set_in_trap_handler(bool value) { in_trap_handler_ = value; }

  bool trap_interrupt_sent() const { return trap_interrupt_sent_; }
  void set_trap_interrupt_sent(bool value) { trap_interrupt_sent_ = value; }
  uint32_t trap_saved_status() const { return trap_saved_status_; }
  void set_trap_saved_status(uint32_t value) { trap_saved_status_ = value; }
  uint64_t trap_saved_exec() const { return trap_saved_exec_; }
  void set_trap_saved_exec(uint64_t value) { trap_saved_exec_ = value; }

  /// @brief Record the trap id supplied by hardware trap entry.
  void set_trap_id(uint32_t value) { trap_id_ = value; }

  /// @brief Read the trap status register (TRAPSTS / EXCP flags).
  uint32_t trapsts() const { return trapsts_; }

  /// @brief Write the trap status register.
  void set_trapsts(uint32_t val) { trapsts_ = val; }

  /// @brief Whether the debugger has stopped this wave (trapped or suspended).
  /// @details A debug-halted wave keeps its slot and all register state; the
  /// scheduler skips it so the CU can go quiescent without retiring the wave.
  bool debug_halted() const { return debug_halted_; }
  void set_debug_halted(bool v) { debug_halted_ = v; }

  /// @brief Whether KFD has temporarily suspended this wave's queue.
  /// @details Queue suspension freezes execution for a stable CWSR snapshot,
  /// but unlike debug_halted it does not imply an architectural stop reason.
  bool debug_suspended() const { return debug_suspended_; }
  void set_debug_suspended(bool v) { debug_suspended_ = v; }
  bool debug_paused() const { return debug_halted_ || debug_suspended_; }

  /// @brief Whether a future debugger resume should request single-step mode.
  bool debug_single_step() const { return single_step_; }
  void set_debug_single_step(bool v) { single_step_ = v; }

  /// @brief The trap id recorded by the last s_trap (breakpoint = 1).
  uint32_t trap_id() const { return trap_id_; }

  /// @brief Stable, unique debugger wave id (planted into TTMP4:5 on stop).
  /// @details Zero until assigned. rocm-dbgapi reads this from the CWSR area as
  /// the wave's identity, so it must be stable across re-serialization of the
  /// same wave. Assigned lazily by the driver on the first stop.
  uint64_t debug_wave_id() const { return debug_wave_id_; }
  void set_debug_wave_id(uint64_t id) { debug_wave_id_ = id; }

  /// @brief Read one allocated SGPR for debugger state capture.
  uint32_t debug_read_sgpr(uint32_t reg) const;

  /// @brief Read one allocated VGPR lane for debugger state capture.
  uint32_t debug_read_vgpr(uint32_t reg, uint32_t lane) const;

  /// @brief Write one allocated SGPR from debugger state restore.
  void debug_write_sgpr(uint32_t reg, uint32_t value);

  /// @brief Write one allocated VGPR lane from debugger state restore.
  void debug_write_vgpr(uint32_t reg, uint32_t lane, uint32_t value);

  /// @brief Stop this wave in the debugger (models the trap handler entry).
  /// @param trap_id Trap id from the s_trap immediate (breakpoint = 1).
  /// @details Records the trap id and halts the wave for debugger inspection.
  /// The PC is advanced past the s_trap by the caller (issue_instruction), so
  /// the saved PC points just after the trap, matching the ROCr trap handler.
  void debug_trap(uint32_t trap_id) {
    trap_id_ = trap_id;
    debug_halted_ = true;
    single_step_ = false;
  }

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
    aql_packet_id_ = 0;
    code_load_bias_ = 0;
    wave_in_group_ = 0;
    process_id_ = 0;
    lds_base_ = 0;
    lds_size_ = 0;
    lds_ = nullptr;
    cluster_rank_ = 0;
    cluster_size_ = 1;
    num_sgprs_ = 0;
    num_vgprs_ = 0;
    sgpr_alloc_ = {};
    vgpr_alloc_ = {};
    exec_ = lane_mask();
    vcc_ = 0;
    m0_ = 0;
    set_mode_raw(0);
    set_wave_sched_mode_raw(0);
    scratch_base_ = 0;
    scratch_lane_size_ = 0;
    scratch_scoreboard_id_ = 0;
    shared_aperture_base_ = 0;
    shared_aperture_limit_ = 0;
    private_aperture_base_ = 0;
    private_aperture_limit_ = 0;
    wait_counters_ = {};
    wait_target_ = {};
    ready_cycle_ = 0;
    state_ = WfState::HALTED;
    for (auto &t : ttmp_)
      t = 0;
    trapsts_ = 0;
    in_trap_handler_ = false;
    trap_interrupt_sent_ = false;
    trap_saved_status_ = 0;
    trap_saved_exec_ = 0;
    debug_halted_ = false;
    debug_suspended_ = false;
    single_step_ = false;
    trap_id_ = 0;
    debug_wave_id_ = 0;
    queue_id_ = 0;
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
      : cu_(cu), cu_view_(cu), wf_id_(wf_id), wf_size_(wf_size), max_sgprs_(max_sgprs),
        max_vgprs_(max_vgprs) {}

  ComputeUnitCore &cu_; ///< Parent CU (permanent, set at construction).
  InstructionComputeUnitView cu_view_;
  uint32_t wf_id_ = 0;          ///< Slot index within the CU (permanent).
  uint32_t wg_id_ = 0;          ///< Workgroup ID (set per dispatch).
  uint32_t dispatch_id_ = 0;    ///< Dispatch ID (set per dispatch, unique per dispatch).
  uint32_t aql_packet_id_ = 0;  ///< AQL ring packet id of the dispatch (debugger correlation).
  uint64_t code_load_bias_ = 0; ///< GPU load bias for code-object-relative call targets.
  uint32_t wave_in_group_ = 0;  ///< Position of this wave within its workgroup (debugger).
  uint32_t process_id_ = 0;     ///< Owning process ID (PASID analog, set per dispatch).
  uint32_t queue_id_ = 0;       ///< KFD queue ID that launched this wave (debugger correlation).
  uint32_t lds_base_ = 0;       ///< Per-WG LDS base offset (set per dispatch).
  uint32_t lds_size_ = 0;       ///< Aligned per-WG LDS allocation size.
  Lds *lds_ = nullptr;          ///< Placement-selected LDS backing; nullptr means CU-local LDS.
  uint32_t cluster_rank_ = 0;   ///< Workgroup rank inside the dispatch cluster.
  uint32_t cluster_size_ = 1;   ///< Number of workgroups in the dispatch cluster.

  uint32_t wf_size_ = 0;   ///< Lanes per wavefront (ISA-fixed).
  uint32_t num_sgprs_ = 0; ///< Allocated scalar registers (set at dispatch).
  uint32_t num_vgprs_ = 0; ///< Allocated vector registers (set at dispatch).
  uint32_t max_sgprs_ = 0; ///< ISA maximum SGPRs per wavefront.
  uint32_t max_vgprs_ = 0; ///< ISA maximum VGPRs per wavefront.

  RegAllocation sgpr_alloc_; ///< Slice in CU's SGPR file.
  RegAllocation vgpr_alloc_; ///< Slice in CU's VGPR file.

private:
  ComputeUnitCore &raw_cu() { return cu_; }
  const ComputeUnitCore &raw_cu() const { return cu_; }

  uint64_t lane_mask() const { return wf_size_ >= 64 ? ~0ULL : ((1ULL << wf_size_) - 1ULL); }

  uint64_t exec_ = ~0ULL;              ///< EXEC mask -- one bit per lane (1 = active).
  uint64_t vcc_ = 0;                   ///< Vector condition code (per-lane comparison result).
  uint32_t m0_ = 0;                    ///< M0 special register (misc addressing).
  uint32_t mode_raw_ = 0;              ///< MODE register state.
  uint8_t vgpr_msb_mode_ = 0;          ///< S_SET_VGPR_MSB layout for MODE VGPR_MSB bits.
  uint32_t wave_sched_mode_raw_ = 0;   ///< WAVE_SCHED_MODE register state.
  uint64_t scratch_base_ = 0;          ///< Per-wavefront scratch (private segment) base address.
  uint32_t scratch_lane_size_ = 0;     ///< Per-lane private scratch allocation size in bytes.
  uint32_t scratch_scoreboard_id_ = 0; ///< Scratch slot index (debugger private-memory mapping).
  uint64_t shared_aperture_base_ = 0;
  uint64_t shared_aperture_limit_ = 0;
  uint64_t private_aperture_base_ = 0;
  uint64_t private_aperture_limit_ = 0;
  WfState state_ = WfState::HALTED; ///< Current execution state.
  WaitCounters wait_counters_;      ///< Outstanding memory operation counters.

  uint32_t ttmp_[16] = {};           ///< Trap temporary registers (TTMP0-15).
  uint32_t trapsts_ = 0;             ///< Trap status register (EXCP flags).
  bool in_trap_handler_ = false;     ///< Executing the configured trap-handler shader.
  bool trap_interrupt_sent_ = false; ///< Handler issued MSG_INTERRUPT for this entry.
  uint32_t trap_saved_status_ = 0;   ///< Interrupted STATUS restored after handler completion.
  uint64_t trap_saved_exec_ = 0;     ///< Interrupted EXEC restored after handler completion.
  bool debug_halted_ = false;        ///< Stopped by the debugger (skipped by scheduler).
  bool debug_suspended_ = false;     ///< Queue-suspended for a stable CWSR snapshot.
  bool single_step_ = false;         ///< Execute one instruction on resume, then re-stop.
  uint32_t trap_id_ = 0;             ///< Trap id from the last s_trap (breakpoint = 1).
  uint64_t debug_wave_id_ = 0;       ///< Stable debugger wave id (TTMP4:5); 0 until assigned.

public:
  uint32_t trace_inst_count_ = 0; ///< Debug: instruction count for trace.

  /// @brief Cycle at which this WF became RUNNING (for scheduler priority).
  uint64_t ready_cycle() const { return ready_cycle_; }
  void set_ready_cycle(uint64_t c) { ready_cycle_ = c; }

  WavefrontState *plugin_state(uint32_t slot) const {
    assert(slot < plugin_states_.size());
    return plugin_states_[slot].get();
  }
  void set_plugin_state(uint32_t slot, std::unique_ptr<WavefrontState> s) {
    if (plugin_states_.size() <= slot)
      plugin_states_.resize(slot + 1);
    plugin_states_[slot] = std::move(s);
  }

private:
  // Mutable: plugin state is externally-attached observer state, not part of
  // the wavefront's GPU simulation contract. The SIMD register-read path is
  // const (it doesn't alter GPU state), but plugins need to update their own
  // tracking during reads.
  mutable std::vector<std::unique_ptr<WavefrontState>> plugin_states_;
  uint64_t ready_cycle_ = 0;
  WaitTarget wait_target_; ///< Current s_waitcnt thresholds.

  friend class ComputeUnitCore; // CU sets allocation fields during dispatch.

  // Memory pipelines complete deferred VM loads into physical SGPR/VGPR
  // storage. They intentionally bypass instruction read-observation because
  // completion writes produced memory results rather than instruction source
  // reads.
  friend class GlobalMemPipeline;
  friend class LocalMemPipeline;
  friend class ScalarMemPipeline;
};

inline uint32_t apply_gpr_idx(const Wavefront &wf, uint32_t vgpr_off, bool is_dst) {
  uint32_t mode = wf.gpr_idx_mode();
  if ((!is_dst && (mode & 0x7)) || (is_dst && (mode & 0x8)))
    return vgpr_off + wf.gpr_idx_offset();
  return vgpr_off;
}

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
