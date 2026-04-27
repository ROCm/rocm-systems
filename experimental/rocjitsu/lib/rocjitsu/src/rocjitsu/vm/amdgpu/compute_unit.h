// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file compute_unit.h
/// @brief AMDGPU compute unit hierarchy: ComputeUnitCore, ExecComputeUnit, and IsaExecComputeUnit.

#ifndef ROCJITSU_VM_AMDGPU_COMPUTE_UNIT_H_
#define ROCJITSU_VM_AMDGPU_COMPUTE_UNIT_H_

#include "rocjitsu/base/api.h"
#include "rocjitsu/vm/execution_plugin.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/mtype.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "simdojo/components/register_file.h"
#include "simdojo/components/vector_reg.h"
#include "util/log.h"

#include "simdojo/sim/component.h"
#include "simdojo/sim/exec_mode.h"
#include "simdojo/sim/simulation.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// @brief Base AMDGPU compute unit that owns wavefront slots and register files.
///
/// @details Owns the physical SGPR and VGPR register files and a fixed array of
/// pre-allocated wavefront slots. Each wavefront holds a permanent reference
/// back to this CU and its slot index (wf_id).
///
/// dispatch_wf() finds the first idle slot, allocates registers, and
/// activates it. retire_halted_wfs() frees register allocations and calls
/// clear() so the slot can be reused.
///
/// Each step() call picks the next active wavefront (round-robin) and executes
/// one instruction using the ISA-specific decoder.
///
/// The execution shell (event scheduling, activation, idle detection) is
/// provided by ExecComputeUnit<Mode> below. ISA-specific parts (VGPR register
/// file type, instruction execution dispatch, wavefront creation) are
/// implemented by IsaExecComputeUnit<Mode, Isa>. Use the create() factory
/// to construct.
class ComputeUnitCore : public simdojo::CompositeComponent {
public:
  /// @brief Configuration for a compute unit.
  struct Config {
    rj_code_arch_t arch;             ///< ISA architecture (determines wave size, decoder).
    uint32_t num_wf_slots;           ///< Number of hardware wavefront slots (contexts).
    uint32_t sgprs_per_wf;           ///< Scalar GPRs per wavefront (allocation granularity).
    uint32_t vgprs_per_wf;           ///< Vector GPRs per wavefront (allocation granularity).
    uint32_t lds_size_kb;            ///< Local Data Share size in kilobytes.
    uint32_t functional_quantum = 0; ///< Max instructions per advance() (0 = unbounded).
  };

  ~ComputeUnitCore() override = default;

  /// @brief Create a compute unit for the given architecture and execution mode.
  /// @param name Human-readable name (e.g., "cu0").
  /// @param config CU configuration parameters.
  /// @param memory Shared GPU memory (not owned).
  /// @param l2 Shared L2 cache (owned by the XCD).
  /// @param exec_mode Execution mode for the CU.
  /// @returns Owning pointer to the created compute unit.
  static std::unique_ptr<ComputeUnitCore>
  create(std::string name, const Config &config, GpuMemory *memory, L2Cache *l2,
         simdojo::ExecMode exec_mode = simdojo::ExecMode::FUNCTIONAL);

  /// @brief Activate an idle wavefront slot with the given dispatch parameters.
  ///
  /// @details Finds the first idle slot, allocates SGPR and VGPR register file blocks,
  /// and initializes the slot's dynamic state (wg_id, pc, allocations).
  /// @param wg_id Workgroup ID for this wavefront.
  /// @param pc Kernel entry point (byte address).
  /// @param sgprs Number of scalar registers to allocate.
  /// @param vgprs Number of vector registers to allocate.
  /// @returns Pointer to the activated wavefront, or nullptr if no free slot
  ///          or insufficient register space.
  Wavefront *dispatch_wf(uint32_t wg_id, uint64_t pc, uint32_t sgprs, uint32_t vgprs);

  /// @brief Execute one instruction on the next active wavefront.
  /// @retval true An instruction was executed.
  /// @retval false No active wavefronts remain.
  bool step() override;

  /// @brief Clear all halted wavefront slots and free their register allocations.
  void retire_halted_wfs();

  /// @brief Check whether this CU can accept an entire workgroup.
  ///
  /// @details Queries the number of free wavefront slots and register file
  /// blocks without modifying any state. The command processor calls this
  /// before dispatching to guarantee all-or-nothing workgroup placement.
  /// @param num_wfs Number of wavefronts in the workgroup.
  /// @returns true if the CU has enough free slots and registers.
  bool can_accept_workgroup(uint32_t num_wfs) const;

  /// @brief Execute work until the next scheduling boundary.
  ///
  /// @details In FUNCTIONAL mode, executes up to `functional_quantum`
  /// instructions via step(), then yields back to the event loop. When
  /// `functional_quantum` is 0 (the default), all active wavefronts are
  /// drained in a single call. A non-zero quantum guarantees forward
  /// progress for inter-CU synchronization (e.g., spin-locks on global
  /// memory) by allowing other CUs' events to interleave.
  /// In CLOCKED mode (future), processes one pipeline cycle.
  /// @retval true More work remains; advance() should be called again.
  /// @retval false No more active wavefronts; CU is idle.
  virtual bool advance() = 0;

  /// @brief Signal that work has been dispatched; begin processing.
  ///
  /// @details Schedules an engine event that calls advance() repeatedly
  /// until all wavefronts are exhausted. Called by the command processor
  /// after dispatch_wf().
  virtual void activate() = 0;

  /// @brief Check whether this CU has no active wavefronts.
  /// @retval true No wavefronts are actively executing.
  /// @retval false At least one wavefront is active.
  virtual bool is_idle() const { return !has_active_wfs(); }

  /// @brief Register a callback invoked when this CU becomes idle.
  ///
  /// @details Called after all dispatched wavefronts complete execution.
  /// The command processor uses this to detect when all CUs are done.
  /// @param cb Callback to invoke when idle.
  void set_on_idle(std::function<void()> cb) { on_idle_ = std::move(cb); }

  /// @brief Set the execution plugin group (shared ownership).
  void set_plugin_group(std::shared_ptr<ExecutionPluginGroup> pg) { plugin_group_ = pg ? pg : ExecutionPluginGroup::empty_group(); }

  /// @brief Return the number of dispatched (active or halted) wavefront slots.
  /// @returns Count of non-idle wavefront slots.
  size_t num_wfs() const;

  /// @brief Return the total number of wavefront slots.
  /// @returns Total hardware wavefront slot count.
  uint32_t num_wf_slots() const { return config_.num_wf_slots; }

  /// @brief Access a wavefront slot by index (always non-null).
  /// @param idx Zero-based wavefront slot index.
  /// @returns Pointer to the wavefront slot.
  Wavefront *wf(size_t idx) { return wfs_[idx].get(); }

  /// @brief Access a wavefront slot by index (const, always non-null).
  /// @param idx Zero-based wavefront slot index.
  /// @returns Const pointer to the wavefront slot.
  const Wavefront *wf(size_t idx) const { return wfs_[idx].get(); }

  /// @brief Return the CU configuration.
  /// @returns Const reference to the CU configuration.
  const Config &config() const { return config_; }

  /// @brief Return the shared GPU memory.
  /// @returns Pointer to the GPU memory.
  GpuMemory *memory() const { return memory_; }

  /// @brief Return the L1 Scalar Cache (K$).
  L1ScalarCache &l1_scalar() { return l1_scalar_; }

  /// @brief Return the L1 Vector Cache (V$).
  L1VectorCache &l1_vector() { return l1_vector_; }

  /// @brief Return the shared L2 cache.
  L2Cache *l2() const { return l2_; }

  /// @brief Return the Local Data Share (LDS).
  Lds &lds() { return lds_; }

  /// @brief Flush all per-CU caches and the shared L2 to backing store.
  ///
  /// @details L1 V$ uses write-through, so flush just invalidates. L2 flushes
  /// all dirty lines to the backing MemoryInterface (MSC or HBM).
  /// Note: prefer flush_l1() + per-XCD L2 flush to avoid redundant L2 flushes
  /// when multiple CUs share the same L2.
  void flush_all() {
    util::Logger::vm([&](auto &os) {
      if (l1_vector_.store_count() > 0)
        os << std::format("CU {}@{} L1 stores: total={} active={} l2_writes={}", this->name(),
                          reinterpret_cast<uintptr_t>(this), l1_vector_.store_count(),
                          l1_vector_.store_active_count(), l1_vector_.store_l2_writes());
    });
    l1_scalar_.invalidate_all();
    l1_vector_.flush_all();
    l2_->flush_all();
  }

  /// @brief Flush only the per-CU L1 caches (invalidate, since L1 is write-through).
  void flush_l1() {
    l1_scalar_.invalidate_all();
    l1_vector_.flush_all();
  }

  /// @brief Set (or replace) the shared GPU memory pointer.
  ///
  /// Used by the config loader for deferred initialization.
  /// @param memory New GPU memory (not owned).
  void set_memory(GpuMemory *memory) { memory_ = memory; }

  /// @brief Set (or replace) the L2 cache pointer.
  ///
  /// Used by the config loader for deferred initialization.
  /// Also updates the L1 caches' backing store and global memory pipeline.
  /// @param l2 New L2 cache (not owned).
  void set_l2(L2Cache *l2) {
    l2_ = l2;
    l1_scalar_.set_l2(l2);
    l1_vector_.set_l2(l2);
    global_mem_pipeline_.set_l2(l2);
  }

  // Memory issue interface for instruction execute() bodies.
  //
  // These provide the public interface through which instruction execute()
  // methods issue memory operations. In FUNCTIONAL mode they perform
  // the memory access synchronously through the appropriate cache level.

  /// @brief Issue a scalar memory load through the L1 scalar cache.
  ///
  /// @param addr Dword-aligned scalar address (computed by smem_calculate_address).
  /// @param dst_sgpr Physical SGPR index to write the first loaded dword.
  /// @param dword_count Number of dwords to load (1, 2, 4, 8, or 16).
  /// @param mtype Memory type (default RW — Phase D fills in correct value).
  void issue_scalar_mem(uint64_t addr, uint32_t dst_sgpr, uint32_t dword_count,
                        Mtype mtype = Mtype::RW);

  /// @brief Issue a per-lane global/buffer memory load through the L1 vector cache.
  ///
  /// @param addrs Per-lane byte addresses (only active lanes are accessed).
  /// @param lane_mask Bitmask of active lanes.
  /// @param dst_vgpr Physical VGPR index to write the first loaded dword.
  /// @param dword_count Dwords per lane (1, 2, 3, or 4).
  /// @param mtype Memory type (default RW — Phase D fills in correct value).
  void issue_global_mem(const std::array<uint64_t, 64> &addrs, uint64_t lane_mask,
                        uint32_t dst_vgpr, uint32_t dword_count, Mtype mtype = Mtype::RW);

  /// @brief Issue a per-lane local (LDS) memory load.
  ///
  /// @param addrs Per-lane byte addresses into the LDS.
  /// @param lane_mask Bitmask of active lanes.
  /// @param dst_vgpr Physical VGPR index to write the first loaded dword.
  /// @param dword_count Dwords per lane (1 or 2).
  void issue_local_mem(const std::array<uint64_t, 64> &addrs, uint64_t lane_mask, uint32_t dst_vgpr,
                       uint32_t dword_count);

  /// @brief Return the ISA architecture.
  /// @returns Architecture enum value.
  rj_code_arch_t arch() const { return config_.arch; }

  /// @brief Return the wave size (lanes per wavefront, ISA-defined).
  /// @returns Lanes per wavefront.
  uint32_t wf_size() const { return wf_size_; }

  /// @brief Check whether any wavefront slot is actively executing.
  /// @retval true At least one wavefront is not halted.
  /// @retval false All wavefronts are halted.
  bool has_active_wfs() const {
    for (const auto &w : wfs_)
      if (!w->is_halted())
        return true;
    return false;
  }

  /// @brief Return the current round-robin scheduling index.
  /// @returns Index of the next wavefront slot to schedule.
  size_t next_wf_index() const { return next_wf_; }

  /// @brief Read a scalar register from the physical SGPR file.
  /// @param reg_idx Physical register index.
  /// @returns Register value.
  uint32_t read_sgpr(uint32_t reg_idx) const { return sgpr_file_[reg_idx]; }

  /// @brief Write a scalar register in the physical SGPR file.
  /// @param reg_idx Physical register index.
  /// @param val Value to write.
  void write_sgpr(uint32_t reg_idx, uint32_t val) { sgpr_file_[reg_idx] = val; }

  /// @brief Read a vector register lane from the physical VGPR file.
  /// @param reg_idx Physical register index.
  /// @param lane Lane index within the wavefront.
  /// @returns Lane value.
  virtual uint32_t read_vgpr(uint32_t reg_idx, uint32_t lane) const = 0;

  /// @brief Write a vector register lane in the physical VGPR file.
  /// @param reg_idx Physical register index.
  /// @param lane Lane index within the wavefront.
  /// @param val Value to write.
  virtual void write_vgpr(uint32_t reg_idx, uint32_t lane, uint32_t val) = 0;

  /// @brief Return a pointer to a wavefront's SGPR data in the physical file.
  /// @param base Base register index in the SGPR file.
  /// @returns Pointer to the contiguous SGPR data.
  const uint32_t *sgpr_data(uint32_t base) const { return &sgpr_file_[base]; }

  /// @brief Return a pointer to a wavefront's VGPR data in the physical file.
  /// @param base Base register index in the VGPR file.
  /// @returns Const pointer to the raw VGPR data.
  virtual const uint8_t *vgpr_data(uint32_t base) const = 0;

  /// @brief Return a mutable pointer to a wavefront's VGPR data.
  /// @param base Base register index in the VGPR file.
  /// @returns Mutable pointer to the raw VGPR data.
  virtual uint8_t *vgpr_data(uint32_t base) = 0;

  /// @brief Return the SGPR register file (for serialization).
  /// @returns Const reference to the SGPR register file.
  const simdojo::RegisterFile<uint32_t> &sgpr_file() const { return sgpr_file_; }

  /// @brief Return the decoder (for external decode if needed).
  /// @returns Const pointer to the ISA decoder.
  const Decoder *decoder() const { return decoder_.get(); }

  /// @brief Return the completer port (receives dispatch requests from CP).
  /// @returns Pointer to the completer port.
  simdojo::Port *cpl_port() { return cpl_; }

  /// @brief Return the requester port (sends requests to L2).
  /// @returns Pointer to the requester port.
  simdojo::Port *req_port() { return req_; }

  /// @brief Execute an instruction on a wavefront (ISA-specific dispatch).
  ///
  /// @warning NOT thread-safe. Must be called from the CU's event-loop
  /// thread or from single-threaded test contexts only.
  /// @param inst The decoded instruction.
  /// @param wf The wavefront executing the instruction.
  virtual void execute_instruction(Instruction *inst, Wavefront &wf) = 0;

  /// @brief Reset all wavefront slots to halted state.
  ///
  /// Frees all register allocations and resets every slot.  Used by the
  /// instruction execution test harness between instructions.
  /// @warning NOT thread-safe.  Must not be called while step() is running.
  void reset_all_wf();

protected:
  ComputeUnitCore(std::string name, const Config &config, GpuMemory *memory, L2Cache *l2,
                  uint32_t wf_size);

  /// @brief Allocate a contiguous block of VGPRs.
  /// @param count Number of VGPRs to allocate.
  /// @returns Base index of the allocated block, or -1 on failure.
  virtual int32_t allocate_vgprs(uint32_t count) = 0;

  /// @brief Free a VGPR allocation.
  /// @param base Base index returned by allocate_vgprs().
  virtual void free_vgprs(uint32_t base) = 0;

  /// @brief Count the number of free VGPR allocation blocks.
  virtual uint32_t free_vgpr_blocks() const = 0;

  /// @brief Tick all memory pipelines (called at the start of step).
  void tick_pipelines();

  /// @brief Route a memory instruction into the appropriate pipeline.
  /// @param inst The memory instruction (ownership transferred).
  /// @param wf The issuing wavefront.
  void route_memory_inst(Instruction *inst, Wavefront &wf);

  /// @brief Fire the on_idle callback if registered.
  void notify_idle() {
    if (on_idle_)
      on_idle_();
  }

  Config config_;
  GpuMemory *memory_;
  uint32_t wf_size_ = 0;
  std::unique_ptr<Decoder> decoder_;
  simdojo::RegisterFile<uint32_t> sgpr_file_{"sgpr"};
  std::vector<std::unique_ptr<Wavefront>> wfs_; ///< Pre-allocated wavefront slots.
  size_t next_wf_ = 0;

  L2Cache *l2_;
  L1ScalarCache l1_scalar_;
  L1VectorCache l1_vector_;
  Lds lds_;
  ScalarMemPipeline scalar_mem_pipeline_;
  GlobalMemPipeline global_mem_pipeline_;
  LocalMemPipeline local_mem_pipeline_;
  std::function<void()> on_idle_; ///< Callback invoked when CU becomes idle.
  std::shared_ptr<ExecutionPluginGroup> plugin_group_ = ExecutionPluginGroup::empty_group();
  simdojo::Port *cpl_ = nullptr;  ///< Completer port: dispatch activation from CP.
  simdojo::Port *req_ = nullptr;  ///< Requester port: L2 cache request (structural).
};

/// @brief Execution-mode-aware compute unit shell.
///
/// @details Adds event-driven activation on top of ComputeUnitCore.
///
/// In FUNCTIONAL mode, advance() executes instructions via step() up to the
/// configured `functional_quantum` limit (or unbounded if quantum is 0).
/// When the quantum is reached with work remaining, advance() returns true
/// and the work_event_ reschedules at `now + 1`, yielding to the simulation
/// event loop. This interleaving ensures forward progress when wavefronts
/// on different CUs synchronize via global memory (e.g., spin-locks,
/// semaphores). activate() schedules the initial work event. When
/// advance() returns false (no more work), the idle callback is notified
/// so the command processor can detect completion.
///
/// In CLOCKED mode (future), advance() will process one pipeline cycle and
/// activate() will resume the clock.
///
/// @tparam Mode Execution mode (FUNCTIONAL or CLOCKED).
template <simdojo::ExecMode Mode> class ExecComputeUnit : public ComputeUnitCore {
public:
  using ComputeUnitCore::ComputeUnitCore;

  /// @brief Execute work up to the quantum limit, then yield.
  ///
  /// @details In FUNCTIONAL mode, executes up to `functional_quantum`
  /// instructions (or all remaining if quantum is 0), retires halted
  /// wavefronts, and returns whether more work remains. If the CU is
  /// idle after execution, fires the on_idle callback.
  /// @retval true More work remains; work_event_ will reschedule.
  /// @retval false CU is idle; all wavefronts have been exhausted.
  bool advance() override {
    if constexpr (Mode == simdojo::ExecMode::FUNCTIONAL) {
      const uint32_t quantum = this->config_.functional_quantum;
      if (quantum == 0) {
        // Unbounded: drain all wavefronts in a single call.
        while (step()) {
        }
      } else {
        // Bounded: execute up to 'quantum' instructions, then yield
        // back to the event loop so other CUs can make progress.
        for (uint32_t i = 0; i < quantum && step(); ++i) {
        }
      }
    } else {
      /// @todo: Support CLOCKED pipeline cycle.
    }
    if (is_idle()) {
      notify_idle();
      return false;
    }
    return true;
  }

  /// @brief Run wavefronts on this CU.
  ///
  /// In unbounded functional mode (quantum == 0), advance directly — the CU
  /// will drain all wavefronts before returning. This avoids scheduling an
  /// engine event and waiting for the LBTS to advance, which can deadlock
  /// when the engine is in await_primaries mode (KFD driver).
  ///
  /// In bounded mode (quantum > 0), schedule a work event to yield between
  /// quanta, ensuring fair interleaving across CUs.
  void activate() override {
    if constexpr (Mode == simdojo::ExecMode::FUNCTIONAL) {
      if (this->config_.functional_quantum == 0) {
        advance();
        return;
      }
    }
    this->schedule_event(&work_event_, this->engine()->global_time() + 1);
  }

private:
  simdojo::Event work_event_{this, simdojo::EventType::TIMER_CALLBACK,
                             [this](simdojo::Tick now, simdojo::Message *) {
                               if (advance())
                                 this->schedule_event(&work_event_, now + 1);
                             }};
};

/// @brief ISA-parameterized compute unit owning the typed VGPR register file.
///
/// @details The Isa trait provides WF_SIZE which sets the VectorReg element
/// count, so each VGPR holds one uint32_t lane per wavefront thread.
/// Pre-allocates all wavefront slots as IsaWavefront<Isa> instances.
///
/// @tparam Mode Execution mode (FUNCTIONAL or CLOCKED).
/// @tparam Isa ISA traits struct satisfying the GpuIsa concept.
template <simdojo::ExecMode Mode, GpuIsa Isa>
class IsaExecComputeUnit : public ExecComputeUnit<Mode> {
public:
  using Vgpr = simdojo::VectorReg<Isa::WF_SIZE, uint32_t>;

  /// @brief Construct an ISA-parameterized compute unit.
  /// @param name Human-readable name (e.g., "cu0").
  /// @param config CU configuration parameters.
  /// @param memory Shared GPU memory (not owned).
  /// @param l2 Shared L2 cache (not owned).
  IsaExecComputeUnit(std::string name, const ComputeUnitCore::Config &config, GpuMemory *memory,
                     L2Cache *l2)
      : ExecComputeUnit<Mode>(std::move(name), config, memory, l2, Isa::WF_SIZE) {
    vgpr_file_.init(config.num_wf_slots * config.vgprs_per_wf, config.vgprs_per_wf);
    for (uint32_t i = 0; i < config.num_wf_slots; ++i)
      this->wfs_[i] = std::make_unique<IsaWavefront<Isa>>(*this, i);
  }

  /// @returns Lane value from the VGPR file.
  uint32_t read_vgpr(uint32_t reg_idx, uint32_t lane) const override {
    return vgpr_file_[reg_idx][lane];
  }

  /// @brief Write a value to the VGPR file.
  void write_vgpr(uint32_t reg_idx, uint32_t lane, uint32_t val) override {
    vgpr_file_[reg_idx][lane] = val;
  }

  /// @returns Const pointer to the raw VGPR data.
  const uint8_t *vgpr_data(uint32_t base) const override {
    return reinterpret_cast<const uint8_t *>(&vgpr_file_[base]);
  }

  /// @returns Mutable pointer to the raw VGPR data.
  uint8_t *vgpr_data(uint32_t base) override {
    return reinterpret_cast<uint8_t *>(&vgpr_file_[base]);
  }

  /// @brief Return the VGPR register file (typed, only on concrete CU).
  /// @returns Const reference to the VGPR register file.
  const simdojo::RegisterFile<Vgpr> &vgpr_file() const { return vgpr_file_; }

  /// @brief Return a mutable reference to the VGPR register file.
  /// @returns Mutable reference to the VGPR register file.
  simdojo::RegisterFile<Vgpr> &vgpr_file() { return vgpr_file_; }

protected:
  /// @returns Base index of the allocated VGPR block, or -1 on failure.
  int32_t allocate_vgprs(uint32_t count) override { return vgpr_file_.allocate(count); }

  /// @brief Return allocated VGPRs to the free pool.
  void free_vgprs(uint32_t base) override { vgpr_file_.free(base); }

  uint32_t free_vgpr_blocks() const override { return vgpr_file_.free_block_count(); }

  /// @brief Execute one instruction on the given wavefront.
  ///
  /// @brief Execute one instruction on the given wavefront via direct dispatch.
  void execute_instruction(Instruction *inst, Wavefront &wf) override { inst->execute(*inst, &wf); }

private:
  simdojo::RegisterFile<Vgpr> vgpr_file_{"vgpr"};
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_COMPUTE_UNIT_H_
