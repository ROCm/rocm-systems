// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file compute_unit.h
/// @brief AMDGPU compute unit hierarchy: ComputeUnitCore, ExecComputeUnit, and IsaExecComputeUnit.

#ifndef ROCJITSU_VM_AMDGPU_COMPUTE_UNIT_H_
#define ROCJITSU_VM_AMDGPU_COMPUTE_UNIT_H_

#include "rocjitsu/base/api.h"
#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
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
#include "rocjitsu/vm/amdgpu/wf_scheduler.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "simdojo/components/register_file.h"
#include "simdojo/components/vector_reg.h"
#include "util/bit.h"
#include "util/log.h"

#include "simdojo/sim/component.h"
#include "simdojo/sim/exec_mode.h"
#include "simdojo/sim/simulation.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

class CommandProcessor;

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
  static constexpr uint32_t kFunctionalQuantum = 16384;

  /// @brief Configuration for a compute unit.
  struct Config {
    rj_code_arch_t arch;   ///< ISA architecture (determines wave size, decoder).
    uint32_t num_wf_slots; ///< Number of hardware wavefront slots (contexts).
    uint32_t sgprs_per_wf; ///< Scalar GPRs per wavefront (allocation granularity).
    uint32_t vgprs_per_wf; ///< Vector GPRs per wavefront (allocation granularity).
    uint32_t lds_size_kb;  ///< Local Data Share size in kilobytes.
    uint32_t functional_quantum = kFunctionalQuantum; ///< Max instructions per functional slice.
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

  /// @brief Like retire_halted_wfs but without resetting the LDS allocator.
  void retire_halted_wfs_no_lds_reset();

  /// @brief Check whether this CU can accept an entire workgroup.
  ///
  /// @details Queries the number of free wavefront slots and register file
  /// blocks without modifying any state. The command processor calls this
  /// before dispatching to guarantee all-or-nothing workgroup placement.
  /// @param num_wfs Number of wavefronts in the workgroup.
  /// @param lds_bytes LDS bytes required by the workgroup.
  /// @returns true if the CU has enough free slots, registers, and LDS.
  bool can_accept_workgroup(uint32_t num_wfs, uint32_t lds_bytes = 0) const;

  /// @brief Execute up to kFunctionalQuantum instructions, then yield.
  virtual bool advance() = 0;

  uint32_t functional_quantum() const {
    return config_.functional_quantum == 0 ? UINT32_MAX : config_.functional_quantum;
  }

  /// @brief Execute up to one functional quantum of instructions on this CU.
  /// @returns true if any active wavefronts were stepped.
  bool run_quantum() {
    bool ran = false;
    for (uint32_t i = 0, quantum = functional_quantum(); i < quantum; ++i) {
      if (!has_active_wfs())
        break;
      ran = true;
      if (!step())
        break;
    }
    return ran;
  }

  /// @brief Execute a shard of this CU's wavefronts on one host thread.
  ///
  /// @details Intra-CU parallelism: when spare host threads exist (active CUs <
  /// dispatch threads), several threads co-run one CU, each owning the active
  /// wavefronts whose index satisfies (idx % shard_count == shard_idx). Each
  /// wavefront is touched by exactly one shard, so per-wavefront register files
  /// and state need no locking; the per-CU caches are kept thread-local to this
  /// call, and the shared memory subsystem (L1/LDS) is serialized by the per-CU
  /// memory mutex while VALU runs fully parallel. Does NOT retire wavefronts —
  /// the command processor retires after the fork-join joins.
  ///
  /// Workgroup barriers are resolved across shards via the per-workgroup
  /// generation barrier (see WgBarrier); call prepare_shard_barriers() once
  /// before launching the shard jobs for this CU.
  /// @returns true if any instruction was issued.
  bool run_quantum_shard(uint32_t shard_idx, uint32_t shard_count);

  /// @brief (Re)build the per-workgroup cross-thread barriers from the current
  /// resident wavefronts. Must be called single-threaded before the CU's shard
  /// jobs run; the map is then only read (its atomics mutated) during the
  /// parallel region.
  void prepare_shard_barriers();

  /// @brief Whether any resident workgroup has allocated LDS. Used as a safe
  /// proxy for "may contain cross-wavefront barriers" — such CUs stay on the
  /// single-threaded path.
  bool uses_lds() const { return next_lds_alloc_ > 0; }

  /// @brief Number of active (allocated) wavefronts resident on this CU.
  uint32_t active_wf_count() const {
    uint32_t n = 0;
    for (const auto &w : wfs_)
      if (w->sgpr_alloc().count > 0)
        ++n;
    return n;
  }

  /// @brief Toggle parallel-execution mode (enables per-CU memory locking).
  void set_parallel_exec(bool on) { parallel_exec_ = on; }

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

  /// @brief Set the command processor for WG completion notification.
  void set_command_processor(CommandProcessor *cp) { cp_ = cp; }

  /// @brief Register a new workgroup with its expected WF count.
  /// @details Called by the DispatchController when assigning a WG to this CU.
  /// Initializes the refcount so retire_halted_wfs() can detect WG completion.
  void begin_workgroup(uint32_t dispatch_id, uint32_t wg_id, uint32_t wf_count,
                       uint64_t entry_pc) {
    // Invalidate per-CU caches once per dispatch (when a new kernel's first WG
    // lands on this CU), not once per wavefront. The L1 scalar cache must be
    // flushed so wavefronts read fresh kernargs every dispatch. Doing this per
    // wavefront (in dispatch_wf) repeated the work ~wfs_per_dispatch times.
    if (dispatch_id != last_invalidated_dispatch_id_) {
      last_invalidated_dispatch_id_ = dispatch_id;
      l1_scalar_.invalidate_all();
      // The instruction-fetch and decoded-instruction caches hold the kernel's
      // code, keyed by PC. Code at a given PC does not change between dispatches
      // of the same kernel, so only drop these caches when the kernel's entry
      // PC actually changes. Workloads that re-dispatch one kernel many times
      // (e.g. per-layer GEMMs in a transformer) thus keep their decode cache
      // warm across dispatches instead of re-decoding on every CU every time.
      if (entry_pc != last_inst_kernel_pc_) {
        last_inst_kernel_pc_ = entry_pc;
        invalidate_inst_fetch_cache();
      }
    }
    active_wgs_[wg_key(dispatch_id, wg_id)] = wf_count;
  }

  /// @brief Called by Wavefront::halt() to decrement the WG refcount.
  /// @details When the refcount reaches zero, all WFs in the WG have halted
  /// and the CP is notified via notify_wg_complete.
  void release_wf(uint32_t dispatch_id, uint32_t wg_id);

  /// @brief Set the execution plugin group (shared ownership).
  void set_plugin_group(std::shared_ptr<ExecutionPluginGroup> pg) {
    auto empty = ExecutionPluginGroup::empty_group();
    plugin_group_ = pg ? std::move(pg) : empty;
    plugin_hooks_enabled_ = plugin_group_.get() != empty.get();
  }

  /// @brief Return the execution plugin group.
  ExecutionPluginGroup &plugin_group() { return *plugin_group_; }

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

  /// @brief Clear LDS contents (zero-fill).
  void clear_lds() { lds_.clear(); }

  /// @brief Allocate a per-WG LDS region and return its base offset.
  uint32_t allocate_lds(uint32_t size_bytes) {
    uint32_t base = next_lds_alloc_;
    uint32_t aligned = util::align_up(size_bytes, 256u);
    lds_.zero_range(base, aligned);
    next_lds_alloc_ += aligned;
    return base;
  }

  /// @brief Reset LDS allocation (called when all WFs retire).
  void reset_lds_alloc() { next_lds_alloc_ = 0; }

  /// @brief Flush all per-CU caches and the shared L2 to backing store.
  ///
  /// @details L1 V$ uses write-through, so flush just invalidates. L2 flushes
  /// all dirty lines to the backing MemoryInterface (MSC or HBM).
  /// Note: prefer flush_l1() + per-XCD L2 flush to avoid redundant L2 flushes
  /// when multiple CUs share the same L2.
  void flush_all(uint32_t vmid = 0) {
    util::Logger::vm([&](auto &os) {
      if (l1_vector_.store_count() > 0)
        os << std::format("CU {}@{} L1 stores: total={} active={} l2_writes={}", this->name(),
                          reinterpret_cast<uintptr_t>(this), l1_vector_.store_count(),
                          l1_vector_.store_active_count(), l1_vector_.store_l2_writes());
    });
    l1_scalar_.writeback_all(vmid);
    l1_scalar_.invalidate_all();
    l1_vector_.flush_all();
    l2_->flush_all(vmid);
  }

  void flush_l1(uint32_t vmid = 0) {
    l1_scalar_.writeback_all(vmid);
    l1_scalar_.invalidate_all();
    l1_vector_.flush_all();
  }

  /// @brief Set (or replace) the shared GPU memory pointer.
  ///
  /// Used by the config loader for deferred initialization.
  /// @param memory New GPU memory (not owned).
  void set_memory(GpuMemory *memory) {
    memory_ = memory;
    l1_vector_.set_memory(memory);
    l1_scalar_.set_memory(memory);
    invalidate_inst_fetch_cache();
  }

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

  /// @brief Set flat-address-space aperture boundaries (SPI programs these once per node).
  void set_apertures(uint64_t shared_base, uint64_t shared_limit, uint64_t private_base,
                     uint64_t private_limit) {
    shared_aperture_base_ = shared_base;
    shared_aperture_limit_ = shared_limit;
    private_aperture_base_ = private_base;
    private_aperture_limit_ = private_limit;
  }

  /// @brief Query SRAM ECC mode. When true, D16 loads zero unused VGPR bits.
  bool sram_ecc() const { return sram_ecc_; }

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

  bool has_active_wfs_for_process(uint32_t process_id) const {
    for (const auto &w : wfs_)
      if (!w->is_halted() && w->process_id() == process_id)
        return true;
    return false;
  }

  /// @brief Return the current round-robin scheduling index.
  /// @returns Index of the next wavefront slot to schedule.
  uint64_t cycle_count() const { return cycle_counter_; }

  /// @brief Read a scalar register from the physical SGPR file.
  /// @param reg_idx Physical register index.
  /// @returns Register value.
  // TODO(newling) consider cmake flag to build without plugins, this call
  // overhead might be non-negligible.
  bool plugin_hooks_enabled() const { return plugin_hooks_enabled_; }

  uint32_t read_sgpr(uint32_t reg_idx) const {
    if (plugin_hooks_enabled_) {
      if (auto *wf = sgpr_to_wave_[reg_idx]) {
        plugin_group_->onAmdgpuReadSgpr(wf, reg_idx);
      }
    }
    return sgpr_file_[reg_idx];
  }

  /// @brief Write a scalar register in the physical SGPR file.
  /// @param reg_idx Physical register index.
  /// @param val Value to write.
  void write_sgpr(uint32_t reg_idx, uint32_t val) { sgpr_file_[reg_idx] = val; }

  void notify_vgpr_read(const Wavefront *wf, uint32_t reg_idx, uint32_t lane_begin,
                        uint32_t lane_end, uint8_t byte_mask = 0xF) const {
    if (plugin_hooks_enabled_ && wf)
      plugin_group_->onAmdgpuReadVgprs(wf, reg_idx, lane_begin, lane_end, byte_mask);
  }

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

  /// @brief Write 32-bit lane values into one VGPR.
  /// @param reg_idx Physical register index.
  /// @param lane_mask Enabled lanes to write.
  /// @param src Source bytes, indexed as src[lane * src_stride].
  /// @param src_stride Byte stride between source lane values.
  virtual void write_vgpr_lanes32(uint32_t reg_idx, uint64_t lane_mask, const uint8_t *src,
                                  uint32_t src_stride) = 0;

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

  /// @brief Number of physical VGPR registers in one allocation block.
  virtual uint32_t vgpr_allocation_block_size() const = 0;

  /// @brief Typed view of a single VGPR as the file's @c simdojo::VectorReg.
  /// @details The abstract CU exposes the VGPR file only as a byte pointer
  /// (@c vgpr_data), which erases the wavefront-size template parameter. The
  /// file actually stores @c simdojo::VectorReg<N,uint32_t>, so this recovers
  /// the typed register with the design's single localized @c reinterpret_cast.
  /// The @c static_assert pins @c VectorReg<N> to @c N contiguous @c uint32_t
  /// (no padding / vtable) so the byte view and the typed view coincide.
  template <size_t N> simdojo::VectorReg<N, uint32_t> &vgpr_reg(uint32_t base) {
    static_assert(sizeof(simdojo::VectorReg<N, uint32_t>) == N * sizeof(uint32_t),
                  "VectorReg must be layout-compatible with raw lane storage");
    return *reinterpret_cast<simdojo::VectorReg<N, uint32_t> *>(vgpr_data(base));
  }
  template <size_t N> const simdojo::VectorReg<N, uint32_t> &vgpr_reg(uint32_t base) const {
    static_assert(sizeof(simdojo::VectorReg<N, uint32_t>) == N * sizeof(uint32_t),
                  "VectorReg must be layout-compatible with raw lane storage");
    return *reinterpret_cast<const simdojo::VectorReg<N, uint32_t> *>(vgpr_data(base));
  }

  const uint32_t *vgpr_lanes32(uint32_t reg_idx) const {
    return reinterpret_cast<const uint32_t *>(vgpr_data(reg_idx));
  }

  uint32_t *vgpr_lanes32(uint32_t reg_idx) {
    return reinterpret_cast<uint32_t *>(vgpr_data(reg_idx));
  }

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

  /// @brief Update wavefront states (WAITCNT, BARRIER, ENDING transitions).
  void update_wf_states();

  /// @brief Fetch, decode, execute one instruction from the given wavefront.
  void issue_instruction(Wavefront *wf);

  /// @brief Fetch the 16-byte decode window for the given PC.
  void fetch_instruction_words(uint64_t pc, uint32_t vmid, uint32_t words[4]);

  /// @brief Drop cached instruction bytes after image or dispatch changes.
  void invalidate_inst_fetch_cache();

  /// @brief Tick all memory pipelines (called at the start of step in clocked mode).
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
  bool sram_ecc_ = false;
  std::unique_ptr<Decoder> decoder_;
  simdojo::RegisterFile<uint32_t> sgpr_file_{"sgpr"};
  std::vector<std::unique_ptr<Wavefront>> wfs_; ///< Pre-allocated wavefront slots.
  std::unique_ptr<WavefrontScheduler> scheduler_ = std::make_unique<OldestFirstScheduler>();
  uint64_t cycle_counter_ = 0;

  L2Cache *l2_;
  L1ScalarCache l1_scalar_;
  L1VectorCache l1_vector_;
  Lds lds_;
  uint32_t next_lds_alloc_ = 0; ///< Next free LDS offset for per-WG allocation.
  ScalarMemPipeline scalar_mem_pipeline_;
  GlobalMemPipeline global_mem_pipeline_;
  LocalMemPipeline local_mem_pipeline_;
  std::function<void()> on_idle_; ///< Callback invoked when CU becomes idle.
  CommandProcessor *cp_ = nullptr;

  static uint64_t wg_key(uint32_t dispatch_id, uint32_t wg_id) {
    return (uint64_t(dispatch_id) << 32) | wg_id;
  }
  std::unordered_map<uint64_t, uint32_t> active_wgs_;

  // Cross-thread workgroup barrier (intra-CU parallel path). When a workgroup's
  // wavefronts are sharded across host threads, s_barrier can no longer be
  // resolved by the single-threaded update_wf_states() scan. Each workgroup gets
  // a generation barrier: a wavefront arriving at the barrier decrements
  // not_yet_here; the one that drives it to zero advances generation (the
  // release), and every shard observes the release by polling generation. Fully
  // cooperative — no thread blocks, so a workgroup whose wavefronts span shards
  // cannot deadlock. live tracks non-halted participants so a wavefront that
  // exits (s_endpgm) before its siblings reach the barrier still lets the round
  // complete.
  struct WgBarrier {
    std::atomic<uint32_t> live{0};       // non-halted wavefronts in the workgroup
    std::atomic<uint32_t> not_yet_here{0}; // live wavefronts not yet at this barrier
    std::atomic<uint32_t> generation{0}; // bumped on each barrier release
  };
  std::unordered_map<uint64_t, std::unique_ptr<WgBarrier>> shard_barriers_;

  uint64_t shared_aperture_base_ = 0;
  uint64_t shared_aperture_limit_ = 0;
  uint64_t private_aperture_base_ = 0;
  uint64_t private_aperture_limit_ = 0;

  std::shared_ptr<ExecutionPluginGroup> plugin_group_ = ExecutionPluginGroup::empty_group();
  bool plugin_hooks_enabled_ = false;

  struct InstFetchCacheLine {
    uint64_t pc = 0;
    uint32_t vmid = 0;
    std::array<uint32_t, 4> words{};
    bool valid = false;
  };
  static constexpr size_t kInstFetchCacheLines = 1024;
  std::array<InstFetchCacheLine, kInstFetchCacheLines> inst_fetch_cache_{};

  // Per-CU decoded-instruction cache. decoder_->decode() heap-allocates an
  // Instruction per call, and a hot loop re-decodes the same static instruction
  // every iteration. Caching the decoded form (keyed by PC) lets repeated
  // executions skip the decode + alloc/free entirely. Only non-memory, non-
  // control-flow instructions are cached: they carry no per-execution state and
  // are immutable during execute(). Memory ops (per-exec DynamicInstState owned
  // by the memory pipeline) and the s_setpc/s_swappc early-return path are never
  // cached. Per-CU and accessed single-threaded (one thread runs a CU per
  // fork-join), so no synchronization is needed.
  struct DecodeCacheLine {
    uint64_t pc = 0;
    uint32_t vmid = 0;
    std::unique_ptr<Instruction> inst; // non-memory decoded instruction, or null
  };
  std::array<DecodeCacheLine, kInstFetchCacheLines> decode_cache_{};

  // Intra-CU parallel variants: execute using caller-supplied (thread-local)
  // fetch/decode caches instead of the per-CU members, so several host threads
  // can issue instructions for different wavefronts of this CU concurrently
  // without racing on the shared caches. The single-argument overloads above
  // delegate here with the per-CU caches for the serial path.
  void issue_instruction(Wavefront *wf, InstFetchCacheLine *fetch_cache,
                         DecodeCacheLine *decode_cache);
  void fetch_instruction_words(uint64_t pc, uint32_t vmid, uint32_t words[4],
                               InstFetchCacheLine *fetch_cache);

  // Cross-thread workgroup-barrier helpers (intra-CU parallel path).
  // arrive: a wavefront reached s_barrier. depart: a wavefront exited (s_endpgm)
  // before the pending barrier. release: drive the round to completion.
  void barrier_arrive(WgBarrier &b, Wavefront &wf);
  void barrier_depart(WgBarrier &b);
  static void barrier_release(WgBarrier &b, uint32_t completed_gen);

  // Serializes the shared per-CU memory subsystem (L1 V$/S$, LDS) while multiple
  // threads co-run this CU's wavefronts. VALU touches only per-wavefront state
  // and runs lock-free; only memory ops (route_memory_inst) take this lock.
  // Atomics already serialize inside L2. Unused on the single-threaded path.
  std::mutex parallel_mem_mutex_;
  bool parallel_exec_ = false;

  // Dispatch id whose arrival last triggered per-CU cache invalidation, so the
  // invalidation happens once per dispatch (in begin_workgroup) rather than per
  // wavefront. UINT32_MAX = none yet.
  uint32_t last_invalidated_dispatch_id_ = UINT32_MAX;

  // PC of the kernel whose code currently populates the instruction-fetch and
  // decoded-instruction caches. Used to skip invalidating those caches when a
  // new dispatch runs the same kernel (same entry PC) as the previous one.
  uint64_t last_inst_kernel_pc_ = UINT64_MAX;

  /// Reverse lookup: physical SGPR index -> owning wavefront (for race detection).
  /// Populated at dispatch_wf time. Null entries mean "not allocated".
  std::vector<Wavefront *> sgpr_to_wave_;
  /// Populated by the ISA-specific subclass (which owns the VGPR file).
  virtual void fill_vgpr_to_wave(uint32_t /*base*/, uint32_t /*count*/, Wavefront * /*wf*/) {}
  simdojo::Port *cpl_ = nullptr; ///< Completer port: dispatch activation from CP.
  simdojo::Port *req_ = nullptr; ///< Requester port: L2 cache request (structural).
  uint64_t step_count_ = 0;
};

/// @brief Execution-mode-aware compute unit shell.
///
/// @details Adds event-driven activation on top of ComputeUnitCore.
///
/// In FUNCTIONAL mode, advance() executes up to kFunctionalQuantum
/// instructions, then yields to the simulation event loop. This
/// interleaving ensures forward progress when wavefronts on different
/// CUs synchronize via global memory (e.g., spin-locks, semaphores).
///
/// @tparam Mode Execution mode (FUNCTIONAL or CLOCKED).
template <simdojo::ExecMode Mode> class ExecComputeUnit : public ComputeUnitCore {
public:
  using ComputeUnitCore::ComputeUnitCore;

  /// @brief Execute work up to the quantum limit, then yield.
  bool advance() override {
    if constexpr (Mode == simdojo::ExecMode::FUNCTIONAL) {
      for (uint32_t i = 0, quantum = this->functional_quantum(); i < quantum && step(); ++i) {
      }
    } else {
      /// @todo: Support CLOCKED pipeline cycle.
    }
    if (is_idle()) {
      notify_idle();
      return !is_idle();
    }
    return true;
  }

  /// @brief Schedule CU execution via the event loop.
  void activate() override {
    auto now = this->engine()->context(this->partition_id()).current_tick();
    this->schedule_event(&work_event_, now + 1);
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
    static_assert(!HasAccVgpr<Isa> || Isa::MAX_VGPRS_PER_WF == ACC_VGPR_OFFSET,
                  "AccVGPR allocation base must match execution-side addressing");
    // AccVGPR operands are addressed after the normal VGPR bank in the same
    // physical file, so acc0 lives at base + ACC_VGPR_OFFSET.
    constexpr uint32_t accvgpr_physical_base = ACC_VGPR_OFFSET;
    constexpr uint32_t accvgpr_physical_limit =
        Isa::MAX_ACC_VGPRS_PER_WF == 0 ? 0 : accvgpr_physical_base + Isa::MAX_ACC_VGPRS_PER_WF;
    vgprs_per_block_ = std::max(config.vgprs_per_wf, accvgpr_physical_limit);
    vgpr_file_.init(config.num_wf_slots * vgprs_per_block_, vgprs_per_block_);
    vgpr_to_wave_.resize(config.num_wf_slots * vgprs_per_block_, nullptr);
    for (uint32_t i = 0; i < config.num_wf_slots; ++i)
      this->wfs_[i] = std::make_unique<IsaWavefront<Isa>>(*this, i);
    this->sram_ecc_ = Isa::SRAM_ECC;
  }

  /// @returns Lane value from the VGPR file.
  uint32_t read_vgpr(uint32_t reg_idx, uint32_t lane) const override {
    if (this->plugin_hooks_enabled_) {
      if (auto *wf = vgpr_to_wave_[reg_idx]) {
        this->plugin_group_->onAmdgpuReadVgprs(wf, reg_idx, lane, lane + 1);
      }
    }
    return vgpr_file_[reg_idx][lane];
  }

  void fill_vgpr_to_wave(uint32_t base, uint32_t count, Wavefront *wf) override {
    std::fill(vgpr_to_wave_.begin() + base, vgpr_to_wave_.begin() + base + count, wf);
  }

  /// @brief Write a value to the VGPR file.
  void write_vgpr(uint32_t reg_idx, uint32_t lane, uint32_t val) override {
    vgpr_file_[reg_idx][lane] = val;
  }

  void write_vgpr_lanes32(uint32_t reg_idx, uint64_t lane_mask, const uint8_t *src,
                          uint32_t src_stride) override {
    auto &reg = vgpr_file_[reg_idx];
    constexpr uint64_t full_lane_mask =
        Isa::WF_SIZE >= 64 ? ~uint64_t{0} : ((uint64_t{1} << Isa::WF_SIZE) - 1);
    if (lane_mask == full_lane_mask && src_stride == sizeof(uint32_t)) {
      std::memcpy(&reg[0], src, Isa::WF_SIZE * sizeof(uint32_t));
      return;
    }
    for (uint32_t lane = 0; lane < Isa::WF_SIZE; ++lane) {
      if (!(lane_mask & (uint64_t{1} << lane)))
        continue;
      std::memcpy(&reg[lane], src + lane * src_stride, sizeof(uint32_t));
    }
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
  int32_t allocate_vgprs(uint32_t count) override { return vgpr_file_.allocate(count, false); }

  /// @brief Return allocated VGPRs to the free pool.
  void free_vgprs(uint32_t base) override { vgpr_file_.free(base); }

  uint32_t free_vgpr_blocks() const override { return vgpr_file_.free_block_count(); }

public:
  uint32_t vgpr_allocation_block_size() const override { return vgprs_per_block_; }

protected:
  /// @brief Execute one instruction on the given wavefront.
  ///
  /// @brief Execute one instruction on the given wavefront via direct dispatch.
  void execute_instruction(Instruction *inst, Wavefront &wf) override { inst->execute(*inst, &wf); }

private:
  simdojo::RegisterFile<Vgpr> vgpr_file_{"vgpr"};
  std::vector<Wavefront *> vgpr_to_wave_; ///< Physical VGPR → owning wavefront.
  uint32_t vgprs_per_block_ = 0;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_COMPUTE_UNIT_H_
