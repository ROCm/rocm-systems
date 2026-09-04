// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file compute_unit.h
/// @brief AMDGPU compute unit hierarchy: ComputeUnitCore, ExecComputeUnit, and IsaExecComputeUnit.

#ifndef ROCJITSU_VM_AMDGPU_COMPUTE_UNIT_H_
#define ROCJITSU_VM_AMDGPU_COMPUTE_UNIT_H_

#include "rocjitsu/base/api.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/cluster_lds_multicast.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/instruction_cache.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/mtype.h"
#include "rocjitsu/vm/amdgpu/physical_register_resources.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/amdgpu/wf_scheduler.h"
#include "rocjitsu/vm/amdgpu/workgroup_key.h"
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
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace test {
class ComputeUnitTestAccess;
}
namespace amdgpu {

class CommandProcessor;

inline constexpr int32_t kWorkgroupBarrierId = -1;
inline constexpr int32_t kWorkgroupTrapBarrierId = -2;
inline constexpr int32_t kClusterBarrierId = -3;
inline constexpr int32_t kClusterTrapBarrierId = -4;

inline constexpr uint8_t kNamedBarrierBit = 0;
inline constexpr uint8_t kWorkgroupBarrierBit = 1;
inline constexpr uint8_t kWorkgroupTrapBarrierBit = 2;
inline constexpr uint8_t kClusterBarrierBit = 3;
inline constexpr uint8_t kClusterTrapBarrierBit = 4;

/// @brief Base AMDGPU compute unit that owns wavefront slots and register files.
///
/// @details Owns the physical SGPR and VGPR register files and a fixed array of
/// pre-allocated wavefront slots. Each wavefront holds a permanent reference
/// back to this CU and its slot index (wf_id).
///
/// dispatch_wf() finds the first idle slot, allocates registers, and
/// activates it. When a wavefront reaches s_endpgm it halts: free_wavefront_resources()
/// frees its register allocations and resets the slot for reuse, mirroring how real
/// hardware reclaims a wave's resources at termination (there is no separate lazy
/// retirement pass).
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
  static constexpr uint32_t kFunctionalQuantum = 1024;
  static constexpr uint32_t kDebugFunctionalQuantum = 64;
  static constexpr uint32_t kMaxNamedBarriers = 16;

  /// @brief Configuration for a compute unit.
  struct Config {
    rj_code_arch_t arch;   ///< ISA architecture (determines wave size, decoder).
    uint32_t num_wf_slots; ///< Number of hardware wavefront slots (contexts).
    uint32_t sgprs_per_wf; ///< Scalar GPRs per wavefront (allocation granularity).
    uint32_t vgprs_per_wf; ///< Vector GPRs per wavefront (allocation granularity).
    uint32_t lds_size_kb;  ///< Local Data Share size in kilobytes.
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
  /// @param num_sgprs Number of scalar registers to allocate.
  /// @param num_vgprs Number of vector registers to allocate.
  /// @returns Pointer to the activated wavefront, or nullptr if no free slot
  ///          or insufficient register space.
  Wavefront *dispatch_wf(uint32_t wg_id, uint64_t pc, uint32_t num_sgprs, uint32_t num_vgprs,
                         uint32_t wave_size = 0);

  /// @brief Activate a wave with an explicit ordinary/accumulator VGPR split.
  Wavefront *dispatch_wf(uint32_t wg_id, uint64_t pc, uint32_t num_sgprs, WaveVgprAllocation vgprs,
                         uint32_t wave_size = 0);

  /// @brief Activate a specific idle wavefront slot.
  /// @details Used by checkpoint restoration when hardware slot identity is
  /// execution state. Returns nullptr when the requested slot is invalid or busy.
  Wavefront *dispatch_wf_at(uint32_t wf_id, uint32_t wg_id, uint64_t pc, uint32_t num_sgprs,
                            uint32_t num_vgprs, uint32_t wave_size = 0);

  /// @brief Activate a specific slot with an explicit VGPR allocation split.
  Wavefront *dispatch_wf_at(uint32_t wf_id, uint32_t wg_id, uint64_t pc, uint32_t num_sgprs,
                            WaveVgprAllocation vgprs, uint32_t wave_size = 0);

  /// @brief Advance every RUNNING wavefront by one instruction, then report
  /// residency.
  /// @details Issues one instruction to each wavefront currently in the RUNNING
  /// state (waves stalled on WAITCNT/BARRIER, or halted, issue nothing this tick).
  /// @retval true At least one wavefront is still resident (active), regardless of
  ///         whether any instruction issued this call — so a fully WAITCNT/BARRIER-
  ///         stalled CU still returns true.
  /// @retval false No wavefronts remain resident (the CU is idle).
  bool step() override;

  /// @brief Free a halted wavefront's register allocations and reset its slot.
  /// @details Called from Wavefront::halt() at s_endpgm so a terminated wave
  /// releases its SGPR/VGPR blocks immediately, exactly as hardware reclaims
  /// resources at wave termination. LDS is per-workgroup and reclaimed separately
  /// via maybe_reset_lds_alloc() once the whole workgroup completes.
  void free_wavefront_resources(Wavefront &wf);

  /// @brief Reset the per-WG LDS bump allocator once the CU has fully drained.
  /// @details No-op while any wavefront is resident or a cluster pin is held (peer
  /// cluster workgroups may still multicast into LDS after the source wave halts).
  void maybe_reset_lds_alloc();

  /// @brief Check whether this CU can accept an entire workgroup.
  ///
  /// @details Queries free wavefront slots, physical SIMD VRF/SRF capacity,
  /// and LDS without modifying any state. The command processor calls this
  /// before dispatching to guarantee all-or-nothing workgroup placement.
  /// @param num_wfs Number of wavefronts in the workgroup.
  /// @param num_sgprs Descriptor-derived SGPR allocation per wavefront.
  /// @param num_vgprs Descriptor-derived VGPR allocation per wavefront.
  /// @param wave_size Architectural wavefront width, or zero for the ISA default.
  /// @param lds_bytes LDS bytes required by the workgroup.
  /// @returns true if the CU has enough free slots, registers, and LDS.
  bool can_accept_workgroup(uint32_t num_wfs, uint32_t num_sgprs, uint32_t num_vgprs,
                            uint32_t wave_size, uint32_t lds_bytes = 0) const;

  /// @brief Execute up to kFunctionalQuantum instructions, then yield.
  virtual bool execute_quantum() = 0;

  /// @brief End the current functional-mode quantum after this instruction.
  ///
  /// Used by wait-like instructions such as s_sleep so other simulated
  /// components can publish the state on which the wavefront is polling.
  void request_functional_yield() { functional_yield_requested_ = true; }

  /// @brief Schedule the tick event if the CU is not already executing.
  /// Called from dispatch_wf(), the cpl_ port handler, and single-threaded VM
  /// initialization after engine creation but before simulation workers start.
  virtual void schedule_work() = 0;

  /// @brief Thread-safe scheduling for debugger resume from an ioctl thread.
  virtual void schedule_work_async() = 0;

  /// @brief Check whether this CU has no runnable wavefronts.
  /// @retval true No wavefront can currently execute.
  /// @retval false At least one wavefront can execute.
  /// @warning NOT thread-safe (see has_runnable_wfs()): engine-thread only.
  virtual bool is_idle() const { return !has_runnable_wfs(); }

  /// @brief Register a callback invoked when this CU becomes idle.
  ///
  /// @details Called after all dispatched wavefronts complete execution.
  /// The command processor uses this to detect when all CUs are done.
  /// @param cb Callback to invoke when idle.
  void set_on_idle(std::function<void()> cb) { on_idle_ = std::move(cb); }

  struct TrapHandlerConfig {
    uint64_t tba = 0;
    uint64_t tma = 0;
    bool debug_enabled = false;
  };

  /// @brief Resolve the KFD trap handler for a wave's process and GPU.
  using TrapHandlerResolver = std::function<std::optional<TrapHandlerConfig>(const Wavefront &wf)>;
  void set_trap_handler_resolver(TrapHandlerResolver cb) { trap_handler_resolver_ = std::move(cb); }

  /// @brief Handle an architected scalar message issued by trap-handler code.
  using SendmsgHandler = std::function<bool(Wavefront &wf, uint32_t message)>;
  void set_sendmsg_handler(SendmsgHandler cb) { sendmsg_handler_ = std::move(cb); }
  bool handle_sendmsg(Wavefront &wf, uint32_t message) {
    return sendmsg_handler_ && sendmsg_handler_(wf, message);
  }

  /// @brief Notify KFD after configured TBA code returns with STATUS.HALT.
  using TrapCompletionHandler = std::function<void(Wavefront &wf)>;
  void set_trap_completion_handler(TrapCompletionHandler cb) {
    trap_completion_handler_ = std::move(cb);
  }
  void notify_trap_complete(Wavefront &wf) {
    if (trap_completion_handler_)
      trap_completion_handler_(wf);
  }

  /// @brief Callback after a single-stepped wave executes one instruction.
  using SingleStepHandler = std::function<bool(Wavefront &wf)>;

  void set_single_step_handler(SingleStepHandler cb) { single_step_handler_ = std::move(cb); }

  using WatchpointHandler = std::function<bool(Wavefront &wf, uint64_t address, uint32_t bytes,
                                               bool is_write, bool is_atomic)>;

  void set_watchpoint_handler(WatchpointHandler cb) { watchpoint_handler_ = std::move(cb); }

  using IllegalInstHandler = std::function<bool(Wavefront &wf)>;
  void set_illegal_inst_handler(IllegalInstHandler cb) { illegal_inst_handler_ = std::move(cb); }

  using MemoryViolationHandler =
      std::function<bool(Wavefront &wf, uint64_t address, bool is_write)>;
  void set_memory_violation_handler(MemoryViolationHandler cb) {
    memory_violation_handler_ = std::move(cb);
  }
  using AluExceptionHandler = std::function<bool(Wavefront &wf)>;
  void set_alu_exception_handler(AluExceptionHandler cb) { alu_exception_handler_ = std::move(cb); }
  /// @brief Mark a debug session as attached to or detached from this CU.
  /// @details Every transition publishes an I$ invalidation. A debugger writes
  /// breakpoints straight into code memory with none of the maintenance that
  /// invalidates the I$, and it can do so without any wave on this CU issuing
  /// during the session -- attach, plant a breakpoint on a stopped wave, detach.
  /// The invalidation is only published here, not applied: this runs on the
  /// ioctl thread, and the I$ is lock-free precisely because the CU's own thread
  /// is its sole accessor. That thread consumes it at its next fetch.
  void set_debug_active(bool active) {
    if (debug_active_.exchange(active, std::memory_order_relaxed) != active)
      inst_cache_debug_epoch_.fetch_add(1, std::memory_order_release);
  }
  bool debug_active() const { return debug_active_.load(std::memory_order_relaxed); }

  /// @brief Set the command processor for WG completion notification.
  void set_command_processor(CommandProcessor *cp) { cp_ = cp; }

  /// @brief Return the command processor that owns this CU's dispatch stream.
  CommandProcessor *command_processor() { return cp_; }

  /// @brief Override the cluster LDS multicast backend.
  ///
  /// @details Passing nullptr restores the immediate functional backend. Timed
  /// models can install a shared fabric object here without changing the ISA
  /// execution path that produces multicast transactions.
  void set_cluster_lds_multicast_engine(ClusterLdsMulticastEngine *engine) {
    cluster_lds_multicast_engine_ = engine ? engine : &default_cluster_lds_multicast_engine_;
  }

  /// @brief Return the active cluster LDS multicast backend.
  ClusterLdsMulticastEngine &cluster_lds_multicast_engine() {
    return *cluster_lds_multicast_engine_;
  }

  /// @brief Register a new workgroup with its expected WF count.
  /// @details Called by the DispatchController when assigning a WG to this CU.
  /// Initializes the refcount so release_wf() can detect WG completion.
  void begin_workgroup(uint32_t dispatch_id, uint32_t wg_id, uint32_t wf_count,
                       uint32_t num_named_barriers = 0);

  /// @brief Initialize a named barrier's member count and clear its signals.
  void named_barrier_init(Wavefront &wf, int32_t barrier_id, uint32_t member_count);

  /// @brief Associate a wave with one named barrier.
  void named_barrier_join(Wavefront &wf, int32_t barrier_id);

  /// @brief Signal a split-barrier domain and return whether this was its first signal.
  bool barrier_signal(Wavefront &wf, int32_t barrier_id, uint32_t member_count);

  /// @brief Return the packed architectural state of a split-barrier domain.
  uint32_t barrier_state(const Wavefront &wf, int32_t barrier_id) const;

  /// @brief Wait on the completion bit selected by a split-barrier ID.
  void barrier_wait(Wavefront &wf, int32_t barrier_id);

  /// @brief Leave the wave's currently joined named barrier.
  bool named_barrier_leave(Wavefront &wf);

  /// @brief Called by Wavefront::halt() to decrement the WG refcount.
  /// @details When the refcount reaches zero, all WFs in the WG have halted
  /// and the CP is notified via notify_wg_complete.
  void release_wf(uint32_t dispatch_id, uint32_t wg_id,
                  Wavefront::CpCompletionNotice notice = Wavefront::CpCompletionNotice::Send);

  /// @brief Roll back a committed-but-never-run workgroup on a dispatch error.
  /// @details Used to unwind an already-committed cluster peer when a later peer in
  /// the same clustered dispatch fails. Frees the WG's resident waves and drops its
  /// refcount WITHOUT firing the completion hook or CP notify (the WG never executed),
  /// then reclaims LDS if the CU is now idle and unpinned. The caller is responsible
  /// for unpinning any CP-side cluster LDS pin.
  void abort_workgroup(uint32_t dispatch_id, uint32_t wg_id);

  /// @brief Set the execution plugin group (shared ownership).
  void set_plugin_group(std::shared_ptr<ExecutionPluginGroup> pg) {
    plugin_group_ = pg ? pg : ExecutionPluginGroup::empty_group();
    observes_sgpr_reads_ = plugin_group_->observes_sgpr_reads();
  }

  /// @brief Return the execution plugin group.
  ExecutionPluginGroup &plugin_group() { return *plugin_group_; }

  /// @brief Return the number of resident (not-yet-halted) wavefront slots.
  /// @details A wave frees its resources and its slot at s_endpgm, so halted
  /// waves are not counted. Equivalent to the number of active wavefronts.
  /// @returns Count of resident wavefront slots.
  size_t num_wfs() const;

  /// @brief Return the total number of wavefront slots.
  /// @returns Total hardware wavefront slot count.
  uint32_t num_wf_slots() const { return config_.num_wf_slots; }

  /// @brief Record this CU's physical location within its XCC.
  /// @param shader_engine_id Zero-based shader-engine index within the XCC.
  /// @param cu_index Zero-based CU index within the shader engine.
  void set_shader_engine_location(uint32_t shader_engine_id, uint32_t cu_index) {
    shader_engine_id_ = shader_engine_id;
    scratch_scoreboard_base_ = cu_index * config_.num_wf_slots;
  }

  /// @brief Return this CU's physical shader-engine index.
  uint32_t shader_engine_id() const { return shader_engine_id_; }

  /// @brief Return the first scratch scoreboard slot owned by this CU.
  uint32_t scratch_scoreboard_base() const { return scratch_scoreboard_base_; }

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

  /// @brief Return the per-CU instruction cache (I$).
  InstructionCache &instruction_cache() { return inst_cache_; }

  /// @brief Return the shared L2 cache.
  L2Cache *l2() const { return l2_; }

  /// @brief Return the Local Data Share (LDS).
  Lds &lds() { return lds_; }
  const Lds &lds() const { return lds_; }

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

  /// @brief Reset LDS allocation when no resident waves or pinned clusters remain.
  void reset_lds_alloc() { next_lds_alloc_ = 0; }

  /// @brief Hold LDS allocation state while a workgroup cluster is resident.
  ///
  /// @details Cluster multicast can target peer workgroups after the source WG
  /// halts. Keep the per-CU LDS allocator pinned until the whole peer cluster
  /// completes so a later WG cannot reuse LDS while peer multicast writes are
  /// still possible, but larger dispatches can reclaim LDS between clusters.
  void pin_lds_until_cluster_retired(uint64_t cluster_key) {
    lds_pinned_clusters_.insert(cluster_key);
  }

  /// @brief Release the LDS allocation pin for a retired workgroup cluster.
  void unpin_lds_for_cluster(uint64_t cluster_key) { lds_pinned_clusters_.erase(cluster_key); }

  /// @brief Return true while any cluster can still receive multicast LDS writes.
  bool lds_allocation_pinned() const { return !lds_pinned_clusters_.empty(); }

  /// @brief Flush all per-CU caches and the shared L2 to backing store.
  ///
  /// @details Both L1 caches use write-through, so flush just invalidates. L2
  /// flushes all dirty lines to the backing MemoryInterface (MSC or HBM).
  /// Note: prefer flush_l1() + per-XCD L2 flush to avoid redundant L2 flushes
  /// when multiple CUs share the same L2.
  void flush_all(uint32_t vmid = 0) {
    util::Logger::vm([&](auto &os) {
      if (l1_vector_.store_count() > 0)
        os << std::format("CU {}@{} L1 stores: total={} active={} l2_writes={}", this->name(),
                          reinterpret_cast<uintptr_t>(this), l1_vector_.store_count(),
                          l1_vector_.store_active_count(), l1_vector_.store_l2_writes());
    });
    l1_scalar_.invalidate_all();
    l1_vector_.flush_all();
    inst_cache_.invalidate_all();
    l2_->flush_all(vmid);
  }

  void flush_l1(uint32_t vmid = 0) {
    (void)vmid;
    l1_scalar_.invalidate_all();
    l1_vector_.flush_all();
    inst_cache_.invalidate_all();
  }

  /// @brief Set (or replace) the shared GPU memory pointer.
  ///
  /// Used by the config loader for deferred initialization.
  /// @param memory New GPU memory (not owned).
  void set_memory(GpuMemory *memory) {
    memory_ = memory;
    l1_vector_.set_memory(memory);
    l1_scalar_.set_memory(memory);
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
  /// @warning NOT thread-safe: reads the non-atomic per-wave state_. Safe only on the
  ///   shared partition engine thread (CP and its CUs share one partition, asserted in
  ///   CommandProcessor::startup()); callers on any other thread would race a halt().
  bool has_active_wfs() const {
    std::lock_guard<std::recursive_mutex> lock(wave_state_mutex_);
    for (const auto &w : wfs_)
      if (!w->is_halted())
        return true;
    return false;
  }

  /// @brief Check whether any wavefront can currently make forward progress.
  /// @details A debug-halted wave occupies its slot (so @ref has_active_wfs
  /// stays true and the wave is not retired) but cannot run, so it must not
  /// keep the CU's event loop spinning. Idle detection uses this instead of
  /// @ref has_active_wfs so the engine can quiesce while a wave is stopped at
  /// a breakpoint. @retval true At least one non-halted, non-debug-halted wave.
  bool has_runnable_wfs() const {
    std::lock_guard<std::recursive_mutex> lock(wave_state_mutex_);
    for (const auto &w : wfs_)
      if (!w->is_halted() && !w->debug_paused())
        return true;
    return false;
  }

  template <typename F> decltype(auto) with_wave_state_locked(F &&fn) {
    WaveStateGuard lock(*this);
    return std::forward<F>(fn)();
  }

  bool has_active_wfs_for_process(uint32_t process_id) const {
    std::lock_guard<std::recursive_mutex> lock(wave_state_mutex_);
    for (const auto &w : wfs_)
      if (!w->is_halted() && w->process_id() == process_id)
        return true;
    return false;
  }

  /// @brief Return the current round-robin scheduling index.
  /// @returns Index of the next wavefront slot to schedule.
  uint64_t cycle_count() const { return cycle_counter_; }

  /// @brief Return the physical SGPR allocation block size per wavefront.
  uint32_t sgpr_allocation_block_size() const { return config_.sgprs_per_wf; }

  /// @brief Number of descriptor-allocation violations observed by this CU.
  [[nodiscard]] uint64_t register_allocation_violation_count() const {
    return register_allocation_violation_count_.load(std::memory_order_relaxed);
  }

  /// @brief Test whether a physical SGPR range is contained in a wavefront's
  /// fixed simulator storage block.
  /// @details This is an isolation boundary, not the descriptor-derived
  /// architectural allocation extent recorded in `wf.sgpr_alloc().count`.
  [[nodiscard]] bool owns_sgpr_range(const Wavefront &wf, uint32_t physical_base,
                                     uint32_t physical_count) const {
    return &wf.raw_cu() == this && wf.sgpr_alloc().count != 0 &&
           RegAllocation{wf.sgpr_alloc().base, sgpr_allocation_block_size()}.contains(
               physical_base, physical_count);
  }

  /// @brief Test whether a wavefront may access a physical VGPR range.
  /// @details Enforces both the fixed simulator storage block and the
  /// descriptor-derived ordinary-VGPR extent. Access beyond the descriptor
  /// allocation is invalid hardware behavior and terminates the simulation.
  [[nodiscard]] bool owns_vgpr_range(const Wavefront &wf, uint32_t physical_base,
                                     uint32_t physical_count) const {
    if (&wf.raw_cu() != this || wf.vgpr_alloc().count == 0 || physical_count == 0)
      return false;
    const bool owns_storage =
        RegAllocation{wf.vgpr_alloc().base, vgpr_allocation_block_size()}.contains(physical_base,
                                                                                   physical_count);
    if (!owns_storage || physical_base < wf.vgpr_alloc().base)
      return false;

    const uint32_t relative_base = physical_base - wf.vgpr_alloc().base;
    const uint32_t ordinary_limit = isa_properties(arch()).max_addressable_vgprs_per_wf;
    if (relative_base < ordinary_limit) {
      const bool owns_ordinary = relative_base < wf.num_ordinary_vgprs() &&
                                 physical_count <= wf.num_ordinary_vgprs() - relative_base;
      if (owns_ordinary)
        return owns_storage;
      report_vgpr_allocation_violation(wf, relative_base, physical_count, false);
      return false;
    }

    if (relative_base >= ACC_VGPR_OFFSET) {
      const uint32_t acc_base = relative_base - ACC_VGPR_OFFSET;
      const bool owns_accumulator =
          acc_base < wf.num_accvgprs() && physical_count <= wf.num_accvgprs() - acc_base;
      if (owns_accumulator)
        return owns_storage;
      report_vgpr_allocation_violation(wf, acc_base, physical_count, true);
      return false;
    }

    if (owns_storage)
      return owns_storage;

    return false;
  }

private:
  // RegisterAccess is the instruction-facing register boundary. These
  // ownership and SGPR access helpers validate the complete operation, fire
  // callbacks, and only then expose or modify storage.
  friend class RegisterAccess;

  /// @brief Resolve one owner for a complete physical SGPR range.
  /// @details Returns null for empty, unallocated, out-of-file, or mixed-owner
  /// ranges. CU-bound RegisterAccess uses this compatibility path; wave-bound
  /// access keeps its explicit owner instead.
  [[nodiscard]] const Wavefront *sgpr_owner_for_range(uint32_t physical_base,
                                                      uint32_t physical_count) const {
    if (physical_count == 0 || physical_base >= sgpr_file_.total_regs() ||
        physical_count > sgpr_file_.total_regs() - physical_base)
      return nullptr;
    const Wavefront *owner = sgpr_owner(physical_base);
    if (!owner || !owns_sgpr_range(*owner, physical_base, physical_count))
      return nullptr;
    return owner;
  }

  /// @brief Resolve one owner for a complete physical VGPR range.
  /// @details Returns null for empty, unallocated, out-of-file, or mixed-owner
  /// ranges. Implemented by the concrete CU that owns the typed VGPR file.
  [[nodiscard]] virtual const Wavefront *vgpr_owner_for_range(uint32_t physical_base,
                                                              uint32_t physical_count) const = 0;

  [[nodiscard]] uint32_t read_sgpr(const Wavefront &wf, uint32_t reg_idx) const {
    if (!owns_sgpr_range(wf, reg_idx, 1))
      return 0;
    notify_scalar_register_read(
        wf, RegisterRef{RegClass::SGPR, static_cast<uint16_t>(reg_idx - wf.sgpr_alloc().base), 1});
    return sgpr_file_[reg_idx];
  }

  void write_sgpr(const Wavefront &wf, uint32_t reg_idx, uint32_t val) {
    if (!owns_sgpr_range(wf, reg_idx, 1))
      return;
    plugin_group_->onAmdgpuWriteScalarRegister(
        &wf, RegisterRef{RegClass::SGPR, static_cast<uint16_t>(reg_idx - wf.sgpr_alloc().base), 1});
    sgpr_file_[reg_idx] = val;
  }

  void write_sgpr64(const Wavefront &wf, uint32_t reg_idx, uint64_t val) {
    if (!owns_sgpr_range(wf, reg_idx, 2))
      return;
    plugin_group_->onAmdgpuWriteScalarRegister(
        &wf, RegisterRef{RegClass::SGPR, static_cast<uint16_t>(reg_idx - wf.sgpr_alloc().base), 2});
    sgpr_file_[reg_idx] = static_cast<uint32_t>(val);
    sgpr_file_[reg_idx + 1] = static_cast<uint32_t>(val >> 32);
  }

  void notify_scalar_register_read(const Wavefront &wf, RegisterRef reg) const {
    if (observes_sgpr_reads_)
      plugin_group_->onAmdgpuReadScalarRegister(&wf, reg);
  }

  void notify_scalar_register_write(const Wavefront &wf, RegisterRef reg) const {
    plugin_group_->onAmdgpuWriteScalarRegister(&wf, reg);
  }

public:
  /// @brief Read a scalar register from the physical SGPR file.
  /// @details VM/diagnostic compatibility accessor. It notifies the plugin
  /// group when the physical register is currently owned by a wavefront.
  /// Instruction code should use Operand or RegisterAccess so complete-range
  /// ownership and the explicit executing wave are preserved.
  /// @param reg_idx Physical register index.
  /// @returns Register value.
  // TODO(newling) consider cmake flag to build without plugins, this call
  // overhead might be non-negligible.
  uint32_t read_sgpr(uint32_t reg_idx) const {
    if (reg_idx >= sgpr_file_.total_regs())
      return 0;
    if (observes_sgpr_reads_)
      if (auto *wf = sgpr_owner(reg_idx))
        return read_sgpr(*wf, reg_idx);
    return sgpr_file_[reg_idx];
  }

  /// @brief Read raw SGPR storage without producing an observation callback.
  uint32_t read_sgpr_storage(uint32_t reg_idx) const {
    return reg_idx < sgpr_file_.total_regs() ? sgpr_file_[reg_idx] : 0;
  }

  /// @brief Write a scalar register in the physical SGPR file.
  /// @details Raw VM-level write used for memory completion and dispatch/runtime
  /// state setup. It deliberately does not fire an instruction callback.
  /// Instruction code must use wave-bound RegisterAccess.
  /// @param reg_idx Physical register index.
  /// @param val Value to write.
  void write_sgpr(uint32_t reg_idx, uint32_t val) {
    if (reg_idx < sgpr_file_.total_regs())
      sgpr_file_[reg_idx] = val;
  }

  /// @brief Notify plugins that a wavefront read lanes of a physical VGPR.
  /// @details Low-level notification primitive used by RegisterAccess and the
  /// concrete CU implementation. Instruction emulators should acquire observed
  /// VGPR values through RegisterAccess rather than manually pairing raw
  /// storage access with this hook.
  void notify_vgpr_read(const Wavefront *wf, uint32_t reg_idx, uint64_t lane_mask,
                        uint8_t byte_mask = rocjitsu::ExecutionPlugin::kFullByteMask) const {
    if (wf && lane_mask != 0 && owns_vgpr_range(*wf, reg_idx, 1))
      plugin_group_->onAmdgpuReadVgprLanes(wf, reg_idx, lane_mask, byte_mask);
  }

  /// @brief Notify plugins that a wavefront wrote lanes of a physical VGPR.
  /// @details Low-level notification primitive used by RegisterAccess.
  /// Raw VM/storage writes deliberately bypass this hook.
  void notify_vgpr_write(const Wavefront *wf, uint32_t reg_idx, uint64_t lane_mask,
                         uint8_t byte_mask = rocjitsu::ExecutionPlugin::kFullByteMask) const {
    if (wf)
      lane_mask &= wf->vgpr_write_mask();
    if (wf && lane_mask != 0 && byte_mask != 0 && owns_vgpr_range(*wf, reg_idx, 1))
      plugin_group_->onAmdgpuWriteVgprLanes(wf, reg_idx, lane_mask, byte_mask);
  }

  /// @brief Report a scalar-lane VGPR write without applying vector write masks.
  /// @details V_WRITELANE ignores EXEC and DPP destination masks, including
  /// when a Wave64 wave selects lanes 32--63.
  void notify_scalar_lane_vgpr_write(
      const Wavefront *wf, uint32_t reg_idx, uint64_t lane_mask,
      uint8_t byte_mask = rocjitsu::ExecutionPlugin::kFullByteMask) const {
    if (wf && lane_mask != 0 && byte_mask != 0 && owns_vgpr_range(*wf, reg_idx, 1))
      plugin_group_->onAmdgpuWriteVgprLanes(wf, reg_idx, lane_mask, byte_mask);
  }

  /// @brief Notify plugins that lanes of a physical VGPR were read.
  /// @details Resolves the owning wavefront from the physical register index.
  /// Transitional single-register compatibility path for CU internals.
  /// Instruction helpers should retain their explicit wave in RegisterAccess.
  virtual void
  notify_vgpr_read_by_reg(uint32_t reg_idx, uint64_t lane_mask,
                          uint8_t byte_mask = rocjitsu::ExecutionPlugin::kFullByteMask) const = 0;

  /// @brief Notify plugins that lanes of a physical VGPR were written.
  /// @details Resolves the owning wavefront from the physical register index.
  /// Transitional single-register compatibility path for CU internals.
  /// Instruction helpers should retain their explicit wave in RegisterAccess.
  virtual void
  notify_vgpr_write_by_reg(uint32_t reg_idx, uint64_t lane_mask,
                           uint8_t byte_mask = rocjitsu::ExecutionPlugin::kFullByteMask) const = 0;

  /// @brief Return the wavefront currently owning a physical VGPR.
  virtual const Wavefront *vgpr_owner(uint32_t reg_idx) const = 0;

  /// @brief Read a vector register lane from the physical VGPR file.
  /// @details VM/storage-level scalar lane accessor. The concrete
  /// implementation reports the read to plugins. Instruction-visible VGPR
  /// reads should still go through Operand or RegisterAccess so the read
  /// intent, lane mask, byte mask, and region lifetime remain explicit and
  /// enforceable.
  /// @param reg_idx Physical register index.
  /// @param lane Lane index within the wavefront.
  /// @returns Lane value.
  virtual uint32_t read_vgpr(uint32_t reg_idx, uint32_t lane) const = 0;

  /// @brief Write a vector register lane in the physical VGPR file.
  /// @details VM/storage-level scalar lane write. Instruction emulators should
  /// prefer Operand or RegisterAccess write APIs for instruction-visible VGPR
  /// writes. Use this directly only in VM/runtime code paths that deliberately
  /// operate on physical register storage, such as dispatch setup or memory
  /// completion.
  /// @param reg_idx Physical register index.
  /// @param lane Lane index within the wavefront.
  /// @param val Value to write.
  virtual void write_vgpr(uint32_t reg_idx, uint32_t lane, uint32_t val) = 0;

  /// @brief Return a pointer to a wavefront's SGPR data in the physical file.
  /// @param base Base register index in the SGPR file.
  /// @returns Pointer to the contiguous SGPR data.
  const uint32_t *sgpr_data(uint32_t base) const { return &sgpr_file_[base]; }

  /// @brief Return a raw pointer to one VGPR in the physical file.
  /// @details This bypasses plugin read hooks and should not be used directly
  /// by instruction emulators. It is reserved for RegisterAccess, VM storage
  /// code, serialization/checkpointing, diagnostics, and tightly controlled
  /// internals that have a separate observation contract.
  /// The returned pointer spans exactly one register's lanes, not a contiguous
  /// multi-register region. For software-lazy storage, an unmaterialized
  /// register may return shared immutable zero backing. Treat the result as an
  /// ephemeral value observation: it is not a persistent storage identity and
  /// need not observe a later write through another handle.
  /// @param base Register index in the VGPR file.
  /// @returns Const pointer to one register's raw lane data.
  virtual const uint8_t *raw_vgpr_data(uint32_t base) const = 0;

  /// @brief Return a mutable raw pointer to one VGPR.
  /// @details This bypasses the instruction-facing RegisterAccess boundary.
  /// It is intended for VM storage operations such as memory completion,
  /// checkpoint restore, RegisterAccess view implementation, and other
  /// tightly controlled internals. Instruction emulators should use Operand or
  /// RegisterAccess write APIs instead. The returned pointer spans exactly one
  /// register's lanes and remains stable until the owning wave retires.
  /// @param base Register index in the VGPR file.
  /// @returns Mutable pointer to one register's raw lane data.
  virtual uint8_t *raw_vgpr_data(uint32_t base) = 0;

  /// @brief Visit a logical physical-VGPR range in register order.
  /// @details Each callback span contains exactly one register's lanes. Backing
  /// storage boundaries are not observable through this interface.
  template <typename Function>
  void for_each_raw_vgpr(uint32_t base, uint32_t count, Function &&function) const {
    using FunctionType = std::remove_reference_t<Function>;
    static_assert(std::is_object_v<FunctionType>, "VGPR visitors must be callable objects");
    static_assert(std::is_invocable_v<FunctionType &, std::span<const uint32_t>>,
                  "VGPR visitor must accept a const lane span");
    for_each_raw_vgpr_impl(base, count, &function,
                           [](const void *context, std::span<const uint32_t> lanes) {
                             if constexpr (std::is_const_v<FunctionType>) {
                               (*static_cast<const FunctionType *>(context))(lanes);
                             } else {
                               (*static_cast<FunctionType *>(const_cast<void *>(context)))(lanes);
                             }
                           });
  }

  /// @brief Copy raw bytes from a logical physical-VGPR range.
  virtual void copy_raw_vgprs_to(uint32_t base, uint32_t count,
                                 std::span<std::byte> destination) const = 0;

  /// @brief Restore raw bytes into a logically zero physical-VGPR range.
  /// @details Zero source runs remain unmaterialized, so this is a full restore
  /// only when every destination byte is initially zero.
  virtual void restore_raw_vgprs_into_zeroed_storage(uint32_t base, uint32_t count,
                                                     std::span<const std::byte> source) = 0;

  /// @brief Read a VGPR lane directly from physical storage.
  /// @details This deliberately bypasses plugin observation and is reserved
  /// for VM storage operations. Instruction code receives
  /// `InstructionComputeUnitView`, which does not expose this API.
  uint32_t read_vgpr_storage(uint32_t reg_idx, uint32_t lane) const {
    return reinterpret_cast<const uint32_t *>(raw_vgpr_data(reg_idx))[lane];
  }

  /// @brief Number of physical VGPR registers in one allocation block.
  virtual uint32_t vgpr_allocation_block_size() const = 0;

  /// @brief Number of physically stored lanes in each VGPR.
  virtual uint32_t vgpr_storage_lane_count() const = 0;

  /// @brief Raw typed view of a single VGPR as the file's @c simdojo::VectorReg.
  /// @details The abstract CU exposes the VGPR file only as a byte pointer
  /// (@c raw_vgpr_data), which erases the wavefront-size template parameter. The
  /// file actually stores @c simdojo::VectorReg<N,uint32_t>, so this recovers
  /// the typed register with the design's single localized @c reinterpret_cast.
  /// Like @c raw_vgpr_data, this bypasses plugin hooks and is for
  /// RegisterAccess/VM internals rather than instruction emulator call sites.
  /// The @c static_assert pins @c VectorReg<N> to @c N contiguous @c uint32_t
  /// (no padding / vtable) so the byte view and the typed view coincide.
  template <size_t N> simdojo::VectorReg<N, uint32_t> &raw_vgpr_reg(uint32_t base) {
    static_assert(sizeof(simdojo::VectorReg<N, uint32_t>) == N * sizeof(uint32_t),
                  "VectorReg must be layout-compatible with raw lane storage");
    return *reinterpret_cast<simdojo::VectorReg<N, uint32_t> *>(raw_vgpr_data(base));
  }
  template <size_t N> const simdojo::VectorReg<N, uint32_t> &raw_vgpr_reg(uint32_t base) const {
    static_assert(sizeof(simdojo::VectorReg<N, uint32_t>) == N * sizeof(uint32_t),
                  "VectorReg must be layout-compatible with raw lane storage");
    return *reinterpret_cast<const simdojo::VectorReg<N, uint32_t> *>(raw_vgpr_data(base));
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

  /// @brief Reserve/release physical SIMD register-file capacity for a wave slot.
  bool reserve_physical_registers(uint32_t wf_id, uint32_t num_sgprs, uint32_t num_vgprs,
                                  uint32_t wave_size);
  void release_physical_registers(uint32_t wf_id);

  using RawVgprVisitor = void (*)(const void *, std::span<const uint32_t>);
  virtual void for_each_raw_vgpr_impl(uint32_t base, uint32_t count, const void *context,
                                      RawVgprVisitor visitor) const = 0;

  /// @brief Update wavefront states (WAITCNT, BARRIER, ENDING transitions).
  void update_wf_states();

  /// @brief Fetch, decode, execute one instruction from the given wavefront.
  void issue_instruction(Wavefront *wf);

  /// @brief Apply any I$ invalidation a debug attach or detach published.
  /// @details Runs on this CU's own thread, which is the I$'s sole accessor.
  /// See set_debug_active() for why the transition cannot invalidate directly.
  void sync_inst_cache_debug_epoch() {
    const uint64_t epoch = inst_cache_debug_epoch_.load(std::memory_order_acquire);
    if (epoch == inst_cache_debug_epoch_seen_)
      return;
    inst_cache_.invalidate_all();
    inst_cache_debug_epoch_seen_ = epoch;
  }

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
  uint32_t shader_engine_id_ = 0;
  uint32_t scratch_scoreboard_base_ = 0;
  bool sram_ecc_ = false;
  std::unique_ptr<Decoder> decoder_;
  simdojo::RegisterFile<uint32_t> sgpr_file_{"sgpr"};
  std::vector<std::unique_ptr<Wavefront>> wfs_; ///< Pre-allocated wavefront slots.
  struct SimdRegisterUsage {
    uint32_t resident_waves = 0;
    uint32_t sgprs = 0;
    uint32_t vgprs = 0;
  };
  struct WaveRegisterReservation {
    uint32_t simd = UINT32_MAX;
    uint32_t sgprs = 0;
    uint32_t vgprs = 0;
  };
  PhysicalRegisterProperties physical_registers_;
  std::vector<SimdRegisterUsage> simd_register_usage_;
  std::vector<WaveRegisterReservation> wave_register_reservations_;

  [[nodiscard]] bool place_physical_registers(std::vector<SimdRegisterUsage> &usage,
                                              uint32_t num_sgprs, uint32_t num_vgprs,
                                              uint32_t wave_size,
                                              WaveRegisterReservation *reservation) const;
  /// @brief Hold the wave-state lock, then notify the CP once it is released.
  /// @details The CP takes hw_queue_mutex_ and then this lock when it dispatches
  /// (handle_doorbell -> dispatch_workgroups -> dispatch_wf), so anything running
  /// under this lock must not reach back into the CP and take hw_queue_mutex_ --
  /// a wave hitting s_endpgm on the engine thread would otherwise close an AB-BA
  /// cycle against a concurrent dispatch or DESTROY_QUEUE. release_wf() therefore
  /// queues its completions instead of sending them, and the outermost guard
  /// delivers them here, after the lock is dropped and in the same order.
  class WaveStateGuard {
  public:
    explicit WaveStateGuard(ComputeUnitCore &cu) : cu_(cu), lock_(cu.wave_state_mutex_) {
      ++cu_.wave_state_depth_;
    }
    WaveStateGuard(const WaveStateGuard &) = delete;
    WaveStateGuard &operator=(const WaveStateGuard &) = delete;
    ~WaveStateGuard() {
      const bool outermost = --cu_.wave_state_depth_ == 0;
      lock_.unlock();
      if (outermost)
        cu_.flush_wg_completions();
    }

  private:
    ComputeUnitCore &cu_;
    std::unique_lock<std::recursive_mutex> lock_;
  };

  /// @brief Send the workgroup completions queued under the wave-state lock.
  /// @warning Must be called with that lock released; it takes hw_queue_mutex_.
  void flush_wg_completions();

  mutable std::recursive_mutex wave_state_mutex_;
  /// @brief Recursion depth of WaveStateGuard on the thread holding the mutex.
  /// @details Only ever touched under @ref wave_state_mutex_, so the value
  /// belongs to whichever thread currently owns it.
  unsigned wave_state_depth_ = 0;
  /// @brief Workgroups that finished while the wave-state lock was held.
  /// @details Drained by @ref flush_wg_completions once the lock is dropped.
  std::vector<std::pair<uint32_t, uint32_t>> pending_wg_completions_;
  std::unique_ptr<WavefrontScheduler> scheduler_ = std::make_unique<OldestFirstScheduler>();
  uint64_t cycle_counter_ = 0;

  L2Cache *l2_;
  L1ScalarCache l1_scalar_;
  L1VectorCache l1_vector_;
  InstructionCache inst_cache_;
  /// @brief Debug attach/detach transitions seen by set_debug_active().
  std::atomic<uint64_t> inst_cache_debug_epoch_{0};
  /// @brief The epoch this CU's thread has already invalidated the I$ for.
  uint64_t inst_cache_debug_epoch_seen_ = 0;
  /// @brief Dispatch whose launch invalidation the I$ has already taken.
  /// @details Held as 64 bits so the initial value is outside the 32-bit
  /// dispatch-ID space and the first dispatch, including ID 0, still
  /// invalidates.
  uint64_t inst_cache_dispatch_id_ = ~uint64_t{0};
  Lds lds_;
  ImmediateClusterLdsMulticastEngine default_cluster_lds_multicast_engine_;
  ClusterLdsMulticastEngine *cluster_lds_multicast_engine_ = &default_cluster_lds_multicast_engine_;

  void report_vgpr_allocation_violation(const Wavefront &wf, uint32_t relative_base,
                                        uint32_t register_count, bool accumulator) const {
    const uint64_t prior =
        register_allocation_violation_count_.fetch_add(1, std::memory_order_relaxed);
    if (prior != 0)
      return;
    util::Logger::warn("RocJitsu register allocation violation: wave accessed ",
                       accumulator ? "accumulator VGPR range acc" : "ordinary VGPR range v",
                       relative_base, ":", accumulator ? "acc" : "v",
                       relative_base + register_count - 1, " outside descriptor allocation (",
                       accumulator ? "accumulator" : "ordinary",
                       " count=", accumulator ? wf.num_accvgprs() : wf.num_ordinary_vgprs(),
                       ") at pc=0x", std::hex, wf.pc, std::dec, " arch=", static_cast<int>(arch()));
    if (engine())
      engine()->request_exit("VGPR access exceeds descriptor allocation", 1);
  }

  mutable std::atomic<uint64_t> register_allocation_violation_count_{0};
  uint32_t next_lds_alloc_ = 0; ///< Next free LDS offset for per-WG allocation.
  std::unordered_set<uint64_t> lds_pinned_clusters_;
  ScalarMemPipeline scalar_mem_pipeline_;
  GlobalMemPipeline global_mem_pipeline_;
  LocalMemPipeline local_mem_pipeline_;
  std::function<void()> on_idle_; ///< Callback invoked when CU becomes idle.
  TrapHandlerResolver trap_handler_resolver_;
  SendmsgHandler sendmsg_handler_;
  TrapCompletionHandler trap_completion_handler_;
  SingleStepHandler single_step_handler_;
  WatchpointHandler watchpoint_handler_;
  IllegalInstHandler illegal_inst_handler_;
  MemoryViolationHandler memory_violation_handler_;
  AluExceptionHandler alu_exception_handler_;
  std::atomic<bool> debug_active_{false};
  CommandProcessor *cp_ = nullptr;

  std::unordered_map<uint64_t, uint32_t> active_wgs_;

  struct BarrierCounter {
    uint32_t member_count = 0;
    uint32_t signal_count = 0;
  };
  struct WorkgroupBarriers {
    uint32_t allocated_count = 0;
    std::array<BarrierCounter, kMaxNamedBarriers + 1> named{};
    std::array<BarrierCounter, 2> workgroup{};
  };
  std::vector<Wavefront *> complete_barrier(uint32_t dispatch_id, uint32_t wg_id,
                                            uint8_t completion_bit, uint32_t named_barrier_id = 0);
  void notify_barrier_complete(std::span<Wavefront *> members);
  std::unordered_map<uint64_t, WorkgroupBarriers> barrier_wgs_;

  uint64_t shared_aperture_base_ = 0;
  uint64_t shared_aperture_limit_ = 0;
  uint64_t private_aperture_base_ = 0;
  uint64_t private_aperture_limit_ = 0;

  std::shared_ptr<ExecutionPluginGroup> plugin_group_ = ExecutionPluginGroup::empty_group();
  bool observes_sgpr_reads_ = false;

  /// @brief Resolve the owner of a physical SGPR from its allocation block.
  /// @details Power-of-two block sizes use a shift on the instruction read path;
  /// other target configurations retain exact division semantics.
  Wavefront *sgpr_owner(uint32_t reg_idx) const {
    const uint32_t block = sgpr_block_shift_ != kVariableSgprBlockShift
                               ? reg_idx >> sgpr_block_shift_
                               : reg_idx / config_.sgprs_per_wf;
    const auto &allocation = sgpr_block_owners_[block];
    return reg_idx < allocation.end ? allocation.owner : nullptr;
  }

  static constexpr uint32_t kVariableSgprBlockShift = std::numeric_limits<uint32_t>::max();
  struct SgprBlockOwner {
    Wavefront *owner = nullptr;
    uint32_t end = 0;
  };
  /// Reverse lookup: SGPR allocation block -> owning wavefront for instruction
  /// observation. One entry per hardware wave slot replaces one pointer per
  /// physical SGPR, while end preserves the physical allocation-block boundary.
  std::vector<SgprBlockOwner> sgpr_block_owners_;
  uint32_t sgpr_block_shift_ = kVariableSgprBlockShift;
  /// Record or clear the owner of one VGPR allocation block.
  /// @param base Allocation-block-aligned physical VGPR base.
  /// @param wf Owning wavefront, or nullptr when the block is freed.
  virtual void set_vgpr_block_owner(uint32_t base, Wavefront *wf) = 0;
  simdojo::Port *cpl_ = nullptr; ///< Completer port: dispatch activation from CP.
  simdojo::Port *req_ = nullptr; ///< Requester port: L2 cache request (structural).
  uint64_t step_count_ = 0;
  bool functional_yield_requested_ = false;

  friend class CommandProcessor;
  friend class ::rocjitsu::test::ComputeUnitTestAccess;
};

inline InstructionCache &InstructionComputeUnitView::instruction_cache() {
  return raw_cu().instruction_cache();
}
inline L1ScalarCache &InstructionComputeUnitView::l1_scalar() { return raw_cu().l1_scalar(); }
inline L1VectorCache &InstructionComputeUnitView::l1_vector() { return raw_cu().l1_vector(); }
inline L2Cache *InstructionComputeUnitView::l2() const { return raw_cu().l2(); }
inline Lds &InstructionComputeUnitView::lds() { return raw_cu().lds(); }
inline bool InstructionComputeUnitView::sram_ecc() const { return raw_cu().sram_ecc(); }
inline rj_code_arch_t InstructionComputeUnitView::arch() const { return raw_cu().arch(); }
inline uint32_t InstructionComputeUnitView::wf_size() const { return raw_cu().wf_size(); }
inline uint32_t InstructionComputeUnitView::sgprs_per_wf() const {
  return raw_cu().config().sgprs_per_wf;
}
inline uint32_t InstructionComputeUnitView::vgpr_allocation_block_size() const {
  return raw_cu().vgpr_allocation_block_size();
}
inline std::string InstructionComputeUnitView::full_path() const { return raw_cu().full_path(); }
inline simdojo::ComponentID InstructionComputeUnitView::id() const { return raw_cu().id(); }
inline simdojo::SimulationEngine *InstructionComputeUnitView::engine() const {
  return raw_cu().engine();
}
inline void InstructionComputeUnitView::request_functional_yield() {
  raw_cu().request_functional_yield();
}
inline bool InstructionComputeUnitView::handle_sendmsg(Wavefront &wf, uint32_t message) {
  return raw_cu().handle_sendmsg(wf, message);
}
inline void InstructionComputeUnitView::notify_trap_complete(Wavefront &wf) {
  raw_cu().notify_trap_complete(wf);
}

/// @brief Execution-mode-aware compute unit shell.
///
/// @details Adds event-driven activation on top of ComputeUnitCore.
///
/// In FUNCTIONAL mode, execute_quantum() runs up to kFunctionalQuantum
/// instructions, then yields to the simulation event loop. This
/// interleaving ensures forward progress when wavefronts on different
/// CUs synchronize via global memory (e.g., spin-locks, semaphores).
///
/// @tparam Mode Execution mode (FUNCTIONAL or CLOCKED).
template <simdojo::ExecMode Mode> class ExecComputeUnit : public ComputeUnitCore {
public:
  using ComputeUnitCore::ComputeUnitCore;

  /// @brief Execute work up to the quantum limit, then yield.
  bool execute_quantum() override {
    if constexpr (Mode == simdojo::ExecMode::FUNCTIONAL) {
      // A request left by direct step() execution must not shorten this quantum.
      functional_yield_requested_ = false;
      last_quantum_executed_ = 0;
      const uint32_t quantum = debug_active() ? kDebugFunctionalQuantum : kFunctionalQuantum;
      for (uint32_t i = 0; i < quantum && step(); ++i) {
        ++last_quantum_executed_;
        if (std::exchange(functional_yield_requested_, false))
          break;
      }
    } else {
      /// @todo: Support CLOCKED pipeline cycle.
    }
    if (is_idle()) {
      notify_idle();
      if (is_idle()) {
        executing_ = false;
        return false;
      }
    }
    return true;
  }

  void schedule_work() override {
    // Never wake an idle CU. The CP nudges the CU through the cp->cu.cpl port,
    // gated on has_active_wfs() when sent, but that port carries link latency, so a
    // nudge can arrive a tick after the last wavefront has halted and freed itself.
    // Waking the CU then would run an empty tick with no wave to issue — pure waste.
    // Work is scheduled only when there is work to do.
    //
    // When simulation is running, executing_ and tick_event_ are engine-thread
    // only: dispatch_wf(), the cpl_ handler, and the CP all run on this CU's owning
    // partition. VM creation may also call this after engine attachment but before
    // any simulation worker starts, when it has exclusive access to the queues. A
    // cross-partition call during execution would be an executing_ data race plus
    // an unsynchronized event-queue push.
    //
    // Runnable, not merely active: a wave the debugger has stopped is still
    // resident on this CU, so scheduling work for it would spin the engine
    // against a wave that cannot retire an instruction until the debugger
    // resumes it.
    if (executing_ || !this->engine() || !this->has_runnable_wfs())
      return;
    executing_ = true;
    auto now = this->engine()->context(this->partition_id()).current_tick();
    this->schedule_event(&tick_event_, now + 1);
  }

  void schedule_work_async() override {
    if (this->engine())
      this->engine()->schedule_event_now(&resume_event_);
  }

private:
  // Reschedule by the number of quantum loop iterations taken, not a fixed
  // kFunctionalQuantum: a wavefront that requests a yield after k<kFunctionalQuantum
  // iterations (e.g. s_sleep, vendor-dep retry) resumes at now+k so a peer
  // component's published state is observed promptly rather than leaping a full
  // quantum ahead. Note this counts loop iterations, not instructions issued —
  // step() advances even when every wave is WAITCNT/BARRIER-stalled — so a fully
  // stalled CU still advances by the full quantum. max(1,...) keeps the event
  // strictly in the future so re-entries never collapse onto one tick.
  simdojo::Event tick_event_{
      this, simdojo::EventType::TIMER_CALLBACK, [this](simdojo::Tick now, simdojo::Message *) {
        if (execute_quantum())
          this->schedule_event(&tick_event_, now + std::max<uint64_t>(1, last_quantum_executed_));
      }};
  // Cross-thread debugger resumes first enter this event. Its handler runs on
  // the CU partition and can safely update executing_ through schedule_work().
  simdojo::Event resume_event_{this, simdojo::EventType::TIMER_CALLBACK,
                               [this](simdojo::Tick, simdojo::Message *) { schedule_work(); }};
  uint64_t last_quantum_executed_ = 0;
  bool executing_ = false;
};

/// @brief ISA-parameterized compute unit owning the typed VGPR register file.
///
/// @details The physical VGPR element uses the architecture's maximum lane
/// count, avoiding unreachable upper-half storage on Wave32-only targets.
/// Pre-allocates all wavefront slots as IsaWavefront<Isa> instances.
///
/// @tparam Mode Execution mode (FUNCTIONAL or CLOCKED).
/// @tparam Isa ISA traits struct satisfying the GpuIsa concept.
template <simdojo::ExecMode Mode, GpuIsa Isa>
class IsaExecComputeUnit : public ExecComputeUnit<Mode> {
public:
  static_assert(Isa::WF_SIZE_MAX <= 64, "AMDGPU VGPR storage supports at most Wave64");
  using Vgpr = simdojo::VectorReg<Isa::WF_SIZE_MAX, uint32_t>;
  static constexpr uint32_t MAX_ACCVGPR_PHYSICAL_LIMIT =
      Isa::MAX_ACC_VGPRS_PER_WF == 0 ? 0 : ACC_VGPR_OFFSET + Isa::MAX_ACC_VGPRS_PER_WF;
  static constexpr uint32_t MAX_VGPRS_PER_BLOCK =
      std::max(Isa::MAX_ADDRESSABLE_VGPRS_PER_WF, MAX_ACCVGPR_PHYSICAL_LIMIT);
  static constexpr size_t MAX_VGPR_FILE_REGISTERS =
      static_cast<size_t>(Isa::MAX_WF_SLOTS) * MAX_VGPRS_PER_BLOCK;
  using VgprFile = simdojo::RegisterFile<Vgpr, simdojo::RegisterFileStorage::SOFTWARE_LAZY,
                                         MAX_VGPR_FILE_REGISTERS>;

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
    for (uint32_t i = 0; i < config.num_wf_slots; ++i)
      this->wfs_[i] = std::make_unique<IsaWavefront<Isa>>(*this, i);
    this->sram_ecc_ = Isa::SRAM_ECC;
  }

  /// @returns Lane value from the VGPR file.
  uint32_t read_vgpr(uint32_t reg_idx, uint32_t lane) const override {
    if (reg_idx >= vgpr_file_.total_regs() || lane >= Isa::WF_SIZE_MAX)
      return 0;
    notify_vgpr_read_by_reg(reg_idx, uint64_t{1} << lane);
    return vgpr_file_[reg_idx][lane];
  }

  void notify_vgpr_read_by_reg(
      uint32_t reg_idx, uint64_t lane_mask,
      uint8_t byte_mask = rocjitsu::ExecutionPlugin::kFullByteMask) const override {
    if (auto *wf = vgpr_owner(reg_idx))
      this->notify_vgpr_read(wf, reg_idx, lane_mask, byte_mask);
  }

  void notify_vgpr_write_by_reg(
      uint32_t reg_idx, uint64_t lane_mask,
      uint8_t byte_mask = rocjitsu::ExecutionPlugin::kFullByteMask) const override {
    if (auto *wf = vgpr_owner(reg_idx))
      this->notify_vgpr_write(wf, reg_idx, lane_mask, byte_mask);
  }

  const Wavefront *vgpr_owner(uint32_t reg_idx) const override {
    if (vgprs_per_block_ == 0)
      return nullptr;
    const size_t block = reg_idx / vgprs_per_block_;
    return block < this->config_.num_wf_slots ? vgpr_block_owners_[block] : nullptr;
  }

private:
  const Wavefront *vgpr_owner_for_range(uint32_t physical_base,
                                        uint32_t physical_count) const override {
    if (physical_count == 0 || physical_base >= vgpr_file_.total_regs() ||
        physical_count > vgpr_file_.total_regs() - physical_base)
      return nullptr;
    const Wavefront *owner = vgpr_owner(physical_base);
    return owner && this->owns_vgpr_range(*owner, physical_base, physical_count) ? owner : nullptr;
  }

public:
  void set_vgpr_block_owner(uint32_t base, Wavefront *wf) override {
    assert(vgprs_per_block_ != 0 && base % vgprs_per_block_ == 0);
    const size_t block = base / vgprs_per_block_;
    assert(block < this->config_.num_wf_slots);
    vgpr_block_owners_[block] = wf;
  }

  /// @brief Write a value to the VGPR file.
  void write_vgpr(uint32_t reg_idx, uint32_t lane, uint32_t val) override {
    if (reg_idx < vgpr_file_.total_regs() && lane < Isa::WF_SIZE_MAX)
      vgpr_file_[reg_idx][lane] = val;
  }

  /// @returns Const pointer to one VGPR's raw lane data.
  const uint8_t *raw_vgpr_data(uint32_t base) const override {
    return reinterpret_cast<const uint8_t *>(&vgpr_file_[base]);
  }

  /// @returns Mutable pointer to one VGPR's raw lane data.
  uint8_t *raw_vgpr_data(uint32_t base) override {
    return reinterpret_cast<uint8_t *>(&vgpr_file_[base]);
  }

  void copy_raw_vgprs_to(uint32_t base, uint32_t count,
                         std::span<std::byte> destination) const override {
    vgpr_file_.copy_to(base, count, destination);
  }

  void restore_raw_vgprs_into_zeroed_storage(uint32_t base, uint32_t count,
                                             std::span<const std::byte> source) override {
    vgpr_file_.copy_nonzero_from(base, count, source);
  }

  /// @brief Return the VGPR register file (typed, only on concrete CU).
  /// @returns Const reference to the VGPR register file.
  const VgprFile &vgpr_file() const { return vgpr_file_; }

  /// @brief Return a mutable reference to the VGPR register file.
  /// @returns Mutable reference to the VGPR register file.
  VgprFile &vgpr_file() { return vgpr_file_; }

protected:
  /// @returns Base index of the allocated VGPR block, or -1 on failure.
  int32_t allocate_vgprs(uint32_t count) override { return vgpr_file_.allocate(count); }

  /// @brief Return allocated VGPRs to the free pool.
  void free_vgprs(uint32_t base) override {
    vgpr_file_.free(base);
    set_vgpr_block_owner(base, nullptr);
  }

  uint32_t free_vgpr_blocks() const override { return vgpr_file_.free_block_count(); }

  void for_each_raw_vgpr_impl(uint32_t base, uint32_t count, const void *context,
                              ComputeUnitCore::RawVgprVisitor visitor) const override {
    static_assert(sizeof(Vgpr) == Isa::WF_SIZE_MAX * sizeof(uint32_t),
                  "VectorReg must be layout-compatible with raw lane storage");
    vgpr_file_.for_each(base, count, [&](const Vgpr &reg) {
      visitor(context,
              {reinterpret_cast<const uint32_t *>(&reg), static_cast<size_t>(Isa::WF_SIZE_MAX)});
    });
  }

public:
  uint32_t vgpr_allocation_block_size() const override { return vgprs_per_block_; }
  uint32_t vgpr_storage_lane_count() const override { return Isa::WF_SIZE_MAX; }

protected:
  /// @brief Execute one instruction on the given wavefront.
  ///
  /// @brief Execute one instruction on the given wavefront via direct dispatch.
  void execute_instruction(Instruction *inst, Wavefront &wf) override {
    assert(inst->execute && "instruction execution backend is not linked");
    inst->execute(*inst, &wf);
  }

private:
  VgprFile vgpr_file_{"vgpr"};
  /// One owner per register-file allocation block. Every VGPR in a block has
  /// the same owner, so a per-register reverse map would duplicate each pointer
  /// @c vgprs_per_block_ times. The array is sized by wave slots because the
  /// register file currently contains exactly one allocation block per slot.
  std::array<Wavefront *, Isa::MAX_WF_SLOTS> vgpr_block_owners_{};
  uint32_t vgprs_per_block_ = 0;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_COMPUTE_UNIT_H_
