// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_COMMAND_PROCESSOR_H_
#define ROCJITSU_VM_AMDGPU_COMMAND_PROCESSOR_H_

/// @file command_processor.h
/// @brief Command processor (CP) component.
///
/// @details Models a CP that works with the ROCm runtime to fetch
/// and process HSA AQL packets and dispatch work to compute units.
///
/// Architecture: the CP directly owns queue state and doorbell monitoring
/// (CP hardware functions). Three sub-blocks handle distinct pipeline stages:
///   - AqlPacketProcessor: ring buffer fetch, packet parse, DispatchEntry creation
///   - DispatchController: SPI+ADC WG iteration, CU resource check, WF creation
///   - CompletionTracker: per-dispatch WG counting, in-order signal retirement
///
/// @see <a
/// href="https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/command-processor.html">ROCm
/// CP documentation</a>

#include "rocjitsu/vm/amdgpu/cluster_lds_multicast.h"
#include "rocjitsu/vm/amdgpu/completion_tracker.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/cpu_dispatch_pool.h"
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/spi.h"
#include "rocjitsu/vm/amdgpu/workgroup_key.h"

#include "simdojo/sim/component.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rocjitsu/base/rj_compiler.h"
#ifndef HSA_LARGE_MODEL
#define HSA_LARGE_MODEL 1
#endif
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
#include "hsa/hsa.h"
RJ_DIAGNOSTIC_POP

namespace rocjitsu {
namespace amdgpu {

/// @brief Description of an AQL hardware queue registered with the CP.
struct HwQueue {
  uint32_t process_id = 0;
  uint32_t queue_id = 0;
  uint64_t ring_base_va = 0;
  uint32_t ring_size = 0;
  uint64_t read_ptr_va = 0;
  uint64_t write_ptr_va = 0;
  uint32_t doorbell_offset = 0;
  void *doorbell_base = nullptr;
  uint64_t doorbell_va = 0;
  uint64_t last_doorbell = 0;
  bool host_accessible = false;
  bool is_sdma = false;
  /// @brief Set when a packet faulted; the queue stops until it is torn down.
  /// @details A faulted packet is retired rather than retried, because its
  /// endpoint will never resolve. Continuing the scan would then run the FENCE
  /// or signal packet behind it and publish completion for work that never
  /// happened, which is the same lie the faulted copy was stopped from telling.
  /// Hardware halts the engine on a VM fault and waits for the driver; this
  /// models that, and the violation has already been reported to the process.
  bool faulted = false;
  bool debug_suspended = false;
  bool runtime_suspended = false;
  /// A command-processor pass observed this queue while its debugger gate was closed.
  /// Cleared on resume after scheduling one pass to process the deferred work.
  bool debug_work_deferred = false;
  uint64_t queue_desc_va = 0;
  uint64_t exception_status_va = 0;
  uint32_t exception_event_id = 0;
  /// CP-private monotonic fetch cursor: the next ring index to fetch. Normally
  /// tracks read_ptr_va exactly, but stays ahead of it while the debugger holds
  /// the queue's read_dispatch_id at a trapped dispatch (so packets are not
  /// re-fetched). See fetch_from_queue and serialize_queue_debug_waves.
  uint64_t fetch_cursor = 0;
  /// Spread each of this queue's dispatches over every XCD of the SoC, the way a
  /// multi-XCD part does when it runs as a single partition. Set by the queue
  /// creation path that models such a device; registering the queue replicates it
  /// onto the peer XCDs.
  bool xcd_fanout = false;
  /// Set on the replicas that xcd_fanout creates. A replica never reads the ring
  /// and never polls a doorbell; work reaches it as dispatch shards from the XCD
  /// that owns the queue.
  bool fanout_replica = false;
};

enum class SdmaPacketDialect {
  Legacy,
  Gfx11Plus,
  Gfx1250,
};

/// @brief AMDGPU command processor that dispatches wavefronts to compute units.
///
/// @details Distributes AQL dispatch packets across the registered compute units in
/// round-robin order, activating pre-allocated wavefront slots.
///
/// Event-driven: the CP monitors registered hardware queue doorbells via a
/// polling thread. When new AQL packets are detected, it fetches them from the
/// ring buffer, parses the kernel descriptor, and dispatches wavefronts to CUs.
///
/// Completion signals fire per-dispatch when all workgroups retire (gem5 model),
/// not on global CU idle. Signals fire in per-queue submission order.
class CommandProcessor : public simdojo::Component {
public:
  explicit CommandProcessor(std::string name,
                            simdojo::ExecMode exec_mode = simdojo::ExecMode::FUNCTIONAL);
  ~CommandProcessor() override;

  void set_memory(GpuMemory *mem) { memory_ = mem; }
  void add_l2_cache(L2Cache *l2) {
    // Idempotent: the config-driven builder and the Xcd full constructor may
    // both attempt to register the same L2. Avoid duplicate entries so cache
    // maintenance does not flush the same L2 twice.
    if (std::find(l2_caches_.begin(), l2_caches_.end(), l2) == l2_caches_.end())
      l2_caches_.push_back(l2);
  }
  void set_packed_tid(bool v) { packed_tid_ = v; }
  bool packed_tid() const { return packed_tid_; }
  void set_sdma_packet_dialect(SdmaPacketDialect dialect) { sdma_packet_dialect_ = dialect; }
  SdmaPacketDialect sdma_packet_dialect() const { return sdma_packet_dialect_; }
  /// @brief Configure launch and packet behavior derived from the GPU architecture.
  void configure_for_arch(rj_code_arch_t arch);
  void set_shared_dispatch_pool(CpuDispatchPool *pool);
  void set_dispatch_threads(uint32_t threads);
  uint32_t dispatch_threads() const { return dispatch_threads_; }
  /// @brief Update doorbell_base for all queues belonging to a process.
  /// @details Called when the doorbell page is mmap'd after queue creation.
  void set_doorbell_base(uint32_t process_id, void *base);

  using InterruptCallback = std::function<void(uint32_t process_id, uint32_t event_id)>;
  void set_interrupt_callback(InterruptCallback cb) { interrupt_cb_ = std::move(cb); }

  using ScratchBackingResolver = std::function<uint64_t(uint32_t process_id)>;
  void set_scratch_backing_resolver(ScratchBackingResolver cb) {
    scratch_resolver_ = std::move(cb);
  }

  using ScratchBackingAllocator =
      std::function<bool(uint32_t process_id, uint64_t gpu_va, size_t size)>;
  void set_scratch_backing_allocator(ScratchBackingAllocator cb) {
    scratch_allocator_ = std::move(cb);
  }

  /// @brief Number of shader engines per XCC (array_count / simd_arrays_per_engine).
  /// Used as the divisor when publishing COMPUTE_TMPRING_SIZE.WAVES so that
  /// rocm-dbgapi's scratch_memory_region does not disable private access.
  void set_scratch_wave_divisor(uint32_t se_per_xcc) {
    scratch_wave_divisor_ = se_per_xcc == 0 ? 1 : se_per_xcc;
  }

  /// @brief Tell this CP where its XCD sits among the SoC's XCDs.
  ///
  /// @details @p peers lists every XCD's command processor in XCD index order and
  /// includes this one at index @p rank. A dispatch on a fanned-out queue is split
  /// so that XCD i takes the grid chunks congruent to i modulo peers.size(); the
  /// rank is the XCD's own index, not its position relative to the queue's owner,
  /// so the workgroup-to-XCD mapping does not depend on which XCD a queue landed on.
  /// @param rank This CP's XCD index.
  /// @param peers All XCD command processors of the SoC, in XCD index order.
  void set_xcd_topology(uint32_t rank, std::vector<CommandProcessor *> peers);

  /// @brief Identify this CP's XCC in the device-wide scratch allocation.
  void set_scratch_xcc_layout(uint32_t xcc_id, uint32_t xcc_count) {
    scratch_xcc_id_ = xcc_id;
    scratch_xcc_count_ = xcc_count == 0 ? 1 : xcc_count;
  }

  void register_queue(HwQueue queue);
  void unregister_queue(uint32_t queue_id, uint32_t process_id);

  /// @brief Take one XCD's share of a dispatch fanned out by a peer XCD.
  ///
  /// @details Thread-safe, and safe to call from another partition's thread while
  /// the caller holds its own CP's hw_queue_mutex_: the shard is parked in an inbox
  /// guarded by a leaf mutex and the CP is woken through the engine's cross-thread
  /// event queue rather than dispatched inline. The shard reaches the queue state
  /// when this CP next drains the inbox on its own thread.
  /// @param shard The share of the grid this XCD is to run.
  void accept_fanout_shard(DispatchEntry shard);
  void update_queue(uint32_t queue_id, uint32_t process_id, uint64_t ring_base_va,
                    uint32_t ring_size, uint32_t queue_percentage);
  void set_queue_debug_suspended(uint32_t queue_id, uint32_t process_id, bool suspended);
  bool signal_queue_exception(uint32_t queue_id, uint32_t process_id, uint64_t status);
  uint64_t read_process_memory64(uint64_t address, uint32_t process_id) const {
    return memory_ && memory_->is_fetchable(address, process_id)
               ? memory_->read64(address, process_id)
               : 0;
  }

  void set_plugin_group(std::shared_ptr<ExecutionPluginGroup> pg) {
    plugin_group_ = pg ? pg : ExecutionPluginGroup::empty_group();
    if (completion_) {
      completion_->set_plugin_group(plugin_group_);
    }
    set_dispatch_threads(dispatch_threads_);
  }

  void add_spi(ShaderProcessorInput *spi) { spis_.push_back(spi); }

  void add_compute_unit(ComputeUnitCore *cu) {
    auto port_id = static_cast<simdojo::PortID>(dispatch_ports_.size());
    auto port = std::make_unique<simdojo::Port>("req_" + cu->name(), port_id, this,
                                                simdojo::PortDirection::OUT,
                                                simdojo::PortProtocol::DISPATCH);
    dispatch_ports_.push_back(add_port(std::move(port)));
    cus_.push_back(cu);
    scratch_shader_engine_count_ =
        std::max(scratch_shader_engine_count_, cu->shader_engine_id() + 1);
    scratch_waves_per_se_ =
        std::max(scratch_waves_per_se_, cu->scratch_scoreboard_base() + cu->num_wf_slots());
    cu->set_pool_driven(dispatch_threads_ > 1);
    cu->set_command_processor(this);
    cu->set_on_idle([this]() { on_cu_idle(); });
  }

  void startup() override;
  void shutdown() override;
  bool step() override;
  simdojo::Event *doorbell_event() { return &doorbell_event_; }

  /// @brief WG completion notification from CU refcount reaching zero.
  void notify_wg_complete(uint32_t dispatch_id, uint32_t wg_id);

  void set_workgroup_id_offset(uint32_t offset) { workgroup_id_offset_ = offset; }

  [[nodiscard]] size_t dispatched_count() const { return total_dispatched_; }

  /// @brief Total workgroups this CP has placed on its own XCD's compute units.
  /// @details Distinct from dispatched_count(), which counts AQL packets. This is
  /// a lifetime running total, not a per-dispatch figure: to see how one grid was
  /// spread, snapshot every XCD's counter before the dispatch and diff afterwards.
  ///
  /// Atomic because the increment happens on the dispatch path under
  /// hw_queue_mutex_ while SoC::dispatched_workgroups_per_xcd() reads every XCD's
  /// counter without that lock. Relaxed ordering is enough: this is a cumulative
  /// statistic, not a synchronization point.
  [[nodiscard]] uint64_t dispatched_workgroups() const {
    return dispatched_workgroups_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] size_t next_cu_index() const { return next_cu_; }

  /// @brief Number of entries (@p queue_id, @p process_id) accepted on this CP.
  ///
  /// @details Test-only. Acceptance is recorded at the queue's single ordered push
  /// site under this CP's queue mutex and read AFTER a run. Unlike queue depth, the
  /// count does not depend on whether a peer retires an earlier entry before the
  /// next one arrives.
  [[nodiscard]] size_t accepted_entry_count_for_test(uint32_t queue_id, uint32_t process_id) {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    const auto *qs = find_queue_state(queue_id, process_id);
    return qs == nullptr ? 0 : qs->accepted_entries;
  }

  /// @brief Kinds of the first two entries accepted on this CP for the queue.
  /// @details Test-only. The order matches the queue's ordered push site.
  [[nodiscard]] std::array<DispatchPacketKind, 2>
  first_accepted_entry_kinds_for_test(uint32_t queue_id, uint32_t process_id) {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    const auto *qs = find_queue_state(queue_id, process_id);
    return qs == nullptr ? std::array<DispatchPacketKind, 2>{} : qs->first_accepted_entry_kinds;
  }

  /// @brief Hardware queues registered with this CP, including fan-out replicas.
  ///
  /// @details Test-only. Whether a queue is present here as an owner or as a
  /// replica is an internal placement detail, not something production code
  /// should branch on.
  [[nodiscard]] size_t registered_queue_count_for_test() const {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    return hw_queues_.size();
  }

  /// @brief Host-accessible queues this CP polls, excluding fan-out replicas.
  ///
  /// @details Test-only, and the count form of polls_kfd_queues(): replication
  /// makes every CP hold a host-accessible queue, so this is what says whether a
  /// CP still has a ring of its own to read after another CP's queue is
  /// destroyed. Deliberately narrower than "queues this CP owns" -- a queue
  /// registered directly against this CP by a test is owned by it but is not
  /// host-accessible, so it is not counted here.
  [[nodiscard]] size_t polled_kfd_queue_count_for_test() const {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    size_t polled = 0;
    for (const auto &q : hw_queues_)
      polled += (q.host_accessible && !q.fanout_replica) ? 1 : 0;
    return polled;
  }

  /// @brief Step a dispatch id within one XCD's residue class.
  ///
  /// @details Public and static only so the wrap can be pinned by a test: it is
  /// otherwise ~2^29 dispatches away, and the case that matters is a
  /// non-power-of-two XCD count, where running off the end of uint32_t would
  /// interleave the classes and let two XCDs mint the same id.
  /// @param current The id just handed out.
  /// @param base First id of this XCD's class, where the sequence restarts.
  /// @param stride Number of participating XCDs.
  /// @returns The next id in the same class.
  [[nodiscard]] static uint32_t step_dispatch_id(uint32_t current, uint32_t base, uint32_t stride) {
    if (current > std::numeric_limits<uint32_t>::max() - stride)
      return base;
    return current + stride;
  }

  const std::vector<simdojo::Port *> &dispatch_ports() const { return dispatch_ports_; }
  const std::vector<ComputeUnitCore *> &compute_units() const { return cus_; }

  /// @brief Return LDS targets selected by a cluster multicast mask.
  std::vector<ClusterLdsTarget> cluster_lds_targets(uint32_t dispatch_id, uint32_t wg_id,
                                                    uint32_t mcast_mask);

  /// @brief Test-only view of the doorbell monitor lifecycle flag.
  ///
  /// @details Exposes doorbell_running_ so a regression test can observe the
  /// monitor stopping after the last polled queue is destroyed and restarting
  /// when a new one registers. Polled, not host-accessible: the monitor stops as
  /// soon as this CP owns no ring of its own, which can leave host-accessible
  /// fan-out replicas registered behind it. Read under doorbell_thread_mutex_ so
  /// it never races monitor teardown or ensure_doorbell_monitor().
  [[nodiscard]] bool doorbell_monitor_running_for_test() {
    std::lock_guard<std::mutex> lock(doorbell_thread_mutex_);
    return doorbell_running_;
  }
  /// @brief Test-only check that teardown reaped the monitor's thread handle.
  [[nodiscard]] bool doorbell_monitor_joinable_for_test() {
    std::lock_guard<std::mutex> lock(doorbell_thread_mutex_);
    return doorbell_thread_.joinable();
  }

  /// @brief Test-only view of one queue's debugger suspension gate.
  [[nodiscard]] bool queue_debug_suspended_for_test(uint32_t queue_id, uint32_t process_id) {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    auto queue = std::find_if(hw_queues_.begin(), hw_queues_.end(), [&](const auto &candidate) {
      return candidate.queue_id == queue_id && candidate.process_id == process_id;
    });
    return queue != hw_queues_.end() && queue->debug_suspended;
  }

  [[nodiscard]] bool queue_runtime_suspended_for_test(uint32_t queue_id, uint32_t process_id) {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    auto queue = std::find_if(hw_queues_.begin(), hw_queues_.end(), [&](const auto &candidate) {
      return candidate.queue_id == queue_id && candidate.process_id == process_id;
    });
    return queue != hw_queues_.end() && queue->runtime_suspended;
  }

  /// @brief Test-only count of executed command-processor doorbell passes.
  [[nodiscard]] uint64_t doorbell_handle_count_for_test() const {
    return doorbell_handle_count_.load(std::memory_order_relaxed);
  }

private:
  struct ClusterWorkgroupPlacement;
  struct ClusterBarrierState;

  /// @brief Initialize a wavefront's registers per the AMDHSA ABI.
  void init_wavefront_regs(ComputeUnitCore *cu, Wavefront *wf, const DispatchEntry &entry,
                           uint32_t global_wg_id, uint32_t wf_index_in_wg);

  void handle_doorbell(simdojo::Tick timestamp);

  /// @brief Re-arm a re-check of a queue stalled on an unsatisfied external wait
  /// (barrier/dependency signal, or an SDMA VA not yet translatable).
  /// @details Runs on the engine thread. When a doorbell poll thread is monitoring
  /// this CP (host-accessible/KFD queues), it sets stall_pending_ so the poll thread
  /// re-nudges the idle engine at its 100us cadence — the engine must NOT reschedule
  /// the doorbell on the main event queue, which models simulated timing and would
  /// spin millions of ticks while wall-clock RPC latency elapses. Internal test
  /// queues have no poll thread and are driven by engine->run()/step(), so there the
  /// re-check must be kept alive by rescheduling the doorbell event at @p now + 1.
  void arm_stall_recheck(simdojo::Tick now);

  /// @brief Re-arm a re-check while this CP holds a shard whose grid is still
  /// running on another XCD.
  /// @details Caller MUST hold hw_queue_mutex_ and MUST be on this CP's own
  /// partition thread. No-op unless a queue head is a share this XCD has
  /// finished but the grid has not retired device-wide.
  void arm_grid_wait_recheck();

  /// @brief Fetch AQL packets from a single HW queue.
  void fetch_from_queue(HwQueue &queue, HwQueueState &qs, simdojo::Tick now);

  /// @brief Process SDMA packets from an SDMA queue's ring buffer.
  void process_sdma_ring(HwQueue &queue, uint64_t read_idx, uint64_t write_idx, simdojo::Tick now);

  /// @brief Coarse invalidate of the GPU data caches (L1 V$ + L2/GL2).
  /// @details Emulated SDMA and CP writes land directly in the backing store,
  /// bypassing the cache hierarchy. Real SDMA does not snoop GL2, so stale
  /// cached copies are knocked out the way HW cache-maintenance does it: coarse
  /// and indiscriminate, not per-range. This is the simulator's stand-in for a
  /// GL2 invalidate; the consuming kernel's acquire fence at dispatch flushes
  /// the remaining per-CU caches (including the scalar K$).
  ///
  /// @warning Drops dirty L2 lines without writeback. Only use after a direct
  /// backing write whose destination is the only stale region; otherwise use
  /// flush_gpu_caches() so dirty L2 lines are published, not lost.
  void invalidate_gpu_caches();

  /// @brief Coarse writeback+invalidate of the GPU data caches (L1 K$/V$ + L2).
  /// @details Like invalidate_gpu_caches(), but publishes dirty data instead of
  /// dropping it. Scalar and vector L1 are write-through and only need
  /// invalidation. Dirty L2 data is flushed to backing before the direct SDMA
  /// write (which runs after this returns), so a later L2 flush cannot overwrite
  /// the direct result. Each L2 line is written back under its owning VMID.
  void flush_gpu_caches();

  /// @brief Parse an AQL dispatch packet, read its kernel descriptor, and create a DispatchEntry.
  /// @param aql_packet_id AQL ring packet id (queue read index) for debugger correlation.
  void process_aql_packet(const hsa_kernel_dispatch_packet_t &pkt, const HwQueue &queue,
                          uint64_t pkt_addr, uint32_t queue_packet_id, HwQueueState &qs,
                          uint64_t aql_packet_id = 0, ClusterDispatchShape cluster_shape = {});

  rocr::llvm::amdhsa::kernel_descriptor_t
  read_kernel_descriptor(uint64_t kernel_object, uint32_t vmid, bool host_accessible = false);
  /// @brief Dispatch workgroups from entry to CUs. Returns number dispatched.
  uint32_t dispatch_workgroups(DispatchEntry &entry);

  /// @brief Split a dispatch across the SoC's XCDs, keeping this XCD's share.
  ///
  /// @details Narrows @p dp to this XCD's share and hands the remaining shares to
  /// the peer XCDs, all sharing one GridCompletion so the completion signal fires
  /// once. Does nothing when the SoC has a single XCD.
  ///
  /// Every XCD gets an entry even when the grid is too small to give it any
  /// workgroups. An empty share is what keeps the replicas' queues in step with
  /// the owner's for the packets that are replicated, and barrier_satisfied()
  /// reads that ordering from the entries sitting ahead of a barrier'd packet;
  /// skipping the empty ones would leave a replica with a shorter prefix than the
  /// owner and let it start a barrier'd packet while a sibling was still running
  /// the one before it. Packets that run no shader are copied across as well, by
  /// replicate_non_kernel_entry(), so a replica's entry list is the owner's whole
  /// sequence rather than a subsequence of it.
  void fan_out_dispatch(DispatchEntry &dp);

  /// @brief Give every peer XCD a copy of a packet that runs no shader.
  ///
  /// @details A kernel dispatch is split across the XCDs, but a barrier or an IB
  /// has no workgroups to split -- what the peers need is the entry itself. The
  /// ordering barrier_satisfied() reads from the entries sitting ahead of a
  /// barrier'd packet is only device-wide if every XCD sees the same packets, so a
  /// replica that skipped these would hold a strict subsequence of the owner's list
  /// and could start a barrier'd packet while the owner still had an unretired one
  /// in front of its own copy.
  ///
  /// The copy carries no completion signal. A packet is owed exactly one, and the
  /// XCD that read it keeps that duty, exactly as it does for a kernel shard.
  void replicate_non_kernel_entry(const DispatchEntry &dp);

  /// @brief Move shards handed over by peer XCDs into their queue states.
  /// @details Runs on this CP's own thread, under hw_queue_mutex_. Kept separate
  /// from accept_fanout_shard() so that no CP ever takes a peer's hw_queue_mutex_.
  void drain_fanout_inbox();

  /// @brief Schedule a doorbell on every XCD of the SoC, this one included.
  /// @details Used when a dispatch retires device-wide: the XCD holding the
  /// completion signal may be parked with nothing left to rouse it, and a peer may
  /// be parked behind a barrier bit this dispatch was blocking. Replicas run no
  /// doorbell poll thread of their own, so nothing else would re-examine them.
  void wake_all_xcds();

  /// @brief Allocate a dispatch id unique across every XCD of the SoC.
  ///
  /// @details Fan-out copies a dispatch id onto peer XCDs, and completion
  /// bookkeeping (notify_wg_complete, the cluster placement keys, the CU's
  /// per-workgroup refcounts) looks entries up by that id. If two XCDs could mint
  /// the same id, a peer holding a shard of one dispatch and an own dispatch with
  /// the same id would credit workgroup completions to whichever it found first.
  /// Seeding each CP at its XCD rank and stepping by the XCD count keeps the id
  /// spaces disjoint without a shared counter: each XCD owns one residue class
  /// modulo the XCD count.
  ///
  /// The wrap is explicit rather than left to the type. Letting the counter run
  /// off the end of uint32_t preserves disjointness only when the XCD count
  /// divides 2^32, and XcdShard deliberately accepts any count >= 1, so the two
  /// contracts would disagree for a non-power-of-two topology -- after the wrap
  /// the classes interleave and two XCDs mint the same id. Returning to the base
  /// of this XCD's own class keeps them disjoint for every count. It is ~2^29
  /// dispatches per XCD away in any case.
  uint32_t allocate_dispatch_id() {
    const uint32_t id = next_dispatch_id_;
    next_dispatch_id_ = step_dispatch_id(next_dispatch_id_, dispatch_id_base_, dispatch_id_stride_);
    return id;
  }

  /// @brief Locate the queue state for a (queue_id, process_id) pair.
  /// @returns Pointer into new_queue_states_, or null when not registered. Caller
  /// must hold hw_queue_mutex_ and must not use the result across a registration
  /// change.
  HwQueueState *find_queue_state(uint32_t queue_id, uint32_t process_id);

  void register_cluster_workgroup(const DispatchEntry &entry, uint32_t local_wg_id,
                                  uint32_t global_wg_id, ComputeUnitCore *cu, uint32_t lds_base);
  bool cluster_barrier_signal(Wavefront &wf, int32_t barrier_id);
  uint32_t cluster_barrier_state(const Wavefront &wf, int32_t barrier_id,
                                 uint32_t allocation_blocks) const;
  bool cluster_barrier_valid(const Wavefront &wf, int32_t barrier_id) const;
  bool find_valid_cluster_barrier_locked(const Wavefront &wf, int32_t barrier_id,
                                         ClusterWorkgroupPlacement *&placement,
                                         ClusterBarrierState *&barriers);
  bool find_valid_cluster_barrier_locked(const Wavefront &wf, int32_t barrier_id,
                                         const ClusterWorkgroupPlacement *&placement,
                                         const ClusterBarrierState *&barriers) const;
  void drain_pending_wg_completions();
  void mark_cluster_workgroup_complete(uint32_t dispatch_id, uint32_t wg_id);
  void erase_cluster_workgroup(uint32_t dispatch_id, uint32_t wg_id);
  void erase_cluster_workgroups(uint32_t dispatch_id);
  /// @brief Drop cluster LDS pins collected under cluster_placements_mutex_.
  /// @warning Must run with that lock released; it reaches the CUs' wave-state lock.
  void release_cluster_lds_pins(const std::vector<std::pair<ComputeUnitCore *, uint64_t>> &unpin);

  /// @brief Asynchronous Compute Engine (ACE): dispatch workgroups from all
  /// active queues to SPIs and run CUs to completion.
  bool ace_dispatch_all();

  /// @brief Process all queues: dispatch undispatched entries, handle non-kernel entries.
  void process_queues();

  bool has_active_cus() const;
  FunctionalQuantumResult run_active_cus_once();

  /// @brief Called from CU on_idle callback. In functional mode with quantum>0,
  /// checks for stalled dispatches that can resume.
  void on_cu_idle();

  /// @brief Queue scheduling: select next queue with undispatched entries.
  HwQueueState *schedule_next_queue();

  void handle_doorbell_sync(simdojo::Tick timestamp);

  /// @brief Check if barrier is satisfied for an entry.
  bool barrier_satisfied(const HwQueueState &qs, size_t idx) const;

  /// @brief Return total pending entries across all queues.
  size_t pending_entries() const {
    size_t total = 0;
    for (auto &qs : new_queue_states_)
      total += qs.entries.size();
    return total;
  }

  /// @brief Whether any host-accessible queue is registered here, replicas included.
  ///
  /// @details Answers whether this CP's lifecycle is anchored by the VM-level
  /// primary, which a fan-out replica does anchor just as its owner does.
  bool has_kfd_queues() const {
    for (const auto &q : hw_queues_)
      if (q.host_accessible)
        return true;
    return false;
  }

  /// @brief Whether any queue here is one whose doorbell this CP actually polls.
  ///
  /// @details A fan-out replica is host-accessible but is never polled: its work
  /// arrives as dispatch shards from the XCD that owns the queue. Anything scoped
  /// to the doorbell monitor must ask this rather than has_kfd_queues(), or a CP
  /// left holding only replicas keeps a monitor alive for a ring it never reads.
  bool polls_kfd_queues() const {
    for (const auto &q : hw_queues_)
      if (q.host_accessible && !q.fanout_replica)
        return true;
    return false;
  }

  bool uses_gfx11_plus_sdma_packets() const {
    return sdma_packet_dialect_ == SdmaPacketDialect::Gfx11Plus ||
           sdma_packet_dialect_ == SdmaPacketDialect::Gfx1250;
  }

  // gfx1250 widens the GCR packet to 6 dwords; gfx11/12 keep the 5-dword layout.
  bool uses_gfx1250_gcr_packet() const {
    return sdma_packet_dialect_ == SdmaPacketDialect::Gfx1250;
  }

  GpuMemory *memory_ = nullptr;
  std::vector<ShaderProcessorInput *> spis_;
  std::vector<L2Cache *> l2_caches_;
  std::vector<HwQueue> hw_queues_;
  std::vector<HwQueueState> new_queue_states_;
  std::vector<ComputeUnitCore *> cus_;
  std::vector<simdojo::Port *> dispatch_ports_;

  uint32_t xcd_rank_ = 0;
  // Every XCD's CP in XCD index order, including this one. Empty until the SoC
  // wires the topology, which leaves fan-out disabled. Raw pointers: the SoC owns
  // every XCD and destroys them together, so a peer outlives any use of it here,
  // including the cross-thread uses in accept_fanout_shard() and wake_all_xcds().
  std::vector<CommandProcessor *> xcd_peers_;

  // Shards handed over by peer XCDs, awaiting this CP's next pass. Guarded by a
  // leaf mutex, never hw_queue_mutex_: a peer appends here while holding its own
  // hw_queue_mutex_, so acquiring anything else under this one would reintroduce
  // the cross-CP lock cycle it exists to avoid.
  std::mutex fanout_inbox_mutex_;
  std::vector<DispatchEntry> fanout_inbox_;

  size_t next_cu_ = 0;
  size_t next_queue_idx_ = 0;
  // Almost always accessed under hw_queue_mutex_, but the teardown path in
  // handle_doorbell() must clear it AFTER unlocking (stop_doorbell_monitor() joins
  // the poll thread, which takes hw_queue_mutex_). Atomic so that lock-held reads in
  // register_queue() cannot data-race that one unlocked write. Only the internal
  // test-queue path (!has_kfd_queues()) ever sets it; KFD queues anchor the primary
  // at the VM level (rj_vm.cpp).
  std::atomic<bool> is_primary_ = false;
  uint32_t workgroup_id_offset_ = 0;
  bool packed_tid_ = false;
  // GFX11+ SDMA GCR keeps the same opcode but changes packet size/layout, so
  // the decoder cannot infer this dialect from the packet header alone.
  SdmaPacketDialect sdma_packet_dialect_ = SdmaPacketDialect::Legacy;
  uint32_t next_dispatch_id_ = 1;
  // Step between successive dispatch ids from this CP. Set to the XCD count when
  // the SoC wires the topology so no two XCDs ever mint the same id.
  uint32_t dispatch_id_stride_ = 1;
  /// First id of this XCD's residue class; where allocate_dispatch_id() restarts.
  uint32_t dispatch_id_base_ = 1;
  size_t total_dispatched_ = 0;
  std::atomic<uint64_t> dispatched_workgroups_{0};
  simdojo::ExecMode exec_mode_ = simdojo::ExecMode::FUNCTIONAL;
  uint32_t dispatch_threads_ = 1;
  CpuDispatchPool *shared_dispatch_pool_ = nullptr;
  std::unique_ptr<CpuDispatchPool> local_dispatch_pool_;
  std::vector<ComputeUnitCore *> active_cu_scratch_;

  struct PendingWorkgroupCompletion {
    uint32_t dispatch_id = 0;
    uint32_t wg_id = 0;
  };
  std::vector<PendingWorkgroupCompletion> pending_wg_completions_;

  struct ClusterWorkgroupPlacement {
    ComputeUnitCore *cu = nullptr;
    uint32_t lds_base = 0;
    uint64_t cluster_key = 0;
    uint32_t cluster_rank = 0;
    uint32_t cluster_size = 1;
    bool completed = false;
    std::vector<uint32_t> peer_wg_ids;
  };
  std::unordered_map<uint64_t, ClusterWorkgroupPlacement> cluster_wg_placements_;
  /// @brief Guards @ref cluster_wg_placements_ and @ref cluster_barriers_, not the
  /// queue state.
  /// @details A multicast LDS write resolves its peers from the CU's execute
  /// path, which already holds that CU's wave-state lock, while a dispatch takes
  /// hw_queue_mutex_ and then the wave-state lock. Sharing hw_queue_mutex_ here
  /// would close that cycle, so the map gets its own lock, ordered after both.
  /// Nothing may call into a CU while holding it -- see
  /// erase_cluster_workgroup(), which collects its LDS cleanup and runs it after
  /// the unlock.
  mutable std::recursive_mutex cluster_placements_mutex_;
  struct ClusterBarrierState {
    uint32_t expected_member_count = 0;
    uint32_t member_count = 0;
    std::unordered_set<uint32_t> registered_workgroups;
    std::array<std::unordered_set<uint32_t>, 2> signaled_workgroups;
  };
  std::unordered_map<uint64_t, ClusterBarrierState> cluster_barriers_;

  simdojo::Event doorbell_event_{this, simdojo::EventType::TIMER_CALLBACK};
  simdojo::Event dispatch_continuation_event_{this, simdojo::EventType::TIMER_CALLBACK};
  bool dispatch_continuation_pending_ = false;
  // Guards changes to the shape of hw_queues_ and new_queue_states_. The
  // dispatch handler holds a shared lock while worker execution temporarily
  // releases hw_queue_mutex_, keeping its vector references stable.
  std::shared_mutex queue_structure_mutex_;
  mutable std::recursive_mutex hw_queue_mutex_;

  std::shared_ptr<ExecutionPluginGroup> plugin_group_ = ExecutionPluginGroup::empty_group();

  friend class ComputeUnitCore;

  /// @brief Read a uint64 from GPU virtual address space via GpuMemory translation.
  uint64_t read_gpu_u64(uint64_t va, uint32_t vmid) const;

  /// @brief Read a uint32 from GPU virtual address space via GpuMemory translation.
  uint32_t read_gpu_u32(uint64_t va, uint32_t vmid) const;

  /// @brief Read a block of bytes from GPU virtual address space into a buffer.
  void read_gpu_block(uint64_t va, void *dst, size_t size, uint32_t vmid) const;

  /// @brief Write a block of bytes to GPU virtual address space from a buffer.
  amdgpu::AccessOutcome write_gpu_block(uint64_t va, const void *src, size_t size, uint32_t vmid);

  void stop_doorbell_monitor();
  /// @brief Stop and join the monitor only when no polled queue remains.
  /// @details Polled rather than host-accessible: a CP left holding only fan-out
  /// replicas still has host-accessible queues registered, but no ring it reads,
  /// so its monitor must retire.
  /// @details Caller MUST NOT hold hw_queue_mutex_: this helper takes that mutex
  /// to recheck the queue set, then may join a poller that needs the same mutex to
  /// finish its current scan. Serializing the recheck with startup ensures a
  /// concurrently registered queue cannot be left without a polling thread.
  void stop_doorbell_monitor_if_idle();
  /// @brief Start the doorbell monitor if one is not already running.
  /// @details Serialized by doorbell_thread_mutex_. Caller MUST NOT hold
  /// hw_queue_mutex_ so lifecycle operations consistently acquire
  /// doorbell_thread_mutex_ before hw_queue_mutex_.
  void ensure_doorbell_monitor();
  bool scan_doorbells();

  InterruptCallback interrupt_cb_;
  ScratchBackingResolver scratch_resolver_;
  ScratchBackingAllocator scratch_allocator_;
  uint32_t scratch_wave_divisor_ = 1;
  uint32_t scratch_shader_engine_count_ = 1;
  uint32_t scratch_waves_per_se_ = 1;
  uint32_t scratch_xcc_id_ = 0;
  uint32_t scratch_xcc_count_ = 1;
  std::unique_ptr<CompletionTracker> completion_;

  std::atomic<bool> invalid_pending_{false};

  std::atomic<uint64_t> doorbell_handle_count_{0};
  /// @brief Ticks to wait before the next stall re-check, doubled on each idle
  /// re-check and reset to 1 whenever work arrives.
  /// @details Only the no-poll-thread path uses this; a CP that polls KFD queues
  /// re-checks on its poll thread's own cadence instead.
  simdojo::Tick stall_recheck_backoff_ = 1;
  static constexpr simdojo::Tick kMaxStallRecheckBackoff = 4096;

  // Set when a queue stalls on an unsatisfied barrier/dependency signal (or an
  // SDMA VA not yet translatable) — a wait on progress that is external to the
  // current engine pass (a peer rank's kernel completion arriving via the daemon,
  // or a producer on another queue). Re-checking such a stall by rescheduling the
  // doorbell event at now+1 spins the main event queue (which models simulated
  // timing) millions of times per collective, pegging a core while wall-clock RPC
  // latency elapses. Instead, like invalid_pending_, the doorbell poll thread
  // re-nudges the (idle) engine at its 100us cadence so the stall is re-evaluated
  // without a busy simulated-time spin.
  std::atomic<bool> stall_pending_{false};

  void doorbell_poll_loop(std::stop_token stop);

  // The doorbell monitor's lifecycle is serialized by its OWN mutex, deliberately
  // distinct from hw_queue_mutex_. Queue removal releases hw_queue_mutex_ before
  // stopping and joining the monitor, so an in-progress scan can finish. The
  // lifecycle path then rechecks the queue set while startup is excluded; this
  // keeps a concurrent registration from losing its monitor.
  std::mutex doorbell_thread_mutex_;
  // True while the lifecycle owns a running monitor.
  bool doorbell_running_ = false;
  std::jthread doorbell_thread_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_COMMAND_PROCESSOR_H_
