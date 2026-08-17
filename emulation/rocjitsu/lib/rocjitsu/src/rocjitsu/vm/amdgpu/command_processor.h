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
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/spi.h"
#include "rocjitsu/vm/amdgpu/workgroup_key.h"

#include "simdojo/sim/component.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
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
  uint64_t queue_desc_va = 0;
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
  explicit CommandProcessor(std::string name) : simdojo::Component(std::move(name)) {
    // Bind the doorbell handler at construction, not in startup(): register_queue()
    // may start the doorbell poll thread (which fires doorbell_event_ via
    // schedule_event_now) as soon as a host-accessible queue is registered, which can
    // happen before startup() runs. Binding here removes that ordering hazard — a
    // handlerless doorbell_event_ would be silently dropped by the engine.
    doorbell_event_.set_handler(
        [this](simdojo::Tick ts, simdojo::Message *) { handle_doorbell(ts); });
  }
  ~CommandProcessor() override { stop_doorbell_monitor(); }

  void set_memory(GpuMemory *mem) { memory_ = mem; }
  void add_l2_cache(L2Cache *l2) {
    // Idempotent: the config-driven builder and the Xcd full constructor may
    // both attempt to register the same L2. Avoid duplicate entries so cache
    // maintenance does not flush the same L2 twice.
    if (std::find(l2_caches_.begin(), l2_caches_.end(), l2) == l2_caches_.end())
      l2_caches_.push_back(l2);
  }
  void set_vgpr_granularity(uint32_t g) { vgpr_granularity_ = g; }
  uint32_t vgpr_granularity() const { return vgpr_granularity_; }
  void set_packed_tid(bool v) { packed_tid_ = v; }
  bool packed_tid() const { return packed_tid_; }
  void set_sdma_packet_dialect(SdmaPacketDialect dialect) { sdma_packet_dialect_ = dialect; }
  SdmaPacketDialect sdma_packet_dialect() const { return sdma_packet_dialect_; }
  /// @brief Configure launch and packet behavior derived from the GPU architecture.
  void configure_for_arch(rj_code_arch_t arch);
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
                    uint32_t ring_size);

  void set_plugin_group(std::shared_ptr<ExecutionPluginGroup> pg) {
    plugin_group_ = pg ? pg : ExecutionPluginGroup::empty_group();
    if (completion_) {
      completion_->set_plugin_group(plugin_group_);
    }
  }

  void add_spi(ShaderProcessorInput *spi) { spis_.push_back(spi); }

  void add_compute_unit(ComputeUnitCore *cu) {
    auto port_id = static_cast<simdojo::PortID>(dispatch_ports_.size());
    auto port = std::make_unique<simdojo::Port>("req_" + cu->name(), port_id, this,
                                                simdojo::PortDirection::OUT,
                                                simdojo::PortProtocol::DISPATCH);
    dispatch_ports_.push_back(add_port(std::move(port)));
    cus_.push_back(cu);
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
  /// @details Distinct from dispatched_count(), which counts AQL packets. Used to
  /// observe how a grid is distributed across the XCDs of a multi-XCD SoC. Atomic
  /// because SoC::dispatched_workgroups_per_xcd() reads every XCD's counter, which
  /// under a partitioned engine means reading counters other threads are writing.
  /// Relaxed throughout: it is a statistic, not a synchronization point.
  [[nodiscard]] uint64_t dispatched_workgroups() const {
    return dispatched_workgroups_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] size_t next_cu_index() const { return next_cu_; }

  /// @brief Hardware queues registered with this CP, including fan-out replicas.
  [[nodiscard]] size_t registered_queue_count() const {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    return hw_queues_.size();
  }

  const std::vector<simdojo::Port *> &dispatch_ports() const { return dispatch_ports_; }
  const std::vector<ComputeUnitCore *> &compute_units() const { return cus_; }

  /// @brief Return LDS targets selected by a cluster multicast mask.
  std::vector<ClusterLdsTarget> cluster_lds_targets(uint32_t dispatch_id, uint32_t wg_id,
                                                    uint32_t mcast_mask);

  /// @brief Test-only view of the doorbell monitor lifecycle flag.
  ///
  /// @details Exposes doorbell_running_ so a regression test can observe the
  /// monitor retiring after the last host-accessible queue is destroyed and
  /// restarting when a new one registers. Read under doorbell_thread_mutex_ so it
  /// never races the loop's self-exit or ensure_doorbell_monitor().
  [[nodiscard]] bool doorbell_monitor_running_for_test() {
    std::lock_guard<std::mutex> lock(doorbell_thread_mutex_);
    return doorbell_running_;
  }

private:
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
  void process_aql_packet(const hsa_kernel_dispatch_packet_t &pkt, const HwQueue &queue,
                          uint64_t pkt_addr, HwQueueState &qs,
                          ClusterDispatchShape cluster_shape = {});

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
  /// the one before it. A replica's entry list is only ever a subsequence of the
  /// owner's -- see the fan-out section of docs/vm-design.md for the invariant the
  /// non-replicated packet types have to satisfy.
  void fan_out_dispatch(DispatchEntry &dp);

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
  /// spaces disjoint without a shared counter. Disjointness survives the 32-bit
  /// wraparound only when the XCD count divides 2^32, which every shipped
  /// configuration satisfies; it is ~2^29 dispatches per XCD away in any case.
  uint32_t allocate_dispatch_id() {
    uint32_t id = next_dispatch_id_;
    next_dispatch_id_ += dispatch_id_stride_;
    return id;
  }

  /// @brief Locate the queue state for a (queue_id, process_id) pair.
  /// @returns Pointer into new_queue_states_, or null when not registered. Caller
  /// must hold hw_queue_mutex_ and must not use the result across a registration
  /// change.
  HwQueueState *find_queue_state(uint32_t queue_id, uint32_t process_id);

  void register_cluster_workgroup(const DispatchEntry &entry, uint32_t local_wg_id,
                                  uint32_t global_wg_id, ComputeUnitCore *cu, uint32_t lds_base);
  void mark_cluster_workgroup_complete(uint32_t dispatch_id, uint32_t wg_id);
  void erase_cluster_workgroup(uint32_t dispatch_id, uint32_t wg_id);
  void erase_cluster_workgroups(uint32_t dispatch_id);

  /// @brief Asynchronous Compute Engine (ACE): dispatch workgroups from all
  /// active queues to SPIs and run CUs to completion.
  bool ace_dispatch_all();

  /// @brief Process all queues: dispatch undispatched entries, handle non-kernel entries.
  void process_queues();

  /// @brief Called from CU on_idle callback. In functional mode with quantum>0,
  /// checks for stalled dispatches that can resume.
  void on_cu_idle();

  /// @brief Queue scheduling: select next queue with undispatched entries.
  HwQueueState *schedule_next_queue();

  /// @brief Check if barrier is satisfied for an entry.
  bool barrier_satisfied(const HwQueueState &qs, size_t idx) const;

  /// @brief Return total pending entries across all queues.
  size_t pending_entries() const {
    size_t total = 0;
    for (auto &qs : new_queue_states_)
      total += qs.entries.size();
    return total;
  }

  bool has_kfd_queues() const {
    for (const auto &q : hw_queues_)
      if (q.host_accessible)
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
  uint32_t vgpr_granularity_ = 8;
  bool packed_tid_ = false;
  // GFX11+ SDMA GCR keeps the same opcode but changes packet size/layout, so
  // the decoder cannot infer this dialect from the packet header alone.
  SdmaPacketDialect sdma_packet_dialect_ = SdmaPacketDialect::Legacy;
  uint32_t next_dispatch_id_ = 1;
  // Step between successive dispatch ids from this CP. Set to the XCD count when
  // the SoC wires the topology so no two XCDs ever mint the same id.
  uint32_t dispatch_id_stride_ = 1;
  size_t total_dispatched_ = 0;
  std::atomic<uint64_t> dispatched_workgroups_{0};

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

  simdojo::Event doorbell_event_{this, simdojo::EventType::TIMER_CALLBACK};
  mutable std::recursive_mutex hw_queue_mutex_;

  std::shared_ptr<ExecutionPluginGroup> plugin_group_ = ExecutionPluginGroup::empty_group();

  /// @brief Read a uint64 from GPU virtual address space via GpuMemory translation.
  uint64_t read_gpu_u64(uint64_t va, uint32_t vmid) const;

  /// @brief Read a uint32 from GPU virtual address space via GpuMemory translation.
  uint32_t read_gpu_u32(uint64_t va, uint32_t vmid) const;

  /// @brief Read a block of bytes from GPU virtual address space into a buffer.
  void read_gpu_block(uint64_t va, void *dst, size_t size, uint32_t vmid) const;

  /// @brief Write a block of bytes to GPU virtual address space from a buffer.
  void write_gpu_block(uint64_t va, const void *src, size_t size, uint32_t vmid);

  void stop_doorbell_monitor();
  /// @brief Start the doorbell monitor if one is not already running, reaping a
  /// previously self-exited thread first. Serialized by doorbell_thread_mutex_.
  /// @details Caller MUST NOT hold hw_queue_mutex_: this may join a monitor that
  /// self-exited, and that monitor's final self-exit check takes hw_queue_mutex_,
  /// so joining under it would deadlock. This also fixes the lock order to
  /// doorbell_thread_mutex_ -> hw_queue_mutex_ (never the reverse).
  void ensure_doorbell_monitor();
  bool scan_doorbells();

  InterruptCallback interrupt_cb_;
  ScratchBackingResolver scratch_resolver_;
  ScratchBackingAllocator scratch_allocator_;
  std::unique_ptr<CompletionTracker> completion_;

  std::atomic<bool> invalid_pending_{false};

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
  // distinct from hw_queue_mutex_. An empty monitor exits itself when it observes
  // the last host-accessible queue gone (so unregister_queue never has to join a
  // thread that may be mid-iteration inside engine/event code — that join could
  // deadlock, which is why queue destruction only removes queue state). A later
  // register_queue reaps the already-exited jthread and starts a fresh one. Taking
  // doorbell_thread_mutex_ (never nested under hw_queue_mutex_) keeps concurrent
  // register/unregister from racing on doorbell_thread_ and doorbell_running_.
  std::mutex doorbell_thread_mutex_;
  // True while a monitor is running or about to run. The monitor clears it under
  // doorbell_thread_mutex_ just before returning, so ensure_doorbell_monitor() can
  // tell a live monitor from a self-exited one that still needs joining.
  bool doorbell_running_ = false;
  std::jthread doorbell_thread_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_COMMAND_PROCESSOR_H_
