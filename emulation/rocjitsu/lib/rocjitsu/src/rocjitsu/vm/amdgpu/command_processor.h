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
/// @see <a
/// href="https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/command-processor.html">ROCm
/// CP documentation</a>

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"

#include "simdojo/sim/component.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rocjitsu/base/rj_compiler.h"
#define HSA_LARGE_MODEL 1
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
#include "hsa/hsa.h"
RJ_DIAGNOSTIC_POP

namespace rocjitsu {
namespace amdgpu {

/// @brief Description of an AQL hardware queue registered with the CP.
struct HwQueue {
  uint32_t queue_id = 0;     ///< Queue ID assigned by the driver.
  uint64_t ring_base_va = 0; ///< GPU VA of the ring buffer.
  uint32_t ring_size = 0;    ///< Ring buffer size in bytes.
  uint64_t read_ptr_va = 0;  ///< GPU VA of the read pointer.
  uint64_t write_ptr_va = 0; ///< GPU VA of the write pointer.
  /// @brief Byte offset of this queue's doorbell slot within the aperture page.
  /// Used by KFD (host_accessible=true) queues. The CP resolves the absolute
  /// address as doorbell_base + doorbell_offset, mirroring how hardware uses
  /// the BAR base + cp_hqd_pq_doorbell_control offset from the MQD.
  uint32_t doorbell_offset = 0;
  /// @brief GPU VA of the doorbell slot for internal simulation queues (host_accessible=false).
  /// Internal test queues write through GpuMemory; KFD queues use doorbell_offset instead.
  uint64_t doorbell_va = 0;
  uint64_t last_doorbell = 0;   ///< Last-seen doorbell value (for change detection).
  bool host_accessible = false; ///< True if pointers are host VAs (KFD queues).
  bool is_sdma = false;         ///< True for SDMA queues (DMA packet format, not AQL).
};

/// @brief AMDGPU command processor that dispatches wavefronts to compute units.
///
/// @details Distributes AQL dispatch packets across the registered compute units in
/// round-robin order, activating pre-allocated wavefront slots.
///
/// Event-driven: the CP monitors registered hardware queue doorbells via a
/// polling thread. When new AQL packets are detected, it fetches them from the
/// ring buffer, parses the kernel descriptor, and dispatches wavefronts to CUs.
/// When all CUs become idle, the on_idle callback signals completion.
class CommandProcessor : public simdojo::Component {
public:
  /// @brief Construct a command processor.
  /// @param name Human-readable name (e.g., "cp").
  explicit CommandProcessor(std::string name) : simdojo::Component(std::move(name)) {}
  ~CommandProcessor() override { stop_doorbell_monitor(); }

  /// @brief Set the GPU memory interface for the fetcher.
  void set_memory(GpuMemory *mem) { memory_ = mem; }

  /// @brief Set the VGPR granularity for decoding the kernel descriptor.
  /// GFX9 (CDNA): 4. GFX940+ (CDNA3/CDNA4): 8.
  void set_vgpr_granularity(uint32_t g) { vgpr_granularity_ = g; }

  /// @brief Enable packed workitem IDs (PackedTID target feature).
  /// @details CDNA3/4 (gfx90a/gfx942/gfx950) pack X/Y/Z workitem IDs into
  /// VGPR0 as 10-bit fields: [9:0]=X, [19:10]=Y, [29:20]=Z.
  void set_packed_tid(bool v) { packed_tid_ = v; }

  /// @brief Set the base address of the doorbell aperture page for KFD queues.
  ///
  /// @details Called by the driver once after mmap'ing the doorbell page. All KFD
  /// queues registered before this call (via register_queue) start being polled
  /// only after the base is set. Mirrors the kernel-side model where the CP knows
  /// the BAR base address and per-queue offsets from the MQD independently.
  /// @param base Host pointer to the start of the doorbell aperture page.
  void set_doorbell_base(void *base);

  /// @brief Register a callback invoked when the CP signals AQL packet completion.
  ///
  /// @details Models the hardware interrupt fired by the CP after writing to a
  /// completion signal's event mailbox. The driver registers this to wake any
  /// threads blocked in WAIT_EVENTS without the 100ms polling delay.
  using InterruptCallback = std::function<void(uint32_t event_id)>;
  void set_interrupt_callback(InterruptCallback cb) { interrupt_cb_ = std::move(cb); }

  /// @brief Register a hardware AQL queue for doorbell monitoring and packet fetching.
  /// Starts the doorbell polling thread on first queue registration.
  void register_queue(HwQueue queue);

  /// @brief Unregister a hardware queue. Stops the polling thread when no queues remain.
  void unregister_queue(uint32_t queue_id);

  /// @brief Update ring buffer parameters for an existing hardware queue.
  /// @details Called by the driver on KFD UPDATE_QUEUE (e.g., after preemption/resume).
  /// Updates ring_base_va and ring_size under hw_queue_mutex_ so the poll thread
  /// sees the new ring without tearing.
  void update_queue(uint32_t queue_id, uint64_t ring_base_va, uint32_t ring_size);

  /// @brief Set the execution plugin group (shared ownership).
  void set_plugin_group(std::shared_ptr<ExecutionPluginGroup> pg) {
    plugin_group_ = pg ? pg : ExecutionPluginGroup::empty_group();
  }

  /// @brief Register a compute unit that this CP can dispatch to.
  ///
  /// @details Creates a dispatch OUT port for the CU and registers an on_idle
  /// callback. The port is wired to the CU's dispatch IN port during
  /// Xcd::initialize() via the topology.
  /// @param cu Pointer to the compute unit (must outlive the CP).
  void add_compute_unit(ComputeUnitCore *cu) {
    auto port_id = static_cast<simdojo::PortID>(dispatch_ports_.size());
    auto port = std::make_unique<simdojo::Port>("req_" + cu->name(), port_id, this,
                                                simdojo::PortDirection::OUT,
                                                simdojo::PortProtocol::DISPATCH);
    dispatch_ports_.push_back(add_port(std::move(port)));
    cus_.push_back(cu);
    cu->set_on_idle([this]() { check_all_idle(); });
  }

  /// @brief Set up the doorbell event handler.
  void startup() override;

  /// @brief Process one dispatch from the internal queue: create wavefronts and assign to CUs.
  /// @retval true More dispatches to process.
  /// @retval false Dispatch queue is empty.
  bool step() override;

  /// @brief Return the reusable doorbell event (for external scheduling).
  /// @returns Pointer to the doorbell event.
  simdojo::Event *doorbell_event() { return &doorbell_event_; }

  /// @brief Check whether all CUs are idle and no dispatches remain.
  ///
  /// Called from CU on_idle callbacks. If all CUs are idle and the dispatch
  /// queue is empty, signals that this primary component is done.
  void check_all_idle();

  /// @brief Set a workgroup ID offset for multi-XCD dispatch.
  /// @details In multi-XCD configurations, each XCD's CP is assigned a different
  /// offset so that workgroup IDs are globally unique across XCDs.
  void set_workgroup_id_offset(uint32_t offset) { workgroup_id_offset_ = offset; }

  /// @brief Return the number of dispatches processed so far.
  size_t dispatched_count() const { return dispatched_; }

  /// @brief Return the next CU index for round-robin dispatch.
  size_t next_cu_index() const { return next_cu_; }

  /// @brief Return the dispatch OUT ports (one per registered CU).
  /// @returns Const reference to the vector of dispatch ports.
  const std::vector<simdojo::Port *> &dispatch_ports() const { return dispatch_ports_; }

  /// @brief Return the registered compute units.
  /// @returns Const reference to the vector of compute unit pointers.
  const std::vector<ComputeUnitCore *> &compute_units() const { return cus_; }

private:
  struct InternalDispatch {
    uint64_t kernel_entry_pc = 0;
    uint32_t workgroup_count = 1;
    uint32_t wfs_per_workgroup = 1;
    uint32_t sgprs_per_wf = 104;
    uint32_t vgprs_per_wf = 256;
    uint64_t kernarg_addr = 0;
    uint32_t num_user_sgprs = 2;
    uint32_t kernel_code_properties = 0; ///< AMDHSA kernel_code_properties bitfield.
    uint64_t dispatch_ptr = 0;           ///< Host address of the AQL dispatch packet.
    uint64_t queue_ptr = 0;              ///< Host address of the amd_queue_t struct.
    uint32_t workgroup_id_offset = 0;
    uint32_t grid_wgs_x = 0;
    uint32_t grid_wgs_y = 1;
    uint32_t grid_wgs_z = 1;
    bool enable_wg_id_x = true;
    bool enable_wg_id_y = false;
    bool enable_wg_id_z = false;
    uint8_t enable_vgpr_workitem_id = 0; ///< 0=X, 1=X+Y, 2=X+Y+Z (from compute_pgm_rsrc2)
    uint16_t workgroup_size_x = 64;      ///< Workgroup dims for workitem ID decomposition.
    uint16_t workgroup_size_y = 1;
    uint16_t workgroup_size_z = 1;
    uint64_t completion_signal = 0;    ///< AQL completion signal handle (0 = none).
    uint64_t scratch_backing_addr = 0; ///< GPU VA of scratch backing memory (from amd_queue_t).
    uint32_t private_segment_fixed_size =
        0; ///< Per-lane scratch size in bytes (from kernel descriptor).
    uint32_t group_segment_fixed_size = 0; ///< Per-WG LDS size in bytes (from kernel descriptor).
    bool host_signal = false;              ///< True if signal is in host memory (KFD path).
    bool ordered = false;                  ///< True for KFD (host-accessible) queue dispatches.
  };

  /// @brief Initialize a wavefront's registers per the AMDHSA ABI.
  void init_wavefront_regs(ComputeUnitCore *cu, Wavefront *wf, const InternalDispatch &pkt,
                           uint32_t global_wg_id, uint32_t wf_index_in_wg);

  /// @brief Doorbell event handler: check HW queues, fetch packets, dispatch, activate CUs.
  void handle_doorbell(simdojo::Tick timestamp);

  /// @brief Fetch AQL packets from all registered HW queues into the dispatch queue.
  void fetch_packets();

  /// @brief Fetch AQL packets from a single HW queue.
  void fetch_from_queue(HwQueue &queue);

  /// @brief Process SDMA packets from an SDMA queue's ring buffer.
  /// Handles COPY_LINEAR (memcpy), FENCE (write), TRAP (interrupt),
  /// POLL_REGMEM (spin-wait), ATOMIC (fetch-add), and CONST_FILL.
  void process_sdma_ring(HwQueue &queue, uint64_t read_idx, uint64_t write_idx);

  /// @brief Parse an AQL dispatch packet, read its kernel descriptor, and enqueue an
  /// InternalDispatch.
  void process_aql_packet(const hsa_kernel_dispatch_packet_t &pkt, const HwQueue &queue,
                          uint64_t pkt_addr);

  /// @brief Read a kernel descriptor from memory.
  rocr::llvm::amdhsa::kernel_descriptor_t read_kernel_descriptor(uint64_t kernel_object,
                                                                 bool host_accessible = false);
  void signal_aql_completion(uint64_t pkt_addr);

  /// @brief Return the number of pending dispatch entries.
  size_t pending_dispatches() const {
    return (dispatched_ <= dispatch_queue_.size()) ? dispatch_queue_.size() - dispatched_ : 0;
  }

  GpuMemory *memory_ = nullptr; ///< GPU memory for the fetcher.
  /// @brief Base host pointer for the doorbell aperture page (KFD queues).
  /// Analogous to the GPU BAR base: per-queue doorbell_offset indexes into it.
  /// Synchronised via release/acquire on the atomic itself — set_doorbell_base
  /// does NOT hold hw_queue_mutex_. All readers use memory_order_acquire.
  std::atomic<void *> doorbell_base_{nullptr};
  std::vector<HwQueue> hw_queues_;               ///< Registered AQL hardware queues.
  std::vector<ComputeUnitCore *> cus_;           ///< Registered compute units.
  std::vector<simdojo::Port *> dispatch_ports_;  ///< One OUT port per CU (parallel to cus_).
  std::vector<InternalDispatch> dispatch_queue_; ///< Pending internal dispatches.

  size_t dispatched_ = 0;            ///< Number of dispatches already processed.
  size_t next_cu_ = 0;               ///< Round-robin CU index.
  bool is_primary_ = false;          ///< True if CP registered as primary.
  uint32_t workgroup_id_offset_ = 0; ///< Multi-XCD workgroup ID offset.
  uint32_t vgpr_granularity_ = 8;    ///< VGPR allocation granularity (4 for GFX9, 8 for GFX940+).
  bool packed_tid_ = false;          ///< PackedTID: pack X/Y/Z into VGPR0 as 10-bit fields.

  /// @brief Reusable doorbell event. Fired when packets need processing.
  simdojo::Event doorbell_event_{this, simdojo::EventType::TIMER_CALLBACK};

  /// @brief Retry queue for workgroups that could not be dispatched because all CUs were full.
  std::vector<InternalDispatch> retry_queue_;

  /// @brief Protects hw_queues_ (accessed from both the polling thread and register_queue).
  std::mutex hw_queue_mutex_;

  std::shared_ptr<ExecutionPluginGroup> plugin_group_ = ExecutionPluginGroup::empty_group();

  uint64_t read_gpu_u64(uint64_t va) const;
  void stop_doorbell_monitor();
  /// @brief Scan all registered HW queues for doorbell changes. Returns true if any changed.
  bool scan_doorbells();

  InterruptCallback interrupt_cb_; ///< Fired after writing to a signal's event mailbox.

  /// @brief Doorbell polling thread. Monitors registered HW queue doorbells
  /// and injects doorbell events when new packets are detected.
  /// Polls at 100µs intervals.
  void doorbell_poll_loop(std::stop_token stop);
  std::jthread doorbell_thread_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_COMMAND_PROCESSOR_H_
