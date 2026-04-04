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

#include "simdojo/sim/component.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// @brief A dispatch packet describing a kernel launch.
///
/// @details Contains the kernel entry point, grid dimensions, wavefront
/// configuration, and register initialization parameters. The command
/// processor reads these to create wavefronts and initialize their user
/// SGPRs (kernarg pointer, etc.), system SGPRs (workgroup ID), and
/// workitem IDs (v0 = lane index).
struct DispatchPacket {
  uint64_t kernel_entry_pc = 0;   ///< Byte address of kernel code.
  uint32_t workgroup_count = 1;   ///< Number of workgroups to dispatch.
  uint32_t wfs_per_workgroup = 1; ///< Wavefronts per workgroup.
  uint32_t sgprs_per_wf = 104;    ///< Scalar GPRs per wavefront (from code object).
  uint32_t vgprs_per_wf = 256;    ///< Vector GPRs per wavefront (from code object).

  /// @brief Kernarg segment base address in GPU memory.
  ///
  /// @details When non-zero, written to s[0:1] as the first two user SGPRs.
  /// On gfx9+, the compiler always places the kernarg pointer in s[0:1]
  /// regardless of kernel_code_properties flags.
  uint64_t kernarg_addr = 0;

  /// @brief Number of user SGPRs preceding the system SGPRs (from RSRC2).
  ///
  /// @details The command processor writes workgroup_id_x at
  /// `sgpr_alloc.base + num_user_sgprs`. This value should match the
  /// `user_sgpr_count` field from COMPUTE_PGM_RSRC2 in the kernel descriptor.
  uint32_t num_user_sgprs = 2;

  /// @brief Offset added to local workgroup IDs to produce global IDs.
  ///
  /// @details When dispatching across multiple XCDs, each XCD receives a
  /// contiguous range of workgroup IDs. The system SGPR (workgroup_id_x)
  /// is written as `local_wg_id + workgroup_id_offset`.
  uint32_t workgroup_id_offset = 0;
};

/// @brief AMDGPU command processor that dispatches wavefronts to compute units.
///
/// @details Distributes dispatch packets across the registered compute units in
/// round-robin order, activating pre-allocated wavefront slots.
///
/// Event-driven: during startup(), the CP schedules a doorbell event if
/// pre-loaded packets exist. External submissions (from the Driver) inject
/// doorbell events via schedule_event_async(). The doorbell handler processes
/// all pending packets and activates CUs. When all CUs become idle,
/// the on_idle callback signals completion to the primary protocol.
class CommandProcessor : public simdojo::Component {
public:
  /// @brief Construct a command processor.
  /// @param name Human-readable name (e.g., "cp").
  explicit CommandProcessor(std::string name) : simdojo::Component(std::move(name)) {}

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

  /// @brief Enqueue a dispatch packet for processing before simulation starts.
  ///
  /// @details Used by the config loader and tests to pre-load work. Must be called
  /// before startup(). During startup(), the CP schedules an initial doorbell
  /// event if any packets are pre-loaded.
  /// @param packet The dispatch parameters.
  void enqueue(DispatchPacket packet) { dispatch_queue_.push_back(std::move(packet)); }

  /// @brief Submit a dispatch packet from an external thread.
  ///
  /// @details Thread-safe. Buffers the packet and injects a doorbell event via
  /// schedule_event_async(). The CP processes it at the next event processing cycle.
  /// @param packet The dispatch parameters.
  void submit(DispatchPacket packet);

  /// @brief Register as primary and schedule initial doorbell if work is pre-loaded.
  void startup() override;

  /// @brief Process one dispatch packet: create wavefronts and assign to CUs.
  /// @retval true More dispatch packets to process.
  /// @retval false Dispatch queue is empty.
  bool step() override;

  /// @brief Return the number of pending dispatch packets.
  /// @returns Number of unprocessed packets in the dispatch queue.
  size_t pending_dispatches() const {
    return (dispatched_ <= dispatch_queue_.size()) ? dispatch_queue_.size() - dispatched_ : 0;
  }

  /// @brief Return the reusable doorbell event (for external scheduling).
  /// @returns Pointer to the doorbell event.
  simdojo::Event *doorbell_event() { return &doorbell_event_; }

  /// @brief Check whether all CUs are idle and no dispatches remain.
  ///
  /// Called from CU on_idle callbacks. If all CUs are idle and the dispatch
  /// queue is empty, signals that this primary component is done.
  void check_all_idle();

  /// @brief Return the number of packets already processed.
  /// @returns Count of dispatched packets.
  size_t dispatched_count() const { return dispatched_; }

  /// @brief Return the current round-robin CU index.
  /// @returns Index of the next CU to receive a dispatch.
  size_t next_cu_index() const { return next_cu_; }

  /// @brief Return the dispatch OUT ports (one per registered CU).
  /// @returns Const reference to the vector of dispatch ports.
  const std::vector<simdojo::Port *> &dispatch_ports() const { return dispatch_ports_; }

  /// @brief Return the registered compute units.
  /// @returns Const reference to the vector of compute unit pointers.
  const std::vector<ComputeUnitCore *> &compute_units() const { return cus_; }

private:
  /// @brief Doorbell event handler: drain async queue, dispatch, activate CUs.
  void handle_doorbell(simdojo::Tick timestamp);

  /// @brief Drain async submissions into the dispatch queue.
  void drain_async_submissions();

  std::vector<ComputeUnitCore *> cus_;          ///< Registered compute units.
  std::vector<simdojo::Port *> dispatch_ports_; ///< One OUT port per CU (parallel to cus_).
  std::vector<DispatchPacket> dispatch_queue_;  ///< Pending dispatch packets.

  /// @brief Dispatch state — accessed only from the owning partition's thread.
  ///
  /// @details Thread safety is guaranteed by the PDES engine's partition isolation:
  /// step() and handle_doorbell() run exclusively on the owning thread.
  /// External submissions go through async_submissions_ (mutex-guarded).
  size_t dispatched_ = 0; ///< Number of packets already processed.
  size_t next_cu_ = 0;    ///< Round-robin CU index.

  /// @brief Reusable doorbell event. Fired when packets need processing.
  simdojo::Event doorbell_event_{this, simdojo::EventType::TIMER_CALLBACK};

  /// @brief Retry queue for workgroups that could not be dispatched because all CUs were full.
  /// Re-drained into dispatch_queue_ on the next doorbell event.
  std::vector<DispatchPacket> retry_queue_;

  /// @brief Async submission buffer for external threads (guarded by async_mutex_).
  std::mutex async_mutex_;
  std::vector<DispatchPacket> async_submissions_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_COMMAND_PROCESSOR_H_
