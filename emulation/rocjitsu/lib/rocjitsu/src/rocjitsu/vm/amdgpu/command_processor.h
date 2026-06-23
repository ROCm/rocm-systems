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

#include "rocjitsu/vm/amdgpu/completion_tracker.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/spi.h"

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
};

enum class SdmaPacketDialect {
  Legacy,
  Gfx11Plus,
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
  explicit CommandProcessor(std::string name) : simdojo::Component(std::move(name)) {}
  ~CommandProcessor() override { stop_doorbell_monitor(); }

  void set_memory(GpuMemory *mem) { memory_ = mem; }
  void add_l2_cache(L2Cache *l2) { l2_caches_.push_back(l2); }
  void set_vgpr_granularity(uint32_t g) { vgpr_granularity_ = g; }
  uint32_t vgpr_granularity() const { return vgpr_granularity_; }
  void set_packed_tid(bool v) { packed_tid_ = v; }
  bool packed_tid() const { return packed_tid_; }
  void set_sdma_packet_dialect(SdmaPacketDialect dialect) { sdma_packet_dialect_ = dialect; }
  SdmaPacketDialect sdma_packet_dialect() const { return sdma_packet_dialect_; }
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

  void register_queue(HwQueue queue);
  void unregister_queue(uint32_t queue_id, uint32_t process_id);
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

  size_t dispatched_count() const { return total_dispatched_; }

  size_t next_cu_index() const { return next_cu_; }

  const std::vector<simdojo::Port *> &dispatch_ports() const { return dispatch_ports_; }
  const std::vector<ComputeUnitCore *> &compute_units() const { return cus_; }

private:
  /// @brief Initialize a wavefront's registers per the AMDHSA ABI.
  void init_wavefront_regs(ComputeUnitCore *cu, Wavefront *wf, const DispatchEntry &entry,
                           uint32_t global_wg_id, uint32_t wf_index_in_wg);

  void handle_doorbell(simdojo::Tick timestamp);

  /// @brief Fetch AQL packets from all registered HW queues.
  void fetch_packets();

  /// @brief Fetch AQL packets from a single HW queue.
  void fetch_from_queue(HwQueue &queue, HwQueueState &qs);

  /// @brief Process SDMA packets from an SDMA queue's ring buffer.
  void process_sdma_ring(HwQueue &queue, uint64_t read_idx, uint64_t write_idx);

  /// @brief Parse an AQL dispatch packet, read its kernel descriptor, and create a DispatchEntry.
  void process_aql_packet(const hsa_kernel_dispatch_packet_t &pkt, const HwQueue &queue,
                          uint64_t pkt_addr, HwQueueState &qs);

  rocr::llvm::amdhsa::kernel_descriptor_t
  read_kernel_descriptor(uint64_t kernel_object, uint32_t vmid, bool host_accessible = false);
  /// @brief Dispatch workgroups from entry to CUs. Returns number dispatched.
  uint32_t dispatch_workgroups(DispatchEntry &entry);

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
    return sdma_packet_dialect_ == SdmaPacketDialect::Gfx11Plus;
  }

  GpuMemory *memory_ = nullptr;
  std::vector<ShaderProcessorInput *> spis_;
  std::vector<L2Cache *> l2_caches_;
  std::vector<HwQueue> hw_queues_;
  std::vector<HwQueueState> new_queue_states_;
  std::vector<ComputeUnitCore *> cus_;
  std::vector<simdojo::Port *> dispatch_ports_;

  size_t next_cu_ = 0;
  size_t next_queue_idx_ = 0;
  bool is_primary_ = false;
  uint32_t workgroup_id_offset_ = 0;
  uint32_t vgpr_granularity_ = 8;
  bool packed_tid_ = false;
  // GFX11+ SDMA GCR keeps the same opcode but changes packet size/layout, so
  // the decoder cannot infer this dialect from the packet header alone.
  SdmaPacketDialect sdma_packet_dialect_ = SdmaPacketDialect::Legacy;
  uint32_t next_dispatch_id_ = 1;
  size_t total_dispatched_ = 0;

  simdojo::Event doorbell_event_{this, simdojo::EventType::TIMER_CALLBACK};
  std::recursive_mutex hw_queue_mutex_;

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
  bool scan_doorbells();

  InterruptCallback interrupt_cb_;
  ScratchBackingResolver scratch_resolver_;
  ScratchBackingAllocator scratch_allocator_;
  std::unique_ptr<CompletionTracker> completion_;

  void doorbell_poll_loop(std::stop_token stop);
  std::jthread doorbell_thread_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_COMMAND_PROCESSOR_H_
