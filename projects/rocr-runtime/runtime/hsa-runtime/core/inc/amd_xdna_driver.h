////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_AMD_XDNA_DRIVER_H_
#define HSA_RUNTIME_CORE_INC_AMD_XDNA_DRIVER_H_

#include <memory>
#include <string>
#include <vector>

#include "core/inc/amd_aie_agent.h"
#include "core/inc/driver.h"
#include "core/inc/memory_region.h"
#include "inc/hsa_ext_amd_aie.h"

namespace rocr {

namespace AMD {

/// @brief AMD XDNA Driver for AMD AIE agents.
///
/// @details The user-mode driver for AMD AIE that provides APIs for the ROCr core to allocate
/// memory, manage DMA buffers, allocate queues, and more.
class XdnaDriver final : public core::Driver {
public:
  XdnaDriver(std::string devnode_name);

  /// @brief Determine if the xdna-driver is present on the system and attempt to open it if found.
  ///
  /// @param[out] driver object
  static hsa_status_t DiscoverDriver(std::unique_ptr<core::Driver>& driver);

  /// @brief Returns the size of the dev heap in bytes. This is the size reserved when a dev heap
  /// exists on this device; check @ref HasDevHeap first, since UMQ (AIE4) devices have none.
  static uint64_t GetDevHeapByteSize();

  /// @brief Returns whether this device has a dev heap. KMQ (aie2/aie2p) devices do; UMQ (AIE4)
  /// devices do not, since the kernel driver rejects AMDXDNA_BO_DEV_HEAP creation for them.
  bool HasDevHeap() const { return has_dev_heap_; }

  hsa_status_t Init() override;
  hsa_status_t ShutDown() override;
  hsa_status_t QueryKernelModeDriver(core::DriverQuery query) override;
  hsa_status_t Open() override;
  hsa_status_t Close() override;
  hsa_status_t GetSystemProperties(HsaSystemProperties& sys_props) const override;
  hsa_status_t GetNodeProperties(HsaNodeProperties& node_props, uint32_t node_id) const override;
  hsa_status_t GetEdgeProperties(std::vector<HsaIoLinkProperties>& io_link_props,
                                 uint32_t node_id) const override;
  hsa_status_t GetMemoryProperties(uint32_t node_id,
                                   std::vector<HsaMemoryProperties>& mem_props) const override;
  hsa_status_t GetCacheProperties(uint32_t node_id, uint32_t processor_id,
                                  std::vector<HsaCacheProperties>& cache_props) const override;
  hsa_status_t AllocateMemory(const core::MemoryRegion& mem_region,
                              core::MemoryRegion::AllocateFlags alloc_flags, size_t size,
                              uint32_t node_id, core::DriverMemoryHandle* handle) override;
  hsa_status_t FreeMemory(const core::DriverMemoryHandle& handle) override;
  hsa_status_t QueryPointerInfo(const void* ptr, const core::MemoryRegion* region,
                                core::MemoryRegion::AllocateFlags alloc_flags,
                                const core::DriverMemoryHandle* handle,
                                HsaPointerInfo* info) const override;
  hsa_status_t CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE type, uint32_t queue_pct,
                           HSA::hsa_amd_queue_priority_internal_t priority, uint32_t sdma_engine_id, void* queue_addr,
                           uint64_t queue_size_bytes, uint64_t queue_metadata_size_bytes, HsaEvent* event,
                           HsaQueueResource& queue_resource) const override;
  hsa_status_t UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_pct, HSA::hsa_amd_queue_priority_internal_t priority,
                           void* queue_addr, uint64_t queue_size, HsaEvent* event) const override;
  hsa_status_t DestroyQueue(HSA_QUEUEID queue_id) const override;

  /// @brief Create Kernel Mode Queue (KMQ) metadata to dispatch packets in a user-mode access agent
  /// dispatch queue.
  ///
  /// @param[in] queue_size size of the dispatch queue in number of packets
  /// @param[in] num_cols number of columns to create the queue's initial hardware context with.
  /// Dispatch resizes the context to the columns its kernels declare.
  /// @param[in] num_core_rows number of core tiles in each column
  /// @param[in] device_id PCI device ID of the agent the queue dispatches to, resolved here so
  /// dispatch does not look the device type up per batch
  /// @param[out] queue_metadata KMQ metadata created for the dispatch queue
  hsa_status_t CreateKernelModeQueue(size_t queue_size, uint32_t num_cols, uint32_t num_core_rows,
                                     uint16_t device_id, void** queue_metadata) const;

  /// @brief Destroy the Kernel Mode Queue (KMQ) metadata.
  ///
  /// @note This function will also destroy the hardware context associated with the KMQ. Even if
  /// that fails, the metadata is still considered destroyed and the function will return the error
  /// from destroying the hardware context.
  ///
  /// @param[in] queue_metadata KMQ metadata to be destroyed
  hsa_status_t DestroyKernelModeQueue(void* queue_metadata) const;

  /// @brief Returns whether this device submits through a user-mode queue (UMQ) rather than a
  /// kernel-mode queue (KMQ), per the kernel driver's own device_type. Defaults to false (KMQ) if
  /// the device_type cannot be read.
  bool UsesUserModeQueue() const;

  /// @brief Create User Mode Queue (UMQ) metadata: the hardware context created against the
  /// queue's own ring, and the mapped doorbell, for an AIE4 agent dispatch queue.
  ///
  /// Unlike KMQ, UMQ has no separate ring: @p ring_buf (the queue's own HSA-visible AQL packet
  /// ring) is handed to the device directly, and dispatch is a single doorbell write with no
  /// per-batch driver call -- the same model a GPU hardware queue uses.
  ///
  /// @param[in] ring_buf the queue's AQL packet ring (hsa_queue_t::base_address)
  /// @param[in] agent agent the queue dispatches to, to resolve @p ring_buf's BO handle
  /// @param[in] num_cols number of columns to create the queue's hardware context with
  /// @param[in] num_core_rows number of core tiles in each column
  /// @param[out] queue_metadata UMQ metadata created for the dispatch queue
  /// @param[out] doorbell the mapped hardware doorbell to ring on dispatch
  hsa_status_t CreateUserModeQueue(void* ring_buf, const core::Agent& agent, uint32_t num_cols,
                                   uint32_t num_core_rows, void** queue_metadata,
                                   volatile uint32_t** doorbell) const;

  /// @brief Destroy the User Mode Queue (UMQ) metadata: its hardware context and doorbell mapping.
  ///
  /// @param[in] queue_metadata UMQ metadata to be destroyed
  hsa_status_t DestroyUserModeQueue(void* queue_metadata) const;

  /// @brief Prepares a range of UMQ packets for CERT to execute.
  ///
  /// hsa_amd_aie_kernel_dispatch_packet_t was written to be binary-compatible with CERT's own
  /// struct host_queue_packet / exec_buf: same bytes, reinterpreted. Every field the application
  /// filled in already carries what CERT wants, except kernel_object -- the application treats it
  /// as a host AieKernelDescriptor* (the same handle KMQ dispatch resolves it as), but CERT reads
  /// the same bytes as a raw device address (struct exec_buf's dpu_control_code_host_addr_low/
  /// high). CERT does not patch argument addresses into that control code itself either (confirmed
  /// on real hardware: it faults on the IOMMU at the ELF's small placeholder offsets if nothing
  /// else patches them first), so this allocates a fresh per-dispatch copy of each packet's
  /// kernel's pristine control code, folds that dispatch's kernarg addresses into it (exactly like
  /// @ref SubmitCmdChain's full-ELF path does for aie2p), and overwrites kernel_object with the
  /// copy's address -- never the shared pristine one.
  ///
  /// Unlike @ref SubmitCmdChain, this never touches the device: there is no per-batch driver call
  /// for UMQ, so the caller still owns publishing the new write index and ringing the doorbell.
  ///
  /// @param[in,out] ring the queue's packet ring; only kernel_object_low/high are modified
  /// @param[in] ring_capacity number of packet slots in @p ring, for wrapping the index
  /// @param[in] first_pkt_idx index of the first packet to translate
  /// @param[in] num_pkts number of packets to translate, starting at @p first_pkt_idx. Must be
  /// greater than 0.
  /// @param[in] agent agent that owns the queue
  /// @retval HSA_STATUS_ERROR_INVALID_CODE_OBJECT a packet's kernel_object does not name a loaded
  /// descriptor
  /// @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES a per-dispatch control-code copy could not be
  /// allocated
  hsa_status_t TranslateUmqPackets(hsa_amd_aie_kernel_dispatch_packet_t* ring,
                                   uint32_t ring_capacity, uint64_t first_pkt_idx,
                                   uint64_t num_pkts, const core::Agent& agent) const;

  hsa_status_t SetQueueCUMask(HSA_QUEUEID queue_id, uint32_t cu_mask_count,
                              uint32_t* queue_cu_mask) const override;
  hsa_status_t AllocQueueGWS(HSA_QUEUEID queue_id, uint32_t num_gws,
                             uint32_t* first_gws) const override;
  hsa_status_t ExportMemoryHandle(const core::Agent& agent, const core::DriverMemoryHandle& handle,
                                  core::ShareType type, void* export_handle) override;
  hsa_status_t ImportMemoryHandle(const core::Agent& agent, core::DriverMemoryHandle* handle,
                                  core::ShareType type, void* import_handle,
                                  void* mem = nullptr) override;
  hsa_status_t Map(const core::DriverMemoryHandle& handle, void *mem, size_t offset,
                   size_t size, hsa_access_permission_t perms, uint32_t node_id) override;
  hsa_status_t Unmap(const core::DriverMemoryHandle& handle, void *mem, size_t offset,
                     size_t size, uint32_t node_id) override;
  hsa_status_t CreateShareableHandle(core::DriverMemoryHandle* handle, const core::Agent& agent,
                                     uint64_t* offset) override;
  hsa_status_t DestroyMemoryHandle(core::DriverMemoryHandle* handle) override;

  /// @brief Submits packets to the driver for execution.
  ///
  /// @note The packets are contiguous in index but not necessarily contiguous in memory.
  ///
  /// @param[in] q queue with packets
  /// @param[in,out] queue_metadata Kernel Mode Queue (KMQ) metadata. It will be updated if the
  /// driver needs to create a new hardware context.
  /// @param[in] first_pkt_idx index of the first packet in the queue
  /// @param[in] num_pkts number of packets in the queue to be submitted. Must be greater than 0.
  /// @param[in] agent agent that owns the queue
  /// @param[out] num_completed how many packets, counting from @p first_pkt_idx, executed and had
  /// their completion signals released. @p num_pkts on success. On failure this is the prefix the
  /// device got through before it stopped, so the caller can consume exactly those and leave the
  /// failing packet and everything after it in the ring.
  hsa_status_t SubmitCmdChain(hsa_queue_t& q, void* queue_metadata, uint64_t first_pkt_idx,
                              uint64_t num_pkts, const core::Agent& agent, uint64_t* num_completed);

  hsa_status_t SPMAcquire(uint32_t preferred_node_id) const override;
  hsa_status_t SPMRelease(uint32_t preferred_node_id) const override;
  hsa_status_t SPMSetDestBuffer(uint32_t preferred_node_id, uint32_t size_bytes, uint32_t* timeout,
                                uint32_t* size_copied, void* dest_mem_addr,
                                bool* is_spm_data_loss) const override;
  hsa_status_t SetTrapHandler(uint32_t node_id, const void* base, uint64_t base_size,
                              const void* buffer_base, uint64_t buffer_base_size) const override;
  hsa_status_t GetDeviceHandle(uint32_t node_id, void** device_handle) const override;
  hsa_status_t GetDeviceFd(uint32_t node_id, int *fd) const override;
  hsa_status_t GetClockCounters(uint32_t node_id, HsaClockCounters* clock_counter) const override;
  hsa_status_t GetTileConfig(uint32_t node_id, HsaGpuTileConfig* config) const override;
  hsa_status_t GetWallclockFrequency(uint32_t node_id, uint64_t* frequency) const override;
  hsa_status_t AllocateScratchMemory(uint32_t node_id, uint64_t size, void** mem) const override;
  hsa_status_t AvailableMemory(uint32_t node_id, uint64_t* available_size) const override;
  hsa_status_t RegisterMemory(void* ptr, uint64_t size, HsaMemFlags mem_flags) const override;
  hsa_status_t DeregisterMemory(void* ptr) const override;
  hsa_status_t MakeMemoryResident(const void* mem, size_t size, uint64_t* alternate_va,
                                  const HsaMemFlags* mem_flags, uint32_t num_nodes,
                                  const uint32_t* nodes) const override;
  hsa_status_t MakeMemoryUnresident(const void* mem) const override;

  hsa_status_t IsModelEnabled(bool* enable) const override;

  hsa_status_t GetQueueSaveAreaInfo(HSA_QUEUEID queue_id, void** address, size_t* size) const override;

  hsa_status_t CheckAcceleratorReadiness(core::Agent& agent, bool* ready) const override;

  hsa_status_t SetPersistingCacheSize(uint32_t node_id, uint64_t cache_size) override {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

 private:
  /// @brief Queries the driver version and updates internal state.
  hsa_status_t QueryDriverVersion();

  /// @brief Allocate device accessible heap (dev heap) space.
  hsa_status_t InitDeviceHeap();

  /// @brief Free device accessible heap (dev heap) space.
  hsa_status_t FreeDeviceHeap();

  /// @brief Device heap BO.
  ///
  /// Its mapping is @ref dev_heap_vaddr, which BO_DEV allocations carve their VA out of.
  uint32_t dev_heap_bo = 0;

  /// @brief The aligned device heap mapping, of size @ref dev_heap_size and alignment
  /// @ref dev_heap_alignment.
  void* dev_heap_vaddr = nullptr;

  /// @brief Whether this device has a dev heap. Set by InitDeviceHeap(); see @ref HasDevHeap.
  bool has_dev_heap_ = false;

  /// @brief Device heap size in bytes.
  static constexpr size_t dev_heap_size = 64 * 1024 * 1024;

  /// @brief Device heap alignment in bytes.
  static constexpr size_t dev_heap_alignment = 64 * 1024 * 1024;
};

} // namespace AMD
} // namespace rocr

#endif // header guard
