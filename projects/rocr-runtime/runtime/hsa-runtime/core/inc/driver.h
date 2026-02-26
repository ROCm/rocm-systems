////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2023-2026, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HSA_RUNTME_CORE_INC_DRIVER_H_
#define HSA_RUNTME_CORE_INC_DRIVER_H_

#include <cstdint>
#include <limits>
#include <string>

#include "core/inc/memory_region.h"
#include "hsakmt/hsakmttypes.h"
#include "inc/hsa.h"
#include "core/inc/hsa_internal.h"

namespace rocr {
namespace core {

class Queue;

enum class DriverQuery { GET_DRIVER_VERSION };

enum class DriverType {
  XDNA = 0,
  KFD,
#ifdef HSAKMT_VIRTIO_ENABLED
  KFD_VIRTIO,
#endif
  NUM_DRIVER_TYPES
};

/// @brief Handle for exported / imported memory.
struct ShareableHandle {
  uint64_t handle{};

  bool IsValid() const { return handle != 0; }
};

/// @brief Kernel driver interface.
///
/// @details A class used to provide an interface between the core runtime
/// and agent kernel drivers. It also maintains state associated with active
/// kernel drivers.
class Driver {
public:
  Driver(DriverType kernel_driver_type, std::string devnode_name);
  virtual ~Driver() = default;

  /// @brief Initialize the driver's state after opening.
  virtual hsa_status_t Init() = 0;

  /// @brief Release the driver's resources and close the kernel-mode
  /// driver.
  virtual hsa_status_t ShutDown() = 0;

  /// @brief Get driver version information.
  /// @retval DriverVersionInfo containing the driver's version information.
  const HsaVersionInfo& Version() const { return version_; }

  /// @brief Query the kernel-model driver.
  /// @retval HSA_STATUS_SUCCESS if the kernel-model driver query was
  /// successful.
  virtual hsa_status_t QueryKernelModeDriver(DriverQuery query) = 0;

  /// @brief Open a connection to the driver using name_.
  /// @retval HSA_STATUS_SUCCESS if the driver was opened successfully.
  virtual hsa_status_t Open() = 0;

  /// @brief Close a connection to the open driver using fd_.
  /// @retval HSA_STATUS_SUCCESS if the driver was opened successfully.
  virtual hsa_status_t Close() = 0;

  /// @brief Get the system properties for nodes managed by this driver.
  virtual hsa_status_t GetSystemProperties(HsaSystemProperties& sys_props) const = 0;

  /// @brief Get the properties for a specific node managed by this driver.
  virtual hsa_status_t GetNodeProperties(HsaNodeProperties& node_props, uint32_t node_id) const = 0;

  /// @brief Get the edge (IO link) properties of a specific node (that is
  /// managed by this driver) in the topology graph.
  /// @param[out] io_link_props IO link properties of the node specified by @p node_id.
  /// @param[in] node_id ID of the node whose link properties are being queried.
  virtual hsa_status_t GetEdgeProperties(std::vector<HsaIoLinkProperties>& io_link_props,
                                         uint32_t node_id) const = 0;

  /// @brief Get the memory properties of a specific node.
  /// @param[in] node_id Node ID of the agent.
  /// @param[out] mem_props Memory properties of the node specified by @p node_id.
  /// @retval HSA_STATUS_SUCCESS if the driver sucessfully returns the node's
  /// memory properties.
  virtual hsa_status_t GetMemoryProperties(uint32_t node_id,
                                           std::vector<HsaMemoryProperties>& mem_props) const = 0;

  /// @brief Get the cache properties of a specific node.
  /// @param[in] node_ide Node ID of the agent.
  /// @param[out] cache_props Cache properties of the node specified by @p node_id.
  /// @retval HSA_STATUS_SUCCESS if the driver successfully returns the node's cache properties.
  virtual hsa_status_t GetCacheProperties(uint32_t node_id, uint32_t processor_id,
                                          std::vector<HsaCacheProperties>& cache_props) const = 0;

  /// @brief Allocate agent-accessible memory (system or agent-local memory).
  /// @param[out] mem pointer to newly allocated memory.
  /// @retval HSA_STATUS_SUCCESS if memory was successfully allocated or
  /// hsa_status_t error code if the memory allocation failed.
  virtual hsa_status_t AllocateMemory(const MemoryRegion &mem_region,
                                      MemoryRegion::AllocateFlags alloc_flags,
                                      void **mem, size_t size,
                                      uint32_t node_id) = 0;

  virtual hsa_status_t FreeMemory(void *mem, size_t size) = 0;

  /// @brief Create an agent dispatch queue with user-mode access rights.
  /// @param[in] node_id Node ID of the agent on which the queue is being created.
  /// @param[in] type Queue's type.
  /// @param[in] queue_pct Maximum percentage of a queue's occupancy allowed.
  /// @param[in] priority Queue's priority for scheduling.
  /// @param[in] sdma_engine_id ID of the SDMA engine on which the queue is being created. Only used
  /// if @p type is one of the SDMA queue types.
  /// @param[in] queue_addr Address of the queue's ring buffer.
  /// @param[in] queue_size_bytes Size of the queue's ring buffer in bytes.
  /// @param[in] event HsaEvent for event-driven callbacks.
  /// @param[out] queue_resource Queue resource information populated by the driver.
  virtual hsa_status_t CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE type, uint32_t queue_pct,
                                   HSA::hsa_amd_queue_priority_internal_t priority, uint32_t sdma_engine_id,
                                   void* queue_addr, uint64_t queue_size_bytes, HsaEvent* event,
                                   HsaQueueResource& queue_resource) const = 0;

  /// @brief Destroy a queue.
  /// @param queue_id Kernel-mode driver's assigned queue ID.
  virtual hsa_status_t DestroyQueue(HSA_QUEUEID queue_id) const = 0;

  /// @brief Update a queue's properties.
  /// @param[in] queue_id Kernel-mode driver's assigned queue ID.
  /// @param[in] queue_pct Maximum percentage of a queue's occupancy allowed.
  /// @param[in] priority Queue's priority for scheduling.
  /// @param[in] queue_addr Queue's ring buffer base address.
  /// @param[in] queue_size_bytes Size of the queue's ring buffer in bytes.
  /// @param[in] event HsaEvent for event-driven callbacks.
  virtual hsa_status_t UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_pct,
                                   HSA::hsa_amd_queue_priority_internal_t priority, void* queue_addr,
                                   uint64_t queue_size_bytes, HsaEvent* event) const = 0;

  /// @brief Set the CU mask for a queue.
  /// @details This sets the CU bitmask for a queue. The CU mask determines which CUs
  /// a queue's dispatches can target. Currently this is only supported for GPU devices.
  /// @param[in] queue_id Kernel-mode driver's assigned queue ID.
  /// @param[in] cu_mask_count Number of CU bits in the mask.
  /// @param[in] queue_cu_mask New CU mask for the queue.
  virtual hsa_status_t SetQueueCUMask(HSA_QUEUEID queue_id, uint32_t cu_mask_count,
                                      uint32_t* queue_cu_mask) const = 0;

  /// @brief Allocate global wave sync (GWS) resource for a queue. This is only supported for GPUs.
  /// GWS can be used to synchronize wavefronts across the entire GPU device.
  /// @param[in] queue_id Kernel-mode driver's assigned queue ID.
  /// @param[in] num_gws Number of GWS slots.
  /// @param[in] first_gws First GWS slot.
  virtual hsa_status_t AllocQueueGWS(HSA_QUEUEID queue_id, uint32_t num_gws,
                                     uint32_t* first_gws) const = 0;

  /// @brief Exports a memory object via dma-buf.
  ///
  /// @param[in] mem virtual address
  /// @param[in] size memory size in bytes
  /// @param[out] dmabuf_fd dma-buf file descriptor
  /// @param[out] offset memory offset in bytes
  virtual hsa_status_t ExportDMABuf(void *mem, size_t size, int *dmabuf_fd,
                                    size_t *offset) = 0;

  /// @brief Imports a memory object via dma-buf.
  ///
  /// @note The handle must be destroyed with @ref DestroyImportedShareableHandle.
  ///
  /// @param[in] dmabuf_fd dma-buf file descriptor
  /// @param[in] agent agent to import the memory for
  /// @param[out] handle handle to the imported memory
  /// @param[in] mem address of existing buffer, used to bypass import
  virtual hsa_status_t ImportDMABuf(int dmabuf_fd, const core::Agent& agent,
                                    core::ShareableHandle* handle, void* mem = nullptr) = 0;

  /// @brief Destroys the handle created during @ref ImportDMABuf.
  ///
  /// @param[in] handle handle of the object to release
  virtual hsa_status_t DestroyImportedShareableHandle(core::ShareableHandle* handle) = 0;

  /// @brief Maps the memory associated with the handle.
  ///
  /// @param[in] handle handle to the memory object
  /// @param[in] mem virtual address associated with the handle
  /// @param[in] offset memory offset in bytes
  /// @param[in] size memory size in bytes
  /// @param[out] perms new permissions
  virtual hsa_status_t Map(core::ShareableHandle handle, void *mem,
                           size_t offset, size_t size,
                           hsa_access_permission_t perms) = 0;

  /// @brief Unmaps the memory associated with the handle.
  ///
  /// @param[in] handle handle to the memory object
  /// @param[in] mem virtual address associated with the handle
  /// @param[in] offset memory offset in bytes
  /// @param[in] size memory size in bytes
  virtual hsa_status_t Unmap(core::ShareableHandle handle, void *mem,
                             size_t offset, size_t size) = 0;

  /// @brief Maps the virtual address to the physical address and creates a handle to share this
  /// mapping.
  ///
  /// @note The handle must be destroyed with @ref DestroyShareableHandle.
  ///
  /// @param[in] va virtual address
  /// @param[in] mem physical memory handle
  /// @param[in] size memory size in bytes
  /// @param[in] agent agent associated with @p mem
  /// @param[out] handle handle of the memory object
  /// @param[out] offset memory offset in bytes
  /// @param[out] drm_fd file descriptor
  /// @param[out] drm_fd_offset offset in @p drm_fd
  virtual hsa_status_t CreateShareableHandle(void* va, void* mem, size_t size,
                                             const core::Agent& agent,
                                             core::ShareableHandle* handle, uint64_t* offset,
                                             int* drm_fd, uint64_t* drm_fd_offset) = 0;

  /// @brief Destroys the handle created during @ref CreateShareableHandle.
  ///
  /// @param[in] handle handle of the object to destroy
  virtual hsa_status_t DestroyShareableHandle(core::ShareableHandle* handle) = 0;

  /// @brief Acquire a streaming performance monitor on an agent.
  /// @param[in] preferred_node_id Node ID of the preferred agent.
  virtual hsa_status_t SPMAcquire(uint32_t preferred_node_id) const = 0;
  /// @brief Release a streaming performance monitor on an agent.
  /// @param[in] preferred_node_id Node ID of the preferred agent.
  virtual hsa_status_t SPMRelease(uint32_t preferred_node_id) const = 0;
  /// @brief Setup the destination user-mode buffer for streaming performance monitor data.
  /// @param[in] preferred_node_id Node ID of the preferred agent.
  /// @param[in] size_bytes Size of the destination buffer in bytes.
  /// @param[in, out] timeout Timeout in milliseconds.
  /// @param[out] size_copied Size of data copied in bytes.
  /// @param[in] dest_mem_addr Destination address for streaming performance data. Set to NULL to
  /// stop copy on previous buffer.
  /// @param[out] is_spm_data_loss Data was lost if true.
  virtual hsa_status_t SPMSetDestBuffer(uint32_t preferred_node_id, uint32_t size_bytes,
                                        uint32_t* timeout, uint32_t* size_copied,
                                        void* dest_mem_addr, bool* is_spm_data_loss) const = 0;

  /// @brief Open anonymous file descriptor to enable events and read SMI events.
  /// @param[in] node_id Node ID to receive the SMI event from.
  /// @param[out] fd Anonymous file descriptor.
  /// @retval HSA_STATUS_ERROR_INVALID_AGENT if the agent's driver doesn't support
  /// SMI events.
  virtual hsa_status_t OpenSMI(uint32_t node_id, int* fd) const {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  /// @brief Sets trap handler and trap buffer to be used for all queues associated
  /// with the specified NodeId within this process context
  /// @param[in] node_id Node ID of the agent
  /// @param[in] base Trap handler base address
  /// @param[in] base_size Trap handler base size
  /// @param[in] buffer_base Trap buffer base address
  /// @param[in] buffer_base_size Trap buffer size
  /// @return HSA_STATUS_SUCCESS if the driver successfully sets the trap handler.
  virtual hsa_status_t SetTrapHandler(uint32_t node_id, const void* base, uint64_t base_size,
                                      const void* buffer_base, uint64_t buffer_base_size) const = 0;

  /// @brief Gets the device handle for a specific node.
  /// @param node_id Node ID of the agent
  /// @param device_handle Device handle
  /// @return HSA_STATUS_SUCCESS if the driver successfully returns the device
  virtual hsa_status_t GetDeviceHandle(uint32_t node_id, void** device_handle) const = 0;


  /// @brief Gets clock counters for particular Node
  /// @param[in] node_id Node ID of the agent
  /// @param[out] clock_counter Clock counter
  /// @return HSA_STATUS_SUCCESS if the driver successfully returns the clock
  virtual hsa_status_t GetClockCounters(uint32_t node_id,
                                        HsaClockCounters* clock_counter) const = 0;

  /// @brief Get the tile configuration for a specific node.
  ///
  /// @param[in] node_id Node ID of the agent
  /// @param[out] config Pointer to tile configuration
  /// @return HSA_STATUS_SUCCESS if the driver successfully returns the tile configuration.
  virtual hsa_status_t GetTileConfig(uint32_t node_id, HsaGpuTileConfig* config) const = 0;

  /// @brief Check if the HSA KMT Model is enabled
  /// @param[out] enable True if the model is enabled, false otherwise
  virtual hsa_status_t IsModelEnabled(bool* enable) const = 0;

  /// @brief Gets the wallclock frequency for a specific node.
  /// @param[in] node_id Node ID of the agent
  /// @param[out] frequency Pointer to the wallclock frequency
  /// @return HSA_STATUS_SUCCESS if the wallclock frequency was successfully retrieved, or an error
  /// code.
  virtual hsa_status_t GetWallclockFrequency(uint32_t node_id, uint64_t* frequency) const = 0;

  /// @brief Allocates scratch memory for the agent.
  /// @param[in] node_id Node ID of the agent
  /// @param[in] size Size of the scratch memory
  /// @param[out] mem Pointer to the scratch memory
  /// @return HSA_STATUS_SUCCESS if scratch memory allocated successfully.
  virtual hsa_status_t AllocateScratchMemory(uint32_t node_id, uint64_t size, void** mem) const = 0;

  /// @brief Inquires memory available for allocation as a memory buffer
  /// @param[in] node_id Node ID of the agent
  /// @param[out] available_size Available memory size in bytes
  /// @return HSA_STATUS_SUCCESS if the driver successfully returns the available memory size.
  virtual hsa_status_t AvailableMemory(uint32_t node_id, uint64_t* available_size) const = 0;

  /// @brief Register memory to GPU
  /// @param[in] ptr Address of memory to be registered
  /// @param[in] size Size of memory
  /// @param[in] mem_flags Flags of memory registering
  /// @return HSA_STATUS_SUCCESS if memory registered successfully.
  virtual hsa_status_t RegisterMemory(void* ptr, uint64_t size, HsaMemFlags mem_flags) const = 0;

  /// @brief Unregisters with a memory
  /// @param[in] ptr Pointer of memory
  /// @return HSA_STATUS_SUCCESS if deregister memory successfully.
  virtual hsa_status_t DeregisterMemory(void* ptr) const = 0;

  /// @brief Make the memory is resident and can be accessed by GPU
  /// @param[in] mem address of memory to be made resident
  /// @param[in] size size of memory
  /// @param[out] alternate_va alternate virtual address
  /// @param[in] mem_flags memory flags can be null
  /// @param[in] num_nodes number of nodes to be used can be 0 if not used
  /// @param[in] nodes nodes to be used can be null
  /// @return HSA_STATUS_SUCCESS if the driver successfully makes the memory
  virtual hsa_status_t MakeMemoryResident(const void* mem, size_t size, uint64_t* alternate_va,
                                          const HsaMemMapFlags* mem_flags = nullptr,
                                          uint32_t num_nodes = 0,
                                          const uint32_t* nodes = nullptr) const = 0;

  /// @brief Releases the residency of the memory
  /// @param[in] mem address of memory to be made unresident
  /// @return HSA_STATUS_SUCCESS if the driver successfully releases the residency
  virtual hsa_status_t MakeMemoryUnresident(const void* mem) const = 0;

  /// @brief Gets the queue save area information for a specific queue.
  /// @param[in]  queue_id Queue ID of the queue
  /// @param[out] address Address of the queue save area
  /// @param[out] size Size of the used queue save area in bytes
  /// @return HSA_STATUS_SUCCESS if the driver successfully returns the queue save area information
  virtual hsa_status_t GetQueueSaveAreaInfo(HSA_QUEUEID queue_id, void** address, size_t* size) const = 0;

  /// @brief Create a kernel event for signal implementation.
  /// @param[in,out] event_descriptor Descriptor specifying event type and properties.
  /// @param[in] manual_reset If true, event must be manually reset after signaling.
  /// @param[out] event Pointer to the newly created event.
  /// @retval HSA_STATUS_SUCCESS if the event was created successfully.
  /// @retval HSA_STATUS_ERROR if event creation failed or is unsupported.
  virtual hsa_status_t CreateEvent(HsaEventDescriptor& event_descriptor, bool manual_reset,
                                   HsaEvent** event) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Destroy a kernel event.
  /// @param[in] event Event to destroy.
  /// @retval HSA_STATUS_SUCCESS if the event was destroyed successfully.
  /// @retval HSA_STATUS_ERROR if event destruction failed or is unsupported.
  virtual hsa_status_t DestroyEvent(HsaEvent* event) const { return HSA_STATUS_ERROR; }

  /// @brief Wait on a single event with timeout.
  /// @param[in] event Event to wait on.
  /// @param[in] timeout_ms Timeout in milliseconds.
  /// @param[in,out] event_age Pointer to track event age across waits.
  /// @retval HSA_STATUS_SUCCESS if the wait completed successfully.
  /// @retval HSA_STATUS_ERROR if the wait failed or is unsupported.
  virtual hsa_status_t WaitOnEvent(HsaEvent* event, uint32_t timeout_ms,
                                   uint64_t* event_age) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Wait on multiple events with timeout.
  /// @param[in] events Array of events to wait on.
  /// @param[in] num_events Number of events in the array.
  /// @param[in] wait_on_all If true, wait for all events; otherwise wait for any.
  /// @param[in] timeout_ms Timeout in milliseconds.
  /// @param[in,out] event_age Pointer to track event age across waits.
  /// @retval HSA_STATUS_SUCCESS if the wait completed successfully.
  /// @retval HSA_STATUS_ERROR if the wait failed or is unsupported.
  virtual hsa_status_t WaitOnMultipleEvents(HsaEvent** events, uint32_t num_events,
                                            bool wait_on_all, uint32_t timeout_ms,
                                            uint64_t* event_age) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Signal an event to wake waiting threads.
  /// @param[in] event Event to signal.
  /// @retval HSA_STATUS_SUCCESS if the event was signaled successfully.
  /// @retval HSA_STATUS_ERROR if signaling failed or is unsupported.
  virtual hsa_status_t SetEvent(HsaEvent* event) const { return HSA_STATUS_ERROR; }

  /// @brief Ring queue doorbell to notify the hardware of new work.
  /// @param[in] queue_id Kernel-mode driver's assigned queue ID.
  /// @param[in] value Doorbell value (typically the new write index).
  /// @retval HSA_STATUS_SUCCESS if the doorbell was rung successfully.
  /// @retval HSA_STATUS_ERROR if the operation failed or is unsupported.
  virtual hsa_status_t RingDoorbell(HSA_QUEUEID queue_id, uint64_t value) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Query information about a memory pointer.
  /// @param[in] ptr Pointer to query.
  /// @param[out] info Pointer information populated by the driver.
  /// @retval HSA_STATUS_SUCCESS if the query was successful.
  /// @retval HSA_STATUS_ERROR if the query failed or is unsupported.
  virtual hsa_status_t QueryPointerInfo(const void* ptr, HsaPointerInfo* info) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Associate user data with a memory allocation.
  /// @param[in] ptr Pointer to the memory allocation.
  /// @param[in] user_data User data to associate with the allocation.
  /// @retval HSA_STATUS_SUCCESS if the user data was set successfully.
  /// @retval HSA_STATUS_ERROR if the operation failed or is unsupported.
  virtual hsa_status_t SetMemoryUserData(const void* ptr, void* user_data) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Free a thunk memory object handle (buffer object).
  /// @param[in] handle Memory object handle to free.
  /// @retval HSA_STATUS_SUCCESS if the handle was freed successfully.
  /// @retval HSA_STATUS_ERROR if the operation failed or is unsupported.
  virtual hsa_status_t FreeMemoryHandle(HsaMemoryObjectHandle handle) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Return an ASAN header page to the driver.
  /// @param[in] addr Address of the ASAN header page.
  /// @retval HSA_STATUS_SUCCESS if the page was returned successfully.
  /// @retval HSA_STATUS_ERROR if the operation failed or is unsupported.
  virtual hsa_status_t ReturnAsanHeaderPage(void* addr) const { return HSA_STATUS_ERROR; }

  /// @brief Map memory to all GPUs in the system.
  /// @param[in] mem Address of the memory to map.
  /// @param[in] size Size of the memory in bytes.
  /// @param[out] alternate_va Alternate virtual address for the mapping.
  /// @retval HSA_STATUS_SUCCESS if the memory was mapped successfully.
  /// @retval HSA_STATUS_ERROR if the mapping failed or is unsupported.
  virtual hsa_status_t MapMemoryToGPU(const void* mem, size_t size, uint64_t* alternate_va) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Map memory to specific GPU nodes.
  /// @param[in] mem Address of the memory to map.
  /// @param[in] size Size of the memory in bytes.
  /// @param[out] alternate_va Alternate virtual address for the mapping.
  /// @param[in] flags Memory mapping flags.
  /// @param[in] num_nodes Number of nodes in the @p nodes array.
  /// @param[in] nodes Array of node IDs to map the memory to.
  /// @retval HSA_STATUS_SUCCESS if the memory was mapped successfully.
  /// @retval HSA_STATUS_ERROR if the mapping failed or is unsupported.
  virtual hsa_status_t MapMemoryToGPUNodes(const void* mem, size_t size, uint64_t* alternate_va,
                                           HsaMemMapFlags flags, uint32_t num_nodes,
                                           const uint32_t* nodes) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Get DRM file descriptor and CPU address from a memory handle.
  /// @param[in] device_handle Device handle for the GPU.
  /// @param[in] mem_handle Memory object handle.
  /// @param[out] drm_fd DRM file descriptor for the memory.
  /// @param[out] cpu_addr CPU-accessible address for the memory.
  /// @retval HSA_STATUS_SUCCESS if the address was retrieved successfully.
  /// @retval HSA_STATUS_ERROR if the operation failed or is unsupported.
  virtual hsa_status_t GetMemoryCpuAddr(void* device_handle, void* mem_handle, int* drm_fd,
                                        uint64_t* cpu_addr) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Allocate memory with a specified alignment.
  /// @param[in] node Node ID for the allocation.
  /// @param[in] size Size of the allocation in bytes.
  /// @param[in] alignment Required alignment in bytes.
  /// @param[in] flags Memory allocation flags.
  /// @param[out] mem Pointer to the allocated memory.
  /// @retval HSA_STATUS_SUCCESS if memory was allocated successfully.
  /// @retval HSA_STATUS_ERROR if the allocation failed or is unsupported.
  virtual hsa_status_t AllocateMemoryAlign(uint32_t node, size_t size, size_t alignment,
                                           HsaMemFlags flags, void** mem) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Create a CPU-accessible mapping for a memory object handle.
  /// @param[in] handle Memory object handle to map.
  /// @param[out] cpu_ptr CPU-accessible pointer to the mapped memory.
  /// @retval HSA_STATUS_SUCCESS if the mapping was created successfully.
  /// @retval HSA_STATUS_ERROR if the mapping failed or is unsupported.
  virtual hsa_status_t MemoryCpuMap(HsaMemoryObjectHandle handle, void** cpu_ptr) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Export memory as a DMA buffer file descriptor for IPC.
  /// @param[in] mem Address of the memory to export.
  /// @param[in] size Size of the memory in bytes.
  /// @param[out] dmabuf_fd DMA buffer file descriptor.
  /// @param[out] offset Offset within the DMA buffer.
  /// @retval HSA_STATUS_SUCCESS if the export was successful.
  /// @retval HSA_STATUS_ERROR if the export failed or is unsupported.
  virtual hsa_status_t ExportDMABufHandle(const void* mem, size_t size, int* dmabuf_fd,
                                          uint64_t* offset) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Import an external memory handle.
  /// @param[in] desc Descriptor for the external handle to import.
  /// @param[out] result Import result containing the memory address and metadata.
  /// @param[in,out] flags Import flags controlling behavior and receiving status.
  /// @retval HSA_STATUS_SUCCESS if the import was successful.
  /// @retval HSA_STATUS_ERROR if the import failed or is unsupported.
  virtual hsa_status_t HandleImport(const HsaExternalHandleDesc* desc,
                                    HsaHandleImportResult* result,
                                    HsaHandleImportFlags* flags) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Create a shareable memory handle for legacy (non-DMA buffer) IPC.
  /// @param[in] mem Address of the memory to share.
  /// @param[in] size Size of the memory in bytes.
  /// @param[out] handle Shared memory handle for the allocation.
  /// @retval HSA_STATUS_SUCCESS if the handle was created successfully.
  /// @retval HSA_STATUS_ERROR if the operation failed or is unsupported.
  virtual hsa_status_t ShareMemory(void* mem, size_t size, HsaSharedMemoryHandle* handle) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Register a shared memory handle for legacy IPC import.
  /// @param[in] handle Shared memory handle to register.
  /// @param[out] address Address of the registered memory.
  /// @param[out] size Size of the registered memory in bytes.
  /// @retval HSA_STATUS_SUCCESS if the handle was registered successfully.
  /// @retval HSA_STATUS_ERROR if the operation failed or is unsupported.
  virtual hsa_status_t RegisterSharedHandle(const HsaSharedMemoryHandle* handle, void** address,
                                            HSAuint64* size) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Register a graphics/DMA-buf handle to GPU nodes.
  /// @param[in] dmabuf_fd DMA buffer file descriptor.
  /// @param[out] info Graphics resource information populated by the driver.
  /// @param[in] num_nodes Number of nodes in the @p nodes array. Pass 0 to register to all nodes.
  /// @param[in] nodes Array of node IDs to register the handle to. May be NULL if @p num_nodes is
  /// 0.
  /// @retval HSA_STATUS_SUCCESS if the handle was registered successfully.
  /// @retval HSA_STATUS_ERROR if the operation failed or is unsupported.
  virtual hsa_status_t RegisterGraphicsHandleToNodes(int dmabuf_fd, HsaGraphicsResourceInfo* info,
                                                     uint32_t num_nodes, uint32_t* nodes) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Register a graphics/DMA-buf handle to GPU nodes with extended flags.
  /// @param[in] dmabuf_fd DMA buffer file descriptor.
  /// @param[out] info Graphics resource information populated by the driver.
  /// @param[in] num_nodes Number of nodes in the @p nodes array.
  /// @param[in] nodes Array of node IDs to register the handle to.
  /// @param[in] flags Registration flags controlling memory mapping behavior.
  /// @retval HSA_STATUS_SUCCESS if the handle was registered successfully.
  /// @retval HSA_STATUS_ERROR if the operation failed or is unsupported.
  virtual hsa_status_t RegisterGraphicsHandleToNodesExt(HSAuint64 dmabuf_fd,
                                                        HsaGraphicsResourceInfo* info,
                                                        HSAuint64 num_nodes, uint32_t* nodes,
                                                        HSA_REGISTER_MEM_FLAGS flags) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Map a virtual address range to a memory object handle.
  /// @param[in] handle Memory object handle.
  /// @param[in] offset Offset within the memory object in bytes.
  /// @param[in] size Size of the mapping in bytes.
  /// @param[in] va Virtual address to map to.
  /// @param[in] access Access flags for the mapping.
  /// @retval HSA_STATUS_SUCCESS if the mapping was created successfully.
  /// @retval HSA_STATUS_ERROR if the mapping failed or is unsupported.
  virtual hsa_status_t MemoryVaMap(HsaMemoryObjectHandle handle, uint64_t offset, uint64_t size,
                                   uint64_t va, uint32_t access) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Unmap a virtual address range from a memory object handle.
  /// @param[in] handle Memory object handle.
  /// @param[in] offset Offset within the memory object in bytes.
  /// @param[in] size Size of the mapping in bytes.
  /// @param[in] va Virtual address to unmap.
  /// @retval HSA_STATUS_SUCCESS if the unmapping was successful.
  /// @retval HSA_STATUS_ERROR if the operation failed or is unsupported.
  virtual hsa_status_t MemoryVaUnmap(HsaMemoryObjectHandle handle, uint64_t offset, uint64_t size,
                                     uint64_t va) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Set SVM (Shared Virtual Memory) attributes for an address range.
  /// @param[in] addr Start address of the memory range.
  /// @param[in] size Size of the memory range in bytes.
  /// @param[in] count Number of attributes in the @p attrs array.
  /// @param[in] attrs Array of SVM attributes to set.
  /// @retval HSA_STATUS_SUCCESS if the attributes were set successfully.
  /// @retval HSA_STATUS_ERROR if the operation failed or is unsupported.
  virtual hsa_status_t SVMSetAttr(void* addr, size_t size, uint32_t count,
                                  HSA_SVM_ATTRIBUTE* attrs) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Get SVM (Shared Virtual Memory) attributes for an address range.
  /// @param[in] addr Start address of the memory range.
  /// @param[in] size Size of the memory range in bytes.
  /// @param[in] count Number of attributes in the @p attrs array.
  /// @param[in,out] attrs Array of SVM attributes to query. The type field specifies
  ///   which attribute to query; the value field is populated by the driver.
  /// @retval HSA_STATUS_SUCCESS if the attributes were retrieved successfully.
  /// @retval HSA_STATUS_ERROR if the operation failed or is unsupported.
  virtual hsa_status_t SVMGetAttr(void* addr, size_t size, uint32_t count,
                                  HSA_SVM_ATTRIBUTE* attrs) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Query PC sampling capabilities for a node.
  /// @param[in] node_id Node ID of the agent.
  /// @param[out] sample_info Buffer to receive sampling capability information.
  /// @param[in] size Size of the @p sample_info buffer in bytes.
  /// @param[in,out] count On input, number of entries that fit in @p sample_info.
  ///   On output, number of entries available.
  /// @retval HSA_STATUS_SUCCESS if the query was successful.
  /// @retval HSA_STATUS_ERROR if the query failed or is unsupported.
  virtual hsa_status_t PcSamplingQueryCapabilities(uint32_t node_id, void* sample_info, size_t size,
                                                   uint32_t* count) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Create a PC sampling session.
  /// @param[in] node_id Node ID of the agent.
  /// @param[in] sample_info Sampling configuration.
  /// @param[out] trace_id Trace ID assigned to the new session.
  /// @retval HSA_STATUS_SUCCESS if the session was created successfully.
  /// @retval HSA_STATUS_ERROR_RESOURCE_BUSY if a session already exists.
  /// @retval HSA_STATUS_ERROR if creation failed or is unsupported.
  virtual hsa_status_t PcSamplingCreate(uint32_t node_id, HsaPcSamplingInfo* sample_info,
                                        HsaPcSamplingTraceId* trace_id) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Destroy a PC sampling session.
  /// @param[in] node_id Node ID of the agent.
  /// @param[in] trace_id Trace ID of the session to destroy.
  /// @retval HSA_STATUS_SUCCESS if the session was destroyed successfully.
  /// @retval HSA_STATUS_ERROR if destruction failed or is unsupported.
  virtual hsa_status_t PcSamplingDestroy(uint32_t node_id, HsaPcSamplingTraceId trace_id) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Start a PC sampling session.
  /// @param[in] node_id Node ID of the agent.
  /// @param[in] trace_id Trace ID of the session to start.
  /// @retval HSA_STATUS_SUCCESS if the session was started successfully.
  /// @retval HSA_STATUS_ERROR if starting failed or is unsupported.
  virtual hsa_status_t PcSamplingStart(uint32_t node_id, HsaPcSamplingTraceId trace_id) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Stop a PC sampling session.
  /// @param[in] node_id Node ID of the agent.
  /// @param[in] trace_id Trace ID of the session to stop.
  /// @retval HSA_STATUS_SUCCESS if the session was stopped successfully.
  /// @retval HSA_STATUS_ERROR if stopping failed or is unsupported.
  virtual hsa_status_t PcSamplingStop(uint32_t node_id, HsaPcSamplingTraceId trace_id) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Enable the debug interface for core dump collection.
  /// @param[out] runtime_ptr Pointer to the runtime debug data.
  /// @param[out] runtime_size Size of the runtime debug data in bytes.
  /// @retval HSA_STATUS_SUCCESS if the debug interface was enabled successfully.
  /// @retval HSA_STATUS_ERROR if enabling failed or is unsupported.
  virtual hsa_status_t DbgEnable(void** runtime_ptr, uint32_t* runtime_size) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Disable the debug interface.
  /// @retval HSA_STATUS_SUCCESS if the debug interface was disabled successfully.
  /// @retval HSA_STATUS_ERROR if disabling failed or is unsupported.
  virtual hsa_status_t DbgDisable() const { return HSA_STATUS_ERROR; }

  /// @brief Get a device data snapshot for core dump.
  /// @param[out] data Pointer to the device data buffer allocated by the driver.
  /// @param[out] count Number of device data entries.
  /// @param[out] entry_size Size of each device data entry in bytes.
  /// @retval HSA_STATUS_SUCCESS if the snapshot was captured successfully.
  /// @retval HSA_STATUS_ERROR if the operation failed or is unsupported.
  virtual hsa_status_t DbgGetDeviceData(void** data, uint32_t* count, uint32_t* entry_size) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Get a queue data snapshot for core dump.
  /// @param[out] data Pointer to the queue data buffer allocated by the driver.
  /// @param[out] count Number of queue data entries.
  /// @param[out] entry_size Size of each queue data entry in bytes.
  /// @param[in] suspend If true, suspend queues before capturing the snapshot.
  /// @retval HSA_STATUS_SUCCESS if the snapshot was captured successfully.
  /// @retval HSA_STATUS_ERROR if the operation failed or is unsupported.
  virtual hsa_status_t DbgGetQueueData(void** data, uint32_t* count, uint32_t* entry_size,
                                       bool suspend) const {
    return HSA_STATUS_ERROR;
  }

  /// @brief Read from or write to an AIS (AI Storage) file via the GPU.
  /// @param[in] device_ptr Device memory address for the transfer.
  /// @param[in] size Size of the transfer in bytes.
  /// @param[in] fd File descriptor of the AIS file.
  /// @param[in] file_offset Offset within the file in bytes.
  /// @param[in] operation Read or write operation flag.
  /// @param[out] size_copied Number of bytes actually transferred.
  /// @param[out] status Driver-specific status code for the operation.
  /// @retval HSA_STATUS_SUCCESS if the transfer completed successfully.
  /// @retval HSA_STATUS_ERROR if the transfer failed or is unsupported.
  virtual hsa_status_t AisReadWriteFile(void* device_ptr, size_t size, int fd, int64_t file_offset,
                                        HsaAisFlags operation, uint64_t* size_copied,
                                        int32_t* status) const {
    return HSA_STATUS_ERROR;
  }

  /// Unique identifier for supported kernel-mode drivers.
  const DriverType kernel_driver_type_;

protected:
 HsaVersionInfo version_{std::numeric_limits<uint32_t>::max(),
                         std::numeric_limits<uint32_t>::max()};

 const std::string devnode_name_;
 int fd_ = -1;
};

} // namespace core
} // namespace rocr

#endif // header guard
