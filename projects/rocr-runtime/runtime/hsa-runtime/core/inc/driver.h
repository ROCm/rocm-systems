////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.
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
                                   HSA_QUEUE_PRIORITY priority, uint32_t sdma_engine_id,
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
                                   HSA_QUEUE_PRIORITY priority, void* queue_addr,
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

  /// @brief Imports memory using dma-buf.
  ///
  /// @param[in] mem virtual address
  /// @param[in] size memory size in bytes
  /// @param[out] dmabuf_fd dma-buf file descriptor
  /// @param[out] offset memory offset in bytes
  virtual hsa_status_t ExportDMABuf(void *mem, size_t size, int *dmabuf_fd,
                                    size_t *offset) = 0;

  /// @brief Imports a memory chunk via dma-buf.
  ///
  /// @param[in] dmabuf_fd dma-buf file descriptor
  /// @param[in] agent agent to import the memory for
  /// @param[out] handle handle to the imported memory
  virtual hsa_status_t ImportDMABuf(int dmabuf_fd, core::Agent &agent,
                                    core::ShareableHandle &handle) = 0;

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

  /// @brief Releases the object associated with the handle.
  ///
  /// @param[in] handle handle of the object to release
  virtual hsa_status_t
  ReleaseShareableHandle(core::ShareableHandle &handle) = 0;

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
  /// @return HSA_STATUS_SUCCESS if the driver successfully makes the memory
  virtual hsa_status_t MakeMemoryUnresident(const void* mem) const = 0;

  /// @brief Shares memory with another process.
  /// @param[in] mem Pointer to the memory to be shared.
  /// @param[in] size Size of the memory to be shared.
  /// @param[out] share_mem Pointer to the shared memory handle.
  /// @return HSA_STATUS_SUCCESS if the memory was successfully shared, or an error code.
  virtual hsa_status_t ShareMemory(void* mem, size_t size, HsaSharedMemoryHandle* share_mem) const {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  /// @brief Registers a shared memory handle.
  /// @param[in] share_mem Pointer to the shared memory handle.
  /// @param[out] mem Pointer to the memory.
  /// @param[out] size Size of the memory.
  /// @return HSA_STATUS_SUCCESS if the memory was successfully registered, or an error code.
  virtual hsa_status_t RegisterSharedHandle(const HsaSharedMemoryHandle* share_mem, void** mem,
                                            uint64_t* size) const {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  /// @brief Replaces the ASAN header page with a valid one.
  /// @param[in] mem Pointer to the memory to be replaced.
  /// @return HSA_STATUS_SUCCESS if the ASAN header page was successfully replaced, or an error
  /// code.
  virtual hsa_status_t ReplaceAsanHeaderPage(void* mem) const {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  /// @brief Returns the ASAN header page to its original state.
  /// @param[in] mem Pointer to the memory to be returned.
  /// @return HSA_STATUS_SUCCESS if the ASAN header page was successfully returned, or an error
  /// code.
  virtual hsa_status_t ReturnAsanHeaderPage(void* mem) const {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  /// @brief Queries the PC sampling capabilities.
  /// @param[in] node_id Node ID of the agent
  /// @param[in] sample_info Pointer to the sample information
  /// @param[in] sample_info_sz Size of the sample information
  /// @param[out] sz_needed Size of the sample information needed
  /// @return HSA_STATUS_SUCCESS if the PC sampling capabilities were successfully queried, or an
  /// error code.
  virtual hsa_status_t PcSamplingQueryCapabilities(uint32_t node_id, void* sample_info,
                                                   uint32_t sample_info_sz,
                                                   uint32_t* sz_needed) const {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  /// @brief Creates a PC sampling session.
  /// @param[in] node_id Node ID of the agent
  /// @param[in] sample_info Pointer to the sample information
  /// @param[out] trace_id Pointer to the trace ID
  /// @return HSA_STATUS_SUCCESS if the PC sampling session was successfully created, or an error
  /// code.
  virtual hsa_status_t PcSamplingCreate(uint32_t node_id, HsaPcSamplingInfo* sample_info,
                                        uint32_t* trace_id) const {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  /// @brief Destroys a PC sampling session.
  /// @param[in] node_id Node ID of the agent
  /// @param[in] trace_id Trace ID of the PC sampling session
  /// @return HSA_STATUS_SUCCESS if the PC sampling session was successfully destroyed, or an error
  /// code.
  virtual hsa_status_t PcSamplingDestroy(uint32_t node_id, uint32_t trace_id) const {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  /// @brief Starts a PC sampling session.
  /// @param[in] node_id Node ID of the agent
  /// @param[in] trace_id Trace ID of the PC sampling session
  /// @return HSA_STATUS_SUCCESS if the PC sampling session was successfully started, or an error
  /// code.
  virtual hsa_status_t PcSamplingStart(uint32_t node_id, uint32_t trace_id) const {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  /// @brief Stops a PC sampling session.
  /// @param[in] node_id Node ID of the agent
  /// @param[in] trace_id Trace ID of the PC sampling session
  /// @return HSA_STATUS_SUCCESS if the PC sampling session was successfully stopped, or an error
  /// code.
  virtual hsa_status_t PcSamplingStop(uint32_t node_id, uint32_t trace_id) const {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  /// Unique identifier for supported kernel-mode drivers.
  const DriverType kernel_driver_type_;

  /// @brief Creates an operating system event associated with a HSA event ID
  /// @param[in] event_desc Pointer to the event descriptor that describes the event
  /// @param[in] manual_reset if true, the event is manually reset; otherwise, it is automatically
  /// reset
  /// @param[in] IsSignaled if true, the event is initially signaled
  /// @param[out] event pointer to the created HsaEvent object
  /// @return HSA_STATUS_SUCCESS if the event was successfully created, or an error code
  static hsa_status_t CreateEvent(HsaEventDescriptor* event_desc, bool manual_reset,
                                  bool IsSignaled, HsaEvent** event) {
    auto create_event = function_table_.create_event;
    if (create_event == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (create_event(event_desc, manual_reset, IsSignaled, event) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

  /// @brief Destroys an operating system event associated with a HSA event ID
  /// @param[in] event pointer to the HsaEvent object to be destroyed
  /// @return HSA_STATUS_SUCCESS if the event was successfully destroyed, or an error code
  static hsa_status_t DestroyEvent(HsaEvent* event) {
    auto destroy_event = function_table_.destroy_event;
    if (destroy_event == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (destroy_event(event) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

  /// @brief Sets the specified event object to the signaled state
  /// @param[in] event pointer to the HsaEvent object to be set
  /// @return HSA_STATUS_SUCCESS if the event was successfully set, or an error code
  static hsa_status_t SetEvent(HsaEvent* event) {
    auto set_event = function_table_.set_event;
    if (set_event == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (set_event(event) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

  /// @brief Checks the current state of the event object. If the object's state is
  /// signaled, the function returns immediately.
  /// @param[in] event pointer to the HsaEvent object to be queried
  /// @param[in] milliseconds time in milliseconds to wait for the event to be signaled
  /// @param[out] event_age pointer to a variable that will hold the event age
  /// @return HSA_STATUS_SUCCESS if the event was successfully queried, or an error code
  static hsa_status_t WaitEventExt(HsaEvent* event, uint32_t milliseconds, uint64_t* event_age) {
    auto wait_event_ext = function_table_.wait_event_ext;
    if (wait_event_ext == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (wait_event_ext(event, milliseconds, event_age) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

  /// @brief Checks the current state of multiple event objects.
  /// @param[in] events array of pointers to HsaEvent objects to be queried
  /// @param[in] num_events number of events in the array
  /// @param[in] wait_on_all if true, the function waits for all events to be signaled;
  /// @param[in] milliseconds time in milliseconds to wait for the events to be signaled
  /// @param[out] event_age pointer to an array that will hold the event ages
  /// @return
  static hsa_status_t WaitEventsExt(HsaEvent* events[], uint32_t num_events, bool wait_on_all,
                                    uint32_t milliseconds, uint64_t* event_age) {
    auto wait_events_ext = function_table_.wait_events_ext;
    if (wait_events_ext == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (wait_events_ext(events, num_events, wait_on_all, milliseconds, event_age) !=
        HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }
    return HSA_STATUS_SUCCESS;
  }

  /// @brief Allocates memory aligned to a specified boundary. Normally used for CPU virtual address
  /// reserve.
  /// @param[in] node_id Node ID of the agent
  /// @param[in] size Size of the memory to be allocated
  /// @param[in] alignment Alignment of the memory to be allocated
  /// @param[in] mem_flags Memory flags for the allocation
  /// @param[out] mem Pointer to store the allocated memory address
  /// @return HSA_STATUS_SUCCESS if the memory was successfully allocated, or an error code
  static hsa_status_t VirtualAddressReserve(uint32_t node_id, uint64_t size, uint64_t alignment,
                                            HsaMemFlags mem_flags, void** mem) {
    auto alloc_mem_align = function_table_.alloc_mem_align;
    if (alloc_mem_align == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (alloc_mem_align(node_id, size, alignment, mem_flags, mem) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }
    return HSA_STATUS_SUCCESS;
  }

  /// @brief Frees memory allocated by the driver. Normally used for free a reserved CPU virtual
  /// address.
  /// @param[in] mem Pointer to the memory to be freed
  /// @param[in] size Size of the memory to be freed
  /// @return HSA_STATUS_SUCCESS if the memory was successfully freed, or an error code
  static hsa_status_t VirtualAddressFree(void* mem, size_t size) {
    auto free_memory = function_table_.free_memory;
    if (free_memory == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (free_memory(mem, size) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

  /// @brief Unmaps the memory associated with the InterOP/IPC handle.
  /// @param[in] mem Pointer to the memory to be unmapped.
  /// @return HSA_STATUS_SUCCESS if the memory was successfully unmapped, or an error code.
  static hsa_status_t ShareableMemoryUnmap(void* mem) {
    auto unmap_mem = function_table_.unmap_mem;
    if (unmap_mem == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (unmap_mem(mem) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

  /// @brief Deregisters the memory associated with the InterOP/IPC handle.
  /// @param[in] mem Pointer to the memory to be deregistered.
  /// @return HSA_STATUS_SUCCESS if the memory was successfully deregistered, or an error code.
  static hsa_status_t ShareableMemoryDeregister(void* mem) {
    auto deregister_mem = function_table_.deregister_mem;
    if (deregister_mem == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (deregister_mem(mem) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

  /// @brief Registers a graphics handle.
  /// @param[in] handle Handle to the graphics resource.
  /// @param[in] info Pointer to the graphics resource info.
  /// @param[in] num_nodes Number of nodes to register the graphics resource on.
  /// @param[in] nodes Array of node IDs to register the graphics resource on.
  /// @return HSA_STATUS_SUCCESS if the graphics handle was successfully registered, or an error
  /// code.
  static hsa_status_t RegisterGraphicsHandle(uint64_t handle, HsaGraphicsResourceInfo* info,
                                             uint64_t num_nodes, uint32_t* nodes) {
    auto register_graphics_handle = function_table_.register_graphics_handle;
    if (register_graphics_handle == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (register_graphics_handle(handle, info, num_nodes, nodes) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

  /// @brief Registers a graphics handle with additional flags.
  /// @param[in] handle Handle to the graphics resource.
  /// @param[in] info Pointer to the graphics resource info.
  /// @param[in] num_nodes Number of nodes to register the graphics resource on.
  /// @param[in] nodes Array of node IDs to register the graphics resource on.
  /// @param[in] flags Flags for the graphics resource.
  /// @return HSA_STATUS_SUCCESS if the graphics handle was successfully registered, or an error
  /// code.
  static hsa_status_t RegisterGraphicsHandleExt(uint64_t handle, HsaGraphicsResourceInfo* info,
                                                uint64_t num_nodes, uint32_t* nodes,
                                                HSA_REGISTER_MEM_FLAGS flags) {
    auto register_graphics_handle_ext = function_table_.register_graphics_handle_ext;
    if (register_graphics_handle_ext == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (register_graphics_handle_ext(handle, info, num_nodes, nodes, flags) !=
        HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

  /// @brief Queries information about a pointer.
  /// @param[in] ptr Pointer to query information about.
  /// @param[out] info Pointer to the pointer info.
  /// @return HSA_STATUS_SUCCESS if the pointer info was successfully queried, or an error code.
  static hsa_status_t QueryPointerInfo(const void* ptr, HsaPointerInfo* info) {
    auto query_pointer_info = function_table_.query_pointer_info;
    if (query_pointer_info == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (query_pointer_info(ptr, info) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

  /// @brief Sets user data for a pointer.
  /// @param[in] ptr Pointer to set user data for.
  /// @param[in] user_data User data to set.
  /// @return HSA_STATUS_SUCCESS if the user data was successfully set, or an error code.
  static hsa_status_t SetMemoryUserData(const void* ptr, void* user_data) {
    auto set_memory_user_data = function_table_.set_memory_user_data;
    if (set_memory_user_data == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (set_memory_user_data(ptr, user_data) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

  /// @brief Sets SVM attributes for a pointer.
  /// @param[in] ptr Pointer to set SVM attributes for.
  /// @param[in] size Size of the memory to set SVM attributes for.
  /// @param[in] nattr Number of attributes to set.
  /// @param[in] attrs Array of attributes to set.
  /// @return HSA_STATUS_SUCCESS if the SVM attributes were successfully set, or an error code.
  static hsa_status_t SVMSetAttr(void* ptr, uint64_t size, unsigned int nattr,
                                 HSA_SVM_ATTRIBUTE* attrs) {
    auto svm_set_attr = function_table_.svm_set_attr;
    if (svm_set_attr == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (svm_set_attr(ptr, size, nattr, attrs) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

  /// @brief Gets SVM attributes for a pointer.
  /// @param[in] ptr Pointer to get SVM attributes for.
  /// @param[in] size Size of the memory to get SVM attributes for.
  /// @param[in] nattr Number of attributes to get.
  /// @param[out] attrs Array of attributes to get.
  /// @return HSA_STATUS_SUCCESS if the SVM attributes were successfully got, or an error code.
  static hsa_status_t SVMGetAttr(void* ptr, uint64_t size, unsigned int nattr,
                                 HSA_SVM_ATTRIBUTE* attrs) {
    auto svm_get_attr = function_table_.svm_get_attr;
    if (svm_get_attr == nullptr) {
      return HSA_STATUS_ERROR_NOT_INITIALIZED;
    }

    if (svm_get_attr(ptr, size, nattr, attrs) != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

 protected:
  HsaVersionInfo version_{std::numeric_limits<uint32_t>::max(),
                          std::numeric_limits<uint32_t>::max()};

  const std::string devnode_name_;
  int fd_ = -1;

  using create_event_fn = HSAKMT_STATUS (*)(HsaEventDescriptor*, bool, bool, HsaEvent**);
  using destroy_event_fn = HSAKMT_STATUS (*)(HsaEvent*);
  using set_event_fn = HSAKMT_STATUS (*)(HsaEvent*);
  using wait_event_ext_fn = HSAKMT_STATUS (*)(HsaEvent*, uint32_t, uint64_t*);
  using wait_events_ext_fn = HSAKMT_STATUS (*)(HsaEvent*[], uint32_t, bool, uint32_t, uint64_t*);
  using alloc_mem_align_fn = HSAKMT_STATUS (*)(uint32_t, uint64_t, uint64_t, HsaMemFlags, void**);
  using free_memory_fn = HSAKMT_STATUS (*)(void*, size_t);
  using unmap_mem_fn = HSAKMT_STATUS (*)(void*);
  using deregister_mem_fn = HSAKMT_STATUS (*)(void*);
  using register_graphics_handle_fn = HSAKMT_STATUS (*)(uint64_t, HsaGraphicsResourceInfo*,
                                                        uint64_t, uint32_t*);
  using register_graphics_handle_ext_fn = HSAKMT_STATUS (*)(uint64_t, HsaGraphicsResourceInfo*,
                                                            uint64_t, uint32_t*,
                                                            HSA_REGISTER_MEM_FLAGS);
  using query_pointer_info_fn = HSAKMT_STATUS (*)(const void*, HsaPointerInfo*);
  using set_memory_user_data_fn = HSAKMT_STATUS (*)(const void*, void*);
  using svm_set_attr_fn = HSAKMT_STATUS (*)(void*, uint64_t, unsigned int, HSA_SVM_ATTRIBUTE*);
  using svm_get_attr_fn = HSAKMT_STATUS (*)(void*, uint64_t, unsigned int, HSA_SVM_ATTRIBUTE*);

  struct DriverFunctionTable {
    create_event_fn create_event;
    destroy_event_fn destroy_event;
    set_event_fn set_event;
    wait_event_ext_fn wait_event_ext;
    wait_events_ext_fn wait_events_ext;
    alloc_mem_align_fn alloc_mem_align;
    free_memory_fn free_memory;
    unmap_mem_fn unmap_mem;
    deregister_mem_fn deregister_mem;
    register_graphics_handle_fn register_graphics_handle;
    register_graphics_handle_ext_fn register_graphics_handle_ext;
    query_pointer_info_fn query_pointer_info;
    set_memory_user_data_fn set_memory_user_data;
    svm_set_attr_fn svm_set_attr;
    svm_get_attr_fn svm_get_attr;
  };

  static DriverFunctionTable function_table_;
};

} // namespace core
} // namespace rocr

#endif // header guard
