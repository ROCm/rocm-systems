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

enum class DriverType { XDNA = 0, KFD, NUM_DRIVER_TYPES };

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
