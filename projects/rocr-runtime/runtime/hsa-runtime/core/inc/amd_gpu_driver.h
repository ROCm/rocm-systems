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

#ifndef HSA_RUNTIME_CORE_INC_AMD_GPU_DRIVER_H_
#define HSA_RUNTIME_CORE_INC_AMD_GPU_DRIVER_H_

#include <memory>
#include <string>

#include "hsakmt/hsakmttypes.h"

#include "core/inc/driver.h"
#include "core/inc/memory_region.h"

namespace rocr {

namespace core {
class Queue;
}

namespace AMD {

/// @brief AMD GPU Driver for AMD GPU and CPU agents.
///
/// @details Provides a driver interface for GPU agents that abstracts the
/// underlying kernel-mode driver. Platform-specific implementations (KFD on
/// Linux, DXG on Windows) are compiled conditionally.
class GpuDriver final : public core::Driver {
 public:
  GpuDriver(std::string devnode_name);

  /// @brief Discover and open the GPU driver on the system.
  ///
  /// @param[out] driver Driver object for the GPU driver.
  /// @return HSA_STATUS_SUCCESS if driver found and opened.
  /// @return HSA_STATUS_ERROR if unable to find or open the driver.
  static hsa_status_t DiscoverDriver(std::unique_ptr<core::Driver>& driver);

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
  hsa_status_t AllocateMemory(const core::MemoryRegion &mem_region,
                              core::MemoryRegion::AllocateFlags alloc_flags,
                              void **mem, size_t size,
                              uint32_t node_id) override;
  hsa_status_t FreeMemory(void *mem, size_t size) override;
  hsa_status_t CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE type, uint32_t queue_pct,
                           HSA::hsa_amd_queue_priority_internal_t priority, uint32_t sdma_engine_id, void* queue_addr,
                           uint64_t queue_size_bytes, HsaEvent* event,
                           HsaQueueResource& queue_resource) const override;
  hsa_status_t UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_pct, HSA::hsa_amd_queue_priority_internal_t priority,
                           void* queue_addr, uint64_t queue_size, HsaEvent* event) const override;
  hsa_status_t DestroyQueue(HSA_QUEUEID queue_id) const override;
  hsa_status_t SetQueueCUMask(HSA_QUEUEID queue_id, uint32_t cu_mask_count,
                              uint32_t* queue_cu_mask) const override;
  hsa_status_t AllocQueueGWS(HSA_QUEUEID queue_id, uint32_t num_gws,
                             uint32_t* first_gws) const override;
  hsa_status_t ExportDMABuf(void *mem, size_t size, int *dmabuf_fd,
                            size_t *offset) override;
  hsa_status_t ImportDMABuf(int dmabuf_fd, core::Agent &agent,
                            core::ShareableHandle &handle) override;
  hsa_status_t Map(core::ShareableHandle handle, void *mem, size_t offset,
                   size_t size, hsa_access_permission_t perms) override;
  hsa_status_t Unmap(core::ShareableHandle handle, void *mem, size_t offset,
                     size_t size) override;
  hsa_status_t ReleaseShareableHandle(core::ShareableHandle &handle) override;
  hsa_status_t GetShareableHandle(void* va, void* mem, size_t size, core::ShareableHandle* handle) override;
  hsa_status_t SPMAcquire(uint32_t preferred_node_id) const override;
  hsa_status_t SPMRelease(uint32_t preferred_node_id) const override;
  hsa_status_t SPMSetDestBuffer(uint32_t preferred_node_id, uint32_t size_bytes, uint32_t* timeout,
                                uint32_t* size_copied, void* dest_mem_addr,
                                bool* is_spm_data_loss) const override;
  hsa_status_t SetTrapHandler(uint32_t node_id, const void* base, uint64_t base_size,
                              const void* buffer_base, uint64_t buffer_base_size) const override;
  hsa_status_t GetDeviceHandle(uint32_t node_id, void** device_handle) const override;
  hsa_status_t GetClockCounters(uint32_t node_id, HsaClockCounters* clock_counter) const override;
  hsa_status_t GetTileConfig(uint32_t node_id, HsaGpuTileConfig* config) const override;
  hsa_status_t GetWallclockFrequency(uint32_t node_id, uint64_t* frequency) const override;
  hsa_status_t AllocateScratchMemory(uint32_t node_id, uint64_t size, void** mem) const override;
  hsa_status_t AvailableMemory(uint32_t node_id, uint64_t* available_size) const override;
  hsa_status_t RegisterMemory(void* ptr, uint64_t size, HsaMemFlags mem_flags) const override;
  hsa_status_t DeregisterMemory(void* ptr) const override;
  hsa_status_t MakeMemoryResident(const void* mem, size_t size, uint64_t* alternate_va,
                                  const HsaMemMapFlags* mem_flags, uint32_t num_nodes,
                                  const uint32_t* nodes) const override;
  hsa_status_t MakeMemoryUnresident(const void* mem) const override;

  hsa_status_t OpenSMI(uint32_t node_id, int* fd) const override;

  hsa_status_t IsModelEnabled(bool* enable) const override;

  hsa_status_t GetQueueSaveAreaInfo(HSA_QUEUEID queue_id, void** address, size_t* size) const override;

  hsa_status_t CreateEvent(HsaEventDescriptor& event_descriptor, bool manual_reset,
                           HsaEvent** event) const override;
  hsa_status_t DestroyEvent(HsaEvent* event) const override;
  hsa_status_t WaitOnEvent(HsaEvent* event, uint32_t timeout_ms,
                           uint64_t* event_age) const override;
  hsa_status_t WaitOnMultipleEvents(HsaEvent** events, uint32_t num_events, bool wait_on_all,
                                    uint32_t timeout_ms, uint64_t* event_age) const override;
  hsa_status_t SetEvent(HsaEvent* event) const override;

  hsa_status_t RingDoorbell(HSA_QUEUEID queue_id, uint64_t value) const override;

  hsa_status_t QueryPointerInfo(const void* ptr, HsaPointerInfo* info) const override;
  hsa_status_t SetMemoryUserData(const void* ptr, void* user_data) const override;
  hsa_status_t FreeMemoryHandle(HsaMemoryObjectHandle handle) const override;
  hsa_status_t ReturnAsanHeaderPage(void* addr) const override;
  hsa_status_t MapMemoryToGPU(const void* mem, size_t size, uint64_t* alternate_va) const override;
  hsa_status_t MapMemoryToGPUNodes(const void* mem, size_t size, uint64_t* alternate_va,
                                   HsaMemMapFlags flags, uint32_t num_nodes,
                                   const uint32_t* nodes) const override;
  hsa_status_t GetMemoryCpuAddr(void* device_handle, void* mem_handle, int* drm_fd,
                                uint64_t* cpu_addr) const override;
  hsa_status_t AllocateMemoryAlign(uint32_t node, size_t size, size_t alignment, HsaMemFlags flags,
                                   void** mem) const override;
  hsa_status_t MemoryCpuMap(HsaMemoryObjectHandle handle, void** cpu_ptr) const override;

  hsa_status_t ExportDMABufHandle(const void* mem, size_t size, int* dmabuf_fd,
                                  uint64_t* offset) const override;
  hsa_status_t HandleImport(const HsaExternalHandleDesc* desc, HsaHandleImportResult* result,
                            HsaHandleImportFlags* flags) const override;
  hsa_status_t ShareMemory(void* mem, size_t size, HsaSharedMemoryHandle* handle) const override;
  hsa_status_t RegisterSharedHandle(const HsaSharedMemoryHandle* handle, void** address,
                                    HSAuint64* size) const override;
  hsa_status_t RegisterGraphicsHandleToNodes(int dmabuf_fd, HsaGraphicsResourceInfo* info,
                                             uint32_t num_nodes, uint32_t* nodes) const override;
  hsa_status_t RegisterGraphicsHandleToNodesExt(HSAuint64 dmabuf_fd, HsaGraphicsResourceInfo* info,
                                                HSAuint64 num_nodes, uint32_t* nodes,
                                                HSA_REGISTER_MEM_FLAGS flags) const override;
  hsa_status_t MemoryVaMap(HsaMemoryObjectHandle handle, uint64_t offset, uint64_t size,
                           uint64_t va, uint32_t access) const override;
  hsa_status_t MemoryVaUnmap(HsaMemoryObjectHandle handle, uint64_t offset, uint64_t size,
                             uint64_t va) const override;

  hsa_status_t SVMSetAttr(void* addr, size_t size, uint32_t count,
                          HSA_SVM_ATTRIBUTE* attrs) const override;
  hsa_status_t SVMGetAttr(void* addr, size_t size, uint32_t count,
                          HSA_SVM_ATTRIBUTE* attrs) const override;

  hsa_status_t PcSamplingQueryCapabilities(uint32_t node_id, void* sample_info, size_t size,
                                           uint32_t* count) const override;
  hsa_status_t PcSamplingCreate(uint32_t node_id, HsaPcSamplingInfo* sample_info,
                                HsaPcSamplingTraceId* trace_id) const override;
  hsa_status_t PcSamplingDestroy(uint32_t node_id, HsaPcSamplingTraceId trace_id) const override;
  hsa_status_t PcSamplingStart(uint32_t node_id, HsaPcSamplingTraceId trace_id) const override;
  hsa_status_t PcSamplingStop(uint32_t node_id, HsaPcSamplingTraceId trace_id) const override;

  hsa_status_t DbgEnable(void** runtime_ptr, uint32_t* runtime_size) const override;
  hsa_status_t DbgDisable() const override;
  hsa_status_t DbgGetDeviceData(void** data, uint32_t* count, uint32_t* entry_size) const override;
  hsa_status_t DbgGetQueueData(void** data, uint32_t* count, uint32_t* entry_size,
                               bool suspend) const override;

  hsa_status_t AisReadWriteFile(void* device_ptr, size_t size, int fd, int64_t file_offset,
                                HsaAisFlags operation, uint64_t* size_copied,
                                int32_t* status) const override;

  /// @brief Convert internal queue priority to KFD queue priority.
  static HSA_QUEUE_PRIORITY HsaInternalToKfdPriority(
      HSA::hsa_amd_queue_priority_internal_t priority);

  /// @brief Convert HSA access permission to memory map flags.
  static HsaMemoryMapFlags mem_perm(hsa_access_permission_t perm);

  /// @brief Query for user preference and use that to determine Xnack mode
  /// of ROCm system. Return true if Xnack mode is ON or false if OFF.
  static bool BindXnackMode();

 private:
  /// @brief Allocate agent accessible memory (system / local memory).
  static void *AllocateKfdMemory(const HsaMemFlags &flags, uint32_t node_id,
                                 size_t size);

  /// @brief Free agent accessible memory (system / local memory).
  static bool FreeKfdMemory(void *mem, size_t size);

  /// @brief Pin memory.
  static bool MakeKfdMemoryResident(size_t num_node, const uint32_t *nodes,
                                    const void *mem, size_t size,
                                    uint64_t *alternate_va,
                                    HsaMemMapFlags map_flag);

  /// @brief Unpin memory.
  static void MakeKfdMemoryUnresident(const void *mem);

  // Minimum acceptable KFD version numbers.
  static const uint32_t kfd_version_major_min = 0;
  static const uint32_t kfd_version_minor_min = 99;
};

} // namespace AMD
} // namespace rocr

#endif // header guard
