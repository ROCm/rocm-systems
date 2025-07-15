////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HSA_RUNTIME_CORE_INC_AMD_KFD_DRIVER_H_
#define HSA_RUNTIME_CORE_INC_AMD_KFD_DRIVER_H_

#include <memory>
#include <string>

#include "hsakmt/hsakmt.h"

#include "core/inc/driver.h"
#include "core/inc/memory_region.h"

namespace rocr {

namespace core {

class Queue;

}

namespace AMD {

/// @brief AMD Kernel Fusion Driver (KFD) for AMD GPU and CPU agents.
///
/// @details The user-mode driver into the Linux KFD for AMD GPU and CPU HSA
/// agents. Provides APIs for the ROCr core to discover the topology produced
/// by the KFD, allocate memory out of the KFD, manage DMA bufs, allocate queues,
/// and more.
class KfdDriver final : public core::Driver {
public:
  KfdDriver(std::string devnode_name);

  /// @brief Determine of the KFD is present on the system and attemp to open it if found.
  ///
  /// @param[out] Driver object for the KFD.
  /// @return HSA_STATUS_SUCCESS if driver found and opened.
  /// @return HSA_STATUS_ERROR if unable to find or open the KFD.
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
                           HSA_QUEUE_PRIORITY priority, uint32_t sdma_engine_id, void* queue_addr,
                           uint64_t queue_size_bytes, HsaEvent* event,
                           HsaQueueResource& queue_resource) const override;
  hsa_status_t UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_pct, HSA_QUEUE_PRIORITY priority,
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

  hsa_status_t OpenSMI(uint32_t node_id, int* fd) const override;

  hsa_status_t IsModelEnabled(bool* enable) const override;

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

  /// @brief Query for user preference and use that to determine Xnack mode
  /// of ROCm system. Return true if Xnack mode is ON or false if OFF. Xnack
  /// mode of a system is orthogonal to devices that do not support Xnack mode.
  /// It is legal for a system with Xnack ON to have devices that do not support
  /// Xnack functionality.
  static bool BindXnackMode();

  // Minimum acceptable KFD version numbers.
  static const uint32_t kfd_version_major_min = 0;
  static const uint32_t kfd_version_minor_min = 99;
};

} // namespace AMD
} // namespace rocr

#endif // header guard
