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

class KfdDriver final : public core::Driver {
public:
  KfdDriver(std::string devnode_name);

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
  hsa_status_t GetAgentProperties(core::Agent &agent) const override;
  hsa_status_t
  GetMemoryProperties(uint32_t node_id,
                      core::MemoryRegion &mem_region) const override;
  hsa_status_t AllocateMemory(const core::MemoryRegion &mem_region,
                              core::MemoryRegion::AllocateFlags alloc_flags,
                              void **mem, size_t size,
                              uint32_t node_id) override;
  hsa_status_t FreeMemory(void *mem, size_t size) override;
  hsa_status_t CreateQueue(core::Queue &queue) const override;
  hsa_status_t DestroyQueue(core::Queue &queue) const override;
  hsa_status_t ExportDMABuf(void *mem, size_t size, int *dmabuf_fd,
                            size_t *offset) override;
  hsa_status_t ImportDMABuf(int dmabuf_fd, core::Agent &agent,
                            core::ShareableHandle &handle) override;
  hsa_status_t Map(core::ShareableHandle handle, void *mem, size_t offset,
                   size_t size, hsa_access_permission_t perms) override;
  hsa_status_t Unmap(core::ShareableHandle handle, void *mem, size_t offset,
                     size_t size) override;
  hsa_status_t ReleaseShareableHandle(core::ShareableHandle &handle) override;

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
