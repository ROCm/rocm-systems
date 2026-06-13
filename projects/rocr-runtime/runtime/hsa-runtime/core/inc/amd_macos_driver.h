////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HSA_RUNTIME_CORE_INC_AMD_MACOS_DRIVER_H_
#define HSA_RUNTIME_CORE_INC_AMD_MACOS_DRIVER_H_

#if !defined(__APPLE__)
#error "amd_macos_driver.h should only be used in the Darwin build"
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/inc/amd_lite_direct_queue.h"
#include "core/inc/driver.h"
#include "core/inc/memory_region.h"
#include "macgpu.h"

namespace rocr {
namespace core {
class Queue;
class Agent;
}  // namespace core

namespace AMD {

/// @brief ROCR driver backend for macOS userspace → DriverKit (DEXT).
///
/// This is the Stage-2B scaffold. It fulfills the core::Driver pure-virtual
/// interface so the rest of ROCR links, and registers in the topology
/// driver-discovery array alongside KfdDriver / XdnaDriver. DiscoverDriver()
/// currently reports "no device" unconditionally — the real DEXT/IOKit
/// plumbing lands when libmacgpu is built (Stage 1).
///
/// Contract when libmacgpu is wired up later:
///   - Open() probes the ROCmGPU.dext user-client via IOServiceNameMatching
///     + IOServiceOpen, storing the mach_port_t in @ref fd_ (cast).
///   - QueryKernelModeDriver fills @ref version_ from the DEXT GetInfo
///     escape.
///   - AllocateMemory → DEXT AllocDMA escape (DART-mapped).
///   - CreateQueue → MEC ring construction via DEXT MMIO escapes, with
///     the doorbell aperture mapped back into the caller.
///   - ExportDMABuf → not supported on Darwin (Thunderbolt-GPU dma-buf
///     sharing requires DriverKit surface sharing which we don't use);
///     return HSA_STATUS_ERROR.
///
/// All of the above is TODO for follow-up commits. Stage 2B is just the
/// skeleton.
class MacOsDriver final : public core::Driver, private lite::DirectQueuePlatform {
 public:
  using DirectComputeQueue = lite::DirectQueueState;

  explicit MacOsDriver(std::string devnode_name);

  /// @brief Probe for a usable DEXT. Returns HSA_STATUS_SUCCESS with a live
  /// driver on success. Returns HSA_STATUS_ERROR (driver left nullptr) when
  /// no ROCmGPU DEXT is installed / registered.
  static hsa_status_t DiscoverDriver(std::unique_ptr<core::Driver>& driver);

  // core::Driver overrides — every method returns HSA_STATUS_ERROR until
  // the libmacgpu IOKit client is wired up.

  hsa_status_t Init() override;
  hsa_status_t ShutDown() override;
  hsa_status_t QueryKernelModeDriver(core::DriverQuery query) override;

  hsa_status_t Open() override;
  hsa_status_t Close() override;

  hsa_status_t GetSystemProperties(HsaSystemProperties& sys_props) const override;
  hsa_status_t GetNodeProperties(HsaNodeProperties& node_props,
                                 uint32_t node_id) const override;
  hsa_status_t GetEdgeProperties(std::vector<HsaIoLinkProperties>& io_link_props,
                                 uint32_t node_id) const override;
  hsa_status_t GetMemoryProperties(uint32_t node_id,
                                   std::vector<HsaMemoryProperties>& mem_props) const override;
  hsa_status_t GetCacheProperties(uint32_t node_id, uint32_t processor_id,
                                  std::vector<HsaCacheProperties>& cache_props) const override;

  hsa_status_t AllocateMemory(const core::MemoryRegion& mem_region,
                              core::MemoryRegion::AllocateFlags alloc_flags,
                              void** mem, size_t size, uint32_t node_id) override;
  hsa_status_t FreeMemory(void* mem, size_t size) override;

  hsa_status_t CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE type, uint32_t queue_pct,
                           HSA::hsa_amd_queue_priority_internal_t priority,
                           uint32_t sdma_engine_id, void* queue_addr,
                           uint64_t queue_size_bytes,
                           uint64_t queue_metadata_size_bytes, HsaEvent* event,
                           HsaQueueResource& queue_resource) const override;
  hsa_status_t DestroyQueue(HSA_QUEUEID queue_id) const override;
  hsa_status_t UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_pct,
                           HSA::hsa_amd_queue_priority_internal_t priority, void* queue_addr,
                           uint64_t queue_size_bytes, HsaEvent* event) const override;
  hsa_status_t SetQueueCUMask(HSA_QUEUEID queue_id, uint32_t cu_mask_count,
                              uint32_t* queue_cu_mask) const override;
  hsa_status_t AllocQueueGWS(HSA_QUEUEID queue_id, uint32_t num_gws,
                             uint32_t* first_gws) const override;

  hsa_status_t ExportDMABuf(void* mem, size_t size, int* dmabuf_fd,
                            size_t* offset) override;
  hsa_status_t ImportDMABuf(int dmabuf_fd, const core::Agent& agent,
                            core::ShareableHandle* handle, void* mem) override;
  hsa_status_t DestroyImportedShareableHandle(core::ShareableHandle* handle) override;
  hsa_status_t Map(core::ShareableHandle handle, void* mem, size_t offset,
                   size_t size, hsa_access_permission_t perms) override;
  hsa_status_t Unmap(core::ShareableHandle handle, void* mem, size_t offset,
                     size_t size) override;
  hsa_status_t CreateShareableHandle(void* va, void* mem, size_t size,
                                     const core::Agent& agent,
                                     core::ShareableHandle* handle, uint64_t* offset,
                                     int* drm_fd, uint64_t* drm_fd_offset) override;
  hsa_status_t DestroyShareableHandle(core::ShareableHandle* handle) override;

  hsa_status_t SPMAcquire(uint32_t preferred_node_id) const override;
  hsa_status_t SPMRelease(uint32_t preferred_node_id) const override;
  hsa_status_t SPMSetDestBuffer(uint32_t preferred_node_id, uint32_t size_bytes,
                                uint32_t* timeout, uint32_t* size_copied,
                                void* dest_mem_addr, bool* is_spm_data_loss) const override;

  hsa_status_t SetTrapHandler(uint32_t node_id, const void* base, uint64_t base_size,
                              const void* buffer_base, uint64_t buffer_base_size) const override;
  hsa_status_t GetDeviceHandle(uint32_t node_id, void** device_handle) const override;
  hsa_status_t GetClockCounters(uint32_t node_id, HsaClockCounters* clock_counter) const override;
  hsa_status_t GetTileConfig(uint32_t node_id, HsaGpuTileConfig* config) const override;
  hsa_status_t IsModelEnabled(bool* enable) const override;
  hsa_status_t GetWallclockFrequency(uint32_t node_id, uint64_t* frequency) const override;
  hsa_status_t AllocateScratchMemory(uint32_t node_id, uint64_t size, void** mem) const override;
  hsa_status_t AvailableMemory(uint32_t node_id, uint64_t* available_size) const override;
  hsa_status_t RegisterMemory(void* ptr, uint64_t size, HsaMemFlags mem_flags) const override;
  hsa_status_t DeregisterMemory(void* ptr) const override;
  hsa_status_t MakeMemoryResident(const void* mem, size_t size, uint64_t* alternate_va,
                                  const HsaMemMapFlags* mem_flags,
                                  uint32_t num_nodes, const uint32_t* nodes) const override;
  hsa_status_t MakeMemoryUnresident(const void* mem) const override;

  hsa_status_t GetQueueSaveAreaInfo(HSA_QUEUEID queue_id, void** address,
                                    size_t* size) const override;

  hsa_status_t AllocateVram(size_t size, size_t align, void** cpu_addr,
                            uint64_t* gpu_addr);
  // macOS-egpu coherent-data (#2): allocate a DART-mapped DMA buffer (host RAM,
  // cache-coherent, GPU-addressable via its IOVA) instead of the cache-inhibited
  // VRAM BAR. The kernel's output then reaches the host WITHOUT a per-dispatch GL2
  // writeback (the post-acquire that cumulatively stalls the CP), matching Linux's
  // coherent-GTT model. *gpu_iova is the single-segment DART bus address.
  hsa_status_t AllocateCoherentDma(size_t size, size_t align, void** cpu_addr,
                                   uint64_t* gpu_iova);
  // True if ROCR_MACOS_COHERENT_DATA is set (route IsLocalMemory tensors through
  // AllocateCoherentDma and drop the per-dispatch output post-acquire).
  static bool CoherentDataEnabled();
  hsa_status_t HostToGpuAddress(const void* ptr, uint64_t* gpu_addr) const;
  // macOS-egpu: true only if `ptr` lies inside a REGISTERED VRAM allocation
  // (vs HostToGpuAddress which accepts any address in the whole BAR window). Used
  // to translate only genuine device pointers in a kernarg blob and avoid
  // mis-translating by-value scalar fields that coincidentally fall in the window.
  bool IsRegisteredVramPointer(const void* ptr) const;
  void RegisterVramShadow(const void* cpu_addr, size_t size, const void* src);
  hsa_status_t VramShadowAddress(const void* cpu_addr, size_t size,
                                 const void** shadow_addr) const;
  hsa_status_t CreateDirectComputeQueue(DirectComputeQueue* queue);
  hsa_status_t DestroyDirectComputeQueue(DirectComputeQueue& queue);
  hsa_status_t SubmitDirectCompute(DirectComputeQueue& queue,
                                   const uint32_t* pm4, size_t dword_count) const;
  hsa_status_t ReadDirectComputeRptr(const DirectComputeQueue& queue,
                                     uint32_t* rptr) const;
  hsa_status_t SetQueueScratch(DirectComputeQueue& queue,
                               uint64_t scratch_base_256,
                               uint32_t tmpring_size) const;

 private:
  struct VramAllocation {
    uint64_t offset = 0;
    uint64_t size = 0;
    uint64_t gpu_addr = 0;
    std::vector<uint8_t> shadow;
  };

  // Coherent-data (#2): a DART-mapped DMA buffer backing a device allocation.
  struct DmaAllocation {
    uint64_t buffer_id = 0;  // macgpu_free_dma handle
    uint64_t iova = 0;       // single-segment DART bus address (GPU-addressable)
    uint64_t size = 0;
  };

  hsa_status_t EnsureBarMappingsLocked();

  hsa_status_t EnsureDoorbellAperture() const override;
  hsa_status_t ReadMmio32(uint32_t base, uint32_t reg,
                          uint32_t* value) const override;
  hsa_status_t WriteMmio32(uint32_t base, uint32_t reg,
                           uint32_t value) const override;
  hsa_status_t ZeroGpuMemory(uint64_t offset, uint64_t size) const override;
  hsa_status_t WriteGpuMemory32(uint64_t offset, uint32_t value) const override;
  void* GpuMemoryCpuPointer(uint64_t offset) const override;
  // Coherent-QUEUE experiment (ROCR_MACOS_COHERENT_QUEUE, default-off): put the
  // direct queue's ring/MQD/rptr/wptr in a DART-coherent DMA buffer (like Linux's
  // GTT) instead of the cache-inhibited VRAM BAR, to test whether host<->CP
  // coherence races on the non-coherent queue memory cause the intermittent
  // activation/completion timeouts (the op-1 flake).
  bool PreferAllocatedQueueMemory() const override;
  hsa_status_t AllocateQueueMemory(uint64_t size,
                                   lite::DirectQueueMemory* memory) const override;
  hsa_status_t FreeQueueMemory(lite::DirectQueueMemory* memory) const override;
  volatile uint64_t* DoorbellCpuPointer(uint32_t doorbell_index) const override;
  void SleepUs(uint32_t usec) const override;

  // Opaque libmacgpu handle. nullptr until Open() succeeds.
  macgpu_device_t* dev_ = nullptr;
  // Cached device info populated on Open(); reused by GetNodeProperties.
  macgpu_device_info_t info_{};
  mutable std::mutex gpu_lock_;
  void* vram_bar_ = nullptr;
  uint64_t vram_bar_size_ = 0;
  void* doorbell_bar_ = nullptr;
  uint64_t doorbell_bar_size_ = 0;
  uint64_t framebuffer_base_ = 0;
  uint64_t next_vram_offset_ = 0;
  uint32_t next_direct_queue_index_ = 0;
  std::unordered_map<void*, VramAllocation> vram_allocations_;
  // cpu_addr -> DART DMA buffer, for coherent-data device allocations.
  std::unordered_map<void*, DmaAllocation> dma_allocations_;
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_MACOS_DRIVER_H_
