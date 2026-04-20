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

#ifdef __APPLE__

#include "core/inc/amd_macos_driver.h"

#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "core/inc/memory_region.h"

namespace rocr {
namespace AMD {

MacOsDriver::MacOsDriver(std::string devnode_name)
    : core::Driver(core::DriverType::MACOS_DEXT, std::move(devnode_name)) {}

hsa_status_t MacOsDriver::DiscoverDriver(std::unique_ptr<core::Driver>& driver) {
  // Construct a tentative instance, attempt to open the DEXT, and keep
  // it if the handshake succeeds. devnode_name_ carries no information
  // for Darwin (IOKit does service matching by class name, not devnode).
  auto tmp = std::make_unique<MacOsDriver>(std::string("ROCmGPUDriver"));
  hsa_status_t s = tmp->Open();
  if (s != HSA_STATUS_SUCCESS) {
    // HSA_STATUS_ERROR for "no DEXT / not authorized" is normal at
    // discovery time — topology layer treats it as "no device."
    return s;
  }
  s = tmp->QueryKernelModeDriver(core::DriverQuery::GET_DRIVER_VERSION);
  if (s != HSA_STATUS_SUCCESS) {
    tmp->Close();
    return s;
  }
  driver = std::move(tmp);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::Init()     { return HSA_STATUS_SUCCESS; }
hsa_status_t MacOsDriver::ShutDown() { return Close(); }

hsa_status_t MacOsDriver::QueryKernelModeDriver(core::DriverQuery query) {
  if (query != core::DriverQuery::GET_DRIVER_VERSION) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (!dev_) return HSA_STATUS_ERROR;
  // The DEXT doesn't currently publish a version field via the escape
  // table — seed a plausible value derived from the device-info handshake
  // we already performed in Open(). Real versioning lands once the DEXT
  // grows a kROCmGPU_GetVersion selector.
  version_.KernelInterfaceMajorVersion = 1;
  version_.KernelInterfaceMinorVersion = 0;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::Open() {
  if (dev_) return HSA_STATUS_SUCCESS;
  macgpu_status_t r = macgpu_open(&dev_);
  if (r != MACGPU_SUCCESS) {
    dev_ = nullptr;
    // NOT_FOUND is the common path on a Mac without the DEXT installed —
    // let the topology layer treat it as "no GPU from this driver" by
    // returning HSA_STATUS_ERROR (same convention KfdDriver uses).
    return HSA_STATUS_ERROR;
  }
  // Cache the device info so GetSystemProperties / GetNodeProperties can
  // answer without additional DEXT round-trips.
  if (macgpu_get_info(dev_, &info_) != MACGPU_SUCCESS) {
    macgpu_close(dev_);
    dev_ = nullptr;
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::Close() {
  if (dev_) {
    macgpu_close(dev_);
    dev_ = nullptr;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::GetSystemProperties(HsaSystemProperties& sys_props) const {
  // Stage-1 MVP: report a single node that carries the host CPU
  // properties. No FCompute cores yet (GPU-agent discovery is a later
  // commit that lands with MacGpuAgent); reporting CPU cores is what
  // gets ROCR's topology layer to build a CPU agent, which in turn
  // installs the "shared" allocator other parts of the runtime rely
  // on during hsa_init().
  sys_props.NumNodes = dev_ ? 1 : 0;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::GetNodeProperties(HsaNodeProperties& node_props,
                                            uint32_t node_id) const {
  if (node_id != 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  std::memset(&node_props, 0, sizeof(node_props));

  // Query host CPU count via sysctl — Darwin has no _SC_NPROCESSORS_ONLN.
  int ncpu = 1;
  size_t len = sizeof(ncpu);
  if (sysctlbyname("hw.ncpu", &ncpu, &len, nullptr, 0) != 0 || ncpu < 1) {
    ncpu = 1;
  }
  node_props.NumCPUCores = static_cast<HSAuint32>(ncpu);
  node_props.NumFComputeCores = 0;   // no GPU agent yet
  // One memory bank so CpuAgent::InitRegionList builds the host-memory
  // regions (fine-grain, coarse-grain, kernargs). Without this, ROCR's
  // shared allocator stays unset and hsa_init crashes on the first
  // async-signal allocation.
  node_props.NumMemoryBanks = 1;
  node_props.NumCaches = 0;
  node_props.NumIOLinks = 0;
  // Vendor / device identification that downstream consumers look at.
  node_props.VendorId = info_.vendor_id;
  node_props.DeviceId = info_.device_id;
  // NumNeuralCores = 0 (no AIE).
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::GetEdgeProperties(std::vector<HsaIoLinkProperties>& io_link_props,
                                            uint32_t) const {
  // Stage-1 MVP: no IO links advertised. Caller expects a non-error
  // empty vector when the node has no peers.
  io_link_props.clear();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::GetMemoryProperties(uint32_t node_id,
                                              std::vector<HsaMemoryProperties>& mem_props) const {
  if (node_id != 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  // GetNodeProperties reported NumMemoryBanks=1 → caller resized the
  // vector to 1. Populate the single entry with system RAM so that
  // CpuAgent::InitRegionList builds its fine-grain / coarse-grain /
  // kernargs regions over host memory.
  if (mem_props.size() < 1) mem_props.resize(1);

  uint64_t mem_bytes = 0;
  size_t len = sizeof(mem_bytes);
  int mib[2] = {CTL_HW, HW_MEMSIZE};
  if (sysctl(mib, 2, &mem_bytes, &len, nullptr, 0) != 0) {
    mem_bytes = 0;  // Leave ROCR with a best-effort zero if sysctl fails.
  }

  std::memset(&mem_props[0], 0, sizeof(HsaMemoryProperties));
  mem_props[0].HeapType = HSA_HEAPTYPE_SYSTEM;
  mem_props[0].SizeInBytes = mem_bytes;
  mem_props[0].VirtualBaseAddress = 0;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::GetCacheProperties(uint32_t, uint32_t,
                                             std::vector<HsaCacheProperties>& cache_props) const {
  cache_props.clear();
  return HSA_STATUS_SUCCESS;
}

namespace {
// Tracks host allocations so FreeMemory can route untyped void* back to
// the right deallocator. MVP: host memory only (system fine/coarse
// grain + kernargs). GPU local memory lands later via macgpu_alloc_dma.
struct HostAllocRegistry {
  std::mutex m;
  std::unordered_map<void*, size_t> allocations;
};
HostAllocRegistry& GetHostAllocRegistry() {
  static HostAllocRegistry reg;
  return reg;
}
}  // namespace

hsa_status_t MacOsDriver::AllocateMemory(const core::MemoryRegion& /*mem_region*/,
                                         core::MemoryRegion::AllocateFlags /*alloc_flags*/,
                                         void** mem, size_t size,
                                         uint32_t /*node_id*/) {
  if (!mem) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *mem = nullptr;
  // MVP: host-backed allocation only. The only memory regions on a
  // Darwin node today are the CPU agent's fine-grain / coarse-grain /
  // kernargs system regions — none of them are GPU-local. GPU VRAM
  // allocation lands later when MacGpuAgent registers its own
  // AMD::MemoryRegion with IsLocalMemory() and this routes through
  // libmacgpu's macgpu_alloc_dma().

  // mmap gives us zero-filled, page-aligned host memory with RW perms —
  // matches what the Linux path produces from hsaKmtAllocMemory for a
  // system-heap region. Round up to page size so munmap() in FreeMemory
  // uses the same size.
  const size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
  const size_t page = page_size > 0 ? page_size : 4096;
  const size_t rounded = (size + page - 1) & ~(page - 1);
  void* ptr = ::mmap(nullptr, rounded, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
  if (ptr == MAP_FAILED) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  {
    auto& reg = GetHostAllocRegistry();
    std::lock_guard<std::mutex> g(reg.m);
    reg.allocations[ptr] = rounded;
  }
  *mem = ptr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::FreeMemory(void* mem, size_t /*size*/) {
  if (!mem) return HSA_STATUS_SUCCESS;
  auto& reg = GetHostAllocRegistry();
  size_t rounded = 0;
  {
    std::lock_guard<std::mutex> g(reg.m);
    auto it = reg.allocations.find(mem);
    if (it == reg.allocations.end()) return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    rounded = it->second;
    reg.allocations.erase(it);
  }
  if (::munmap(mem, rounded) != 0) return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::CreateQueue(uint32_t, HSA_QUEUE_TYPE, uint32_t,
                                      HSA::hsa_amd_queue_priority_internal_t,
                                      uint32_t, void*, uint64_t, HsaEvent*,
                                      HsaQueueResource&) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::DestroyQueue(HSA_QUEUEID) const { return HSA_STATUS_ERROR; }

hsa_status_t MacOsDriver::UpdateQueue(HSA_QUEUEID, uint32_t,
                                      HSA::hsa_amd_queue_priority_internal_t,
                                      void*, uint64_t, HsaEvent*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::SetQueueCUMask(HSA_QUEUEID, uint32_t, uint32_t*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::AllocQueueGWS(HSA_QUEUEID, uint32_t, uint32_t*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::ExportDMABuf(void*, size_t, int*, size_t*) {
  // Darwin has no dma-buf fd passing between processes for GPU memory —
  // IOSurface is the nearest analog but requires a different
  // handshake. Out of scope.
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::ImportDMABuf(int, const core::Agent&,
                                       core::ShareableHandle*, void*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::DestroyImportedShareableHandle(core::ShareableHandle*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::Map(core::ShareableHandle, void*, size_t, size_t,
                              hsa_access_permission_t) {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::Unmap(core::ShareableHandle, void*, size_t, size_t) {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::CreateShareableHandle(void*, void*, size_t,
                                                const core::Agent&,
                                                core::ShareableHandle*, uint64_t*,
                                                int*, uint64_t*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::DestroyShareableHandle(core::ShareableHandle*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::SPMAcquire(uint32_t) const { return HSA_STATUS_ERROR; }
hsa_status_t MacOsDriver::SPMRelease(uint32_t) const { return HSA_STATUS_ERROR; }
hsa_status_t MacOsDriver::SPMSetDestBuffer(uint32_t, uint32_t, uint32_t*, uint32_t*,
                                           void*, bool*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::SetTrapHandler(uint32_t, const void*, uint64_t,
                                         const void*, uint64_t) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::GetDeviceHandle(uint32_t, void**) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::GetClockCounters(uint32_t, HsaClockCounters*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::GetTileConfig(uint32_t, HsaGpuTileConfig*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::IsModelEnabled(bool* enable) const {
  if (enable) *enable = false;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::GetWallclockFrequency(uint32_t, uint64_t*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::AllocateScratchMemory(uint32_t, uint64_t, void**) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::AvailableMemory(uint32_t, uint64_t*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::RegisterMemory(void*, uint64_t, HsaMemFlags) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::DeregisterMemory(void*) const { return HSA_STATUS_ERROR; }

hsa_status_t MacOsDriver::MakeMemoryResident(const void*, size_t, uint64_t*,
                                             const HsaMemMapFlags*,
                                             uint32_t, const uint32_t*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::MakeMemoryUnresident(const void*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::GetQueueSaveAreaInfo(HSA_QUEUEID, void**, size_t*) const {
  return HSA_STATUS_ERROR;
}

}  // namespace AMD
}  // namespace rocr

#endif  // __APPLE__
