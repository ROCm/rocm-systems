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

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "core/inc/amd_memory_region.h"
#include "core/inc/memory_region.h"

namespace rocr {
namespace AMD {
namespace {

constexpr uint32_t kDoorbellBar = 2;
constexpr uint32_t kMmioBar = 5;
constexpr uint32_t kVramBar = 0;

constexpr uint32_t GC_B0 = 0x1260;
constexpr uint32_t GC_B1 = 0xA000;
constexpr uint32_t NBIO_B2 = 0xD20;

constexpr uint32_t regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN = 0x00c0;
constexpr uint32_t regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL = 0x01cb;
constexpr uint32_t regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL = 0x01ce;

constexpr uint32_t regGRBM_GFX_CNTL = 0x0900;
constexpr uint32_t regCP_MQD_BASE_ADDR = 0x1fa9;
constexpr uint32_t regCP_MQD_BASE_ADDR_HI = 0x1faa;
constexpr uint32_t regCP_HQD_ACTIVE = 0x1fab;
constexpr uint32_t regCP_HQD_VMID = 0x1fac;
constexpr uint32_t regCP_HQD_PERSISTENT_STATE = 0x1fad;
constexpr uint32_t regCP_HQD_PQ_BASE = 0x1fb1;
constexpr uint32_t regCP_HQD_PQ_BASE_HI = 0x1fb2;
constexpr uint32_t regCP_HQD_PQ_RPTR = 0x1fb3;
constexpr uint32_t regCP_HQD_PQ_RPTR_REPORT_ADDR = 0x1fb4;
constexpr uint32_t regCP_HQD_PQ_RPTR_REPORT_ADDR_HI = 0x1fb5;
constexpr uint32_t regCP_HQD_PQ_WPTR_POLL_ADDR = 0x1fb6;
constexpr uint32_t regCP_HQD_PQ_WPTR_POLL_ADDR_HI = 0x1fb7;
constexpr uint32_t regCP_HQD_PQ_DOORBELL_CONTROL = 0x1fb8;
constexpr uint32_t regCP_HQD_PQ_CONTROL = 0x1fba;
constexpr uint32_t regCP_HQD_DEQUEUE_REQUEST = 0x1fc1;
constexpr uint32_t regCP_MQD_CONTROL = 0x1fcb;
constexpr uint32_t regCP_HQD_EOP_BASE_ADDR = 0x1fce;
constexpr uint32_t regCP_HQD_EOP_BASE_ADDR_HI = 0x1fcf;
constexpr uint32_t regCP_HQD_EOP_CONTROL = 0x1fd0;
constexpr uint32_t regCP_HQD_PQ_WPTR_LO = 0x1fdf;
constexpr uint32_t regCP_HQD_PQ_WPTR_HI = 0x1fe0;
constexpr uint32_t regCP_HQD_DEQUEUE_STATUS = 0x1fe8;
constexpr uint32_t regCP_MEC_DOORBELL_RANGE_LOWER = 0x1dfc;
constexpr uint32_t regCP_MEC_DOORBELL_RANGE_UPPER = 0x1dfd;

constexpr uint32_t CP_HQD_PERSISTENT_STATE_DEFAULT = 0x0be05501;
constexpr uint32_t MQD_SIZE = 0x1000;
constexpr uint32_t DIRECT_COMPUTE_RING_SIZE = 0x1000;
constexpr uint32_t DIRECT_COMPUTE_EOP_SIZE = 0x1000;

constexpr uint64_t DIRECT_COMPUTE_BASE_OFF = 0x1900000;
constexpr uint64_t DIRECT_COMPUTE_STRIDE = 0x40000;
constexpr uint64_t DIRECT_COMPUTE_MQD_REL = 0x00000;
constexpr uint64_t DIRECT_COMPUTE_RING_REL = 0x02000;
constexpr uint64_t DIRECT_COMPUTE_EOP_REL = 0x10000;
constexpr uint64_t DIRECT_COMPUTE_RPTR_REL = 0x20000;
constexpr uint64_t DIRECT_COMPUTE_WPTR_REL = 0x21000;
constexpr uint32_t DIRECT_COMPUTE_DOORBELL = 0x20;

// Keep the general-purpose VRAM bump allocator away from the low queue
// scratch window used by the proven direct-compute bring-up scripts.
constexpr uint64_t kVramAllocBaseOffset = 64ull * 1024 * 1024;

uint64_t AlignUpU64(uint64_t value, uint64_t align) {
  return (value + align - 1) & ~(align - 1);
}

size_t AlignUpSize(size_t value, size_t align) {
  return static_cast<size_t>(AlignUpU64(value, align));
}

bool TraceGpuAllocs() { return std::getenv("ROCR_MACOS_TRACE_GPU_ALLOC") != nullptr; }
bool TraceDirectQueue() { return std::getenv("ROCR_MACOS_TRACE_DIRECT_QUEUE") != nullptr; }
bool EnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}
bool TraceDirectQueueVerbose() {
  return EnvEnabled("ROCR_MACOS_TRACE_DIRECT_QUEUE_VERBOSE");
}
bool UseDirectQueueDequeue() {
  // Directly clearing CP_HQD_ACTIVE can leave gfx12 HQDs half-reclaimed:
  // ACTIVE relatches, but the doorbell enable bit may not stick and the
  // queue stops consuming. Keep the firmware dequeue path as the default,
  // with an escape hatch for A/B testing.
  if (EnvEnabled("ROCR_MACOS_DIRECT_QUEUE_DISABLE_DEQUEUE")) return false;
  const char* value = std::getenv("ROCR_MACOS_DIRECT_QUEUE_DEQUEUE");
  if (value != nullptr && value[0] != '\0') return std::strcmp(value, "0") != 0;
  return true;
}
bool SkipDirectQueueDestroy() {
  return EnvEnabled("ROCR_MACOS_DIRECT_QUEUE_SKIP_DESTROY");
}
uint32_t DirectQueueDequeueSettleUs() {
  const char* value = std::getenv("ROCR_MACOS_DIRECT_QUEUE_DEQUEUE_SETTLE_US");
  if (value == nullptr || value[0] == '\0') return 100000;
  char* end = nullptr;
  errno = 0;
  unsigned long parsed = std::strtoul(value, &end, 0);
  if (errno != 0 || end == value || parsed == 0 || parsed > 5000000ul) return 100000;
  return static_cast<uint32_t>(parsed);
}

}  // namespace

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
  // Non-zero NumFComputeCores is the signal to DiscoverGpu() that this
  // node has a GPU. Use a conservative Navi48/RDNA4 shape until libmacgpu
  // exposes a real CU-count escape.
  node_props.NumFComputeCores = 64 * 2;
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
  node_props.LocalMemSize = info_.vram_size;
  node_props.EngineId.ui32.Major = 12;
  node_props.EngineId.ui32.Minor = 0;
  node_props.EngineId.ui32.Stepping = 1;
  node_props.WaveFrontSize = 32;
  node_props.NumSIMDPerCU = 2;
  node_props.NumCUPerArray = 8;
  node_props.NumArrays = 2;
  node_props.NumShaderBanks = 4;
  node_props.MaxWavesPerSIMD = 16;
  node_props.LDSSizeInKB = 64;
  node_props.MaxEngineClockMhzFCompute = 2500;
  node_props.NumSdmaEngines = 1;
  node_props.NumSdmaQueuesPerEngine = 1;
  node_props.NumCpQueues = 1;
  node_props.NumXcc = 1;
  node_props.Capability.ui32.QueueSizePowerOfTwo = 1;
  node_props.Capability.ui32.QueueSize32bit = 1;
  node_props.Capability.ui32.ASICRevision = info_.revision_id & 0xF;
  node_props.Capability.ui32.SVMAPISupported = 1;
  std::snprintf(reinterpret_cast<char*>(node_props.AMDName), HSA_PUBLIC_NAME_SIZE, "gfx1201");
  const char name[] = "AMD Radeon RX 9000 (macOS eGPU)";
  for (size_t i = 0; i < sizeof(name) && i < HSA_PUBLIC_NAME_SIZE; ++i) {
    node_props.MarketingName[i] = static_cast<HSAuint16>(name[i]);
  }
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
  // HIP/ROCclr tears down devices from a different dylib's static destructor.
  // Keep this registry alive until process exit so late frees do not race C++
  // static destruction order.
  static HostAllocRegistry* reg = new HostAllocRegistry();
  return *reg;
}
bool TraceHostAllocs() { return std::getenv("ROCR_MACOS_TRACE_ALLOC") != nullptr; }
}  // namespace

hsa_status_t MacOsDriver::AllocateMemory(const core::MemoryRegion& mem_region,
                                         core::MemoryRegion::AllocateFlags alloc_flags,
                                         void** mem, size_t size,
                                         uint32_t /*node_id*/) {
  if (!mem) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *mem = nullptr;

  const auto& amd_region = static_cast<const AMD::MemoryRegion&>(mem_region);
  if (amd_region.IsLocalMemory()) {
    uint64_t gpu_addr = 0;
    hsa_status_t status = AllocateVram(size, 4096, mem, &gpu_addr);
    if (status == HSA_STATUS_SUCCESS && TraceGpuAllocs()) {
      std::fprintf(stderr, "ROCR macOS vram alloc %p gpu=0x%llx size=%zu flags=0x%x\n", *mem,
                   static_cast<unsigned long long>(gpu_addr), size, alloc_flags);
    }
    return status;
  }

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
  if (TraceHostAllocs()) {
    std::fprintf(stderr, "ROCR macOS alloc %p size=%zu rounded=%zu\n", ptr, size, rounded);
  }
  *mem = ptr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::FreeMemory(void* mem, size_t /*size*/) {
  if (!mem) return HSA_STATUS_SUCCESS;
  {
    std::lock_guard<std::mutex> g(gpu_lock_);
    auto it = vram_allocations_.find(mem);
    if (it != vram_allocations_.end()) {
      if (TraceGpuAllocs()) {
        std::fprintf(stderr, "ROCR macOS vram free %p gpu=0x%llx size=%llu\n", mem,
                     static_cast<unsigned long long>(it->second.gpu_addr),
                     static_cast<unsigned long long>(it->second.size));
      }
      vram_allocations_.erase(it);
      // Bump-allocated BAR0 VRAM is intentionally not reused yet.
      return HSA_STATUS_SUCCESS;
    }
  }
  auto& reg = GetHostAllocRegistry();
  size_t rounded = 0;
  {
    std::lock_guard<std::mutex> g(reg.m);
    auto it = reg.allocations.find(mem);
    if (it == reg.allocations.end()) {
      if (TraceHostAllocs()) {
        std::fprintf(stderr, "ROCR macOS free unknown %p\n", mem);
      }
      return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    }
    rounded = it->second;
    reg.allocations.erase(it);
  }
  if (TraceHostAllocs()) {
    std::fprintf(stderr, "ROCR macOS free %p rounded=%zu\n", mem, rounded);
  }
  if (::munmap(mem, rounded) != 0) {
    if (TraceHostAllocs()) {
      std::fprintf(stderr, "ROCR macOS munmap failed %p rounded=%zu errno=%d\n", mem, rounded,
                   errno);
    }
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::EnsureBarMappingsLocked() {
  if (!dev_) return HSA_STATUS_ERROR;
  if (vram_bar_ == nullptr) {
    macgpu_status_t r = macgpu_map_bar(dev_, kVramBar, &vram_bar_, &vram_bar_size_);
    if (r != MACGPU_SUCCESS) return HSA_STATUS_ERROR;
  }
  if (doorbell_bar_ == nullptr) {
    macgpu_status_t r = macgpu_map_bar(dev_, kDoorbellBar, &doorbell_bar_, &doorbell_bar_size_);
    if (r != MACGPU_SUCCESS) return HSA_STATUS_ERROR;
  }
  if (framebuffer_base_ == 0) {
    uint32_t fb = 0;
    macgpu_status_t r = macgpu_mmio_read32(dev_, kMmioBar, (0x1A000 + 0x0554) * 4, &fb);
    if (r != MACGPU_SUCCESS) return HSA_STATUS_ERROR;
    framebuffer_base_ = static_cast<uint64_t>(fb & 0xFFFFFF) << 24;
  }
  if (next_vram_offset_ == 0) {
    next_vram_offset_ = kVramAllocBaseOffset;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::AllocateVram(size_t size, size_t align, void** cpu_addr,
                                       uint64_t* gpu_addr) {
  if (cpu_addr == nullptr || gpu_addr == nullptr || size == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  *cpu_addr = nullptr;
  *gpu_addr = 0;
  align = std::max<size_t>(align, 4096);
  if ((align & (align - 1)) != 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  std::lock_guard<std::mutex> g(gpu_lock_);
  hsa_status_t status = EnsureBarMappingsLocked();
  if (status != HSA_STATUS_SUCCESS) return status;

  const uint64_t rounded = AlignUpSize(size, align);
  const uint64_t offset = AlignUpU64(next_vram_offset_, align);
  if (offset + rounded > vram_bar_size_) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  auto* ptr = static_cast<char*>(vram_bar_) + offset;
  next_vram_offset_ = offset + rounded;
  VramAllocation alloc;
  alloc.offset = offset;
  alloc.size = rounded;
  alloc.gpu_addr = framebuffer_base_ + offset;
  vram_allocations_[ptr] = alloc;
  *cpu_addr = ptr;
  *gpu_addr = alloc.gpu_addr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::HostToGpuAddress(const void* ptr, uint64_t* gpu_addr) const {
  if (ptr == nullptr || gpu_addr == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *gpu_addr = 0;
  std::lock_guard<std::mutex> g(gpu_lock_);
  if (vram_bar_ == nullptr || framebuffer_base_ == 0) return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  const auto base = reinterpret_cast<uintptr_t>(vram_bar_);
  const auto p = reinterpret_cast<uintptr_t>(ptr);
  if (p < base || p >= base + vram_bar_size_) return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  *gpu_addr = framebuffer_base_ + (p - base);
  return HSA_STATUS_SUCCESS;
}

void MacOsDriver::RegisterVramShadow(const void* cpu_addr, size_t size, const void* src) {
  if (cpu_addr == nullptr || src == nullptr || size == 0) return;

  std::lock_guard<std::mutex> g(gpu_lock_);
  const auto p = reinterpret_cast<uintptr_t>(cpu_addr);
  for (auto& kv : vram_allocations_) {
    const auto base = reinterpret_cast<uintptr_t>(kv.first);
    auto& alloc = kv.second;
    if (p < base || p + size > base + alloc.size) continue;

    if (alloc.shadow.empty()) alloc.shadow.resize(static_cast<size_t>(alloc.size));
    std::memcpy(alloc.shadow.data() + (p - base), src, size);
    return;
  }
}

hsa_status_t MacOsDriver::VramShadowAddress(const void* cpu_addr, size_t size,
                                            const void** shadow_addr) const {
  if (cpu_addr == nullptr || shadow_addr == nullptr || size == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  *shadow_addr = nullptr;

  std::lock_guard<std::mutex> g(gpu_lock_);
  const auto p = reinterpret_cast<uintptr_t>(cpu_addr);
  for (const auto& kv : vram_allocations_) {
    const auto base = reinterpret_cast<uintptr_t>(kv.first);
    const auto& alloc = kv.second;
    if (alloc.shadow.empty() || p < base || p + size > base + alloc.size) continue;

    *shadow_addr = alloc.shadow.data() + (p - base);
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ALLOCATION;
}

hsa_status_t MacOsDriver::ReadMmio32(uint32_t base, uint32_t reg, uint32_t* value) const {
  if (value == nullptr || dev_ == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return macgpu_mmio_read32(dev_, kMmioBar, static_cast<uint64_t>(base + reg) * 4, value) ==
                 MACGPU_SUCCESS
             ? HSA_STATUS_SUCCESS
             : HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::WriteMmio32(uint32_t base, uint32_t reg, uint32_t value) const {
  if (dev_ == nullptr) return HSA_STATUS_ERROR;
  return macgpu_mmio_write32(dev_, kMmioBar, static_cast<uint64_t>(base + reg) * 4, value) ==
                 MACGPU_SUCCESS
             ? HSA_STATUS_SUCCESS
             : HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::EnsureDoorbellApertureLocked() {
  hsa_status_t status = WriteMmio32(NBIO_B2, regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN, 1);
  if (status != HSA_STATUS_SUCCESS) return status;
  status = WriteMmio32(NBIO_B2, regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL,
                       (1u << 0) | (3u << 1) | (3u << 28));
  if (status != HSA_STATUS_SUCCESS) return status;
  status = WriteMmio32(NBIO_B2, regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL,
                       (1u << 0) | (6u << 1) | (3u << 28));
  if (status != HSA_STATUS_SUCCESS) return status;
  status = WriteMmio32(GC_B0, regCP_MEC_DOORBELL_RANGE_LOWER, 0);
  if (status != HSA_STATUS_SUCCESS) return status;
  return WriteMmio32(GC_B0, regCP_MEC_DOORBELL_RANGE_UPPER, (0x8Au * 2u) << 2);
}

hsa_status_t MacOsDriver::SelectHqdLocked(uint32_t me, uint32_t pipe, uint32_t queue) const {
  return WriteMmio32(GC_B1, regGRBM_GFX_CNTL,
                     ((pipe & 0x3u) << 0) | ((me & 0x3u) << 2) |
                         ((0u & 0xFu) << 4) | ((queue & 0x7u) << 8));
}

hsa_status_t MacOsDriver::WaitForDirectHqdIdleLocked(uint32_t pipe, uint32_t queue,
                                                     const char* phase) const {
  const uint32_t timeout_us = DirectQueueDequeueSettleUs();
  constexpr uint32_t kStepUs = 1000;
  const uint32_t max_samples = std::max<uint32_t>(1, timeout_us / kStepUs);
  uint32_t active = 0;
  uint32_t pq_control = 0;
  uint32_t doorbell_control = 0;
  uint32_t dequeue_status = 0;
  for (uint32_t i = 0; i < max_samples; ++i) {
    hsa_status_t status = ReadMmio32(GC_B0, regCP_HQD_ACTIVE, &active);
    if (status != HSA_STATUS_SUCCESS) return status;
    status = ReadMmio32(GC_B0, regCP_HQD_PQ_CONTROL, &pq_control);
    if (status != HSA_STATUS_SUCCESS) return status;
    status = ReadMmio32(GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
    if (status != HSA_STATUS_SUCCESS) return status;
    status = ReadMmio32(GC_B0, regCP_HQD_DEQUEUE_STATUS, &dequeue_status);
    if (status != HSA_STATUS_SUCCESS) return status;
    if (active == 0 && (doorbell_control & 0x40000000u) == 0) {
      if (TraceDirectQueue() && i > 0) {
        std::fprintf(stderr,
                     "ROCR macOS direct queue hqd idle phase=%s pipe=%u hqd=%u "
                     "samples=%u active=0x%x pq_control=0x%x doorbell_control=0x%x "
                     "dequeue_status=0x%x\n",
                     phase ? phase : "unknown", pipe, queue, i + 1, active, pq_control,
                     doorbell_control, dequeue_status);
      }
      return HSA_STATUS_SUCCESS;
    }
    ::usleep(kStepUs);
  }
  if (TraceDirectQueue()) {
    std::fprintf(stderr,
                 "ROCR macOS direct queue hqd idle timeout phase=%s pipe=%u hqd=%u "
                 "active=0x%x pq_control=0x%x doorbell_control=0x%x "
                 "dequeue_status=0x%x timeout_us=%u\n",
                 phase ? phase : "unknown", pipe, queue, active, pq_control,
                 doorbell_control, dequeue_status, timeout_us);
  }
  return HSA_STATUS_ERROR;
}

void MacOsDriver::VramWrite32Locked(uint64_t offset, uint32_t value) const {
  auto* ptr = reinterpret_cast<volatile uint32_t*>(static_cast<char*>(vram_bar_) + offset);
  *ptr = value;
}

void MacOsDriver::ZeroVramLocked(uint64_t offset, uint64_t size) const {
  for (uint64_t i = 0; i < size; i += sizeof(uint32_t)) {
    VramWrite32Locked(offset + i, 0);
  }
}

hsa_status_t MacOsDriver::CreateDirectComputeQueue(DirectComputeQueue* queue) {
  if (queue == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> g(gpu_lock_);
  hsa_status_t status = EnsureBarMappingsLocked();
  if (status != HSA_STATUS_SUCCESS) return status;
  status = EnsureDoorbellApertureLocked();
  if (status != HSA_STATUS_SUCCESS) return status;
  if (next_direct_queue_index_ >= 8) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  queue->queue_index = next_direct_queue_index_++;
  queue->queue_id = queue->queue_index + 1;
  queue->doorbell_index = DIRECT_COMPUTE_DOORBELL + queue->queue_index * 2;
  queue->ring_size_bytes = DIRECT_COMPUTE_RING_SIZE;

  status = ActivateDirectComputeQueueLocked(queue);
  if (status != HSA_STATUS_SUCCESS) {
    --next_direct_queue_index_;
    *queue = {};
  }
  return status;
}

hsa_status_t MacOsDriver::ActivateDirectComputeQueueLocked(DirectComputeQueue* queue) {
  const uint32_t me = 1;
  const uint32_t pipe = queue->queue_index / 4;
  const uint32_t hqd_queue = queue->queue_index % 4;
  hsa_status_t status = SelectHqdLocked(me, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) return status;

  uint32_t active = 0;
  status = ReadMmio32(GC_B0, regCP_HQD_ACTIVE, &active);
  if (status != HSA_STATUS_SUCCESS) return status;
  const bool force = std::getenv("AMD_GPU_MACOS_FORCE_DIRECT_COMPUTE") != nullptr;
  if (active != 0 && !force) {
    WriteMmio32(GC_B1, regGRBM_GFX_CNTL, 0);
    return HSA_STATUS_ERROR;
  }
  if (active != 0) {
    if (UseDirectQueueDequeue()) {
      if (TraceDirectQueue()) {
        std::fprintf(stderr,
                     "ROCR macOS direct queue reclaim active HQD with dequeue "
                     "index=%u pipe=%u hqd=%u active=0x%x\n",
                     queue->queue_index, pipe, hqd_queue, active);
      }
      WriteMmio32(GC_B0, regCP_HQD_DEQUEUE_REQUEST, 1);
      uint32_t now_active = active;
      for (uint32_t i = 0; i < 1000; ++i) {
        ReadMmio32(GC_B0, regCP_HQD_ACTIVE, &now_active);
        if (now_active == 0) break;
        ::usleep(1000);
      }
      WriteMmio32(GC_B0, regCP_HQD_DEQUEUE_REQUEST, 0);
      hsa_status_t idle_status = WaitForDirectHqdIdleLocked(pipe, hqd_queue, "activate-reclaim");
      if (idle_status != HSA_STATUS_SUCCESS) {
        WriteMmio32(GC_B1, regGRBM_GFX_CNTL, 0);
        return idle_status;
      }
      if (TraceDirectQueue()) {
        std::fprintf(stderr,
                     "ROCR macOS direct queue dequeue reclaim complete "
                     "index=%u pipe=%u hqd=%u active=0x%x\n",
                     queue->queue_index, pipe, hqd_queue, now_active);
      }
    } else {
      if (TraceDirectQueue()) {
        std::fprintf(stderr,
                     "ROCR macOS direct queue reclaim active HQD without dequeue "
                     "index=%u pipe=%u hqd=%u active=0x%x\n",
                     queue->queue_index, pipe, hqd_queue, active);
      }
      WriteMmio32(GC_B0, regCP_HQD_ACTIVE, 0);
      uint32_t now_active = active;
      for (uint32_t i = 0; i < 1000; ++i) {
        ReadMmio32(GC_B0, regCP_HQD_ACTIVE, &now_active);
        if (now_active == 0) break;
        ::usleep(1000);
      }
      if (TraceDirectQueue()) {
        std::fprintf(stderr,
                     "ROCR macOS direct queue direct-disable reclaim complete "
                     "index=%u pipe=%u hqd=%u active=0x%x\n",
                     queue->queue_index, pipe, hqd_queue, now_active);
      }
    }
  }

  const uint64_t base_off = DIRECT_COMPUTE_BASE_OFF + queue->queue_index * DIRECT_COMPUTE_STRIDE;
  const uint64_t mqd_off = base_off + DIRECT_COMPUTE_MQD_REL;
  const uint64_t ring_off = base_off + DIRECT_COMPUTE_RING_REL;
  const uint64_t eop_off = base_off + DIRECT_COMPUTE_EOP_REL;
  const uint64_t rptr_off = base_off + DIRECT_COMPUTE_RPTR_REL;
  const uint64_t wptr_off = base_off + DIRECT_COMPUTE_WPTR_REL;
  const uint64_t mqd_mc = framebuffer_base_ + mqd_off;
  const uint64_t ring_mc = framebuffer_base_ + ring_off;
  const uint64_t eop_mc = framebuffer_base_ + eop_off;
  const uint64_t rptr_mc = framebuffer_base_ + rptr_off;
  const uint64_t wptr_mc = framebuffer_base_ + wptr_off;

  ZeroVramLocked(mqd_off, MQD_SIZE);
  ZeroVramLocked(ring_off, DIRECT_COMPUTE_RING_SIZE);
  ZeroVramLocked(eop_off, DIRECT_COMPUTE_EOP_SIZE);
  ZeroVramLocked(rptr_off, 0x20);
  ZeroVramLocked(wptr_off, 0x20);

  std::array<uint32_t, MQD_SIZE / sizeof(uint32_t)> mqd{};
  mqd[0] = 0xC0310800;
  mqd[1] = 1;
  for (uint32_t dw : {0x17u, 0x18u, 0x1Au, 0x1Bu}) mqd[dw] = 0xFFFFFFFFu;
  mqd[0x2C] = 7;

  const uint64_t eop_base_shifted = eop_mc >> 8;
  mqd[0xA5] = static_cast<uint32_t>(eop_base_shifted);
  mqd[0xA6] = static_cast<uint32_t>(eop_base_shifted >> 32);
  mqd[0xA7] = 9;  // bit_length(4 KiB / 4) - 2, matching the Python bring-up path.

  mqd[0x80] = static_cast<uint32_t>(mqd_mc) & 0xFFFFFFFCu;
  mqd[0x81] = static_cast<uint32_t>(mqd_mc >> 32);
  mqd[0x82] = 1;
  mqd[0x84] = (CP_HQD_PERSISTENT_STATE_DEFAULT & ~(0x3FFu << 8)) | (0x55u << 8);

  const uint64_t pq_base_shifted = ring_mc >> 8;
  mqd[0x88] = static_cast<uint32_t>(pq_base_shifted);
  mqd[0x89] = static_cast<uint32_t>(pq_base_shifted >> 32);
  mqd[0x8B] = static_cast<uint32_t>(rptr_mc) & 0xFFFFFFFCu;
  mqd[0x8C] = static_cast<uint32_t>(rptr_mc >> 32) & 0xFFFFu;
  mqd[0x8D] = static_cast<uint32_t>(wptr_mc) & 0xFFFFFFF8u;
  mqd[0x8E] = static_cast<uint32_t>(wptr_mc >> 32) & 0xFFFFu;
  mqd[0x8F] = ((queue->doorbell_index & 0x03FFFFFFu) << 2) | (1u << 30);

  const uint32_t ring_dw = DIRECT_COMPUTE_RING_SIZE / sizeof(uint32_t);
  const uint32_t queue_size_val = 9;  // bit_length(0x1000 / 4) - 2.
  mqd[0x91] = queue_size_val | (5u << 8) | (1u << 27) | (1u << 28) | (1u << 30) |
              (1u << 31) | 0x300000u | 0x8000u;
  (void)ring_dw;
  mqd[0x95] = 0x00300000;
  mqd[0xA2] = 0x100;
  mqd[0xB8] = 1u << 15;

  for (size_t i = 0; i < mqd.size(); ++i) VramWrite32Locked(mqd_off + i * 4, mqd[i]);
  std::atomic_thread_fence(std::memory_order_seq_cst);

  WriteMmio32(GC_B0, regCP_HQD_ACTIVE, 0);
  WriteMmio32(GC_B0, regCP_HQD_PQ_RPTR, 0);
  WriteMmio32(GC_B0, regCP_HQD_PQ_WPTR_LO, 0);
  WriteMmio32(GC_B0, regCP_HQD_PQ_WPTR_HI, 0);
  uint32_t vmid = 0;
  ReadMmio32(GC_B0, regCP_HQD_VMID, &vmid);
  WriteMmio32(GC_B0, regCP_HQD_VMID, vmid & ~0xFu);
  uint32_t doorbell_ctl = 0;
  ReadMmio32(GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_ctl);
  WriteMmio32(GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL, doorbell_ctl & ~0x40000000u);
  WriteMmio32(GC_B0, regCP_MQD_BASE_ADDR, mqd[0x80]);
  WriteMmio32(GC_B0, regCP_MQD_BASE_ADDR_HI, mqd[0x81]);
  WriteMmio32(GC_B0, regCP_MQD_CONTROL, 0);
  WriteMmio32(GC_B0, regCP_HQD_EOP_BASE_ADDR, mqd[0xA5]);
  WriteMmio32(GC_B0, regCP_HQD_EOP_BASE_ADDR_HI, mqd[0xA6]);
  WriteMmio32(GC_B0, regCP_HQD_EOP_CONTROL, mqd[0xA7]);
  WriteMmio32(GC_B0, regCP_HQD_PQ_BASE, mqd[0x88]);
  WriteMmio32(GC_B0, regCP_HQD_PQ_BASE_HI, mqd[0x89]);
  WriteMmio32(GC_B0, regCP_HQD_PQ_RPTR_REPORT_ADDR, mqd[0x8B]);
  WriteMmio32(GC_B0, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, mqd[0x8C]);
  WriteMmio32(GC_B0, regCP_HQD_PQ_CONTROL, mqd[0x91]);
  WriteMmio32(GC_B0, regCP_HQD_PQ_WPTR_POLL_ADDR, mqd[0x8D]);
  WriteMmio32(GC_B0, regCP_HQD_PQ_WPTR_POLL_ADDR_HI, mqd[0x8E]);
  WriteMmio32(GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL, mqd[0x8F]);
  WriteMmio32(GC_B0, regCP_HQD_PERSISTENT_STATE, mqd[0x84]);
  if (TraceDirectQueueVerbose()) {
    uint32_t mqd_base = 0;
    uint32_t mqd_base_hi = 0;
    uint32_t pq_base = 0;
    uint32_t pq_base_hi = 0;
    uint32_t pq_control = 0;
    uint32_t doorbell_control = 0;
    uint32_t persistent = 0;
    uint32_t vmid = 0;
    uint32_t active_before = 0;
    ReadMmio32(GC_B0, regCP_HQD_ACTIVE, &active_before);
    ReadMmio32(GC_B0, regCP_MQD_BASE_ADDR, &mqd_base);
    ReadMmio32(GC_B0, regCP_MQD_BASE_ADDR_HI, &mqd_base_hi);
    ReadMmio32(GC_B0, regCP_HQD_PQ_BASE, &pq_base);
    ReadMmio32(GC_B0, regCP_HQD_PQ_BASE_HI, &pq_base_hi);
    ReadMmio32(GC_B0, regCP_HQD_PQ_CONTROL, &pq_control);
    ReadMmio32(GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
    ReadMmio32(GC_B0, regCP_HQD_PERSISTENT_STATE, &persistent);
    ReadMmio32(GC_B0, regCP_HQD_VMID, &vmid);
    std::fprintf(stderr,
                 "ROCR macOS direct queue pre-active readback qid=%u "
                 "active=0x%x mqd=0x%08x:%08x pq=0x%08x:%08x "
                 "pq_control=0x%08x doorbell_control=0x%08x "
                 "persistent=0x%08x vmid=0x%08x\n",
                 queue->queue_id, active_before, mqd_base_hi, mqd_base,
                 pq_base_hi, pq_base, pq_control, doorbell_control, persistent, vmid);
  }
  WriteMmio32(GC_B0, regCP_HQD_ACTIVE, 1);
  if (TraceDirectQueueVerbose()) {
    uint32_t active_immediate = 0;
    uint32_t pq_control = 0;
    uint32_t doorbell_control = 0;
    ReadMmio32(GC_B0, regCP_HQD_ACTIVE, &active_immediate);
    ReadMmio32(GC_B0, regCP_HQD_PQ_CONTROL, &pq_control);
    ReadMmio32(GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
    std::fprintf(stderr,
                 "ROCR macOS direct queue post-active-write readback qid=%u "
                 "active=0x%x pq_control=0x%08x doorbell_control=0x%08x\n",
                 queue->queue_id, active_immediate, pq_control, doorbell_control);
  }

  ::usleep(10000);
  uint32_t post_active = 0;
  status = ReadMmio32(GC_B0, regCP_HQD_ACTIVE, &post_active);
  if (status != HSA_STATUS_SUCCESS) {
    WriteMmio32(GC_B1, regGRBM_GFX_CNTL, 0);
    return status;
  }
  if (post_active == 0) {
    if (TraceDirectQueue()) {
      uint32_t mqd_base = 0;
      uint32_t mqd_base_hi = 0;
      uint32_t eop_base = 0;
      uint32_t eop_base_hi = 0;
      uint32_t eop_control = 0;
      uint32_t pq_base = 0;
      uint32_t pq_base_hi = 0;
      uint32_t pq_control = 0;
      uint32_t doorbell_control = 0;
      uint32_t rptr_report = 0;
      uint32_t rptr_report_hi = 0;
      uint32_t wptr_poll = 0;
      uint32_t wptr_poll_hi = 0;
      uint32_t persistent = 0;
      uint32_t vmid = 0;
      uint32_t rptr = 0;
      uint32_t wptr = 0;
      uint32_t wptr_hi = 0;
      ReadMmio32(GC_B0, regCP_MQD_BASE_ADDR, &mqd_base);
      ReadMmio32(GC_B0, regCP_MQD_BASE_ADDR_HI, &mqd_base_hi);
      ReadMmio32(GC_B0, regCP_HQD_EOP_BASE_ADDR, &eop_base);
      ReadMmio32(GC_B0, regCP_HQD_EOP_BASE_ADDR_HI, &eop_base_hi);
      ReadMmio32(GC_B0, regCP_HQD_EOP_CONTROL, &eop_control);
      ReadMmio32(GC_B0, regCP_HQD_PQ_BASE, &pq_base);
      ReadMmio32(GC_B0, regCP_HQD_PQ_BASE_HI, &pq_base_hi);
      ReadMmio32(GC_B0, regCP_HQD_PQ_CONTROL, &pq_control);
      ReadMmio32(GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
      ReadMmio32(GC_B0, regCP_HQD_PQ_RPTR_REPORT_ADDR, &rptr_report);
      ReadMmio32(GC_B0, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, &rptr_report_hi);
      ReadMmio32(GC_B0, regCP_HQD_PQ_WPTR_POLL_ADDR, &wptr_poll);
      ReadMmio32(GC_B0, regCP_HQD_PQ_WPTR_POLL_ADDR_HI, &wptr_poll_hi);
      ReadMmio32(GC_B0, regCP_HQD_PERSISTENT_STATE, &persistent);
      ReadMmio32(GC_B0, regCP_HQD_VMID, &vmid);
      ReadMmio32(GC_B0, regCP_HQD_PQ_RPTR, &rptr);
      ReadMmio32(GC_B0, regCP_HQD_PQ_WPTR_LO, &wptr);
      ReadMmio32(GC_B0, regCP_HQD_PQ_WPTR_HI, &wptr_hi);
      std::fprintf(stderr,
                   "ROCR macOS direct queue activate failed qid=%u index=%u me=1 pipe=%u hqd=%u "
                   "doorbell=0x%x base_off=0x%llx ring=0x%llx active=0x0\n",
                   queue->queue_id, queue->queue_index, pipe, hqd_queue, queue->doorbell_index,
                   static_cast<unsigned long long>(base_off),
                   static_cast<unsigned long long>(ring_mc));
      std::fprintf(stderr,
                   "ROCR macOS direct queue failed readback "
                   "mqd=0x%08x:%08x eop=0x%08x:%08x eop_ctl=0x%08x "
                   "pq=0x%08x:%08x pq_ctl=0x%08x doorbell_ctl=0x%08x "
                   "rptr_report=0x%08x:%08x wptr_poll=0x%08x:%08x "
                   "persistent=0x%08x vmid=0x%08x rptr=0x%x wptr=0x%08x:%08x\n",
                   mqd_base_hi, mqd_base, eop_base_hi, eop_base, eop_control,
                   pq_base_hi, pq_base, pq_control, doorbell_control, rptr_report_hi,
                   rptr_report, wptr_poll_hi, wptr_poll, persistent, vmid, rptr, wptr_hi,
                   wptr);
    }
    WriteMmio32(GC_B1, regGRBM_GFX_CNTL, 0);
    return HSA_STATUS_ERROR;
  }

  if (TraceDirectQueue()) {
    uint32_t pq_base = 0;
    uint32_t pq_base_hi = 0;
    uint32_t pq_control = 0;
    uint32_t doorbell_control = 0;
    uint32_t rptr = 0;
    uint32_t wptr = 0;
    uint32_t wptr_hi = 0;
    ReadMmio32(GC_B0, regCP_HQD_PQ_BASE, &pq_base);
    ReadMmio32(GC_B0, regCP_HQD_PQ_BASE_HI, &pq_base_hi);
    ReadMmio32(GC_B0, regCP_HQD_PQ_CONTROL, &pq_control);
    ReadMmio32(GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
    ReadMmio32(GC_B0, regCP_HQD_PQ_RPTR, &rptr);
    ReadMmio32(GC_B0, regCP_HQD_PQ_WPTR_LO, &wptr);
    ReadMmio32(GC_B0, regCP_HQD_PQ_WPTR_HI, &wptr_hi);
    std::fprintf(stderr,
                 "ROCR macOS direct queue activate qid=%u index=%u me=1 pipe=%u hqd=%u "
                 "doorbell=0x%x base_off=0x%llx ring=0x%llx active=0x%x "
                 "pq_base=0x%08x:%08x pq_control=0x%08x doorbell_control=0x%08x "
                 "rptr=0x%x wptr=0x%08x:%08x\n",
                 queue->queue_id, queue->queue_index, pipe, hqd_queue, queue->doorbell_index,
                 static_cast<unsigned long long>(base_off),
                 static_cast<unsigned long long>(ring_mc), post_active, pq_base_hi, pq_base,
                 pq_control, doorbell_control, rptr, wptr_hi, wptr);
  }
  WriteMmio32(GC_B1, regGRBM_GFX_CNTL, 0);

  queue->ring_gpu = ring_mc;
  queue->wptr = 0;
  queue->ring_cpu = reinterpret_cast<volatile uint32_t*>(static_cast<char*>(vram_bar_) + ring_off);
  queue->rptr_cpu = reinterpret_cast<volatile uint64_t*>(static_cast<char*>(vram_bar_) + rptr_off);
  queue->wptr_cpu = reinterpret_cast<volatile uint64_t*>(static_cast<char*>(vram_bar_) + wptr_off);
  queue->doorbell_cpu =
      reinterpret_cast<volatile uint64_t*>(static_cast<char*>(doorbell_bar_) +
                                           queue->doorbell_index * sizeof(uint32_t));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::DestroyDirectComputeQueue(const DirectComputeQueue& queue) {
  if (queue.queue_id == 0) return HSA_STATUS_SUCCESS;
  if (SkipDirectQueueDestroy()) {
    if (TraceDirectQueue()) {
      std::fprintf(stderr,
                   "ROCR macOS direct queue destroy skipped qid=%u index=%u\n",
                   queue.queue_id, queue.queue_index);
    }
    return HSA_STATUS_SUCCESS;
  }
  std::lock_guard<std::mutex> g(gpu_lock_);
  const uint32_t pipe = queue.queue_index / 4;
  const uint32_t hqd_queue = queue.queue_index % 4;
  hsa_status_t status = SelectHqdLocked(1, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) return status;
  uint32_t active = 0;
  ReadMmio32(GC_B0, regCP_HQD_ACTIVE, &active);
  if (UseDirectQueueDequeue()) {
    WriteMmio32(GC_B0, regCP_HQD_DEQUEUE_REQUEST, 1);
    for (uint32_t i = 0; i < 100; ++i) {
      ReadMmio32(GC_B0, regCP_HQD_ACTIVE, &active);
      if (active == 0) break;
      ::usleep(1000);
    }
    WriteMmio32(GC_B0, regCP_HQD_DEQUEUE_REQUEST, 0);
    status = WaitForDirectHqdIdleLocked(pipe, hqd_queue, "destroy");
    if (status != HSA_STATUS_SUCCESS) {
      WriteMmio32(GC_B1, regGRBM_GFX_CNTL, 0);
      return status;
    }
  } else {
    WriteMmio32(GC_B0, regCP_HQD_ACTIVE, 0);
    WriteMmio32(GC_B0, regCP_HQD_PQ_RPTR, 0);
    WriteMmio32(GC_B0, regCP_HQD_PQ_WPTR_LO, 0);
    WriteMmio32(GC_B0, regCP_HQD_PQ_WPTR_HI, 0);
    uint32_t doorbell_ctl = 0;
    ReadMmio32(GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_ctl);
    WriteMmio32(GC_B0, regCP_HQD_PQ_DOORBELL_CONTROL, doorbell_ctl & ~0x40000000u);
  }
  if (TraceDirectQueue()) {
    uint32_t post_active = 0;
    uint32_t rptr = 0;
    ReadMmio32(GC_B0, regCP_HQD_ACTIVE, &post_active);
    ReadMmio32(GC_B0, regCP_HQD_PQ_RPTR, &rptr);
    std::fprintf(stderr,
                 "ROCR macOS direct queue destroy qid=%u index=%u pipe=%u hqd=%u "
                 "mode=%s pre_active=0x%x post_active=0x%x rptr=0x%x\n",
                 queue.queue_id, queue.queue_index, pipe, hqd_queue,
                 UseDirectQueueDequeue() ? "dequeue" : "disable", active,
                 post_active, rptr);
  }
  WriteMmio32(GC_B1, regGRBM_GFX_CNTL, 0);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::SubmitDirectCompute(DirectComputeQueue& queue,
                                              const uint32_t* pm4,
                                              size_t dword_count) const {
  if (queue.queue_id == 0 || queue.ring_cpu == nullptr || queue.wptr_cpu == nullptr ||
      queue.doorbell_cpu == nullptr || pm4 == nullptr || dword_count == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> g(gpu_lock_);
  const uint64_t ring_dw = queue.ring_size_bytes / sizeof(uint32_t);
  const uint64_t wptr = queue.wptr;
  const uint64_t start = wptr % ring_dw;
  for (size_t i = 0; i < dword_count; ++i) {
    queue.ring_cpu[(start + i) % ring_dw] = pm4[i];
  }
  std::atomic_thread_fence(std::memory_order_release);
  const uint64_t new_wptr = wptr + dword_count;
  *queue.wptr_cpu = new_wptr;
  std::atomic_thread_fence(std::memory_order_release);
  const uint32_t pipe = queue.queue_index / 4;
  const uint32_t hqd_queue = queue.queue_index % 4;
  if (TraceDirectQueue()) {
    const size_t sample = std::min<size_t>(dword_count, 8);
    std::fprintf(stderr,
                 "ROCR macOS direct queue submit qid=%u index=%u pipe=%u hqd=%u "
                 "doorbell=0x%x wptr=%llu new_wptr=%llu dwords=%zu first_pm4=",
                 queue.queue_id, queue.queue_index, pipe, hqd_queue, queue.doorbell_index,
                 static_cast<unsigned long long>(wptr),
                 static_cast<unsigned long long>(new_wptr), dword_count);
    for (size_t i = 0; i < sample; ++i) {
      std::fprintf(stderr, "%s0x%08x", i == 0 ? "" : ",", pm4[i]);
    }
    std::fprintf(stderr, "\n");
  }
  hsa_status_t status = SelectHqdLocked(1, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) return status;
  uint32_t selected_active = 0;
  status = ReadMmio32(GC_B0, regCP_HQD_ACTIVE, &selected_active);
  if (status != HSA_STATUS_SUCCESS) {
    WriteMmio32(GC_B1, regGRBM_GFX_CNTL, 0);
    return status;
  }
  if (selected_active == 0) {
    if (TraceDirectQueue()) {
      std::fprintf(stderr,
                   "ROCR macOS direct queue submit rejected inactive HQD qid=%u index=%u "
                   "pipe=%u hqd=%u doorbell=0x%x\n",
                   queue.queue_id, queue.queue_index, pipe, hqd_queue, queue.doorbell_index);
    }
    WriteMmio32(GC_B1, regGRBM_GFX_CNTL, 0);
    return HSA_STATUS_ERROR;
  }
  status = WriteMmio32(GC_B0, regCP_HQD_PQ_WPTR_LO, static_cast<uint32_t>(new_wptr));
  if (status == HSA_STATUS_SUCCESS) {
    status = WriteMmio32(GC_B0, regCP_HQD_PQ_WPTR_HI, static_cast<uint32_t>(new_wptr >> 32));
  }
  if (TraceDirectQueue()) {
    uint32_t active = 0;
    uint32_t rptr = 0;
    uint32_t mmio_wptr = 0;
    uint32_t mmio_wptr_hi = 0;
    ReadMmio32(GC_B0, regCP_HQD_ACTIVE, &active);
    ReadMmio32(GC_B0, regCP_HQD_PQ_RPTR, &rptr);
    ReadMmio32(GC_B0, regCP_HQD_PQ_WPTR_LO, &mmio_wptr);
    ReadMmio32(GC_B0, regCP_HQD_PQ_WPTR_HI, &mmio_wptr_hi);
    std::fprintf(stderr,
                 "ROCR macOS direct queue after mmio-wptr qid=%u active=0x%x "
                 "rptr=0x%x wptr=0x%08x:%08x status=%u\n",
                 queue.queue_id, active, rptr, mmio_wptr_hi, mmio_wptr, status);
  }
  WriteMmio32(GC_B1, regGRBM_GFX_CNTL, 0);
  if (status != HSA_STATUS_SUCCESS) return status;
  *queue.doorbell_cpu = new_wptr;
  queue.wptr = new_wptr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::ReadDirectComputeRptr(const DirectComputeQueue& queue,
                                                uint32_t* rptr) const {
  if (queue.queue_id == 0 || rptr == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> g(gpu_lock_);
  const uint32_t pipe = queue.queue_index / 4;
  const uint32_t hqd_queue = queue.queue_index % 4;
  hsa_status_t status = SelectHqdLocked(1, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) return status;
  status = ReadMmio32(GC_B0, regCP_HQD_PQ_RPTR, rptr);
  WriteMmio32(GC_B1, regGRBM_GFX_CNTL, 0);
  return status;
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

hsa_status_t MacOsDriver::GetWallclockFrequency(uint32_t, uint64_t* frequency) const {
  if (frequency == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *frequency = 1000000000ull;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::AllocateScratchMemory(uint32_t, uint64_t, void**) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::AvailableMemory(uint32_t, uint64_t* available_size) const {
  if (available_size == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *available_size = info_.vram_size;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::RegisterMemory(void*, uint64_t, HsaMemFlags) const {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::DeregisterMemory(void*) const { return HSA_STATUS_SUCCESS; }

hsa_status_t MacOsDriver::MakeMemoryResident(const void* mem, size_t, uint64_t* alternate_va,
                                             const HsaMemMapFlags*,
                                             uint32_t, const uint32_t*) const {
  if (alternate_va != nullptr) {
    uint64_t gpu_addr = 0;
    *alternate_va = HostToGpuAddress(mem, &gpu_addr) == HSA_STATUS_SUCCESS ? gpu_addr : 0;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::MakeMemoryUnresident(const void*) const {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::GetQueueSaveAreaInfo(HSA_QUEUEID, void**, size_t*) const {
  return HSA_STATUS_ERROR;
}

}  // namespace AMD
}  // namespace rocr

#endif  // __APPLE__
