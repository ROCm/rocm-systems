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
#include <mutex>
#include <unordered_map>
#include <utility>

#include "core/inc/amd_lite_direct_queue.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/memory_region.h"

namespace rocr {
namespace AMD {
namespace {

constexpr uint32_t kDoorbellBar = 2;
constexpr uint32_t kMmioBar = 5;
constexpr uint32_t kVramBar = 0;

constexpr uint32_t GC_B0 = 0x1260;
constexpr uint32_t NBIO_B2 = 0xD20;

constexpr uint32_t regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN = 0x00c0;
constexpr uint32_t regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL = 0x01cb;
constexpr uint32_t regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL = 0x01ce;

constexpr uint32_t regCP_MEC_DOORBELL_RANGE_LOWER = 0x1dfc;
constexpr uint32_t regCP_MEC_DOORBELL_RANGE_UPPER = 0x1dfd;

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

lite::DirectQueueOptions MacDirectQueueOptions() {
  lite::DirectQueueOptions options;
  options.force_reclaim = std::getenv("AMD_GPU_MACOS_FORCE_DIRECT_COMPUTE") != nullptr;
  options.use_firmware_dequeue = UseDirectQueueDequeue();
  options.skip_destroy = SkipDirectQueueDestroy();
  options.trace = TraceDirectQueue();
  options.trace_verbose = TraceDirectQueueVerbose();
  options.dequeue_settle_us = DirectQueueDequeueSettleUs();
  options.trace_prefix = "ROCR macOS direct queue";
  // M2: route compute dispatch through MES (lite:: MapLegacyQueueWithMes) instead
  // of the hand-rolled direct HQD. Env-gated, default off (direct HQD remains the
  // proven fallback). When on, MES owns HQD activation -> clears the ~13-submit
  // multi-dispatch ceiling.
  options.use_mes_queue = std::getenv("ROCR_MACOS_USE_MES_QUEUE") != nullptr;
  return options;
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

bool MacOsDriver::IsRegisteredVramPointer(const void* ptr) const {
  if (ptr == nullptr) return false;
  std::lock_guard<std::mutex> g(gpu_lock_);
  const auto p = reinterpret_cast<uintptr_t>(ptr);
  for (const auto& kv : vram_allocations_) {
    const auto base = reinterpret_cast<uintptr_t>(kv.first);
    const auto& alloc = kv.second;
    if (p >= base && p < base + alloc.size) return true;
  }
  return false;
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

hsa_status_t MacOsDriver::EnsureDoorbellAperture() const {
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

hsa_status_t MacOsDriver::WriteGpuMemory32(uint64_t offset, uint32_t value) const {
  if (vram_bar_ == nullptr || offset + sizeof(uint32_t) > vram_bar_size_) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }
  auto* ptr = reinterpret_cast<volatile uint32_t*>(static_cast<char*>(vram_bar_) + offset);
  *ptr = value;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::ZeroGpuMemory(uint64_t offset, uint64_t size) const {
  for (uint64_t i = 0; i < size; i += sizeof(uint32_t)) {
    hsa_status_t status = WriteGpuMemory32(offset + i, 0);
    if (status != HSA_STATUS_SUCCESS) return status;
  }
  return HSA_STATUS_SUCCESS;
}

void* MacOsDriver::GpuMemoryCpuPointer(uint64_t offset) const {
  if (vram_bar_ == nullptr || offset >= vram_bar_size_) return nullptr;
  return static_cast<char*>(vram_bar_) + offset;
}

volatile uint64_t* MacOsDriver::DoorbellCpuPointer(uint32_t doorbell_index) const {
  const uint64_t byte_offset = static_cast<uint64_t>(doorbell_index) * sizeof(uint32_t);
  if (doorbell_bar_ == nullptr || byte_offset + sizeof(uint64_t) > doorbell_bar_size_) {
    return nullptr;
  }
  return reinterpret_cast<volatile uint64_t*>(static_cast<char*>(doorbell_bar_) + byte_offset);
}

void MacOsDriver::SleepUs(uint32_t usec) const {
  ::usleep(usec);
}

hsa_status_t MacOsDriver::CreateDirectComputeQueue(DirectComputeQueue* queue) {
  if (queue == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> g(gpu_lock_);
  hsa_status_t status = EnsureBarMappingsLocked();
  if (status != HSA_STATUS_SUCCESS) return status;
  if (next_direct_queue_index_ >= 8) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  const uint32_t queue_index = next_direct_queue_index_++;
  status = lite::CreateDirectQueue(*this, queue, queue_index, framebuffer_base_,
                                   MacDirectQueueOptions());
  if (status != HSA_STATUS_SUCCESS) {
    --next_direct_queue_index_;
    *queue = {};
  }
  return status;
}

hsa_status_t MacOsDriver::DestroyDirectComputeQueue(DirectComputeQueue& queue) {
  std::lock_guard<std::mutex> g(gpu_lock_);
  return lite::DestroyDirectQueue(*this, queue, MacDirectQueueOptions());
}

hsa_status_t MacOsDriver::SubmitDirectCompute(DirectComputeQueue& queue,
                                              const uint32_t* pm4,
                                              size_t dword_count) const {
  std::lock_guard<std::mutex> g(gpu_lock_);
  return lite::SubmitDirectQueue(*this, queue, pm4, dword_count,
                                 MacDirectQueueOptions());
}

hsa_status_t MacOsDriver::ReadDirectComputeRptr(const DirectComputeQueue& queue,
                                                uint32_t* rptr) const {
  std::lock_guard<std::mutex> g(gpu_lock_);
  return lite::ReadDirectQueueRptr(*this, queue, rptr);
}

hsa_status_t MacOsDriver::CreateQueue(uint32_t, HSA_QUEUE_TYPE, uint32_t,
                                      HSA::hsa_amd_queue_priority_internal_t,
                                      uint32_t, void*, uint64_t, uint64_t, HsaEvent*,
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
