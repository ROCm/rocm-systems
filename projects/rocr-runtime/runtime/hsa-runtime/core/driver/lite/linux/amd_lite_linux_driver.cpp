////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#if defined(__linux__)

#include "core/inc/amd_lite_linux_driver.h"

#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "core/inc/amd_memory_region.h"
#include "core/inc/memory_region.h"

namespace rocr {
namespace AMD {
namespace {

constexpr uint32_t kDefaultCuCount = 64;
constexpr uint32_t kDefaultSimdPerCu = 2;
constexpr uint32_t kDefaultWavefrontSize = 32;
constexpr uint32_t kDefaultMaxWavesPerSimd = 16;
constexpr uint64_t kFallbackLocalMemSize = 256ull * 1024 * 1024;

uint64_t AlignUpU64(uint64_t value, uint64_t align) {
  return (value + align - 1) & ~(align - 1);
}

bool EnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool TraceHostAllocs() { return EnvEnabled("ROCR_AMDGPU_LITE_TRACE_ALLOC"); }
bool TraceGpuAllocs() { return EnvEnabled("ROCR_AMDGPU_LITE_TRACE_GPU_ALLOC"); }
bool TraceDirectQueue() { return EnvEnabled("ROCR_AMDGPU_LITE_TRACE_DIRECT_QUEUE"); }
bool TraceDirectQueueVerbose() {
  return EnvEnabled("ROCR_AMDGPU_LITE_TRACE_DIRECT_QUEUE_VERBOSE");
}
bool UseDirectQueueDequeue() {
  if (EnvEnabled("ROCR_AMDGPU_LITE_DIRECT_QUEUE_DISABLE_DEQUEUE")) return false;
  const char* value = std::getenv("ROCR_AMDGPU_LITE_DIRECT_QUEUE_DEQUEUE");
  if (value != nullptr && value[0] != '\0') return std::strcmp(value, "0") != 0;
  return true;
}
bool SkipDirectQueueDestroy() {
  return EnvEnabled("ROCR_AMDGPU_LITE_DIRECT_QUEUE_SKIP_DESTROY");
}

uint32_t DirectQueueDequeueSettleUs() {
  const char* value = std::getenv("ROCR_AMDGPU_LITE_DIRECT_QUEUE_DEQUEUE_SETTLE_US");
  if (value == nullptr || value[0] == '\0') return 100000;
  char* end = nullptr;
  errno = 0;
  unsigned long parsed = std::strtoul(value, &end, 0);
  if (errno != 0 || end == value || parsed == 0 || parsed > 5000000ul) return 100000;
  return static_cast<uint32_t>(parsed);
}

lite::DirectQueueOptions LinuxDirectQueueOptions() {
  lite::DirectQueueOptions options;
  options.force_reclaim = EnvEnabled("ROCR_AMDGPU_LITE_FORCE_DIRECT_COMPUTE");
  options.use_firmware_dequeue = UseDirectQueueDequeue();
  options.skip_destroy = SkipDirectQueueDestroy();
  options.trace = TraceDirectQueue();
  options.trace_verbose = TraceDirectQueueVerbose();
  options.dequeue_settle_us = DirectQueueDequeueSettleUs();
  options.trace_prefix = "ROCR amdgpu_lite direct queue";
  return options;
}

struct HostAllocRegistry {
  std::mutex m;
  std::unordered_map<void*, size_t> allocations;
};

HostAllocRegistry& GetHostAllocRegistry() {
  static HostAllocRegistry* reg = new HostAllocRegistry();
  return *reg;
}

}  // namespace

LinuxAmdgpuLiteDriver::LinuxAmdgpuLiteDriver(std::string devnode_name)
    : core::Driver(core::DriverType::LINUX_AMDGPU_LITE, devnode_name),
      transport_(std::move(devnode_name)) {}

hsa_status_t LinuxAmdgpuLiteDriver::DiscoverDriver(
    std::unique_ptr<core::Driver>& driver) {
  if (EnvEnabled("ROCR_AMDGPU_LITE_DISABLE")) return HSA_STATUS_ERROR;

  const char* devnode = std::getenv("ROCR_AMDGPU_LITE_DEVNODE");
  auto tmp = std::make_unique<LinuxAmdgpuLiteDriver>(
      devnode != nullptr && devnode[0] != '\0' ? devnode : "/dev/amdgpu_lite0");
  hsa_status_t status = tmp->Open();
  if (status != HSA_STATUS_SUCCESS) return status;
  status = tmp->QueryKernelModeDriver(core::DriverQuery::GET_DRIVER_VERSION);
  if (status != HSA_STATUS_SUCCESS) {
    tmp->Close();
    return status;
  }
  driver = std::move(tmp);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteDriver::Init() { return HSA_STATUS_SUCCESS; }
hsa_status_t LinuxAmdgpuLiteDriver::ShutDown() { return Close(); }

hsa_status_t LinuxAmdgpuLiteDriver::QueryKernelModeDriver(core::DriverQuery query) {
  if (query != core::DriverQuery::GET_DRIVER_VERSION) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (!transport_.is_open()) return HSA_STATUS_ERROR;
  version_.KernelInterfaceMajorVersion = 1;
  version_.KernelInterfaceMinorVersion = 0;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteDriver::Open() { return transport_.Open(); }

hsa_status_t LinuxAmdgpuLiteDriver::Close() {
  transport_.Close();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteDriver::GetSystemProperties(
    HsaSystemProperties& sys_props) const {
  sys_props.NumNodes = transport_.is_open() ? 1 : 0;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteDriver::GetNodeProperties(
    HsaNodeProperties& node_props, uint32_t node_id) const {
  if (node_id != 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::memset(&node_props, 0, sizeof(node_props));

  long ncpu = ::sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpu < 1) ncpu = 1;
  node_props.NumCPUCores = static_cast<HSAuint32>(ncpu);
  node_props.NumFComputeCores = kDefaultCuCount * kDefaultSimdPerCu;
  node_props.NumMemoryBanks = 1;
  node_props.NumCaches = 0;
  node_props.NumIOLinks = 0;
  node_props.VendorId = transport_.vendor_id();
  node_props.DeviceId = transport_.device_id();
  node_props.LocalMemSize = transport_.vram_size() != 0 ? transport_.vram_size()
                                                       : kFallbackLocalMemSize;
  node_props.EngineId.ui32.Major = 12;
  node_props.EngineId.ui32.Minor = 0;
  node_props.EngineId.ui32.Stepping = 1;
  node_props.WaveFrontSize = kDefaultWavefrontSize;
  node_props.NumSIMDPerCU = kDefaultSimdPerCu;
  node_props.NumCUPerArray = 8;
  node_props.NumArrays = 2;
  node_props.NumShaderBanks = 4;
  node_props.MaxWavesPerSIMD = kDefaultMaxWavesPerSimd;
  node_props.LDSSizeInKB = 64;
  node_props.MaxEngineClockMhzFCompute = 2500;
  node_props.NumSdmaEngines = 1;
  node_props.NumSdmaQueuesPerEngine = 1;
  node_props.NumCpQueues = 1;
  node_props.NumXcc = 1;
  node_props.Capability.ui32.QueueSizePowerOfTwo = 1;
  node_props.Capability.ui32.QueueSize32bit = 1;
  node_props.Capability.ui32.SVMAPISupported = 1;
  std::snprintf(reinterpret_cast<char*>(node_props.AMDName),
                HSA_PUBLIC_NAME_SIZE, "gfx1201");
  const char name[] = "AMD Radeon RX 9000 (amdgpu_lite)";
  for (size_t i = 0; i < sizeof(name) && i < HSA_PUBLIC_NAME_SIZE; ++i) {
    node_props.MarketingName[i] = static_cast<HSAuint16>(name[i]);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteDriver::GetEdgeProperties(
    std::vector<HsaIoLinkProperties>& io_link_props, uint32_t) const {
  io_link_props.clear();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteDriver::GetMemoryProperties(
    uint32_t node_id, std::vector<HsaMemoryProperties>& mem_props) const {
  if (node_id != 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (mem_props.size() < 1) mem_props.resize(1);

  const long pages = ::sysconf(_SC_PHYS_PAGES);
  const long page_size = ::sysconf(_SC_PAGESIZE);
  uint64_t mem_bytes = 0;
  if (pages > 0 && page_size > 0) {
    mem_bytes = static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
  }

  std::memset(&mem_props[0], 0, sizeof(HsaMemoryProperties));
  mem_props[0].HeapType = HSA_HEAPTYPE_SYSTEM;
  mem_props[0].SizeInBytes = mem_bytes;
  mem_props[0].VirtualBaseAddress = 0;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteDriver::GetCacheProperties(
    uint32_t, uint32_t, std::vector<HsaCacheProperties>& cache_props) const {
  cache_props.clear();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteDriver::AllocateMemory(
    const core::MemoryRegion& mem_region,
    core::MemoryRegion::AllocateFlags alloc_flags,
    void** mem, size_t size, uint32_t) {
  if (mem == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *mem = nullptr;

  const auto& amd_region = static_cast<const AMD::MemoryRegion&>(mem_region);
  if (amd_region.IsLocalMemory()) {
    uint64_t gpu_addr = 0;
    hsa_status_t status = AllocateVram(size, 4096, mem, &gpu_addr);
    if (status == HSA_STATUS_SUCCESS && TraceGpuAllocs()) {
      std::fprintf(stderr, "ROCR amdgpu_lite vram alloc %p gpu=0x%llx size=%zu flags=0x%x\n",
                   *mem, static_cast<unsigned long long>(gpu_addr), size, alloc_flags);
    }
    return status;
  }

  const size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
  const size_t page = page_size > 0 ? page_size : 4096;
  const size_t rounded = (size + page - 1) & ~(page - 1);
  void* ptr = ::mmap(nullptr, rounded, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  {
    auto& reg = GetHostAllocRegistry();
    std::lock_guard<std::mutex> g(reg.m);
    reg.allocations[ptr] = rounded;
  }
  if (TraceHostAllocs()) {
    std::fprintf(stderr, "ROCR amdgpu_lite alloc %p size=%zu rounded=%zu\n",
                 ptr, size, rounded);
  }
  *mem = ptr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteDriver::FreeMemory(void* mem, size_t) {
  if (mem == nullptr) return HSA_STATUS_SUCCESS;
  {
    std::lock_guard<std::mutex> g(gpu_lock_);
    auto it = vram_allocations_.find(mem);
    if (it != vram_allocations_.end()) {
      lite::LinuxLiteBuffer buffer = it->second.buffer;
      vram_allocations_.erase(it);
      return transport_.FreeVram(&buffer);
    }
  }

  auto& reg = GetHostAllocRegistry();
  size_t rounded = 0;
  {
    std::lock_guard<std::mutex> g(reg.m);
    auto it = reg.allocations.find(mem);
    if (it == reg.allocations.end()) return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    rounded = it->second;
    reg.allocations.erase(it);
  }
  return ::munmap(mem, rounded) == 0 ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

hsa_status_t LinuxAmdgpuLiteDriver::AllocateVram(size_t size, size_t align,
                                                 void** cpu_addr,
                                                 uint64_t* gpu_addr) {
  if (cpu_addr == nullptr || gpu_addr == nullptr || size == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  *cpu_addr = nullptr;
  *gpu_addr = 0;
  align = std::max<size_t>(align, 4096);
  if ((align & (align - 1)) != 0 || align > 4096) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  lite::LinuxLiteBuffer buffer;
  hsa_status_t status = transport_.AllocVram(AlignUpU64(size, align), &buffer);
  if (status != HSA_STATUS_SUCCESS) return status;

  std::lock_guard<std::mutex> g(gpu_lock_);
  VramAllocation alloc;
  alloc.buffer = buffer;
  vram_allocations_[buffer.cpu] = std::move(alloc);
  *cpu_addr = buffer.cpu;
  *gpu_addr = buffer.gpu_addr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LinuxAmdgpuLiteDriver::HostToGpuAddress(const void* ptr,
                                                     uint64_t* gpu_addr) const {
  if (ptr == nullptr || gpu_addr == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *gpu_addr = 0;
  std::lock_guard<std::mutex> g(gpu_lock_);
  const auto p = reinterpret_cast<uintptr_t>(ptr);
  for (const auto& kv : vram_allocations_) {
    const auto base = reinterpret_cast<uintptr_t>(kv.first);
    const auto& buffer = kv.second.buffer;
    if (p < base || p >= base + buffer.size) continue;
    *gpu_addr = buffer.gpu_addr + (p - base);
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ALLOCATION;
}

void LinuxAmdgpuLiteDriver::RegisterVramShadow(const void* cpu_addr,
                                               size_t size, const void* src) {
  if (cpu_addr == nullptr || src == nullptr || size == 0) return;
  std::lock_guard<std::mutex> g(gpu_lock_);
  const auto p = reinterpret_cast<uintptr_t>(cpu_addr);
  for (auto& kv : vram_allocations_) {
    const auto base = reinterpret_cast<uintptr_t>(kv.first);
    auto& alloc = kv.second;
    if (p < base || p + size > base + alloc.buffer.size) continue;
    if (alloc.shadow.empty()) alloc.shadow.resize(static_cast<size_t>(alloc.buffer.size));
    std::memcpy(alloc.shadow.data() + (p - base), src, size);
    return;
  }
}

hsa_status_t LinuxAmdgpuLiteDriver::VramShadowAddress(
    const void* cpu_addr, size_t size, const void** shadow_addr) const {
  if (cpu_addr == nullptr || shadow_addr == nullptr || size == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  *shadow_addr = nullptr;
  std::lock_guard<std::mutex> g(gpu_lock_);
  const auto p = reinterpret_cast<uintptr_t>(cpu_addr);
  for (const auto& kv : vram_allocations_) {
    const auto base = reinterpret_cast<uintptr_t>(kv.first);
    const auto& alloc = kv.second;
    if (alloc.shadow.empty() || p < base || p + size > base + alloc.buffer.size) continue;
    *shadow_addr = alloc.shadow.data() + (p - base);
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ALLOCATION;
}

hsa_status_t LinuxAmdgpuLiteDriver::CreateDirectComputeQueue(
    DirectComputeQueue* queue) {
  if (queue == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> g(gpu_lock_);
  auto slot = std::find(direct_queue_in_use_.begin(), direct_queue_in_use_.end(), false);
  if (slot == direct_queue_in_use_.end()) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  const uint32_t queue_index =
      static_cast<uint32_t>(std::distance(direct_queue_in_use_.begin(), slot));
  direct_queue_in_use_[queue_index] = true;
  hsa_status_t status = lite::CreateDirectQueue(
      transport_, queue, queue_index, transport_.framebuffer_base(),
      LinuxDirectQueueOptions());
  if (status != HSA_STATUS_SUCCESS) {
    direct_queue_in_use_[queue_index] = false;
    *queue = {};
  }
  return status;
}

hsa_status_t LinuxAmdgpuLiteDriver::DestroyDirectComputeQueue(
    const DirectComputeQueue& queue) {
  std::lock_guard<std::mutex> g(gpu_lock_);
  hsa_status_t status = lite::DestroyDirectQueue(transport_, queue,
                                                 LinuxDirectQueueOptions());
  if (status == HSA_STATUS_SUCCESS &&
      queue.queue_index < direct_queue_in_use_.size()) {
    direct_queue_in_use_[queue.queue_index] = false;
  }
  return status;
}

hsa_status_t LinuxAmdgpuLiteDriver::SubmitDirectCompute(
    DirectComputeQueue& queue, const uint32_t* pm4, size_t dword_count) const {
  std::lock_guard<std::mutex> g(gpu_lock_);
  return lite::SubmitDirectQueue(transport_, queue, pm4, dword_count,
                                 LinuxDirectQueueOptions());
}

hsa_status_t LinuxAmdgpuLiteDriver::ReadDirectComputeRptr(
    const DirectComputeQueue& queue, uint32_t* rptr) const {
  std::lock_guard<std::mutex> g(gpu_lock_);
  return lite::ReadDirectQueueRptr(transport_, queue, rptr);
}

hsa_status_t LinuxAmdgpuLiteDriver::CreateQueue(uint32_t, HSA_QUEUE_TYPE, uint32_t,
                                                HSA::hsa_amd_queue_priority_internal_t,
                                                uint32_t, void*, uint64_t,
                                                uint64_t,
                                                HsaEvent*, HsaQueueResource&) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t LinuxAmdgpuLiteDriver::DestroyQueue(HSA_QUEUEID) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t LinuxAmdgpuLiteDriver::UpdateQueue(
    HSA_QUEUEID, uint32_t, HSA::hsa_amd_queue_priority_internal_t,
    void*, uint64_t, HsaEvent*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t LinuxAmdgpuLiteDriver::SetQueueCUMask(HSA_QUEUEID, uint32_t,
                                                   uint32_t*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t LinuxAmdgpuLiteDriver::AllocQueueGWS(HSA_QUEUEID, uint32_t,
                                                  uint32_t*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t LinuxAmdgpuLiteDriver::ExportDMABuf(void*, size_t, int*, size_t*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t LinuxAmdgpuLiteDriver::ImportDMABuf(int, const core::Agent&,
                                                 core::ShareableHandle*, void*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t LinuxAmdgpuLiteDriver::DestroyImportedShareableHandle(
    core::ShareableHandle*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t LinuxAmdgpuLiteDriver::Map(core::ShareableHandle, void*, size_t,
                                        size_t, hsa_access_permission_t) {
  return HSA_STATUS_ERROR;
}

hsa_status_t LinuxAmdgpuLiteDriver::Unmap(core::ShareableHandle, void*, size_t,
                                          size_t) {
  return HSA_STATUS_ERROR;
}

hsa_status_t LinuxAmdgpuLiteDriver::CreateShareableHandle(
    void*, void*, size_t, const core::Agent&, core::ShareableHandle*,
    uint64_t*, int*, uint64_t*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t LinuxAmdgpuLiteDriver::DestroyShareableHandle(
    core::ShareableHandle*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t LinuxAmdgpuLiteDriver::SPMAcquire(uint32_t) const {
  return HSA_STATUS_ERROR;
}
hsa_status_t LinuxAmdgpuLiteDriver::SPMRelease(uint32_t) const {
  return HSA_STATUS_ERROR;
}
hsa_status_t LinuxAmdgpuLiteDriver::SPMSetDestBuffer(
    uint32_t, uint32_t, uint32_t*, uint32_t*, void*, bool*) const {
  return HSA_STATUS_ERROR;
}
hsa_status_t LinuxAmdgpuLiteDriver::SetTrapHandler(uint32_t, const void*,
                                                   uint64_t, const void*,
                                                   uint64_t) const {
  return HSA_STATUS_ERROR;
}
hsa_status_t LinuxAmdgpuLiteDriver::GetDeviceHandle(uint32_t,
                                                    void**) const {
  return HSA_STATUS_ERROR;
}
hsa_status_t LinuxAmdgpuLiteDriver::GetClockCounters(uint32_t,
                                                     HsaClockCounters*) const {
  return HSA_STATUS_ERROR;
}
hsa_status_t LinuxAmdgpuLiteDriver::GetTileConfig(uint32_t,
                                                  HsaGpuTileConfig*) const {
  return HSA_STATUS_ERROR;
}
hsa_status_t LinuxAmdgpuLiteDriver::IsModelEnabled(bool* enable) const {
  if (enable) *enable = false;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t LinuxAmdgpuLiteDriver::GetWallclockFrequency(
    uint32_t, uint64_t* frequency) const {
  if (frequency == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *frequency = 1000000000ull;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t LinuxAmdgpuLiteDriver::AllocateScratchMemory(uint32_t, uint64_t,
                                                          void**) const {
  return HSA_STATUS_ERROR;
}
hsa_status_t LinuxAmdgpuLiteDriver::AvailableMemory(
    uint32_t, uint64_t* available_size) const {
  if (available_size == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *available_size = transport_.vram_size();
  return HSA_STATUS_SUCCESS;
}
hsa_status_t LinuxAmdgpuLiteDriver::RegisterMemory(void*, uint64_t,
                                                   HsaMemFlags) const {
  return HSA_STATUS_SUCCESS;
}
hsa_status_t LinuxAmdgpuLiteDriver::DeregisterMemory(void*) const {
  return HSA_STATUS_SUCCESS;
}
hsa_status_t LinuxAmdgpuLiteDriver::MakeMemoryResident(
    const void* mem, size_t, uint64_t* alternate_va, const HsaMemMapFlags*,
    uint32_t, const uint32_t*) const {
  if (alternate_va != nullptr) {
    uint64_t gpu_addr = 0;
    *alternate_va = HostToGpuAddress(mem, &gpu_addr) == HSA_STATUS_SUCCESS ? gpu_addr : 0;
  }
  return HSA_STATUS_SUCCESS;
}
hsa_status_t LinuxAmdgpuLiteDriver::MakeMemoryUnresident(const void*) const {
  return HSA_STATUS_SUCCESS;
}
hsa_status_t LinuxAmdgpuLiteDriver::GetQueueSaveAreaInfo(HSA_QUEUEID, void**,
                                                         size_t*) const {
  return HSA_STATUS_ERROR;
}

}  // namespace AMD
}  // namespace rocr

#endif  // defined(__linux__)
