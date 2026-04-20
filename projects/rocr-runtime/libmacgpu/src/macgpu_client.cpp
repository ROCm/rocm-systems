/*
 * macgpu_client.cpp — IOKit client implementation for libmacgpu.
 *
 * Mirrors the Python reference implementation at
 * userspace_driver/python/amd_gpu_driver/backends/macos/iokit_client.py.
 */

#include "macgpu.h"

#include <IOKit/IOKitLib.h>
#include <mach/mach.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "rocmgpu_shared.h"

namespace {

// DEXT service name. The DEXT registers its IOUserClass name as the
// service name in the IOKit registry — matches what Info.plist declares.
constexpr const char kDextServiceName[] = "ROCmGPUDriver";

}  // namespace

struct macgpu_device {
  std::mutex       lock;
  io_connect_t     connection = IO_OBJECT_NULL;
  io_service_t     service    = IO_OBJECT_NULL;

  // Tracks active mappings so macgpu_close() can tear them down even if
  // the caller forgot.  Key is the memory-type passed to
  // IOConnectMapMemory64() (BAR index, or DMA_BASE + buffer_id).
  std::unordered_map<uint32_t, std::pair<mach_vm_address_t, mach_vm_size_t>> mappings;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static macgpu_status_t IOReturnToStatus(IOReturn r) {
  switch (r) {
    case kIOReturnSuccess:        return MACGPU_SUCCESS;
    case kIOReturnNotFound:       return MACGPU_ERROR_NOT_FOUND;
    case kIOReturnNotPermitted:   // fallthrough — access-denied family
    case kIOReturnNotPrivileged:
    case kIOReturnExclusiveAccess:
      return MACGPU_ERROR_ACCESS_DENIED;
    case kIOReturnNoMemory:
    case kIOReturnNoResources:    return MACGPU_ERROR_OUT_OF_MEMORY;
    case kIOReturnBadArgument:    return MACGPU_ERROR_INVALID_ARG;
    default:                      return MACGPU_ERROR;
  }
}

// Wrapper around IOConnectCallScalarMethod that handles the const-ness
// mismatch between the const inputs we have and the non-const pointer
// the API wants.
static IOReturn CallScalar(io_connect_t conn, uint32_t selector,
                            const uint64_t* inputs, uint32_t input_cnt,
                            uint64_t* outputs, uint32_t* output_cnt) {
  return IOConnectCallScalarMethod(
      conn, selector,
      const_cast<uint64_t*>(inputs), input_cnt,
      outputs, output_cnt);
}

// Tears down one mapping without touching the `mappings` map (caller
// owns the map iteration).
static void UnmapOne(io_connect_t conn, uint32_t memory_type,
                     mach_vm_address_t addr) {
  if (conn == IO_OBJECT_NULL) return;
  (void)IOConnectUnmapMemory64(conn, memory_type, mach_task_self(), addr);
}

// ---------------------------------------------------------------------------
// Status strings
// ---------------------------------------------------------------------------

extern "C" const char* macgpu_status_string(macgpu_status_t status) {
  switch (status) {
    case MACGPU_SUCCESS:             return "MACGPU_SUCCESS";
    case MACGPU_ERROR:               return "MACGPU_ERROR";
    case MACGPU_ERROR_NOT_FOUND:     return "MACGPU_ERROR_NOT_FOUND";
    case MACGPU_ERROR_ACCESS_DENIED: return "MACGPU_ERROR_ACCESS_DENIED";
    case MACGPU_ERROR_INVALID_ARG:   return "MACGPU_ERROR_INVALID_ARG";
    case MACGPU_ERROR_OUT_OF_MEMORY: return "MACGPU_ERROR_OUT_OF_MEMORY";
    default:                         return "MACGPU_ERROR_UNKNOWN";
  }
}

// ---------------------------------------------------------------------------
// open / close
// ---------------------------------------------------------------------------

extern "C" macgpu_status_t macgpu_open(macgpu_device_t** out_dev) {
  if (!out_dev) return MACGPU_ERROR_INVALID_ARG;
  *out_dev = nullptr;

  // IOServiceNameMatching matches on the registry-entry name — which
  // for a DEXT is the IOUserClass value (set in Info.plist). Using
  // IOServiceMatching("IOUserService") would instead match every
  // DEXT in the system.
  CFMutableDictionaryRef matching = IOServiceNameMatching(kDextServiceName);
  if (!matching) return MACGPU_ERROR_NOT_FOUND;

  // IOServiceGetMatchingService consumes one reference on `matching`.
  io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault, matching);
  if (service == IO_OBJECT_NULL) {
    return MACGPU_ERROR_NOT_FOUND;
  }

  io_connect_t conn = IO_OBJECT_NULL;
  IOReturn r = IOServiceOpen(service, mach_task_self(), 0, &conn);
  if (r != kIOReturnSuccess) {
    IOObjectRelease(service);
    return IOReturnToStatus(r);
  }

  auto* dev = new macgpu_device();
  dev->service = service;
  dev->connection = conn;
  *out_dev = dev;
  return MACGPU_SUCCESS;
}

extern "C" void macgpu_close(macgpu_device_t* dev) {
  if (!dev) return;

  std::lock_guard<std::mutex> guard(dev->lock);

  // Tear down lingering mappings first so the DEXT teardown path
  // doesn't fight with still-mapped BAR windows.
  for (auto& kv : dev->mappings) {
    UnmapOne(dev->connection, kv.first, kv.second.first);
  }
  dev->mappings.clear();

  if (dev->connection != IO_OBJECT_NULL) {
    IOServiceClose(dev->connection);
    dev->connection = IO_OBJECT_NULL;
  }
  if (dev->service != IO_OBJECT_NULL) {
    IOObjectRelease(dev->service);
    dev->service = IO_OBJECT_NULL;
  }

  // We deliberately leave the mutex locked for the delete — any thread
  // that races a close() against another call is buggy, and this way
  // we at least catch the double-close case (mutex destruction on a
  // locked mutex aborts under libc++).
  delete dev;
}

// ---------------------------------------------------------------------------
// Device info
// ---------------------------------------------------------------------------

extern "C" macgpu_status_t macgpu_get_info(macgpu_device_t* dev,
                                           macgpu_device_info_t* out_info) {
  if (!dev || !out_info) return MACGPU_ERROR_INVALID_ARG;

  std::lock_guard<std::mutex> guard(dev->lock);
  if (dev->connection == IO_OBJECT_NULL) return MACGPU_ERROR;

  ROCmGPUDeviceInfo raw{};
  size_t out_size = sizeof(raw);
  IOReturn r = IOConnectCallStructMethod(
      dev->connection, kROCmGPU_GetInfo,
      /*inputStruct=*/nullptr, /*inputStructCnt=*/0,
      &raw, &out_size);
  if (r != kIOReturnSuccess) return IOReturnToStatus(r);

  std::memset(out_info, 0, sizeof(*out_info));
  out_info->vendor_id           = raw.vendorID;
  out_info->device_id           = raw.deviceID;
  out_info->subsystem_vendor_id = raw.subsystemVendorID;
  out_info->subsystem_device_id = raw.subsystemDeviceID;
  out_info->revision_id         = raw.revisionID;
  out_info->vram_size           = raw.vramSize;
  for (int i = 0; i < 6; ++i) {
    out_info->bars[i].size         = raw.bars[i].size;
    out_info->bars[i].memory_index = raw.bars[i].memoryIndex;
    out_info->bars[i].type         = raw.bars[i].type;
    out_info->bars[i].is_64bit     = raw.bars[i].is64bit;
    out_info->bars[i].prefetchable = raw.bars[i].prefetchable;
  }
  return MACGPU_SUCCESS;
}

// ---------------------------------------------------------------------------
// MMIO
// ---------------------------------------------------------------------------

extern "C" macgpu_status_t macgpu_mmio_read32(macgpu_device_t* dev,
                                              uint32_t bar_index,
                                              uint64_t byte_offset,
                                              uint32_t* out_value) {
  if (!dev || !out_value) return MACGPU_ERROR_INVALID_ARG;

  std::lock_guard<std::mutex> guard(dev->lock);
  if (dev->connection == IO_OBJECT_NULL) return MACGPU_ERROR;

  uint64_t inputs[2]  = {bar_index, byte_offset};
  uint64_t outputs[1] = {0};
  uint32_t out_count  = 1;
  IOReturn r = CallScalar(dev->connection, kROCmGPU_MMIORead32,
                          inputs, 2, outputs, &out_count);
  if (r != kIOReturnSuccess) return IOReturnToStatus(r);
  *out_value = static_cast<uint32_t>(outputs[0]);
  return MACGPU_SUCCESS;
}

extern "C" macgpu_status_t macgpu_mmio_write32(macgpu_device_t* dev,
                                               uint32_t bar_index,
                                               uint64_t byte_offset,
                                               uint32_t value) {
  if (!dev) return MACGPU_ERROR_INVALID_ARG;

  std::lock_guard<std::mutex> guard(dev->lock);
  if (dev->connection == IO_OBJECT_NULL) return MACGPU_ERROR;

  uint64_t inputs[3] = {bar_index, byte_offset, value};
  IOReturn r = CallScalar(dev->connection, kROCmGPU_MMIOWrite32,
                          inputs, 3, nullptr, nullptr);
  return IOReturnToStatus(r);
}

// ---------------------------------------------------------------------------
// BAR mapping
// ---------------------------------------------------------------------------

extern "C" macgpu_status_t macgpu_map_bar(macgpu_device_t* dev,
                                          uint32_t bar_index,
                                          void** out_addr,
                                          uint64_t* out_size) {
  if (!dev || !out_addr || !out_size) return MACGPU_ERROR_INVALID_ARG;
  if (bar_index > 5)                   return MACGPU_ERROR_INVALID_ARG;

  std::lock_guard<std::mutex> guard(dev->lock);
  if (dev->connection == IO_OBJECT_NULL) return MACGPU_ERROR;

  // If this BAR is already mapped, reuse the existing mapping rather
  // than asking the DEXT to create a second one (which ld64 on some
  // Darwin builds handles by returning the same window, but on others
  // allocates a parallel mapping — better to be explicit).
  auto it = dev->mappings.find(bar_index);
  if (it != dev->mappings.end()) {
    *out_addr = reinterpret_cast<void*>(it->second.first);
    *out_size = it->second.second;
    return MACGPU_SUCCESS;
  }

  // Ask the DEXT to validate the BAR and return its size. (This is the
  // mKROCmGPU_MapBAR escape; the actual mmap happens via
  // IOConnectMapMemory64 below.)
  uint64_t inputs[1]  = {bar_index};
  uint64_t outputs[1] = {0};
  uint32_t out_count  = 1;
  IOReturn r = CallScalar(dev->connection, kROCmGPU_MapBAR,
                          inputs, 1, outputs, &out_count);
  if (r != kIOReturnSuccess) return IOReturnToStatus(r);

  mach_vm_address_t addr = 0;
  mach_vm_size_t    size = 0;
  r = IOConnectMapMemory64(dev->connection, bar_index, mach_task_self(),
                           &addr, &size, kIOMapAnywhere);
  if (r != kIOReturnSuccess) {
    // Best-effort cleanup of the MapBAR reservation — the DEXT's
    // UnmapBAR path is idempotent, so calling it here won't break
    // anything if the actual reservation never happened.
    uint64_t u_in[1] = {bar_index};
    (void)CallScalar(dev->connection, kROCmGPU_UnmapBAR, u_in, 1, nullptr, nullptr);
    return IOReturnToStatus(r);
  }

  dev->mappings[bar_index] = {addr, size};
  *out_addr = reinterpret_cast<void*>(addr);
  *out_size = size;
  return MACGPU_SUCCESS;
}

extern "C" macgpu_status_t macgpu_unmap_bar(macgpu_device_t* dev,
                                            uint32_t bar_index) {
  if (!dev) return MACGPU_ERROR_INVALID_ARG;
  if (bar_index > 5) return MACGPU_ERROR_INVALID_ARG;

  std::lock_guard<std::mutex> guard(dev->lock);
  if (dev->connection == IO_OBJECT_NULL) return MACGPU_ERROR;

  auto it = dev->mappings.find(bar_index);
  if (it == dev->mappings.end()) return MACGPU_SUCCESS;

  UnmapOne(dev->connection, bar_index, it->second.first);
  dev->mappings.erase(it);

  uint64_t inputs[1] = {bar_index};
  IOReturn r = CallScalar(dev->connection, kROCmGPU_UnmapBAR,
                          inputs, 1, nullptr, nullptr);
  return IOReturnToStatus(r);
}

// ---------------------------------------------------------------------------
// DMA buffers
// ---------------------------------------------------------------------------

extern "C" macgpu_status_t macgpu_alloc_dma(macgpu_device_t* dev,
                                            uint64_t size,
                                            uint32_t flags,
                                            macgpu_dma_buffer_t* out_buf) {
  if (!dev || !out_buf) return MACGPU_ERROR_INVALID_ARG;
  if (size == 0)        return MACGPU_ERROR_INVALID_ARG;

  std::lock_guard<std::mutex> guard(dev->lock);
  if (dev->connection == IO_OBJECT_NULL) return MACGPU_ERROR;

  // AllocDMA is the one selector whose dispatch-table entry mixes
  // scalar inputs and a struct output, so we need the universal
  // IOConnectCallMethod (not IOConnectCallStructMethod or
  // IOConnectCallScalarMethod alone).
  uint64_t scalar_in[2] = {size, flags};
  ROCmGPUDMAInfo raw{};
  size_t out_size = sizeof(raw);
  IOReturn r = IOConnectCallMethod(
      dev->connection, kROCmGPU_AllocDMA,
      scalar_in, 2,             // scalar in
      nullptr, 0,               // struct in
      nullptr, nullptr,         // scalar out
      &raw, &out_size);         // struct out
  if (r != kIOReturnSuccess) return IOReturnToStatus(r);

  const uint32_t memory_type = kROCmGPU_MemType_DMABase + raw.bufferID;
  mach_vm_address_t cpu_addr = 0;
  mach_vm_size_t    mapped_size = 0;
  r = IOConnectMapMemory64(dev->connection, memory_type, mach_task_self(),
                           &cpu_addr, &mapped_size, kIOMapAnywhere);
  if (r != kIOReturnSuccess) {
    // Release the DEXT-side allocation so it doesn't leak.
    uint64_t free_in[1] = {raw.bufferID};
    (void)CallScalar(dev->connection, kROCmGPU_FreeDMA, free_in, 1, nullptr, nullptr);
    return IOReturnToStatus(r);
  }

  dev->mappings[memory_type] = {cpu_addr, mapped_size};

  std::memset(out_buf, 0, sizeof(*out_buf));
  out_buf->buffer_id = raw.bufferID;
  out_buf->size      = raw.size;
  out_buf->cpu_addr  = reinterpret_cast<void*>(cpu_addr);

  const uint32_t n = (raw.segmentCount > 64) ? 64 : raw.segmentCount;
  for (uint32_t i = 0; i < n; ++i) {
    out_buf->segments[i].address = raw.segments[i].address;
    out_buf->segments[i].length  = raw.segments[i].length;
  }
  out_buf->segment_count = n;
  return MACGPU_SUCCESS;
}

extern "C" macgpu_status_t macgpu_free_dma(macgpu_device_t* dev, uint64_t buffer_id) {
  if (!dev) return MACGPU_ERROR_INVALID_ARG;
  if (buffer_id == 0) return MACGPU_SUCCESS;

  std::lock_guard<std::mutex> guard(dev->lock);
  if (dev->connection == IO_OBJECT_NULL) return MACGPU_ERROR;

  const uint32_t memory_type = kROCmGPU_MemType_DMABase + buffer_id;
  auto it = dev->mappings.find(memory_type);
  if (it != dev->mappings.end()) {
    UnmapOne(dev->connection, memory_type, it->second.first);
    dev->mappings.erase(it);
  }

  uint64_t inputs[1] = {buffer_id};
  IOReturn r = CallScalar(dev->connection, kROCmGPU_FreeDMA,
                          inputs, 1, nullptr, nullptr);
  return IOReturnToStatus(r);
}
