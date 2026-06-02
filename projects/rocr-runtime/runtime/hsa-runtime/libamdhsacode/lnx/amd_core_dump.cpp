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

#include <unistd.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <libgen.h>
#include <limits.h>
#include <elf.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>
#include <ctime>
#include <vector>
#include <sstream>
#include <fstream>
#include <memory>
#include "core/util/utils.h"
#include "core/inc/runtime.h"
#include "core/inc/agent.h"
#include "./amd_hsa_code_util.hpp"
#include "core/inc/amd_core_dump.hpp"
#include "hsakmt/hsakmt.h"
#include "hsakmt/linux/kfd_ioctl.h"
#include "core/inc/amd_gpu_agent.h"
#include "core/inc/amd_aql_queue.h"

constexpr char SNAPSHOT_INFO_ALIGNMENT = 0x8;
constexpr uint32_t LOAD_ALIGNMENT_SHIFT = 4;
constexpr uint32_t NOTE_ALIGNMENT_SHIFT = 2;
const std::string PREFIX_FILE_NAME = "gpucore";
constexpr size_t MAX_BUFFER_SIZE = 4 * 1024 * 1024;

namespace rocr {
namespace amd {
namespace coredump {

namespace {
[[nodiscard]] std::string custom_core_dump() {
  return core::Runtime::runtime_singleton_->flag().core_dump_pattern();
}
}

/* Implementation details */
namespace impl {

// Optional: Detect if running in a container
namespace {
[[nodiscard]] bool is_running_in_container() {
  std::ifstream cgroup("/proc/1/cgroup");
  if (!cgroup.is_open()) return false;

  std::string line;
  while (std::getline(cgroup, line)) {
    if (line.find("docker") != std::string::npos ||
        line.find("lxc") != std::string::npos ||
        line.find("kubepods") != std::string::npos) {
      return true;
    }
  }
  return false;
}
} // anonymous namespace

// Read kernel core pattern from /proc/sys/kernel/core_pattern
static std::string read_kernel_core_pattern() {
  std::ifstream pattern_file("/proc/sys/kernel/core_pattern");
  if (!pattern_file.is_open()) {
    return "";
  }

  std::string pattern;
  std::getline(pattern_file, pattern);
  return pattern;
}

// Substitute format specifiers in core pattern
namespace {
std::string substitute_core_pattern(const std::string& pattern) {
  std::string result;
  pid_t pid = getpid();
  // Use gettid() if available (glibc >= 2.30), otherwise fallback to syscall
#if defined(__GLIBC__) && \
       (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 30))
    pid_t tid = gettid();
#else
  pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
#endif
  time_t now = time(nullptr);
  // Get hostname
  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) != 0) {
    strncpy(hostname.data(), "unknown", hostname.size() - 1);
  }
  hostname[hostname.size() - 1] = '\0';
  // Get executable name
  char exe_path[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
  std::string exe_name;
  if (len > 0) {
    exe_path[len] = '\0';
    char* base = basename(exe_path);
    exe_name = base ? std::string(base) : "unknown";
  } else {
    exe_name = "unknown";
  }
  // Parse pattern character by character
  for (size_t i = 0; i < pattern.length(); i++) {
    if (pattern[i] == '%' && i + 1 < pattern.length()) {
      switch (pattern[i + 1]) {
        case '%':
          result += '%';
          break;
        case 'p':
          result += std::to_string(pid);
          break;
        case 'i':
          result += std::to_string(tid);
          break;
        case 'h':
          result += hostname.data();
          break;
        case 'e':
          result += exe_name;
          break;
        case 't':
          result += std::to_string(now);
          break;
        // Unsupported specifiers are dropped (including %<NUL>)
        default:
          break;
      }
      i++;  // Skip next character
    } else {
      result += pattern[i];
    }
  }
  return result;
}
}  // anonymous namespace

namespace {
[[nodiscard]] bool validate_dump_path(const std::string& filepath) {
  // Reject pipe patterns
  if (!filepath.empty() && filepath[0] == '|') {
    fprintf(stderr, "GPU coredump: Pipe patterns not supported\n");
    return false;
  }
  // Extract directory path
  std::string dir;
  size_t last_slash = filepath.find_last_of('/');
  if (last_slash != std::string::npos) {
    dir = filepath.substr(0, last_slash);
  } else {
    dir = ".";
  }
  // Check if directory exists and is writable
  if (access(dir.c_str(), W_OK) != 0) {
    fprintf(stderr, "GPU coredump: Directory %s not writable or does not exist\n", dir.c_str());
    return false;
  }
  return true;
}
} // anonymous namespace

// Parse command line for pipe handler
namespace {
[[nodiscard]] std::vector<std::string> parse_command_line(const std::string& cmd) {
  std::vector<std::string> args;
  std::string current;
  bool in_quotes = false;
  bool escaped = false;
  for (char c : cmd) {
    if (escaped) {
      current += c;
      escaped = false;
    } else if (c == '\\') {
      escaped = true;
    } else if (c == '"') {
      in_quotes = !in_quotes;
    } else if (c == ' ' && !in_quotes) {
      if (!current.empty()) {
        args.push_back(current);
        current.clear();
      }
    } else {
      current += c;
    }
  }
  if (!current.empty()) {
    args.push_back(current);
  }
  return args;
}
}  // anonymous namespace
class PackageBuilder {
 public:
  PackageBuilder() : st_(std::stringstream::out | std::stringstream::binary) {}
  size_t Size() const { return st_.str().size(); }
  template <typename T, typename = typename std::enable_if<!std::is_pointer<T>::value>::type>
  void Write(const T& v) {
    st_.write((char*)&v, sizeof(T));
  }
  void Write(const std::vector<uint8_t>& v) { st_.write((const char*)v.data(), v.size()); }
  void Write(void* data, uint32_t size) { st_.write((const char*)data, size); }
  bool GetBuffer(void* out) {
    size_t sz = Size();

    if (!sz) return false;
    std::memcpy(out, st_.str().c_str(), sz);
    return true;
  }
  void Print(void* buf, uint64_t size) {
    int i;
    for (i = 0; i < size; i++) debug_print("%02x ", 0xFF & ((uint8_t*)buf)[i]);
    debug_print("\n");
  }
 private:
  std::stringstream st_;
};

// Track memory regions that should be included in lightweight coredumps (Scratch + CWSR)
class MemoryRegionFilter {
public:
  struct AddressRange {
    uint64_t start;
    uint64_t end;
    
    bool contains(uint64_t addr, uint64_t size) const {
      uint64_t addr_end = addr + size;
        // Check if the segment overlaps with this range
        return !(addr_end <= start || addr >= end);
    }
  };

  bool address_in_map(const std::map<uint64_t, AddressRange>& ranges,
                     uint64_t addr, uint64_t size) const {
    if (ranges.empty()) return false;
    auto it = ranges.upper_bound(addr);
    if (it != ranges.begin()) {
      it--;
      if (it->second.contains(addr, size)) {
        return true;
      }
    }     
    return false;
  }

  // Helper to check if address range overlaps with any scratch or CWSR memory ranges
  bool should_include(const uint64_t& addr, const uint64_t& size) {
    return address_in_map(scratch_ranges, addr, size) ||
           address_in_map(code_object_ranges, addr, size);
  }

  void add_scratch_range(uint64_t start, size_t size) {
    if (size > 0) {
      scratch_ranges.emplace(start, AddressRange{start, start + size});
    }
  }

  void add_code_object_range(uint64_t start, size_t size) {
    if (size > 0) {
      code_object_ranges.emplace(start, AddressRange{start, start + size});
    }
  }

private:
  std::map<uint64_t, AddressRange> scratch_ranges;
  std::map<uint64_t, AddressRange> code_object_ranges;
}; 

// Build list of memory regions to be included in the lightweight coredump
static hsa_status_t build_lightweight_coredump_ranges(MemoryRegionFilter& filter) {
  // Get all the GPU agents from runtime
  const auto& gpu_agents = core::Runtime::runtime_singleton_->gpu_agents();

  for(const core::Agent* agent : gpu_agents) {
    const AMD::GpuAgent* gpu_agent = static_cast<const AMD::GpuAgent*>(agent);

    // Add scratch memory range for this agent by getting the 
    // size and base_address from agent's scratch_pool_
    void* scratch_base = nullptr;
    size_t scratch_size = 0;
    gpu_agent->GetScratchAperture(&scratch_base, &scratch_size);

    if (scratch_base && scratch_size > 0) {
      // Filter will hold all of the reserved address range, even if unused 
      filter.add_scratch_range(reinterpret_cast<uint64_t>(scratch_base), scratch_size);
      debug_print("Added scratch range: 0x%lx - 0x%lx (size: %zu)\n",
                  reinterpret_cast<uint64_t>(scratch_base),
                  reinterpret_cast<uint64_t>(scratch_base) + scratch_size,
                  scratch_size);
    }
  }

  // Add code object allocations from allocation_map_
  core::Runtime::runtime_singleton_->IterateCodeObjectAllocations(
    [&filter](uint64_t start, size_t size) {
      filter.add_code_object_range(start, size);
      debug_print("Added code object range: 0x%lx - 0x%lx (size: %zu)\n",
                    start, start + size, size);
  });
  return HSA_STATUS_SUCCESS;
}

enum SegmentType { LOAD, NOTE };
struct SegmentBuilder;

struct SegmentInfo {
  SegmentType stype;
  uint64_t vaddr = 0;
  uint64_t size = 0;
  uint32_t flags = 0;
  SegmentBuilder* builder;
};

using SegmentsInfo = std::vector<SegmentInfo>;
using rocr::amd::hsa::alignUp;

// Helper function to get device info from GpuAgent without debug mode
static void GetCoreDeviceInfo(const AMD::GpuAgent* agent, kfd_dbg_device_info_entry& entry) {
  // Call thunk API to get device info from topology/FMM/libdrm
  HSAuint32 gpu_id = agent->properties().KFDGpuID;
  HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtGetCoreDeviceInfo(gpu_id, &entry));

  fprintf(stderr, "[GetCoreDeviceInfo] GPU %u: hsaKmtGetCoreDeviceInfo returned status=%d\n", gpu_id, status);
  fprintf(stderr, "[GetCoreDeviceInfo] GPU %u: BEFORE override: simd_count=%u, max_waves_per_simd=%u, array_count=%u, num_xcc=%u\n",
          gpu_id, entry.simd_count, entry.max_waves_per_simd, entry.array_count, entry.num_xcc);


  // BUG WORKAROUND: hsaKmtGetCoreDeviceInfo returns zeros for many fields
  // Manually populate from GpuAgent properties instead
  const HsaNodeProperties& props = agent->properties();

  entry.gpu_id = gpu_id;
  entry.vendor_id = props.VendorId;
  entry.device_id = props.DeviceId;
  entry.array_count = props.NumArrays * props.NumShaderBanks;

  // Calculate gfx version from EngineId - decimal encoding for KFD 1.13 compatibility
  entry.gfx_target_version = (props.EngineId.ui32.Major * 10000
                               + props.EngineId.ui32.Minor * 100
                               + props.EngineId.ui32.Stepping);

  fprintf(stderr, "[GetCoreDeviceInfo] GPU %u: vendor_id=0x%04x, device_id=0x%04x, gfx_version=%u\n",
          gpu_id, entry.vendor_id, entry.device_id, entry.gfx_target_version);
  fprintf(stderr, "[GetCoreDeviceInfo] GPU %u: AFTER override: simd_count=%u, max_waves_per_simd=%u, array_count=%u, num_xcc=%u\n",
          gpu_id, entry.simd_count, entry.max_waves_per_simd, entry.array_count, entry.num_xcc);

  // Exception status - TODO: Get from agent exception tracking when implemented
  entry.exception_status = 0;  // Placeholder until agent exception tracking is implemented
}

// Minimal struct queue layout from libhsakmt/src/queues.c
// Used to access cached CWSR info without ioctl that hangs on faulted queues
struct queue_cwsr_info {
  uint32_t queue_id;              // Offset 0
  uint64_t wptr;                  // Offset 4 (with padding)
  uint64_t rptr;                  // Offset 12
  void *eop_buffer;               // Offset 20
  void *ctx_save_restore;         // Offset 24/28 (CWSR address - cached at queue creation)
  uint32_t ctx_save_restore_size; // Offset 32/36 (allocated size - never changes)
  // ... rest of struct not needed
};

// Helper function to get queue snapshot from AqlQueue without debug mode
static void GetCoreQueueInfo(AMD::AqlQueue* queue, kfd_queue_snapshot_entry& entry) {
  // Zero out the structure first
  memset(&entry, 0, sizeof(entry));

  // Runtime direct fields (7 fields) - zero cost access from queue object
  entry.ring_base_address = (uint64_t)queue->amd_queue_.hsa_queue.base_address;
  entry.write_pointer_address = (uint64_t)&queue->amd_queue_.write_dispatch_id;
  entry.read_pointer_address = (uint64_t)&queue->amd_queue_.read_dispatch_id;
  entry.queue_id = (uint32_t)queue->aql_queue_id();
  entry.gpu_id = static_cast<const AMD::GpuAgent*>(queue->GetAgent())->properties().KFDGpuID;
  entry.ring_size = queue->amd_queue_.hsa_queue.size;
  entry.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;

  // Runtime cached (1 field) - TODO: Get exception status when implemented
  // entry.exception_status = queue->GetExceptionStatus();
  entry.exception_status = 0;  // Placeholder until exception caching is implemented

  // CWSR fields (2 fields) - Read directly from cached struct queue data
  // This avoids calling hsaKmtGetQueueInfo which uses AMDKFD_IOC_GET_QUEUE_WAVE_STATE
  // ioctl that hangs when GPU hardware is in faulted state
  fprintf(stderr, "[Core Dump] Queue %u: Accessing cached CWSR info from struct queue\n", entry.queue_id);

  struct queue_cwsr_info* q = (struct queue_cwsr_info*)(uintptr_t)queue->aql_queue_id();
  entry.ctx_save_restore_address = (uint64_t)q->ctx_save_restore;
  entry.ctx_save_restore_area_size = q->ctx_save_restore_size;

  fprintf(stderr, "[Core Dump] Queue %u: CWSR address=0x%lx, allocated_size=%u (from cache, no ioctl)\n",
          entry.queue_id,
          (unsigned long)entry.ctx_save_restore_address,
          entry.ctx_save_restore_area_size);

  // Reserved field
  entry.reserved = 0;
}

// DEBUG: Print device snapshot for comparison
static void PrintDeviceSnapshot(const char* label, const kfd_dbg_device_info_entry* entry) {
  fprintf(stderr, "\n=== %s DEVICE SNAPSHOT ===\n", label);
  fprintf(stderr, "  gpu_id:                   0x%08x\n", entry->gpu_id);
  fprintf(stderr, "  location_id:              0x%08x\n", entry->location_id);
  fprintf(stderr, "  vendor_id:                0x%04x\n", entry->vendor_id);
  fprintf(stderr, "  device_id:                0x%04x\n", entry->device_id);
  fprintf(stderr, "  revision_id:              0x%02x\n", entry->revision_id);
  fprintf(stderr, "  subsystem_vendor_id:      0x%04x\n", entry->subsystem_vendor_id);
  fprintf(stderr, "  subsystem_device_id:      0x%04x\n", entry->subsystem_device_id);
  fprintf(stderr, "  gfx_target_version:       0x%08x\n", entry->gfx_target_version);
  fprintf(stderr, "  fw_version:               0x%08x\n", entry->fw_version);
  fprintf(stderr, "  simd_count:               %u\n", entry->simd_count);
  fprintf(stderr, "  max_waves_per_simd:       %u\n", entry->max_waves_per_simd);
  fprintf(stderr, "  array_count:              %u\n", entry->array_count);
  fprintf(stderr, "  simd_arrays_per_engine:   %u\n", entry->simd_arrays_per_engine);
  fprintf(stderr, "  num_xcc:                  %u\n", entry->num_xcc);
  fprintf(stderr, "  capability:               0x%08x\n", entry->capability);
  fprintf(stderr, "  debug_prop:               0x%08x\n", entry->debug_prop);
  fprintf(stderr, "  lds_base:                 0x%016llx\n", (unsigned long long)entry->lds_base);
  fprintf(stderr, "  lds_limit:                0x%016llx\n", (unsigned long long)entry->lds_limit);
  fprintf(stderr, "  scratch_base:             0x%016llx\n", (unsigned long long)entry->scratch_base);
  fprintf(stderr, "  scratch_limit:            0x%016llx\n", (unsigned long long)entry->scratch_limit);
  fprintf(stderr, "  gpuvm_base:               0x%016llx\n", (unsigned long long)entry->gpuvm_base);
  fprintf(stderr, "  gpuvm_limit:              0x%016llx\n", (unsigned long long)entry->gpuvm_limit);
  fprintf(stderr, "  exception_status:         0x%016llx (IGNORED)\n", (unsigned long long)entry->exception_status);
}

// DEBUG: Print queue snapshot for comparison
static void PrintQueueSnapshot(const char* label, const kfd_queue_snapshot_entry* entry) {
  fprintf(stderr, "\n=== %s QUEUE SNAPSHOT ===\n", label);
  fprintf(stderr, "  queue_id:                     0x%08x\n", entry->queue_id);
  fprintf(stderr, "  gpu_id:                       0x%08x\n", entry->gpu_id);
  fprintf(stderr, "  ring_base_address:            0x%016llx\n", (unsigned long long)entry->ring_base_address);
  fprintf(stderr, "  write_pointer_address:        0x%016llx\n", (unsigned long long)entry->write_pointer_address);
  fprintf(stderr, "  read_pointer_address:         0x%016llx\n", (unsigned long long)entry->read_pointer_address);
  fprintf(stderr, "  ring_size:                    %u\n", entry->ring_size);
  fprintf(stderr, "  queue_type:                   %u\n", entry->queue_type);
  fprintf(stderr, "  ctx_save_restore_address:     0x%016llx\n", (unsigned long long)entry->ctx_save_restore_address);
  fprintf(stderr, "  ctx_save_restore_area_size:   %u\n", entry->ctx_save_restore_area_size);
  fprintf(stderr, "  exception_status:             0x%016llx (IGNORED)\n", (unsigned long long)entry->exception_status);
  fprintf(stderr, "  reserved:                     0x%08x\n", entry->reserved);
}

struct SegmentBuilder {
  virtual ~SegmentBuilder() = default;
  /* Find which segments needs to be created.  */
  virtual hsa_status_t Collect(SegmentsInfo& segments) = 0;
  /* Called to read a given SegmentInfo's data.  */
  virtual hsa_status_t Read(void* buf, size_t buf_size, off_t offset) = 0;
};

struct NoteSegmentBuilder : public SegmentBuilder {
  hsa_status_t Collect(SegmentsInfo& segments) override {
    // Hardcode KFD version to 1.13 for note format compatibility
    HsaVersionInfo versionInfo = {1, 18};

    // ========================================================================
    // PHASE 2: Runtime-Only Approach (Debug API Disabled for Stability)
    // ========================================================================
    // Note: Debug API comparison was causing double-free crashes.
    // Using runtime helpers only - they work with or without debugger attached.
    fprintf(stderr, "[Core Dump] Using runtime helpers only (debug API disabled)\n");

    // ========================================================================
    // ALWAYS Collect Runtime-Based Data (Works With or Without Debugger)
    // ========================================================================

    // 1. Get runtime info (cached from runtime_enable)
    kfd_runtime_info runtime_info;
    memset(&runtime_info, 0, sizeof(runtime_info));

    if (HSAKMT_CALL(hsaKmtGetCoreRuntimeInfo(&runtime_info)) != HSAKMT_STATUS_SUCCESS) {
      fprintf(stderr, "Failed to get runtime info via hsaKmtGetCoreRuntimeInfo()\n");
      return HSA_STATUS_ERROR;
    }

    // 2. Build device snapshots from GPU agents
    std::vector<kfd_dbg_device_info_entry> device_snapshots;
    const auto& gpu_agents = core::Runtime::runtime_singleton_->gpu_agents();

    for (const core::Agent* agent : gpu_agents) {
      const AMD::GpuAgent* gpu_agent = static_cast<const AMD::GpuAgent*>(agent);
      kfd_dbg_device_info_entry entry;
      GetCoreDeviceInfo(gpu_agent, entry);
      device_snapshots.push_back(entry);
    }

    // 3. Collect all queues and suspend them (via queue percentage)
    fprintf(stderr, "\n[Core Dump] ========== PHASE 3: Queue Collection and Suspension ==========\n");
    std::vector<AMD::AqlQueue*> all_queues;
    for (const core::Agent* agent : gpu_agents) {
      const AMD::GpuAgent* gpu_agent = static_cast<const AMD::GpuAgent*>(agent);
      const auto& aql_queues = gpu_agent->GetAqlQueues();
      fprintf(stderr, "[Core Dump] GPU %u: Found %zu queue(s)\n",
              gpu_agent->properties().KFDGpuID, aql_queues.size());

      for (core::Queue* q : aql_queues) {
        AMD::AqlQueue* aql_queue = static_cast<AMD::AqlQueue*>(q);
        all_queues.push_back(aql_queue);
        uint32_t qid = (uint32_t)aql_queue->aql_queue_id();
        fprintf(stderr, "[Core Dump] Queue %u: Suspending via Suspend() method...\n", qid);
        // Suspend queue using Suspend() method which sets suspended_ flag properly
        aql_queue->Suspend();
        fprintf(stderr, "[Core Dump] Queue %u: Suspension complete\n", qid);
      }
    }
    fprintf(stderr, "[Core Dump] Total queues collected: %zu\n", all_queues.size());


    // 4. Build queue snapshots (reading cached data, no ioctls)
    fprintf(stderr, "\n[Core Dump] ========== PHASE 4: Building Queue Snapshots ==========\n");
    std::vector<kfd_queue_snapshot_entry> queue_snapshots;
    for (size_t i = 0; i < all_queues.size(); i++) {
      AMD::AqlQueue* queue = all_queues[i];
      fprintf(stderr, "[Core Dump] Processing queue %zu/%zu (queue_id=%u)...\n",
              i+1, all_queues.size(), (uint32_t)queue->aql_queue_id());

      kfd_queue_snapshot_entry entry;
      GetCoreQueueInfo(queue, entry);
      queue_snapshots.push_back(entry);

      fprintf(stderr, "[Core Dump] Queue %u snapshot complete\n", entry.queue_id);
    }
    fprintf(stderr, "[Core Dump] All queue snapshots built successfully\n");

    // 5. Try queue resumption 
    fprintf(stderr, "\n[Core Dump] ========== PHASE 5: Queue Resumption TEST ==========\n");
    fprintf(stderr, "[Core Dump]  resuming queues\n");

    for (size_t i = 0; i < all_queues.size(); i++) {
      AMD::AqlQueue* queue = all_queues[i];
      fprintf(stderr, "[Core Dump] Resuming queue %zu/%zu (queue_id=%u)...\n",
              i+1, all_queues.size(), (uint32_t)queue->aql_queue_id());

      queue->Resume();

      fprintf(stderr, "[Core Dump] Queue %u resume complete\n", (uint32_t)queue->aql_queue_id());
    }

    // ========================================================================
    // Package Runtime Data into PT_NOTE (Works in Both Cases)
    // ========================================================================

    /* Note version */
    note_package_builder_.Write<uint64_t>(1);
    /* Store version_major in PT_NOTE package */
    note_package_builder_.Write<uint32_t>(versionInfo.KernelInterfaceMajorVersion);
    /* Store version_minor in PT_NOTE package */
    note_package_builder_.Write<uint32_t>(versionInfo.KernelInterfaceMinorVersion);
    /* Store runtime_info_size in PT_NOTE package */
    static_assert(16 <= sizeof(kfd_runtime_info));
    fprintf(stderr, "[Core Dump] runtime_info: declared_size=%u, actual_sizeof=%zu\n",
            16, sizeof(kfd_runtime_info));
    note_package_builder_.Write<uint64_t>(16 /* sizeof(kfd_runtime_info) for version 1.18 */);

    /* Store n_agents in PT_NOTE package */
    note_package_builder_.Write<uint32_t>(device_snapshots.size());
    /* Store agent_info_entry_size in PT_NOTE package */
    static_assert(120 <= sizeof(kfd_dbg_device_info_entry));
    fprintf(stderr, "[Core Dump] device_info: declared_size=%u, actual_sizeof=%zu, n_agents=%zu\n",
            120, sizeof(kfd_dbg_device_info_entry), device_snapshots.size());
    note_package_builder_.Write<uint32_t>(120 /* sizeof(kfd_dbg_device_info_entry) at version 1.18 */);

    /* Store n_queues in PT_NOTE package */
    note_package_builder_.Write<uint32_t>(queue_snapshots.size());
    /* Store queue_info_entry_size in PT_NOTE package */
    static_assert(64 <= sizeof(kfd_queue_snapshot_entry));
    fprintf(stderr, "[Core Dump] queue_info: declared_size=%u, actual_sizeof=%zu, n_queues=%zu\n",
            64, sizeof(kfd_queue_snapshot_entry), queue_snapshots.size());
    note_package_builder_.Write<uint32_t>(64 /* sizeof(kfd_queue_snapshot_entry) at version 1.18 */);

    // Push runtime info
    fprintf(stderr, "[Core Dump] Pushing runtime_info: %zu bytes\n", 16);
    PushInfo(&runtime_info, 16);

    // Push device snapshots
    if (!device_snapshots.empty()) {
      size_t total_bytes = device_snapshots.size() * sizeof(kfd_dbg_device_info_entry);
      fprintf(stderr, "[Core Dump] Pushing device_snapshots: %zu bytes (%zu agents * %zu)\n",
              total_bytes, device_snapshots.size(), sizeof(kfd_dbg_device_info_entry));
      fprintf(stderr, "[Core Dump] Agent[0] before push: simd_count=%u, max_waves_per_simd=%u\n",
              device_snapshots[0].simd_count, device_snapshots[0].max_waves_per_simd);
      PushInfo(device_snapshots.data(), total_bytes);
    }

    // Push queue snapshots
    if (!queue_snapshots.empty()) {
      size_t total_bytes = queue_snapshots.size() * sizeof(kfd_queue_snapshot_entry);
      fprintf(stderr, "[Core Dump] Pushing queue_snapshots: %zu bytes (%zu queues * %zu)\n",
              total_bytes, queue_snapshots.size(), sizeof(kfd_queue_snapshot_entry));
      PushInfo(queue_snapshots.data(), total_bytes);
    }

    fprintf(stderr, "\n[Core Dump] ========== PHASE 6: PT_NOTE Packaging ==========\n");
    fprintf(stderr, "[Core Dump] All data pushed to note_package_builder successfully\n");

    /* With note content, package this in the PT_NOTE.  */
    PackageBuilder noteHeaderBuilder;
    noteHeaderBuilder.Write<uint32_t> (7);  /* namesz */
    noteHeaderBuilder.Write<uint32_t> (note_package_builder_.Size());
    noteHeaderBuilder.Write<uint32_t> (NT_AMDGPU_CORE_STATE);  /* type.  */
    noteHeaderBuilder.Write<char[8]> ("AMDGPU\0");

    raw_.resize(noteHeaderBuilder.Size() + note_package_builder_.Size());
    if (!(noteHeaderBuilder.GetBuffer(raw_.data())
          && note_package_builder_.GetBuffer(&raw_[noteHeaderBuilder.Size()]))) {
      fprintf(stderr, "Failed to build the NT_AMDGPU_CORE_STATE note.\n");
      return HSA_STATUS_ERROR;
    }

    SegmentInfo s;
    s.stype = NOTE;
    s.vaddr = 0;
    s.size = raw_.size();
    s.flags = 0;
    s.builder = this;
    segments.push_back(s);

    fprintf(stderr, "[Core Dump] ========== NoteSegmentBuilder::Collect() COMPLETE ==========\n");
    fprintf(stderr, "[Core Dump] Returning HSA_STATUS_SUCCESS - no errors detected\n");

    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t Read(void* buf, size_t buf_size, off_t offset) override {
    if (offset + buf_size >raw_.size ()) return HSA_STATUS_ERROR;
    memcpy(buf, raw_.data() + offset, buf_size);
    return HSA_STATUS_SUCCESS;
  }

 private:
  PackageBuilder note_package_builder_;
  std::vector<unsigned char> raw_;

  void PushInfo(void *data, uint32_t size) {
    note_package_builder_.Write(data, size);
    size = alignUp(size, SNAPSHOT_INFO_ALIGNMENT) - size;
    for (int i = 0; i < size; i++)
      note_package_builder_.Write<uint8_t>(0);
  }
};

struct LoadSegmentBuilder : public SegmentBuilder {
  LoadSegmentBuilder() : fd_(open("/proc/self/mem", O_RDONLY)) {}

  ~LoadSegmentBuilder() {
    if (fd_ != -1) close(fd_);
  }
  hsa_status_t Collect(SegmentsInfo& segments) override {
    const bool lightweight_dump = core::Runtime::runtime_singleton_->flag().lightweight_core_dump_enable();

    impl::MemoryRegionFilter filter;
    if (lightweight_dump) {
      // build memory region filter to only include Scratch + Code object allocations
      hsa_status_t status = impl::build_lightweight_coredump_ranges(filter);
      if (status != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to build lightweight core dump filter\n");
        return status;
      }
    }
    
    const std::string maps_path = "/proc/self/maps";
    std::ifstream maps(maps_path);
    if (!maps.is_open()) {
      fprintf(stderr, "Could not open '%s'", maps_path.c_str());
      return HSA_STATUS_ERROR;
    }

    std::string line;
    while (std::getline(maps, line)) {
      std::istringstream isl{ line };
      std::string address, perms, offset, dev, inode, path;
      if (!(isl >> address >> perms >> offset >> dev >> inode)) {
        fprintf(stderr, "Failed to parse '%s'", maps_path.c_str());
        return HSA_STATUS_ERROR;
      }

      std::getline(isl >> std::ws, path);

      uint64_t start, end;
      if (sscanf(address.c_str(), "%lx-%lx", &start, &end) != 2) {
        fprintf(stderr, "Failed to parse '%s'", maps_path.c_str());
        return HSA_STATUS_ERROR;
      }

      uint64_t size = end - start;

      bool is_gpu_mapping = (path.rfind("/dev/dri/renderD", 0) == 0);

      // Skip non-GPU path mappings
      if (!is_gpu_mapping) continue;

      // For lightweight dumps, keep only scratch allocations
      if (lightweight_dump && !filter.should_include(start, size)) {
        debug_print("Skipping load 0x%lx size: %ld (Not Scratch or Code Object)\n", start, size);
        continue;
      }

      // Add all gpu mappings to the full coredump
      uint32_t flags = SHF_ALLOC;
      flags |= (perms.find('w', 0) != std::string::npos) ? SHF_WRITE : 0;
      flags |= (perms.find('x', 0) != std::string::npos) ? SHF_EXECINSTR : 0;

      debug_print("LOAD 0x%lx size: %ld\n", start, size);
      SegmentInfo s;
      s.stype = LOAD;
      s.vaddr = start;
      s.size = size;
      s.flags = flags;
      s.builder = this;
      segments.push_back(s);
    }
    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t Read(void* buf, size_t buf_size, off_t offset) override {
    if (fd_ == -1) return HSA_STATUS_ERROR;

    size_t done = 0;
    size_t read;
    do {
      read = pread(fd_, static_cast<char *>(buf) + done, buf_size - done,
                   offset + done);
      if (read == -1 && errno != EINTR) {
        perror("Failed to read GPU memory");
        return HSA_STATUS_ERROR;
      }
      else if (read > 0)
        done += read;
    } while (read != 0 && done < buf_size);

    if (read == 0 && done < buf_size) {
      fprintf(stderr, "Reached unexpected EOF while reading VRAM.\n");
      return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
  }

 private:
  int fd_ = -1;
};

// Write core dump to a file descriptor (for pipe handler)
namespace {
// Use size_limit of -1 for no limit (e.g, for pipes)
hsa_status_t write_core_dump_to_fd(int fd, const SegmentsInfo& segments,
                                          size_t size_limit, bool show_progress) {
  if (!segments.size()) return HSA_STATUS_SUCCESS;
  auto copy_buffer = std::make_unique<unsigned char[]>(MAX_BUFFER_SIZE);
  SegmentInfo front = segments.front();
  off_t offset = sizeof(Elf64_Ehdr) + segments.size() * sizeof(Elf64_Phdr);

  if (size_limit != -1 && (offset + front.size > size_limit)) {
    debug_print("Core file size over limit\n");
    return HSA_STATUS_SUCCESS;
  }

  // Use posix_fallocate for regular files
  struct stat fd_stat;
  bool is_reg_file = false;
  if (fstat(fd, &fd_stat) == 0 && S_ISREG(fd_stat.st_mode)) {
    is_reg_file = true;
  }

  // Write ELF header
  Elf64_Ehdr ehdr{};
  ehdr.e_ident[EI_MAG0] = ELFMAG0;
  ehdr.e_ident[EI_MAG1] = ELFMAG1;
  ehdr.e_ident[EI_MAG2] = ELFMAG2;
  ehdr.e_ident[EI_MAG3] = ELFMAG3;
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
  ehdr.e_ident[EI_VERSION] = EV_CURRENT;
  ehdr.e_ident[EI_OSABI] = ELF::ELFOSABI_AMDGPU_HSA;
  ehdr.e_ident[EI_ABIVERSION] = 0;
  ehdr.e_type = ET_CORE;
  ehdr.e_machine = ELF::EM_AMDGPU;
  ehdr.e_version = EV_CURRENT;
  ehdr.e_entry = 0;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = 0;
  ehdr.e_flags = 0;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = segments.size();
  ehdr.e_shentsize = 0;
  ehdr.e_shnum = 0;
  ehdr.e_shstrndx = 0;

  if (write(fd, &ehdr, sizeof(ehdr)) != sizeof(ehdr)) {
    perror("Failed to write ELF header to pipe");
    return HSA_STATUS_ERROR;
  }

  if (is_reg_file) {
    int error = posix_fallocate(fd, sizeof(Elf64_Ehdr), segments.size() * sizeof(Elf64_Phdr));
    if (error != 0) {
      fprintf(stderr, "Failed to allocate file: %s\n", strerror(error));
      return HSA_STATUS_ERROR;
    }
  }

  // Write program headers
  std::vector<Elf64_Phdr> phdrs;
  phdrs.reserve(segments.size());
  for (const SegmentInfo& seg : segments) {
    Elf64_Phdr phdr{};
    phdr.p_type = [](SegmentType s) {
      switch (s) {
        case LOAD:
          return PT_LOAD;
        case NOTE:
          return PT_NOTE;
        default:
          assert(false);
          return PT_NULL;
      }
    }(seg.stype);
    phdr.p_flags = seg.flags;
    phdr.p_vaddr = seg.vaddr;
    phdr.p_paddr = 0;
    phdr.p_memsz = seg.size;
    phdr.p_filesz = seg.size;
    phdr.p_align = [](SegmentType s) {
      switch (s) {
        case LOAD:
          return LOAD_ALIGNMENT_SHIFT;
        case NOTE:
          return NOTE_ALIGNMENT_SHIFT;
        default:
          assert(false);
          return (uint32_t)0;
      }
    } (seg.stype);
    phdr.p_offset = alignUp(offset, (uint64_t)1 << phdr.p_align);
    phdrs.push_back(phdr);
    offset += phdr.p_filesz;
  }

  // Write all program headers
  if (is_reg_file) {
    // For regular files, use pwrite to write at specific offset
    for (size_t i = 0; i < phdrs.size(); i++) {
      off_t phdr_offset = sizeof(Elf64_Ehdr) + i * sizeof(Elf64_Phdr);
      if (pwrite(fd, &phdrs[i], sizeof(Elf64_Phdr), phdr_offset) != sizeof(Elf64_Phdr)) {
        perror("Failed to write program header");
        return HSA_STATUS_ERROR;
      }
    }
  } else {
    // For pipes, use sequential write
    for (const auto& phdr : phdrs) {
      if (write(fd, &phdr, sizeof(phdr)) != sizeof(phdr)) {
        perror("Failed to write program header to pipe");
        return HSA_STATUS_ERROR;
      }
    }
  }

  // Write segment data
  for (size_t idx = 0; idx < segments.size(); idx++) {
    const SegmentInfo& seg = segments[idx];
    const Elf64_Phdr& phdr = phdrs[idx];

    // Check if this segment would exceed size limit
    if (size_limit != -1 && (phdr.p_offset + phdr.p_filesz > size_limit)) {
      if (show_progress) {
        fprintf(stderr, "Core file size limit reached, truncating at segment %zu\n", idx);
      }
      // Stop writing segments but return success - we wrote valid headers
      return HSA_STATUS_SUCCESS;
    }

    if (is_reg_file) {
      int error = posix_fallocate(fd, phdr.p_offset, phdr.p_filesz);
      if (error != 0) {
        fprintf(stderr, "Failed to allocate file: %s\n", strerror(error));
        return HSA_STATUS_ERROR;
      }
    }

    size_t remaining = phdr.p_filesz;
    while (remaining > 0) {
      size_t curr_chunk = std::min(remaining, MAX_BUFFER_SIZE);
      hsa_status_t st = seg.builder->Read(copy_buffer.get(), curr_chunk,
                                          phdr.p_vaddr + phdr.p_filesz - remaining);
      if (st != HSA_STATUS_SUCCESS) {
        return st;
      }

      if (is_reg_file) {
        // For regular files, use pwrite to write at specific offset
        if (pwrite(fd, copy_buffer.get(), curr_chunk,
            phdr.p_offset + phdr.p_filesz - remaining) !=
                                                        (ssize_t)curr_chunk) {
          perror("Failed to write segment data");
          return HSA_STATUS_ERROR;
        }
      } else {
        // For pipes, use sequential write
        if (write(fd, copy_buffer.get(), curr_chunk) != (ssize_t)curr_chunk) {
          perror("Failed to write segment data to pipe");
          return HSA_STATUS_ERROR;
        }
      }
      remaining -= curr_chunk;
    }
  }

  return HSA_STATUS_SUCCESS;

}
} // anonymous namespace

static hsa_status_t
build_core_dump(const std::string& filename, const SegmentsInfo& segments,
                                        size_t size_limit, bool show_progress);
// Handle pipe pattern - fork/exec handler and pipe dump to it
namespace {
hsa_status_t write_to_pipe_handler(const std::string& pattern,
                                          const SegmentsInfo& segments,
                                          size_t size_limit,
                                          bool show_progress) {
  // Check if we're in a container
  if (is_running_in_container() && custom_core_dump().empty()) {
    fprintf(stderr,
      "GPU coredump: System pipe patterns not supported in containers.\n"
      "Falling back to file-based dump. Use custom pattern (HSA_COREDUMP_FILE)"
      " to override.\n");
    // Fall back to file-based dump
    std::string filename = PREFIX_FILE_NAME + "." + std::to_string(getpid()) + ".gpu";
    return build_core_dump(filename, segments, size_limit, show_progress);
  }

  // Extract program and arguments (remove leading '|')
  std::string command = pattern.substr(1);
  std::string substituted = substitute_core_pattern(command);
  // Parse into program and args
  std::vector<std::string> args = parse_command_line(substituted);
  if (args.empty()) {
    fprintf(stderr, "GPU coredump: Invalid pipe pattern\n");
    return HSA_STATUS_ERROR;
  }
  // Create pipe for communication
  int pipefd[2];
  if (pipe(pipefd) == -1) {
    perror("GPU coredump: pipe creation failed");
    return HSA_STATUS_ERROR;
  }
  pid_t pid = fork();
  if (pid == -1) {
    perror("GPU coredump: fork failed");
    close(pipefd[0]);
    close(pipefd[1]);
    return HSA_STATUS_ERROR;
  }
  if (pid == 0) {
    // Child process - execute handler
    close(pipefd[1]);  // Close write end
    // Redirect stdin to read end of pipe
    if (dup2(pipefd[0], STDIN_FILENO) == -1) {
      perror("GPU coredump: dup2 failed");
      _exit(1);
    }
    close(pipefd[0]);
    // Convert args to char* array for execvp
    std::vector<char*> argv;
    for (auto& arg : args) {
      argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);
    // Execute handler
    execvp(argv[0], argv.data());
    // If we get here, exec failed
    perror("GPU coredump: execvp failed");
    _exit(1);
  } else {
    hsa_status_t status;
    // Parent process - write core dump to pipe
    close(pipefd[0]);  // Close read end
    // Write core dump data to pipe
    status = write_core_dump_to_fd(pipefd[1], segments, -1, show_progress);
    close(pipefd[1]);
    // Wait for child to finish
    int child_status;
    if (waitpid(pid, &child_status, 0) == -1) {
      perror("GPU coredump: waitpid failed");
      return HSA_STATUS_ERROR;
    }
    if (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
      fprintf(stderr, "GPU coredump: handler exited with error (status: %d)\n",
                     WIFEXITED(child_status) ? WEXITSTATUS(child_status) : -1);
      return HSA_STATUS_ERROR;
    }
    if (show_progress && status == HSA_STATUS_SUCCESS) {
      printf("GPU core dump sent to pipe handler\n");
    }
      return status;
  }
}
}  // anonymous namespace

static hsa_status_t build_core_dump(const std::string& filename, const SegmentsInfo& segments,
                                    size_t size_limit, bool show_progress) {
  int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    perror("Failed to create GPU coredump");
    return HSA_STATUS_ERROR;
  }

  hsa_status_t result = write_core_dump_to_fd(fd, segments, size_limit, show_progress);
  close(fd);

  if (show_progress && result == HSA_STATUS_SUCCESS) {
    printf("GPU core dump created: %s\n", filename.c_str());
  }

  return result;
}
}   //  namespace impl

hsa_status_t dump_gpu_core() {
  if (core::Runtime::runtime_singleton_->flag().core_dump_disable()) {
    return HSA_STATUS_SUCCESS;
  }

  // Check ulimit -c
  struct rlimit rlimit;
  if (getrlimit(RLIMIT_CORE, &rlimit)) {
    perror("Could not get core file size");
    return HSA_STATUS_ERROR;
  }
  debug_print("core file size: %ld\n", rlimit.rlim_cur);

  if (rlimit.rlim_cur == 0) {
    return HSA_STATUS_SUCCESS;
  }

  impl::NoteSegmentBuilder nbuilder;
  impl::LoadSegmentBuilder lbuilder;
  impl::SegmentsInfo segments;

  fprintf(stderr, "\n[CALLER] About to call NoteSegmentBuilder::Collect()...\n");
  hsa_status_t status = nbuilder.Collect(segments);
  fprintf(stderr, "[CALLER] NoteSegmentBuilder::Collect() returned with status=%d\n", status);
  if (status != HSA_STATUS_SUCCESS) return status;

  fprintf(stderr, "[CALLER] About to call LoadSegmentBuilder::Collect()...\n");
  status = lbuilder.Collect(segments);
  fprintf(stderr, "[CALLER] LoadSegmentBuilder::Collect() returned with status=%d\n", status);
  if (status != HSA_STATUS_SUCCESS) return status;

  fprintf(stderr, "[CALLER] Both Collect() calls completed successfully - continuing to file write...\n");

  // Determine output pattern
  std::string pattern;
  bool kernel_pattern = false;
  bool use_custom_pattern = !custom_core_dump().empty();
  if (use_custom_pattern) {
    pattern = custom_core_dump();
  } else {
    // Fallback to kernel core pattern
    pattern = impl::read_kernel_core_pattern();
    if (pattern.empty()) {
      // If we can't read kernel pattern, use default
      pattern = PREFIX_FILE_NAME + ".%p";
    } else {
      kernel_pattern = true;
    }
  }

  bool show_progress = core::Runtime::runtime_singleton_->flag().enable_core_dump_progress();

  if (!pattern.empty() && pattern[0] == '|') {
    if (show_progress) {
      fprintf(stderr, "Generating GPU core dump via pipe handler\n");
    }
    return impl::write_to_pipe_handler(pattern, segments, rlimit.rlim_cur, show_progress);
  } else {
    // Regular file output
    std::string filename = impl::substitute_core_pattern(pattern);

    if (kernel_pattern && !use_custom_pattern) {
      filename += ".gpu";
    }

    if (!impl::validate_dump_path(filename)) {
      return HSA_STATUS_ERROR;
    }
    if (show_progress) {
      fprintf(stderr, "Generating GPU core dump to: %s\n", filename.c_str());
    }
    return impl::build_core_dump(filename, segments, rlimit.rlim_cur, show_progress);
  }
}
}   //  namespace coredump
}   //  namespace amd
}   //  namespace rocr
