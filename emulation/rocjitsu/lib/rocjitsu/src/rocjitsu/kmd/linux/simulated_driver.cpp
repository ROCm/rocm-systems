// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/simulated_driver.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_hsa_queue.h"
RJ_DIAGNOSTIC_POP
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "util/except.h"
#include "util/log.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <format>
#include <linux/types.h>
#include <sstream>
#include <string_view>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
#include <thread>
#include <unistd.h>
#include <vector>

namespace rocjitsu {

constexpr const char *const kKfdSysfsPrefix = "/sys/devices/virtual/kfd/kfd/topology";

namespace {

bool vm_trace_enabled() {
  static const bool enabled = (std::getenv("RJ_VMEM_TRACE") != nullptr);
  return enabled;
}

constexpr const char *const kDrmSysfsPrefix = "/sys/class/drm";
constexpr const char *const kKfdSysfsPrefixAlt = "/sys/class/kfd/kfd/topology";
constexpr uint32_t kTileConfigCount = 32;
constexpr uint32_t kMacroTileConfigCount = 16;

/// @brief Derive PTE MTYPE from KFD allocation flags (mirrors amdgpu driver).
amdgpu::Mtype pte_mtype_for_flags(uint32_t flags) {
  if (flags & KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED)
    return amdgpu::Mtype::UC;
  if (flags & (KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_USERPTR))
    return amdgpu::Mtype::UC;
  if (flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL)
    return amdgpu::Mtype::UC;
  if (flags & KFD_IOC_ALLOC_MEM_FLAGS_COHERENT)
    return amdgpu::Mtype::CC;
  return amdgpu::Mtype::RW;
}

void *safe_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  long rc = syscall(SYS_mmap, addr, length, prot, flags, fd, offset);
  if (rc < 0)
    return MAP_FAILED;
  return reinterpret_cast<void *>(static_cast<uintptr_t>(rc));
}

} // namespace

std::shared_ptr<KfdProcess> SimulatedDriver::find_process(uint32_t process_id) const {
  std::lock_guard<std::mutex> lk(process_mutex_);
  auto it = processes_.find(process_id);
  return (it != processes_.end()) ? it->second : nullptr;
}

void SimulatedDriver::map_to_gpu(KfdProcess &proc, uint64_t gpu_va, void *host_ptr, size_t size,
                                 amdgpu::Mtype mtype) {
  util::Logger::cp("MAP pid=", proc.process_id(), " va=0x", std::hex, gpu_va, " size=0x", size,
                   std::dec, " mtype=", static_cast<int>(mtype));
  proc.map_pages(gpu_va, host_ptr, size, mtype);
}

void SimulatedDriver::unmap_from_gpu(KfdProcess &proc, uint64_t gpu_va, size_t size) {
  util::Logger::cp("UNMAP pid=", proc.process_id(), " va=0x", std::hex, gpu_va, " size=0x", size,
                   std::dec);
  proc.unmap_pages(gpu_va, size);
}

std::string SimulatedDriver::redirect_sysfs_path(const char *path) const {
  std::string_view sv(path);
  std::string_view kfd_prefix(kKfdSysfsPrefix);
  if (sv.starts_with(kfd_prefix)) {
    auto result = topology_path() + std::string(sv.substr(kfd_prefix.size()));
    util::Logger::vm("sysfs redirect: ", path, " -> ", result);
    return result;
  }
  std::string_view kfd_alt_prefix(kKfdSysfsPrefixAlt);
  if (sv.starts_with(kfd_alt_prefix))
    return topology_path() + std::string(sv.substr(kfd_alt_prefix.size()));

  const auto &drm = topology().drm_path();
  if (!drm.empty()) {
    std::string_view drm_prefix(kDrmSysfsPrefix);
    if (sv.starts_with(drm_prefix))
      return drm + std::string(sv.substr(drm_prefix.size()));
  }

  return {};
}

void SimulatedDriver::setup_topology(const config::KfdDeviceConfig &dev, uint32_t num_xcc) {
  if (!dev.present)
    return;

  Sysfs::GpuInfo gpu{};
  gpu.gpu_id = dev.gpu_id;
  gpu.gfx_target_version = dev.gfx_target_version;
  gpu.vendor_id = dev.vendor_id;
  gpu.device_id = dev.device_id;
  gpu.family_id = dev.family_id;
  gpu.unique_id = dev.unique_id;
  gpu.marketing_name = dev.marketing_name;
  gpu.drm_render_minor = dev.drm_render_minor;
  gpu.revision_id = dev.revision_id;
  gpu.pci_revision_id = dev.pci_revision_id;
  gpu.simd_count = dev.simd_count;
  gpu.max_waves_per_simd = dev.max_waves_per_simd;
  gpu.num_shader_engines = dev.num_shader_engines;
  gpu.num_shader_arrays_per_engine = dev.num_shader_arrays_per_engine;
  gpu.num_cu_per_sh = dev.num_cu_per_sh;
  gpu.simd_per_cu = dev.simd_per_cu;
  gpu.wave_front_size = dev.wave_front_size;
  gpu.max_slots_scratch_cu = dev.max_slots_scratch_cu;
  gpu.local_mem_size = dev.local_mem_size;
  gpu.vram_type = dev.vram_type;
  gpu.lds_size_kb = dev.lds_size_kb;
  gpu.mem_width = dev.mem_width;
  gpu.mem_clk_max = dev.mem_clk_max;
  gpu.l1_size_kb = dev.l1_size_kb;
  gpu.l1_line_size = dev.l1_line_size;
  gpu.l1_assoc = dev.l1_assoc;
  gpu.l2_size_kb = dev.l2_size_kb;
  gpu.l2_line_size = dev.l2_line_size;
  gpu.l2_assoc = dev.l2_assoc;
  gpu.num_sdma_engines = dev.num_sdma_engines;
  gpu.num_sdma_xgmi_engines = dev.num_sdma_xgmi_engines;
  gpu.num_cp_queues = dev.num_cp_queues;
  gpu.max_engine_clk_fcompute = dev.max_engine_clk_fcompute;
  gpu.location_id = dev.location_id;
  gpu.hive_id = dev.hive_id;
  gpu.domain = dev.domain;
  gpu.num_xcc = num_xcc;

  setup_topology(gpu);
}

SimulatedDriver::SimulatedDriver(SoC &soc, bool daemon_mode) : daemon_mode_(daemon_mode) {
  gpus_.push_back({&soc, 0, false, {}});
}

SimulatedDriver::SimulatedDriver(std::vector<SoC *> socs, std::vector<uint32_t> gpu_ids,
                                 bool daemon_mode)
    : daemon_mode_(daemon_mode) {
  for (size_t i = 0; i < socs.size(); ++i)
    gpus_.push_back({socs[i], i < gpu_ids.size() ? gpu_ids[i] : socs[i]->gpu_id(), false, {}});
}

SimulatedDriver::GpuDevice *SimulatedDriver::find_gpu(uint32_t gpu_id) {
  for (auto &g : gpus_)
    if (g.gpu_id == gpu_id)
      return &g;
  return nullptr;
}

const SimulatedDriver::GpuDevice *SimulatedDriver::find_gpu(uint32_t gpu_id) const {
  for (auto &g : gpus_)
    if (g.gpu_id == gpu_id)
      return &g;
  return nullptr;
}

SimulatedDriver::~SimulatedDriver() {
  while (!processes_.empty())
    close(processes_.begin()->first);
}

void SimulatedDriver::setup_topology(const Sysfs::GpuInfo &gpu) {
  if (!gpus_.empty())
    gpus_[0].gpu_id = gpu.gpu_id;
  topology_.generate(gpu);
  topology_.setup_environment();
}

void SimulatedDriver::setup_topology(const std::vector<config::KfdDeviceConfig> &devs,
                                     uint32_t num_xcc) {
  std::vector<Sysfs::GpuInfo> infos;
  infos.reserve(devs.size());
  for (auto &dev : devs) {
    if (!dev.present)
      continue;
    Sysfs::GpuInfo gpu{};
    gpu.gpu_id = dev.gpu_id;
    gpu.gfx_target_version = dev.gfx_target_version;
    gpu.vendor_id = dev.vendor_id;
    gpu.device_id = dev.device_id;
    gpu.family_id = dev.family_id;
    gpu.unique_id = dev.unique_id;
    gpu.marketing_name = dev.marketing_name;
    gpu.drm_render_minor = dev.drm_render_minor;
    gpu.revision_id = dev.revision_id;
    gpu.pci_revision_id = dev.pci_revision_id;
    gpu.simd_count = dev.simd_count;
    gpu.max_waves_per_simd = dev.max_waves_per_simd;
    gpu.num_shader_engines = dev.num_shader_engines;
    gpu.num_shader_arrays_per_engine = dev.num_shader_arrays_per_engine;
    gpu.num_cu_per_sh = dev.num_cu_per_sh;
    gpu.simd_per_cu = dev.simd_per_cu;
    gpu.wave_front_size = dev.wave_front_size;
    gpu.max_slots_scratch_cu = dev.max_slots_scratch_cu;
    gpu.local_mem_size = dev.local_mem_size;
    gpu.vram_type = dev.vram_type;
    gpu.lds_size_kb = dev.lds_size_kb;
    gpu.mem_width = dev.mem_width;
    gpu.mem_clk_max = dev.mem_clk_max;
    gpu.l1_size_kb = dev.l1_size_kb;
    gpu.l1_line_size = dev.l1_line_size;
    gpu.l1_assoc = dev.l1_assoc;
    gpu.l2_size_kb = dev.l2_size_kb;
    gpu.l2_line_size = dev.l2_line_size;
    gpu.l2_assoc = dev.l2_assoc;
    gpu.num_sdma_engines = dev.num_sdma_engines;
    gpu.num_sdma_xgmi_engines = dev.num_sdma_xgmi_engines;
    gpu.num_cp_queues = dev.num_cp_queues;
    gpu.max_engine_clk_fcompute = dev.max_engine_clk_fcompute;
    gpu.location_id = dev.location_id;
    gpu.hive_id = dev.hive_id;
    gpu.domain = dev.domain;
    gpu.num_xcc = num_xcc;
    infos.push_back(gpu);
  }
  if (infos.empty())
    return;
  for (size_t i = 0; i < infos.size() && i < gpus_.size(); ++i)
    gpus_[i].gpu_id = infos[i].gpu_id;
  topology_.generate(infos);
  topology_.setup_environment();
}

bool SimulatedDriver::is_doorbell_range(const void *addr, size_t length) const {
  auto p = find_process(local_process_id_);
  if (!p)
    return false;
  auto &gs = p->gpu(0);
  if (!gs.doorbell_page || gs.doorbell_page_size == 0 || !addr || length == 0)
    return false;
  auto base = reinterpret_cast<uintptr_t>(gs.doorbell_page);
  auto end = base + gs.doorbell_page_size;
  auto query_base = reinterpret_cast<uintptr_t>(addr);
  auto query_end = query_base + length;
  return query_base < end && query_end > base;
}

int SimulatedDriver::open() {
  static std::once_flag raise_nofile_flag;
  std::call_once(raise_nofile_flag, [] {
    struct rlimit rl {};
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < 8192) {
      rl.rlim_cur = std::min<rlim_t>(rl.rlim_max, 65536);
      setrlimit(RLIMIT_NOFILE, &rl);
    }
  });

  if (fd_ < 0) {
    fd_ = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_kfd", 0));
    if (fd_ < 0)
      return -1;
  }

  std::lock_guard<std::mutex> lk(process_mutex_);
  if (!daemon_mode_ && local_process_id_ != 0 && processes_.contains(local_process_id_)) {
    processes_[local_process_id_]->event_state_.reset();
    return fd_;
  }
  uint32_t pid = next_process_id_++;
  auto proc = std::make_shared<KfdProcess>(pid, static_cast<uint32_t>(gpus_.size()));
  proc->event_state_.reset();
  for (auto &g : gpus_) {
    if (auto *mem = g.soc ? g.soc->memory() : nullptr) {
      mem->register_process(pid, &proc->page_table_, &proc->page_table_mutex_);
      if (!daemon_mode_)
        mem->set_passthrough(true);
    }
  }
  processes_[pid] = proc;
  local_process_id_ = pid;

  {
    std::lock_guard<std::mutex> ilk(interrupt_mutex_);
    event_dispatch_[pid] = &proc->event_state_;
  }

  for (size_t i = 0; i < gpus_.size(); ++i) {
    auto &g = gpus_[i];
    if (g.cps_initialized)
      continue;
    if (!g.soc)
      continue;
    uint64_t lds_base = 0x1000000000000ULL + i * 0x10000000000ULL;
    uint64_t scratch_base = 0x2000000000000ULL + i * 0x10000000000ULL;
    g.soc->set_apertures(lds_base, lds_base + 0xFFFFFFFFULL, scratch_base,
                         scratch_base + 0xFFFFFFFFULL);
    g.soc->for_each_cp([this](amdgpu::CommandProcessor *cp) {
      cp->set_interrupt_callback([this](uint32_t process_id, uint32_t event_id) {
        std::lock_guard<std::mutex> ilk(interrupt_mutex_);
        auto it = event_dispatch_.find(process_id);
        if (it != event_dispatch_.end()) {
          util::Logger::cp("INTERRUPT_ROUTE: pid=", process_id, " event_id=", event_id,
                           " found=true");
          it->second->signal_interrupt(event_id);
        } else {
          util::Logger::cp("INTERRUPT_ROUTE: pid=", process_id, " event_id=", event_id,
                           " found=false");
        }
      });
      cp->set_scratch_backing_resolver([this](uint32_t process_id) -> uint64_t {
        std::lock_guard<std::mutex> plk(process_mutex_);
        for (auto &[fd, proc] : processes_) {
          if (proc->process_id() == process_id) {
            for (auto &gs : proc->gpu_state_) {
              if (gs.scratch_backing_va != 0)
                return gs.scratch_backing_va << 16;
            }
          }
        }
        return 0;
      });
      cp->set_scratch_backing_allocator(
          [this](uint32_t process_id, uint64_t gpu_va, size_t size) -> bool {
            return allocate_scratch_backing(process_id, gpu_va, size);
          });
    });
    g.cps_initialized = true;
  }

  return fd_;
}

uint32_t SimulatedDriver::open_process() {
  if (fd_ < 0) {
    fd_ = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_kfd", 0));
    if (fd_ < 0)
      return 0;
  }

  uint32_t pid;
  {
    std::lock_guard<std::mutex> lk(process_mutex_);
    pid = next_process_id_++;
    auto proc = std::make_shared<KfdProcess>(pid, static_cast<uint32_t>(gpus_.size()));
    proc->event_state_.reset();
    for (auto &g : gpus_) {
      if (auto *mem = g.soc ? g.soc->memory() : nullptr)
        mem->register_process(pid, &proc->page_table_, &proc->page_table_mutex_);
    }
    processes_[pid] = proc;

    {
      std::lock_guard<std::mutex> ilk(interrupt_mutex_);
      event_dispatch_[pid] = &proc->event_state_;
    }
  }

  for (size_t i = 0; i < gpus_.size(); ++i) {
    auto &g = gpus_[i];
    if (g.cps_initialized)
      continue;
    if (!g.soc)
      continue;
    uint64_t lds_base = 0x1000000000000ULL + i * 0x10000000000ULL;
    uint64_t scratch_base = 0x2000000000000ULL + i * 0x10000000000ULL;
    g.soc->set_apertures(lds_base, lds_base + 0xFFFFFFFFULL, scratch_base,
                         scratch_base + 0xFFFFFFFFULL);
    g.soc->for_each_cp([this](amdgpu::CommandProcessor *cp) {
      cp->set_interrupt_callback([this](uint32_t process_id, uint32_t event_id) {
        std::lock_guard<std::mutex> ilk(interrupt_mutex_);
        auto it = event_dispatch_.find(process_id);
        if (it != event_dispatch_.end()) {
          util::Logger::cp("INTERRUPT_ROUTE: pid=", process_id, " event_id=", event_id,
                           " found=true");
          it->second->signal_interrupt(event_id);
        } else {
          util::Logger::cp("INTERRUPT_ROUTE: pid=", process_id, " event_id=", event_id,
                           " found=false");
        }
      });
      cp->set_scratch_backing_resolver([this](uint32_t process_id) -> uint64_t {
        std::lock_guard<std::mutex> plk(process_mutex_);
        for (auto &[fd, proc] : processes_) {
          if (proc->process_id() == process_id) {
            for (auto &gs : proc->gpu_state_) {
              if (gs.scratch_backing_va != 0)
                return gs.scratch_backing_va << 16;
            }
          }
        }
        return 0;
      });
      cp->set_scratch_backing_allocator(
          [this](uint32_t process_id, uint64_t gpu_va, size_t size) -> bool {
            return allocate_scratch_backing(process_id, gpu_va, size);
          });
    });
    g.cps_initialized = true;
  }

  return pid;
}

int SimulatedDriver::close() { return close(local_process_id_); }

int SimulatedDriver::close(uint32_t process_id) {
  std::shared_ptr<KfdProcess> extracted;
  std::vector<uint32_t> queue_ids;

  {
    std::lock_guard<std::mutex> lk(process_mutex_);
    auto it = processes_.find(process_id);
    if (it == processes_.end())
      return 0;
    extracted = std::move(it->second);
    processes_.erase(it);
  }

  {
    std::lock_guard<std::mutex> ilk(interrupt_mutex_);
    event_dispatch_.erase(process_id);
  }

  for (auto &g : gpus_) {
    if (auto *mem = g.soc ? g.soc->memory() : nullptr)
      mem->unregister_process(process_id);
  }

  auto &proc = *extracted;
  const bool trace_enabled = vm_trace_enabled();
  size_t leaked_allocations = 0;
  uint64_t leaked_bytes = 0;
  size_t leaked_queues = 0;
  std::vector<uint64_t> leaked_handles;
  proc.event_state_.notify_closing();
  proc.event_state_.signal_page_shutdown();

  {
    std::lock_guard<std::mutex> alk(proc.alloc_mutex_);
    queue_ids.assign(proc.active_queue_ids_.begin(), proc.active_queue_ids_.end());
    proc.active_queue_ids_.clear();

    if (trace_enabled)
      leaked_handles.reserve(proc.allocations_.size());
    for (auto &[handle, alloc] : proc.allocations_) {
      ++leaked_allocations;
      leaked_bytes += alloc.size;
      if (trace_enabled)
        leaked_handles.push_back(handle);
      if (alloc.host_ptr && !(alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR)) {
        unmap_from_gpu(proc, alloc.gpu_va, alloc.size);
        syscall(SYS_munmap, alloc.host_ptr, alloc.size);
        alloc.host_ptr = nullptr;
      }
      if (alloc.memfd >= 0) {
        {
          std::lock_guard<std::mutex> flk(owned_fds_mutex_);
          owned_fds_.erase(alloc.memfd);
        }
        syscall(SYS_close, alloc.memfd);
        alloc.memfd = -1;
      }
    }
    proc.allocations_.clear();
  }

  for (uint32_t qid : queue_ids) {
    for (auto &g : gpus_)
      if (g.soc)
        g.soc->for_each_cp([qid, process_id](amdgpu::CommandProcessor *cp) {
          cp->unregister_queue(qid, process_id);
        });
  }

  for (auto &gs : proc.gpu_state_) {
    if (gs.doorbell_page && gs.doorbell_page_size)
      syscall(SYS_munmap, gs.doorbell_page, gs.doorbell_page_size);
  }

  leaked_queues = queue_ids.size();
  if (trace_enabled) {
    if (leaked_allocations == 0 && leaked_queues == 0) {
      util::Logger::vm("kfd.close: no outstanding GPUVM allocations or queues");
    } else {
      util::Logger::vm("kfd.close: leaked_allocations=", leaked_allocations,
                       " leaked_bytes=", leaked_bytes, " leaked_queues=", leaked_queues);
      if (!leaked_handles.empty()) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < leaked_handles.size(); ++i) {
          oss << leaked_handles[i];
          if (i + 1 < leaked_handles.size())
            oss << ",";
        }
        oss << "]";
        util::Logger::vm("kfd.close: leaked_handles=", oss.str());
      }
    }
  }

  for (auto &[handle, dmabuf] : proc.imported_dmabufs_) {
    [[maybe_unused]] auto &_ = handle;
    if (dmabuf.fd >= 0)
      syscall(SYS_close, dmabuf.fd);
  }

  return 0;
}

int SimulatedDriver::ioctl(unsigned long request, void *arg) {
  return ioctl(local_process_id_, request, arg);
}

int SimulatedDriver::ioctl(uint32_t process_id, unsigned long request, void *arg) {
  auto proc = find_process(process_id);
  if (!proc)
    return -ESRCH;
  return dispatch_ioctl(*proc, request, arg);
}

static const char *ioctl_name(unsigned long req) {
  switch (canonical_ioctl_request(req)) {
  case AMDKFD_IOC_GET_VERSION:
    return "GET_VERSION";
  case AMDKFD_IOC_GET_CLOCK_COUNTERS:
    return "GET_CLOCK_COUNTERS";
  case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
    return "GET_APERTURES";
  case AMDKFD_IOC_ACQUIRE_VM:
    return "ACQUIRE_VM";
  case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU:
    return "ALLOC_MEMORY";
  case AMDKFD_IOC_FREE_MEMORY_OF_GPU:
    return "FREE_MEMORY";
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
    return "MAP_MEMORY";
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
    return "UNMAP_MEMORY";
  case AMDKFD_IOC_CREATE_QUEUE:
    return "CREATE_QUEUE";
  case AMDKFD_IOC_UPDATE_QUEUE:
    return "UPDATE_QUEUE";
  case AMDKFD_IOC_DESTROY_QUEUE:
    return "DESTROY_QUEUE";
  case AMDKFD_IOC_CREATE_EVENT:
    return "CREATE_EVENT";
  case AMDKFD_IOC_DESTROY_EVENT:
    return "DESTROY_EVENT";
  case AMDKFD_IOC_SET_EVENT:
    return "SET_EVENT";
  case AMDKFD_IOC_RESET_EVENT:
    return "RESET_EVENT";
  case AMDKFD_IOC_WAIT_EVENTS:
    return "WAIT_EVENTS";
  case AMDKFD_IOC_RUNTIME_ENABLE:
    return "RUNTIME_ENABLE";
  case AMDKFD_IOC_SET_SCRATCH_BACKING_VA:
    return "SET_SCRATCH_VA";
  case AMDKFD_IOC_SET_TRAP_HANDLER:
    return "SET_TRAP_HANDLER";
  case AMDKFD_IOC_SET_XNACK_MODE:
    return "SET_XNACK";
  case AMDKFD_IOC_SET_MEMORY_POLICY:
    return "SET_MEM_POLICY";
  case AMDKFD_IOC_AVAILABLE_MEMORY:
    return "AVAIL_MEMORY";
  case AMDKFD_IOC_GET_TILE_CONFIG:
    return "GET_TILE_CONFIG";
  case AMDKFD_IOC_SVM:
    return "SVM";
  default:
    return "UNKNOWN";
  }
}

int SimulatedDriver::dispatch_ioctl(KfdProcess &proc, unsigned long request, void *arg) {
  util::Logger::cp("IOCTL pid=", proc.process_id(), " ", ioctl_name(request));

  switch (canonical_ioctl_request(request)) {
  case AMDKFD_IOC_GET_VERSION:
    return get_version_ioctl(arg);
  case AMDKFD_IOC_GET_CLOCK_COUNTERS:
    return get_clock_counters_ioctl(arg);
  case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
    return get_apertures_ioctl(arg);
  case AMDKFD_IOC_ACQUIRE_VM:
    return acquire_vm_ioctl(arg);
  case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU:
    return alloc_memory_ioctl(proc, arg);
  case AMDKFD_IOC_FREE_MEMORY_OF_GPU:
    return free_memory_ioctl(proc, arg);
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
    return map_memory_ioctl(proc, arg);
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
    return unmap_memory_ioctl(proc, arg);
  case AMDKFD_IOC_CREATE_QUEUE:
    return create_queue_ioctl(proc, arg);
  case AMDKFD_IOC_UPDATE_QUEUE:
    return update_queue_ioctl(proc, arg);
  case AMDKFD_IOC_DESTROY_QUEUE:
    return destroy_queue_ioctl(proc, arg);
  case AMDKFD_IOC_CREATE_EVENT:
    return create_event_ioctl(proc, arg);
  case AMDKFD_IOC_DESTROY_EVENT:
    return destroy_event_ioctl(proc, arg);
  case AMDKFD_IOC_SET_EVENT:
    return set_event_ioctl(proc, arg);
  case AMDKFD_IOC_RESET_EVENT:
    return reset_event_ioctl(proc, arg);
  case AMDKFD_IOC_WAIT_EVENTS:
    return wait_events_ioctl(proc, arg);
  case AMDKFD_IOC_SET_XNACK_MODE:
    return set_xnack_mode_ioctl(arg);
  case AMDKFD_IOC_SET_MEMORY_POLICY:
    return set_memory_policy_ioctl(proc, arg);
  case AMDKFD_IOC_AVAILABLE_MEMORY: {
    auto *args = static_cast<kfd_ioctl_get_available_memory_args *>(arg);
    uint64_t allocated = 0;
    {
      std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
      for (auto &[handle, alloc] : proc.allocations_)
        allocated += alloc.size;
    }
    constexpr uint64_t kVramBytes = 64ULL << 30;
    args->available = kVramBytes - std::min(allocated, kVramBytes);
    return 0;
  }
  case AMDKFD_IOC_RUNTIME_ENABLE:
    return runtime_enable_ioctl(proc, arg);
  case AMDKFD_IOC_SET_SCRATCH_BACKING_VA: {
    auto *a = static_cast<kfd_ioctl_set_scratch_backing_va_args *>(arg);
    uint32_t ord = gpu_ordinal(a->gpu_id);
    proc.gpu(ord).scratch_backing_va = a->va_addr;
    util::Logger::vm([&](auto &os) {
      os << "SET_SCRATCH_BACKING_VA pid=" << proc.process_id() << " gpu_id=" << a->gpu_id
         << " va=" << std::hex << a->va_addr << std::dec;
    });
    return 0;
  }
  case AMDKFD_IOC_SET_TRAP_HANDLER: {
    auto *a = static_cast<kfd_ioctl_set_trap_handler_args *>(arg);
    uint32_t ord = gpu_ordinal(a->gpu_id);
    proc.gpu(ord).trap_tba_addr = a->tba_addr;
    proc.gpu(ord).trap_tma_addr = a->tma_addr;
    return 0;
  }
  case AMDKFD_IOC_GET_TILE_CONFIG:
    return get_tile_config_ioctl(arg);
  case AMDKFD_IOC_GET_DMABUF_INFO:
    return get_dmabuf_info_ioctl(proc, arg);
  case AMDKFD_IOC_IMPORT_DMABUF:
    return import_dmabuf_ioctl(proc, arg);
  case AMDKFD_IOC_EXPORT_DMABUF:
    return export_dmabuf_ioctl(proc, arg);
  case AMDKFD_IOC_IPC_EXPORT_HANDLE:
    return ipc_export_handle_ioctl(proc, arg);
  case AMDKFD_IOC_IPC_IMPORT_HANDLE:
    return ipc_import_handle_ioctl(proc, arg);
  case AMDKFD_IOC_SVM:
    // SVM requests carry a trailing attribute array, so libhsakmt sets _IOC_SIZE
    // to the actual buffer size. canonical_ioctl_request() lets this follow the
    // normal switch-dispatch style while still accepting those runtime-sized
    // request values.
    return svm_ioctl(proc, arg);
  default:
    util::Logger::debug_print("rocjitsu: unhandled ioctl 0x", std::hex, request);
    return 0;
  }
}

void *SimulatedDriver::mmap(void *addr, size_t length, int prot, int flags, off_t offset) {
  return mmap(local_process_id_, addr, length, prot, flags, offset);
}

void *SimulatedDriver::mmap(uint32_t process_id, void *addr, size_t length, int prot, int flags,
                            off_t offset) {
  auto p = find_process(process_id);
  if (!p)
    return MAP_FAILED;
  if (daemon_mode_)
    return dispatch_mmap(*p, nullptr, length, prot, flags & ~MAP_FIXED, offset);
  return dispatch_mmap(*p, addr, length, prot, flags, offset);
}

void *SimulatedDriver::dispatch_mmap(KfdProcess &proc, void *addr, size_t length, int prot,
                                     int flags, off_t offset) {
  uint64_t type = static_cast<uint64_t>(offset) & KFD_MMAP_TYPE_MASK;
  util::Logger::vm("SimulatedDriver::mmap type=0x", std::hex, type, " offset=0x", offset,
                   " length=", std::dec, length, " addr=", addr);

  if (type == KFD_MMAP_TYPE_DOORBELL) {
    uint64_t encoded_gpu =
        (static_cast<uint64_t>(offset) & ~KFD_MMAP_TYPE_MASK) >> KFD_MMAP_GPU_ID_SHIFT;
    uint32_t db_gpu_id = static_cast<uint32_t>(encoded_gpu);
    uint32_t ord = gpu_ordinal(db_gpu_id);
    auto *gpu = find_gpu(db_gpu_id);

    int doorbell_fd = -1;
    {
      std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
      for (auto &[handle, alloc] : proc.allocations_) {
        if ((alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL) && alloc.gpu_id == db_gpu_id) {
          doorbell_fd = alloc.memfd;
          break;
        }
      }
    }

    if (doorbell_fd >= 0) {
      off_t cur_size = 0;
      {
        struct stat st {};
        if (fstat(doorbell_fd, &st) == 0)
          cur_size = st.st_size;
      }
      if (static_cast<off_t>(length) > cur_size) {
        if (ftruncate(doorbell_fd, static_cast<off_t>(length)) != 0) {
          errno = ENOMEM;
          return MAP_FAILED;
        }
      }
      // Initialize doorbell backing to 0xFF via a temporary mapping. This
      // avoids SIGBUS on the final MAP_SHARED mmap on Linux 6.17+ where
      // shmem large folio allocation can fail during a bulk memset on a
      // freshly-mapped region. Writing through a separate PROT_WRITE
      // mapping forces page allocation before the final shared mapping.
      auto *init_ptr = static_cast<uint8_t *>(
          safe_mmap(nullptr, length, PROT_WRITE, MAP_SHARED, doorbell_fd, 0));
      if (init_ptr != MAP_FAILED) {
        std::memset(init_ptr, 0xFF, length);
        syscall(SYS_munmap, init_ptr, length);
      }
    }

    int db_mflags = MAP_SHARED;
    if (flags & MAP_FIXED)
      db_mflags |= MAP_FIXED;

    void *ptr = safe_mmap(addr, length, PROT_READ | PROT_WRITE,
                          doorbell_fd >= 0 ? db_mflags : (db_mflags | MAP_ANONYMOUS),
                          doorbell_fd >= 0 ? doorbell_fd : -1, 0);
    if (ptr != MAP_FAILED) {
      // Initialize doorbell backing to 0xFF so each uint64_t slot starts
      // at ~0ULL, matching the HwQueue::last_doorbell sentinel. Without
      // this, MAP_ANONYMOUS gives zero-filled pages and the CP's first
      // scan falsely consumes the 0 vs ~0 transition, leaving
      // last_doorbell==0. When ROCR later rings the doorbell with
      // write_idx==0 (first packet), the CP sees no change and never
      // processes the submission.
      std::memset(ptr, 0xFF, length);

      auto &gs = proc.gpu(ord);
      gs.doorbell_page = ptr;
      gs.doorbell_page_size = length;
      gs.doorbell_gpu_va = reinterpret_cast<uint64_t>(ptr);
      map_to_gpu(proc, gs.doorbell_gpu_va, ptr, length, amdgpu::Mtype::UC);
      if (gpu && gpu->soc)
        gpu->soc->for_each_cp(
            [&](amdgpu::CommandProcessor *cp) { cp->set_doorbell_base(proc.process_id(), ptr); });
    }
    return ptr;
  }

  if (type == KFD_MMAP_TYPE_EVENTS) {
    if (proc.event_state_.memfd < 0) {
      auto raw_events_fd = static_cast<int>(
          syscall(SYS_memfd_create, "rocjitsu_events", MFD_CLOEXEC | MFD_ALLOW_SEALING));
      if (raw_events_fd < 0)
        return MAP_FAILED;
      proc.event_state_.memfd = fcntl(raw_events_fd, F_DUPFD_CLOEXEC, 4096);
      if (proc.event_state_.memfd < 0)
        proc.event_state_.memfd = raw_events_fd;
      else
        syscall(SYS_close, raw_events_fd);
      {
        std::lock_guard<std::mutex> lk(owned_fds_mutex_);
        owned_fds_.insert(proc.event_state_.memfd);
      }
      if (ftruncate(proc.event_state_.memfd, static_cast<off_t>(length)) != 0) {
        {
          std::lock_guard<std::mutex> lk(owned_fds_mutex_);
          owned_fds_.erase(proc.event_state_.memfd);
        }
        syscall(SYS_close, proc.event_state_.memfd);
        proc.event_state_.memfd = -1;
        return MAP_FAILED;
      }
      fallocate(proc.event_state_.memfd, 0, 0, static_cast<off_t>(length));
      {
        auto *init_ptr = static_cast<uint8_t *>(
            safe_mmap(nullptr, length, PROT_WRITE, MAP_SHARED, proc.event_state_.memfd, 0));
        if (init_ptr != MAP_FAILED) {
          syscall(SYS_madvise, init_ptr, length, MADV_POPULATE_WRITE);
          std::memset(init_ptr, 0xFF, length);
          syscall(SYS_munmap, init_ptr, length);
        }
      }
      fcntl(proc.event_state_.memfd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW);
    }
    int mflags = MAP_SHARED;
    if (flags & MAP_FIXED)
      mflags |= MAP_FIXED;
    void *ptr = safe_mmap(addr, length, PROT_READ | PROT_WRITE, mflags, proc.event_state_.memfd, 0);
    if (ptr != MAP_FAILED)
      proc.event_state_.adopt_page(ptr, length);
    return ptr;
  }

  std::lock_guard<std::mutex> lock(proc.alloc_mutex_);

  uint64_t handle = static_cast<uint64_t>(offset) >> 12;
  auto it = proc.allocations_.find(handle);
  if (it == proc.allocations_.end()) {
    errno = EINVAL;
    return MAP_FAILED;
  }

  auto &alloc = it->second;

  if (daemon_mode_ && alloc.memfd >= 0 && alloc.host_ptr != nullptr)
    return alloc.host_ptr;

  void *host_ptr;

  if (alloc.memfd >= 0) {
    if (length > alloc.size) {
      if (ftruncate(alloc.memfd, static_cast<off_t>(length)) != 0) {
        errno = ENOMEM;
        return MAP_FAILED;
      }
    }
    if (alloc.user_va && (flags & MAP_FIXED) && addr != nullptr) {
      auto prot_rc = syscall(SYS_mprotect, addr, length, PROT_READ | PROT_WRITE);
      if (prot_rc == 0) {
        constexpr size_t page_size = 4096;
        size_t num_pages = (length + page_size - 1) / page_size;
        std::vector<uint8_t> page_resident(num_pages);
        auto mc_rc = syscall(SYS_mincore, addr, length, page_resident.data());

        auto *temp_mapping = static_cast<uint8_t *>(
            safe_mmap(nullptr, length, PROT_WRITE, MAP_SHARED, alloc.memfd, 0));
        if (temp_mapping != MAP_FAILED) {
          if (mc_rc == 0) {
            auto *source = static_cast<uint8_t *>(addr);
            for (size_t i = 0; i < num_pages; ++i) {
              if (page_resident[i] & 1) {
                size_t off = i * page_size;
                size_t copy_len = std::min(page_size, length - off);
                std::memcpy(temp_mapping + off, source + off, copy_len);
              }
            }
          }
          syscall(SYS_munmap, temp_mapping, length);
        }
      }
    }

    int mflags = MAP_SHARED;
    if (flags & MAP_FIXED)
      mflags |= MAP_FIXED;
    host_ptr = safe_mmap(addr, length, prot, mflags, alloc.memfd, 0);
    if (host_ptr == MAP_FAILED)
      return MAP_FAILED;
  } else {
    bool reuse_pages = false;
    if (alloc.user_va && (flags & MAP_FIXED) && addr != nullptr) {
      auto rc = syscall(SYS_mprotect, addr, length, PROT_READ | PROT_WRITE);
      reuse_pages = (rc == 0);
    }
    if (reuse_pages) {
      host_ptr = addr;
    } else {
      int mflags = MAP_ANONYMOUS;
      mflags |= (flags & MAP_SHARED) ? MAP_SHARED : MAP_PRIVATE;
      if (flags & MAP_FIXED)
        mflags |= MAP_FIXED;
      host_ptr = safe_mmap(addr, length, prot, mflags, -1, 0);
      if (host_ptr == MAP_FAILED)
        return MAP_FAILED;
    }
  }

  alloc.host_ptr = host_ptr;

  util::Logger::vm([&](auto &os) {
    os << std::format("mmap: gpu_va={:#x} host_ptr={:#x} size={} flags={:#x}"
                      " MAP_FIXED={} user_va={} memfd={}",
                      alloc.gpu_va, reinterpret_cast<uintptr_t>(host_ptr), length, alloc.flags,
                      bool(flags & MAP_FIXED), alloc.user_va, alloc.memfd);
  });

  map_to_gpu(proc, alloc.gpu_va, host_ptr, length, pte_mtype_for_flags(alloc.flags));

  return host_ptr;
}

int SimulatedDriver::munmap(void *addr, size_t length) {
  return munmap(local_process_id_, addr, length);
}

int SimulatedDriver::munmap(uint32_t process_id, void *addr, size_t length) {
  auto p = find_process(process_id);
  if (!p)
    return -ESRCH;
  return dispatch_munmap(*p, addr, length);
}

int SimulatedDriver::dispatch_munmap(KfdProcess &proc, void *addr, size_t length) {
  for (auto &gs : proc.gpu_state_) {
    if (gs.doorbell_page == addr) {
      if (!proc.event_state_.is_closing()) {
        errno = EPERM;
        return -1;
      }
      if (gs.doorbell_gpu_va && gs.doorbell_page_size)
        unmap_from_gpu(proc, gs.doorbell_gpu_va, gs.doorbell_page_size);
      gs.doorbell_page = nullptr;
      gs.doorbell_gpu_va = 0;
      gs.doorbell_page_size = 0;
      syscall(SYS_munmap, addr, length);
      return 0;
    }
  }
  if (addr == proc.event_state_.page) {
    proc.event_state_.page = nullptr;
    proc.event_state_.page_size = 0;
    syscall(SYS_munmap, addr, length);
    return 0;
  }
  std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
  for (auto &[handle, alloc] : proc.allocations_) {
    if (alloc.host_ptr == addr) {
      unmap_from_gpu(proc, alloc.gpu_va, alloc.size);
      syscall(SYS_munmap, addr, length);
      alloc.host_ptr = nullptr;
      return 0;
    }
  }
  return -ENOENT;
}

int SimulatedDriver::get_version_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_version_args *>(arg);
  args->major_version = KFD_IOCTL_MAJOR_VERSION;
  args->minor_version = KFD_IOCTL_MINOR_VERSION;
  return 0;
}

int SimulatedDriver::get_clock_counters_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_clock_counters_args *>(arg);
  auto now = std::chrono::steady_clock::now().time_since_epoch();
  uint64_t ns =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  args->system_clock_freq = 1000000000ULL;
  args->system_clock_counter = ns;
  args->cpu_clock_counter = ns;
  args->gpu_clock_counter = ns;
  return 0;
}

int SimulatedDriver::get_apertures_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_process_apertures_new_args *>(arg);
  auto n = static_cast<uint32_t>(gpus_.size());

  if (args->num_of_nodes == 0) {
    args->num_of_nodes = n;
    return 0;
  }

  auto *apertures =
      reinterpret_cast<kfd_process_device_apertures *>(args->kfd_process_device_apertures_ptr);
  for (uint32_t i = 0; i < n && i < args->num_of_nodes; ++i) {
    apertures[i].lds_base = 0x1000000000000ULL + static_cast<uint64_t>(i) * 0x10000000000ULL;
    apertures[i].lds_limit = apertures[i].lds_base + 0xFFFFFFFFULL;
    apertures[i].scratch_base = 0x2000000000000ULL + static_cast<uint64_t>(i) * 0x10000000000ULL;
    apertures[i].scratch_limit = apertures[i].scratch_base + 0xFFFFFFFFULL;
    apertures[i].gpuvm_base = 0x1000000000ULL;
    apertures[i].gpuvm_limit = 0x3FFFFFFFFFFFULL;
    apertures[i].gpu_id = gpus_[i].gpu_id;
    apertures[i].pad = 0;
  }

  args->num_of_nodes = n;
  return 0;
}

int SimulatedDriver::get_tile_config_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_tile_config_args *>(arg);
  if (daemon_mode_)
    return -ENOTSUP;

  auto *gpu = find_gpu(args->gpu_id);
  if (!gpu || !gpu->soc)
    return -EINVAL;

  uint32_t tile_write_count = std::min(args->num_tile_configs, kTileConfigCount);
  uint32_t macro_write_count = std::min(args->num_macro_tile_configs, kMacroTileConfigCount);

  // ROCr needs gb_addr_config for swizzled-address calculation. Tile-mode arrays are stubbed until
  // a simulator consumer needs their packed register encodings.
  if (args->tile_config_ptr && tile_write_count > 0) {
    auto *tile_config = reinterpret_cast<uint32_t *>(args->tile_config_ptr);
    std::fill_n(tile_config, tile_write_count, 0u);
  }
  if (args->macro_tile_config_ptr && macro_write_count > 0) {
    auto *macro_tile_config = reinterpret_cast<uint32_t *>(args->macro_tile_config_ptr);
    std::fill_n(macro_tile_config, macro_write_count, 0u);
  }

  args->num_tile_configs = tile_write_count;
  args->num_macro_tile_configs = macro_write_count;
  args->gb_addr_config = kmd::gb_addr_config_for_arch(gpu->soc->arch());
  args->num_banks = 0;
  args->num_ranks = 0;
  return 0;
}

int SimulatedDriver::acquire_vm_ioctl([[maybe_unused]] void *arg) {
  (void)arg;
  return 0;
}

int SimulatedDriver::alloc_memory_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_alloc_memory_of_gpu_args *>(arg);

  std::lock_guard<std::mutex> lock(proc.alloc_mutex_);

  bool user_provided_va = (args->va_addr != 0);
  uint64_t va = args->va_addr;
  if (va == 0) {
    va = proc.next_gpu_va_;
    proc.next_gpu_va_ += (args->size + 0xFFF) & ~0xFFFULL;
  }

  KfdProcess::GpuAllocation alloc{};
  alloc.gpu_va = va;
  alloc.size = args->size;
  alloc.flags = args->flags;
  alloc.handle = proc.next_handle_++;
  alloc.host_ptr = nullptr;
  alloc.gpu_id = args->gpu_id;
  alloc.user_va = user_provided_va;

  auto alloc_mtype = pte_mtype_for_flags(args->flags);
  if ((args->flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR) && !daemon_mode_) {
    alloc.host_ptr = reinterpret_cast<void *>(va);
    map_to_gpu(proc, va, reinterpret_cast<void *>(va), args->size, alloc_mtype);
  } else if (daemon_mode_ || !user_provided_va) {
    auto raw_fd = static_cast<int>(
        syscall(SYS_memfd_create, "rocjitsu_alloc", MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (raw_fd >= 0) {
      alloc.memfd = fcntl(raw_fd, F_DUPFD_CLOEXEC, 4096);
      if (alloc.memfd < 0)
        alloc.memfd = raw_fd;
      else
        syscall(SYS_close, raw_fd);
      {
        std::lock_guard<std::mutex> lk(owned_fds_mutex_);
        owned_fds_.insert(alloc.memfd);
      }
      if (alloc.memfd >= 0) {
        [[maybe_unused]] auto ft_rc = ftruncate(alloc.memfd, static_cast<off_t>(alloc.size));
        fallocate(alloc.memfd, 0, 0, static_cast<off_t>(alloc.size));
        fcntl(alloc.memfd, F_ADD_SEALS, F_SEAL_SHRINK);

        if (daemon_mode_ && !(args->flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL)) {
          auto *mapped =
              safe_mmap(nullptr, alloc.size, PROT_READ | PROT_WRITE, MAP_SHARED, alloc.memfd, 0);
          if (mapped != MAP_FAILED) {
            alloc.host_ptr = mapped;
            map_to_gpu(proc, va, alloc.host_ptr, alloc.size, alloc_mtype);
          }
        }
      }
    }
  }

  proc.allocations_[alloc.handle] = alloc;

  args->handle = alloc.handle;
  args->va_addr = va;
  if (args->flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL) {
    args->mmap_offset = KFD_MMAP_TYPE_DOORBELL | kfd_mmap_gpu_id(args->gpu_id);
  } else {
    args->mmap_offset = alloc.handle << 12;
  }

  util::Logger::vm([&](auto &os) {
    os << std::format(
        "ALLOC pid={} handle={} gpu_va={:#x} size={} flags={:#x} memfd={} host_ptr={}",
        proc.process_id(), alloc.handle, va, args->size, args->flags, alloc.memfd,
        reinterpret_cast<uintptr_t>(alloc.host_ptr));
  });

  return 0;
}

bool SimulatedDriver::allocate_scratch_backing(uint32_t process_id, uint64_t gpu_va, size_t size) {
  if (size == 0)
    return false;

  std::shared_ptr<KfdProcess> proc;
  {
    std::lock_guard<std::mutex> plk(process_mutex_);
    for (auto &[fd, p] : processes_) {
      if (p->process_id() == process_id) {
        proc = p;
        break;
      }
    }
  }
  if (!proc)
    return false;

  size_t aligned_size = (size + 0xFFF) & ~0xFFFULL;
  auto raw_fd = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_scratch", MFD_CLOEXEC));
  if (raw_fd < 0)
    return false;

  int memfd = fcntl(raw_fd, F_DUPFD_CLOEXEC, 4096);
  if (memfd < 0)
    memfd = raw_fd;
  else
    syscall(SYS_close, raw_fd);
  {
    std::lock_guard<std::mutex> lk(owned_fds_mutex_);
    owned_fds_.insert(memfd);
  }

  if (ftruncate(memfd, static_cast<off_t>(aligned_size)) != 0) {
    {
      std::lock_guard<std::mutex> lk(owned_fds_mutex_);
      owned_fds_.erase(memfd);
    }
    syscall(SYS_close, memfd);
    return false;
  }
  auto *host_ptr = safe_mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0);
  if (host_ptr == MAP_FAILED) {
    {
      std::lock_guard<std::mutex> lk(owned_fds_mutex_);
      owned_fds_.erase(memfd);
    }
    syscall(SYS_close, memfd);
    return false;
  }
  {
    std::lock_guard<std::mutex> lk(owned_fds_mutex_);
    owned_fds_.erase(memfd);
  }
  syscall(SYS_close, memfd);
  std::memset(host_ptr, 0, aligned_size);
  proc->map_pages(gpu_va, host_ptr, aligned_size);

  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    KfdProcess::GpuAllocation alloc{};
    alloc.gpu_va = gpu_va;
    alloc.size = aligned_size;
    alloc.host_ptr = host_ptr;
    alloc.handle = proc->next_handle_++;
    alloc.memfd = -1;
    proc->allocations_[alloc.handle] = alloc;
  }

  util::Logger::vm([&](auto &os) {
    os << "SCRATCH_BACKING pid=" << process_id << " gpu_va=0x" << std::hex << gpu_va << " size=0x"
       << aligned_size << std::dec << " host=" << host_ptr;
  });

  return true;
}

int SimulatedDriver::free_memory_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_free_memory_of_gpu_args *>(arg);

  std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
  auto it = proc.allocations_.find(args->handle);
  if (it != proc.allocations_.end()) {
    auto &alloc = it->second;
    if (alloc.imported && alloc.dmabuf_fd >= 0) {
      syscall(SYS_close, alloc.dmabuf_fd);
      if (auto dmabuf_it = proc.imported_dmabufs_.find(args->handle);
          dmabuf_it != proc.imported_dmabufs_.end()) {
        proc.fd_to_import_handle_.erase(dmabuf_it->second.fd);
        proc.imported_dmabufs_.erase(dmabuf_it);
      }
    }
    if (alloc.host_ptr && !alloc.user_va)
      unmap_from_gpu(proc, alloc.gpu_va, alloc.size);
    if (alloc.memfd >= 0) {
      {
        std::lock_guard<std::mutex> lk(owned_fds_mutex_);
        owned_fds_.erase(alloc.memfd);
      }
      syscall(SYS_close, alloc.memfd);
    }

    uint32_t freed_process_id = proc.process_id();
    uint64_t freed_handle = args->handle;
    proc.allocations_.erase(it);

    {
      std::lock_guard<std::mutex> ilk(ipc_mutex_);
      for (auto ipc_it = ipc_store_.begin(); ipc_it != ipc_store_.end();) {
        if (ipc_it->second.source_process_id == freed_process_id &&
            ipc_it->second.source_alloc_handle == freed_handle) {
          if (ipc_it->second.backing_memfd >= 0)
            syscall(SYS_close, ipc_it->second.backing_memfd);
          ipc_it = ipc_store_.erase(ipc_it);
        } else {
          ++ipc_it;
        }
      }
    }
  }
  return 0;
}

int SimulatedDriver::map_memory_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arg);

  std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
  auto it = proc.allocations_.find(args->handle);
  if (it == proc.allocations_.end())
    return -EINVAL;
  auto &alloc = it->second;
  util::Logger::vm([&](auto &os) {
    os << std::format("MAP_MEMORY handle={} gpu_va={:#x} size={} flags={:#x} host_ptr={:#x}",
                      alloc.handle, alloc.gpu_va, alloc.size, alloc.flags,
                      reinterpret_cast<uintptr_t>(alloc.host_ptr));
  });
  if (alloc.host_ptr)
    map_to_gpu(proc, alloc.gpu_va, alloc.host_ptr, alloc.size, pte_mtype_for_flags(alloc.flags));
  args->n_success = args->n_devices;
  return 0;
}

int SimulatedDriver::unmap_memory_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_unmap_memory_from_gpu_args *>(arg);
  std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
  auto it = proc.allocations_.find(args->handle);
  if (it != proc.allocations_.end()) {
    // UNMAP only tears down GPU page-table mappings; the allocation record
    // (and its backing memfd/dmabuf_fd) stays tracked until FREE_MEMORY_OF_GPU
    // releases it. Erasing here would leak those fds and make a later FREE a
    // no-op for this handle.
    auto &alloc = it->second;
    unmap_from_gpu(proc, alloc.gpu_va, alloc.size);
  }
  args->n_success = args->n_devices;
  return 0;
}

int SimulatedDriver::create_queue_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_create_queue_args *>(arg);
  auto *gpu = find_gpu(args->gpu_id);
  if (!gpu || !gpu->soc)
    return -EINVAL;

  std::lock_guard<std::mutex> lk(proc.alloc_mutex_);

  if (!daemon_mode_) {
    map_to_gpu(proc, args->ring_base_address, reinterpret_cast<void *>(args->ring_base_address),
               args->ring_size, amdgpu::Mtype::UC);
    uint64_t rptr_page = args->read_pointer_address & ~0xFFFULL;
    uint64_t wptr_page = args->write_pointer_address & ~0xFFFULL;
    map_to_gpu(proc, rptr_page, reinterpret_cast<void *>(rptr_page), 4096, amdgpu::Mtype::UC);
    if (wptr_page != rptr_page)
      map_to_gpu(proc, wptr_page, reinterpret_cast<void *>(wptr_page), 4096, amdgpu::Mtype::UC);
  }

  uint32_t queue_id = proc.next_queue_id_++;
  uint32_t ord = gpu_ordinal(args->gpu_id);
  auto &gs = proc.gpu(ord);
  uint32_t db_offset;
  if (!gs.free_doorbell_offsets.empty()) {
    db_offset = gs.free_doorbell_offsets.back();
    gs.free_doorbell_offsets.pop_back();
  } else {
    if (gs.doorbell_page_size > 0 &&
        gs.next_doorbell_offset + sizeof(uint64_t) > gs.doorbell_page_size)
      return -ENOSPC;
    db_offset = static_cast<uint32_t>(gs.next_doorbell_offset);
    gs.next_doorbell_offset += sizeof(uint64_t);
  }

  amdgpu::HwQueue hw{};
  hw.process_id = proc.process_id();
  hw.queue_id = queue_id;
  hw.ring_base_va = args->ring_base_address;
  hw.ring_size = args->ring_size;
  hw.read_ptr_va = args->read_pointer_address;
  hw.write_ptr_va = args->write_pointer_address;
  hw.doorbell_offset = db_offset;
  hw.doorbell_base = gs.doorbell_page;
  hw.last_doorbell = ~uint64_t(0);
  hw.host_accessible = true;
  hw.is_sdma = (args->queue_type == 1 /*KFD_IOC_QUEUE_TYPE_SDMA*/ ||
                args->queue_type == 3 /*KFD_IOC_QUEUE_TYPE_SDMA_XGMI*/ ||
                args->queue_type == 4 /*KFD_IOC_QUEUE_TYPE_SDMA_BY_ENG_ID*/);
  // amd_queue_t base: write_pointer_address points to write_dispatch_id.
  if (!hw.is_sdma)
    hw.queue_desc_va = args->write_pointer_address - offsetof(amd_queue_t, write_dispatch_id);
  if (hw.is_sdma && !daemon_mode_) {
    auto *wptr = reinterpret_cast<uint64_t *>(args->write_pointer_address);
    auto *rptr = reinterpret_cast<uint64_t *>(args->read_pointer_address);
    util::Logger::vm("SDMA wptr before init: addr=0x", std::hex, args->write_pointer_address,
                     " val=", std::dec, *wptr, " rptr val=", *rptr);
    *wptr = 0;
    *rptr = 0;
  } else if (hw.is_sdma && daemon_mode_) {
    auto *mem = gpu->soc ? gpu->soc->memory() : nullptr;
    if (mem) {
      mem->write64(args->write_pointer_address, 0, proc.process_id());
      mem->write64(args->read_pointer_address, 0, proc.process_id());
    }
  }
  auto *target_cp = gpu->soc->assign_queue_cp();
  if (!target_cp)
    return -EINVAL;
  target_cp->register_queue(std::move(hw));

  args->queue_id = queue_id;
  args->doorbell_offset = KFD_MMAP_TYPE_DOORBELL | kfd_mmap_gpu_id(gpu->gpu_id) | db_offset;
  proc.active_queue_ids_.push_back(queue_id);
  proc.queue_doorbell_map_[queue_id] = {ord, db_offset};
  return 0;
}

int SimulatedDriver::update_queue_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_update_queue_args *>(arg);
  for (auto &g : gpus_)
    if (g.soc)
      g.soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
        cp->update_queue(args->queue_id, proc.process_id(), args->ring_base_address,
                         args->ring_size);
      });
  return 0;
}

int SimulatedDriver::destroy_queue_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_destroy_queue_args *>(arg);
  for (auto &g : gpus_)
    if (g.soc)
      g.soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
        cp->unregister_queue(args->queue_id, proc.process_id());
      });
  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    std::erase(proc.active_queue_ids_, args->queue_id);
    auto it = proc.queue_doorbell_map_.find(args->queue_id);
    if (it != proc.queue_doorbell_map_.end()) {
      auto &gs = proc.gpu(it->second.gpu_ordinal);
      gs.free_doorbell_offsets.push_back(it->second.doorbell_offset);
      proc.queue_doorbell_map_.erase(it);
    }
  }
  // Real CP sends EOP interrupt when queue is deactivated; KFD broadcasts to
  // all type-0 events. This wakes ROCR's signal threads blocked on queue events.
  proc.event_state_.signal_interrupt(0);
  return 0;
}

int SimulatedDriver::set_memory_policy_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_set_memory_policy_args *>(arg);
  if (!find_gpu(args->gpu_id))
    return -EINVAL;
  KfdProcess::MemoryPolicy policy{};
  policy.alternate_base = args->alternate_aperture_base;
  policy.alternate_size = args->alternate_aperture_size;
  policy.default_policy = args->default_policy;
  policy.alternate_policy = args->alternate_policy;
  proc.memory_policies_[args->gpu_id] = policy;
  return 0;
}

int SimulatedDriver::import_dmabuf_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_import_dmabuf_args *>(arg);
  if (!find_gpu(args->gpu_id))
    return -EINVAL;

  struct stat st {};
  if (fstat(args->dmabuf_fd, &st) != 0)
    return -errno;
  uint64_t size = static_cast<uint64_t>(st.st_size);

  int dupfd = fcntl(args->dmabuf_fd, F_DUPFD_CLOEXEC, 0);
  if (dupfd < 0)
    return -errno;

  uint64_t handle;
  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    handle = proc.next_handle_++;
    KfdProcess::GpuAllocation alloc{};
    alloc.gpu_va = args->va_addr;
    alloc.size = size;
    alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
    alloc.handle = handle;
    alloc.user_va = true;
    alloc.imported = true;
    alloc.dmabuf_fd = dupfd;
    alloc.host_ptr = reinterpret_cast<void *>(args->va_addr);
    proc.allocations_[handle] = alloc;
  }

  if (args->va_addr)
    map_to_gpu(proc, args->va_addr, reinterpret_cast<void *>(args->va_addr), size,
               amdgpu::Mtype::UC);

  KfdProcess::ImportedDmabuf info{};
  info.handle = handle;
  info.fd = dupfd;
  info.size = size;
  info.va = args->va_addr;
  info.gpu_id = args->gpu_id;
  proc.imported_dmabufs_[handle] = info;
  proc.fd_to_import_handle_[dupfd] = handle;

  args->handle = handle;
  return 0;
}

int SimulatedDriver::export_dmabuf_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_export_dmabuf_args *>(arg);

  std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
  auto it = proc.allocations_.find(args->handle);
  if (it == proc.allocations_.end())
    return -EINVAL;
  const auto &alloc = it->second;
  if (alloc.memfd < 0)
    return -EINVAL;
  int dupfd = fcntl(alloc.memfd, F_DUPFD_CLOEXEC, 0);
  if (dupfd < 0)
    return -errno;
  args->dmabuf_fd = dupfd;
  return 0;
}

int SimulatedDriver::ipc_export_handle_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_ipc_export_handle_args *>(arg);

  uint64_t alloc_size = 0;
  uint32_t alloc_flags = 0;
  uint32_t alloc_gpu_id = 0;
  int dup_fd = -1;

  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    auto it = proc.allocations_.find(args->handle);
    if (it == proc.allocations_.end())
      return -EINVAL;
    auto &alloc = it->second;

    if (alloc.memfd < 0 && alloc.host_ptr) {
      int promoted_fd = static_cast<int>(
          syscall(SYS_memfd_create, "rocjitsu_ipc_promote", MFD_CLOEXEC | MFD_ALLOW_SEALING));
      if (promoted_fd < 0)
        return -errno;
      if (ftruncate(promoted_fd, static_cast<off_t>(alloc.size)) != 0) {
        syscall(SYS_close, promoted_fd);
        return -errno;
      }
      auto *new_host_ptr =
          safe_mmap(nullptr, alloc.size, PROT_READ | PROT_WRITE, MAP_SHARED, promoted_fd, 0);
      if (new_host_ptr == MAP_FAILED) {
        syscall(SYS_close, promoted_fd);
        return -ENOMEM;
      }
      std::memcpy(new_host_ptr, alloc.host_ptr, alloc.size);

      if (alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR) {
        util::Logger::vm("ipc_export: promoting USERPTR to memfd-backed (snapshot copy, not "
                         "true sharing)");
      }

      {
        std::unique_lock ptlk(proc.page_table_mutex_);
        auto *old_base = static_cast<uint8_t *>(alloc.host_ptr);
        auto *new_base = static_cast<uint8_t *>(new_host_ptr);
        for (size_t off = 0; off < alloc.size; off += KfdProcess::kPageSize) {
          uint64_t page_num = (alloc.gpu_va + off) >> KfdProcess::kPageShift;
          auto pt_it = proc.page_table_.find(page_num);
          if (pt_it != proc.page_table_.end() && pt_it->second.host_ptr == old_base + off)
            pt_it->second.host_ptr = new_base + off;
        }
      }

      if (!(alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR))
        syscall(SYS_munmap, alloc.host_ptr, alloc.size);

      alloc.host_ptr = new_host_ptr;
      alloc.memfd = promoted_fd;
      {
        std::lock_guard<std::mutex> flk(owned_fds_mutex_);
        owned_fds_.insert(promoted_fd);
      }
    } else if (alloc.memfd < 0) {
      int new_fd = static_cast<int>(
          syscall(SYS_memfd_create, "rocjitsu_ipc_lazy", MFD_CLOEXEC | MFD_ALLOW_SEALING));
      if (new_fd < 0)
        return -errno;
      if (ftruncate(new_fd, static_cast<off_t>(alloc.size)) != 0) {
        syscall(SYS_close, new_fd);
        return -errno;
      }
      alloc.memfd = new_fd;
      {
        std::lock_guard<std::mutex> flk(owned_fds_mutex_);
        owned_fds_.insert(new_fd);
      }
    }

    // Upgrade the exporter's PTE mtype to CC (cache coherent) so that
    // the local GPU sees writes from the importing GPU.  On real hardware
    // xGMI snoops handle this; in the simulator CC forces L2 invalidate
    // before every refetch, emulating the cross-GPU coherence protocol.
    {
      std::unique_lock ptlk(proc.page_table_mutex_);
      for (size_t off = 0; off < alloc.size; off += KfdProcess::kPageSize) {
        uint64_t page_num = (alloc.gpu_va + off) >> KfdProcess::kPageShift;
        auto pt_it = proc.page_table_.find(page_num);
        if (pt_it != proc.page_table_.end())
          pt_it->second.mtype = amdgpu::Mtype::CC;
      }
    }

    alloc_size = alloc.size;
    alloc_flags = alloc.flags;
    alloc_gpu_id = alloc.gpu_id;
    dup_fd = fcntl(alloc.memfd, F_DUPFD_CLOEXEC, 0);
  }

  if (dup_fd < 0)
    return -errno;

  IpcHandleKey key{};
  if (getrandom(key.words, sizeof(key.words), 0) != sizeof(key.words)) {
    syscall(SYS_close, dup_fd);
    return -errno;
  }

  IpcObject obj{};
  std::memcpy(obj.share_handle, key.words, sizeof(key.words));
  obj.backing_memfd = dup_fd;
  obj.allocation_size = alloc_size;
  obj.allocation_flags = alloc_flags;
  obj.source_gpu_id = alloc_gpu_id;
  obj.source_process_id = proc.process_id();
  obj.source_alloc_handle = args->handle;

  {
    std::lock_guard<std::mutex> lk(ipc_mutex_);
    ipc_store_[key] = obj;
  }

  std::memcpy(args->share_handle, key.words, sizeof(key.words));
  util::Logger::vm("ipc_export: handle=", args->handle, " size=", alloc_size,
                   " gpu_id=", alloc_gpu_id);
  return 0;
}

int SimulatedDriver::ipc_import_handle_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_ipc_import_handle_args *>(arg);

  IpcHandleKey key{};
  std::memcpy(key.words, args->share_handle, sizeof(key.words));

  int dup_fd = -1;
  uint64_t alloc_size = 0;
  uint32_t alloc_flags = 0;
  uint32_t source_gpu_id = 0;

  {
    std::lock_guard<std::mutex> lk(ipc_mutex_);
    auto it = ipc_store_.find(key);
    if (it == ipc_store_.end())
      return -EINVAL;
    alloc_size = it->second.allocation_size;
    alloc_flags = it->second.allocation_flags;
    source_gpu_id = it->second.source_gpu_id;
    dup_fd = fcntl(it->second.backing_memfd, F_DUPFD_CLOEXEC, 0);
  }

  if (args->gpu_id != 0 && args->gpu_id != source_gpu_id) {
    util::Logger::vm("ipc_import: gpu_id mismatch: requested=", args->gpu_id,
                     " source=", source_gpu_id);
    return -EINVAL;
  }

  if (dup_fd < 0)
    return -errno;

  {
    std::lock_guard<std::mutex> flk(owned_fds_mutex_);
    owned_fds_.insert(dup_fd);
  }

  auto *host_ptr = safe_mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, dup_fd, 0);
  if (host_ptr == MAP_FAILED) {
    {
      std::lock_guard<std::mutex> flk(owned_fds_mutex_);
      owned_fds_.erase(dup_fd);
    }
    syscall(SYS_close, dup_fd);
    return -ENOMEM;
  }

  uint64_t gpu_va;
  uint64_t handle;
  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    if (args->va_addr != 0)
      gpu_va = args->va_addr;
    else {
      gpu_va = proc.next_gpu_va_;
      proc.next_gpu_va_ += (alloc_size + 0xFFF) & ~0xFFFULL;
    }
    handle = proc.next_handle_++;

    KfdProcess::GpuAllocation alloc{};
    alloc.gpu_va = gpu_va;
    alloc.size = alloc_size;
    alloc.flags = alloc_flags;
    alloc.handle = handle;
    alloc.host_ptr = host_ptr;
    alloc.memfd = dup_fd;
    alloc.gpu_id = source_gpu_id;
    alloc.imported = true;
    proc.allocations_[handle] = alloc;
  }

  // IPC-imported memory uses CC (cache coherent) mtype to emulate the
  // cross-GPU coherence that real hardware provides via xGMI snoops.
  // Without this, the importing GPU's L2 cache serves stale data when
  // the exporting GPU writes to the shared buffer.
  map_to_gpu(proc, gpu_va, host_ptr, alloc_size, amdgpu::Mtype::CC);

  args->handle = handle;
  args->mmap_offset = handle << 12;
  args->flags = alloc_flags;

  util::Logger::vm("ipc_import: handle=", handle, " gpu_va=0x", std::hex, gpu_va,
                   " size=", std::dec, alloc_size, " gpu_id=", source_gpu_id);
  return 0;
}

int SimulatedDriver::get_dmabuf_info_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_get_dmabuf_info_args *>(arg);
  uint64_t size = 0;
  uint32_t gpu_id = gpus_.empty() ? 0 : gpus_[0].gpu_id;

  bool found = false;
  for (const auto &[handle, info] : proc.imported_dmabufs_) {
    [[maybe_unused]] auto &_ = handle;
    if (info.fd >= 0 && static_cast<uint32_t>(info.fd) == args->dmabuf_fd) {
      size = info.size;
      gpu_id = info.gpu_id;
      found = true;
      break;
    }
  }

  if (!found) {
    struct stat st {};
    if (fstat(args->dmabuf_fd, &st) != 0)
      return -errno;
    size = static_cast<uint64_t>(st.st_size);
  }

  args->size = size;
  args->gpu_id = gpu_id;
  args->flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT;
  // metadata_ptr is a client-process address that cannot be dereferenced in
  // daemon mode. ROCR currently queries with metadata_size == 0; reject
  // metadata-bearing calls rather than risk a cross-process pointer deref.
  if (args->metadata_size > 0 && daemon_mode_)
    return -EINVAL;
  if (args->metadata_ptr && args->metadata_size && !daemon_mode_) {
    std::memset(reinterpret_cast<void *>(args->metadata_ptr), 0,
                static_cast<size_t>(args->metadata_size));
  }
  args->metadata_size = 0;
  return 0;
}

int SimulatedDriver::svm_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_svm_args *>(arg);
  auto *attrs = reinterpret_cast<kfd_ioctl_svm_attribute *>(args + 1);

  if (args->op == KFD_IOCTL_SVM_OP_SET_ATTR) {
    KfdProcess::SvmRange range{};
    range.size = args->size;
    for (uint32_t i = 0; i < args->nattr; ++i)
      range.attributes[attrs[i].type] = attrs[i].value;
    proc.svm_ranges_[args->start_addr] = std::move(range);
    return 0;
  }

  if (args->op == KFD_IOCTL_SVM_OP_GET_ATTR) {
    auto it = proc.svm_ranges_.find(args->start_addr);
    for (uint32_t i = 0; i < args->nattr; ++i) {
      uint32_t type = attrs[i].type;
      uint32_t value = 0;
      if (it != proc.svm_ranges_.end()) {
        if (auto vit = it->second.attributes.find(type); vit != it->second.attributes.end())
          value = vit->second;
      }
      switch (type) {
      case KFD_IOCTL_SVM_ATTR_PREFERRED_LOC:
      case KFD_IOCTL_SVM_ATTR_PREFETCH_LOC:
        attrs[i].value = value ? value : KFD_IOCTL_SVM_LOCATION_UNDEFINED;
        break;
      default:
        attrs[i].value = value;
        break;
      }
    }
    return 0;
  }

  return -EINVAL;
}

int SimulatedDriver::runtime_enable_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_runtime_enable_args *>(arg);

  std::lock_guard<std::mutex> lock(proc.runtime_mutex_);

  if (args->mode_mask & KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK) {
    if (proc.runtime_state_.pending)
      return -EBUSY;
    bool has_queues = [&] {
      std::lock_guard<std::mutex> alock(proc.alloc_mutex_);
      return !proc.active_queue_ids_.empty();
    }();
    if (!proc.runtime_state_.enabled && has_queues)
      return -EEXIST;
    proc.runtime_state_.enabled = true;
    proc.runtime_state_.pending = false;
    proc.runtime_state_.mode_mask = args->mode_mask;
    proc.runtime_state_.capabilities_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
    proc.runtime_state_.r_debug = args->r_debug;
    args->capabilities_mask = proc.runtime_state_.capabilities_mask;
    return 0;
  }

  proc.runtime_state_ = KfdProcess::RuntimeState{};
  args->capabilities_mask = 0;
  return 0;
}

int SimulatedDriver::set_xnack_mode_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_set_xnack_mode_args *>(arg);
  args->xnack_enabled = 0;
  return 0;
}

bool SimulatedDriver::owns_fd(int fd) const {
  if (fd < 0)
    return false;
  std::lock_guard<std::mutex> lock(owned_fds_mutex_);
  return owned_fds_.contains(fd);
}

void SimulatedDriver::init_reserved_fd_range() {
  struct rlimit rl {};
  getrlimit(RLIMIT_NOFILE, &rl);
  reserved_fd_base_ = static_cast<int>(rl.rlim_cur) - kReservedFdCount;
  next_reserved_fd_ = reserved_fd_base_;
}

int SimulatedDriver::claim_fd(int real_fd) {
  if (reserved_fd_base_ == 0)
    init_reserved_fd_range();
  int vfd = next_reserved_fd_++;
  assert(vfd < reserved_fd_base_ + kReservedFdCount && "reserved fd range exhausted");
  syscall(SYS_dup2, real_fd, vfd);
  syscall(SYS_close, real_fd);
  return vfd;
}

bool SimulatedDriver::owns_reserved_fd(int fd) const {
  return reserved_fd_base_ > 0 && fd >= reserved_fd_base_ &&
         fd < reserved_fd_base_ + kReservedFdCount;
}

int SimulatedDriver::get_mmap_memfd(off_t offset) const {
  return get_mmap_memfd(local_process_id_, offset);
}

int SimulatedDriver::get_mmap_memfd(uint32_t process_id, off_t offset) const {
  auto p = find_process(process_id);
  if (!p)
    return -1;
  return dispatch_get_mmap_memfd(*p, offset);
}

int SimulatedDriver::dispatch_get_mmap_memfd(KfdProcess &proc, off_t offset) const {
  uint64_t type = static_cast<uint64_t>(offset) & KFD_MMAP_TYPE_MASK;

  if (type == KFD_MMAP_TYPE_EVENTS)
    return proc.event_state_.memfd;

  if (type == KFD_MMAP_TYPE_DOORBELL) {
    uint64_t encoded_gpu =
        (static_cast<uint64_t>(offset) & ~KFD_MMAP_TYPE_MASK) >> KFD_MMAP_GPU_ID_SHIFT;
    uint32_t db_gpu_id = static_cast<uint32_t>(encoded_gpu);
    std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
    for (auto &[handle, alloc] : proc.allocations_) {
      if ((alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL) && alloc.gpu_id == db_gpu_id) {
        util::Logger::cp("MEMFD_LOOKUP: pid=", proc.process_id(), " DOORBELL match handle=", handle,
                         " gpu_id=", db_gpu_id, " memfd=", alloc.memfd);
        return alloc.memfd;
      }
    }
    util::Logger::cp("MEMFD_LOOKUP: pid=", proc.process_id(),
                     " DOORBELL NO MATCH gpu_id=", db_gpu_id,
                     " allocations=", proc.allocations_.size());
    return -1;
  }

  uint64_t handle = static_cast<uint64_t>(offset) >> 12;
  std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
  auto it = proc.allocations_.find(handle);
  if (it != proc.allocations_.end())
    return it->second.memfd;

  return -1;
}

} // namespace rocjitsu
