// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/simulated_driver.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/cwsr.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/kfd_topology.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

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
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <format>
#include <fstream>
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

/// @brief Read the TracerPid field from /proc/<pid>/status.
/// @returns The pid of the process ptrace-attached to @p pid, or 0 if none (or
/// the target is gone). This exposes the real kernel ptrace relationship that
/// the KFD debug ioctl authorizes against (kfd_ioctl_set_debug_trap uses
/// ptrace_parent(target->lead_thread) == current). Both the debugger and the
/// target are real Linux processes under the emulator, so consulting the live
/// /proc state models the exact relationship the kernel checks — no mock.
pid_t tracer_pid_of(pid_t pid) {
  if (pid <= 0)
    return 0;
  std::ifstream status("/proc/" + std::to_string(pid) + "/status");
  std::string line;
  constexpr std::string_view kKey = "TracerPid:";
  while (std::getline(status, line)) {
    if (std::string_view(line).substr(0, kKey.size()) == kKey)
      return static_cast<pid_t>(std::strtol(line.c_str() + kKey.size(), nullptr, 10));
  }
  return 0;
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
  gpu.capability = dev.capability;
  gpu.capability2 = dev.capability2;
  gpu.debug_prop = dev.debug_prop;
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
  gpu_infos_ = {gpu};
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
    gpu.capability = dev.capability;
    gpu.capability2 = dev.capability2;
    gpu.debug_prop = dev.debug_prop;
    gpu.num_xcc = num_xcc;
    infos.push_back(gpu);
  }
  if (infos.empty())
    return;
  for (size_t i = 0; i < infos.size() && i < gpus_.size(); ++i)
    gpus_[i].gpu_id = infos[i].gpu_id;
  gpu_infos_ = infos;
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
    processes_[local_process_id_]->retain_open();
    return fd_;
  }
  uint32_t pid = next_process_id_++;
  auto proc = std::make_shared<KfdProcess>(pid, static_cast<uint32_t>(gpus_.size()));
  // Record the caller's Linux pid so AMDKFD_IOC_DBG_TRAP can resolve this
  // process as a self-debug target (args->pid == getpid()).
  proc->set_client_pid(static_cast<pid_t>(getpid()));
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
      for (auto *cu : cp->compute_units()) {
        cu->set_trap_handler(
            [this](amdgpu::Wavefront &wf, uint32_t trap_id) { return on_wave_trap(wf, trap_id); });
        cu->set_single_step_handler(
            [this](amdgpu::Wavefront &wf) { return on_wave_single_step_complete(wf); });
        cu->set_watchpoint_handler([this](amdgpu::Wavefront &wf, uint64_t addr, uint32_t bytes,
                                          bool is_write, bool is_atomic) {
          return on_wave_watchpoint(wf, addr, bytes, is_write, is_atomic);
        });
        cu->set_illegal_inst_handler(
            [this](amdgpu::Wavefront &wf) { return on_wave_illegal_instruction(wf); });
        cu->set_memory_violation_handler(
            [this](amdgpu::Wavefront &wf, uint64_t addr, bool is_write) {
              return on_wave_memory_violation(wf, addr, is_write);
            });
      }
    });
    g.cps_initialized = true;
  }

  return fd_;
}

void SimulatedDriver::set_process_client_pid(uint32_t process_id, pid_t client_pid) {
  std::lock_guard<std::mutex> lk(process_mutex_);
  auto it = processes_.find(process_id);
  if (it != processes_.end()) {
    it->second->set_client_pid(client_pid);
    for (auto &g : gpus_) {
      if (auto *mem = g.soc ? g.soc->memory() : nullptr)
        mem->set_process_client_pid(process_id, client_pid);
    }
  }
}

uint32_t SimulatedDriver::open_process(pid_t client_pid) {
  if (fd_ < 0) {
    fd_ = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_kfd", 0));
    if (fd_ < 0)
      return 0;
  }

  uint32_t pid;
  {
    std::lock_guard<std::mutex> lk(process_mutex_);
    // Client-PID process reuse (and its matching retain) is a daemon-mode
    // feature: multiple client opens of the same PID share one process and
    // balance against multiple close()/release_open() calls. Gating reuse on
    // daemon_mode_ keeps it symmetric with close() — outside daemon mode every
    // open creates a fresh process so the first close cannot tear down a
    // still-referenced one.
    if (daemon_mode_ && client_pid > 0) {
      for (auto &[id, proc] : processes_) {
        if (proc->client_pid() == client_pid) {
          proc->retain_open();
          return id;
        }
      }
    }
    pid = next_process_id_++;
    auto proc = std::make_shared<KfdProcess>(pid, static_cast<uint32_t>(gpus_.size()));
    if (client_pid > 0)
      proc->set_client_pid(client_pid);
    proc->event_state_.reset();
    for (auto &g : gpus_) {
      if (auto *mem = g.soc ? g.soc->memory() : nullptr) {
        mem->register_process(pid, &proc->page_table_, &proc->page_table_mutex_);
        if (client_pid > 0)
          mem->set_process_client_pid(pid, client_pid);
      }
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
      for (auto *cu : cp->compute_units()) {
        cu->set_trap_handler(
            [this](amdgpu::Wavefront &wf, uint32_t trap_id) { return on_wave_trap(wf, trap_id); });
        cu->set_single_step_handler(
            [this](amdgpu::Wavefront &wf) { return on_wave_single_step_complete(wf); });
        cu->set_watchpoint_handler([this](amdgpu::Wavefront &wf, uint64_t addr, uint32_t bytes,
                                          bool is_write, bool is_atomic) {
          return on_wave_watchpoint(wf, addr, bytes, is_write, is_atomic);
        });
        cu->set_illegal_inst_handler(
            [this](amdgpu::Wavefront &wf) { return on_wave_illegal_instruction(wf); });
        cu->set_memory_violation_handler(
            [this](amdgpu::Wavefront &wf, uint64_t addr, bool is_write) {
              return on_wave_memory_violation(wf, addr, is_write);
            });
      }
    });
    g.cps_initialized = true;
  }

  return pid;
}

void SimulatedDriver::retain_local_open() {
  std::lock_guard<std::mutex> lk(process_mutex_);
  if (local_process_id_ == 0)
    return;
  auto it = processes_.find(local_process_id_);
  if (it != processes_.end())
    it->second->retain_open();
}

uint32_t SimulatedDriver::local_open_ref_count() const {
  std::lock_guard<std::mutex> lk(process_mutex_);
  if (local_process_id_ == 0)
    return 0;
  auto it = processes_.find(local_process_id_);
  return it != processes_.end() ? it->second->open_ref_count() : 0;
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
    if (!it->second->release_open())
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
  case AMDKFD_IOC_DBG_TRAP:
    return "DBG_TRAP";
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
  case AMDKFD_IOC_DBG_TRAP:
    return debug_trap_ioctl(proc, arg);
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

kfd_process_device_apertures SimulatedDriver::gpu_apertures(uint32_t ordinal) const {
  kfd_process_device_apertures a = default_apertures_;
  const uint64_t offset = static_cast<uint64_t>(ordinal) * kApertureStride;
  a.lds_base += offset;
  a.lds_limit += offset;
  a.scratch_base += offset;
  a.scratch_limit += offset;
  a.gpu_id = ordinal < gpus_.size() ? gpus_[ordinal].gpu_id : 0;
  return a;
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
  for (uint32_t i = 0; i < n && i < args->num_of_nodes; ++i)
    apertures[i] = gpu_apertures(i);

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
    if (alloc.host_ptr)
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

  // Record debug-relevant queue geometry for GET_QUEUE_SNAPSHOT. rocm-dbgapi
  // reads ctx_save_restore_address/size to locate the queue's CWSR area and the
  // ring pointers to correlate dispatches (kfd_debug.c: get_queue_snapshot).
  KfdProcess::QueueSnapshotInfo qinfo{};
  qinfo.ring_base_address = args->ring_base_address;
  qinfo.write_pointer_address = args->write_pointer_address;
  qinfo.read_pointer_address = args->read_pointer_address;
  qinfo.ctx_save_restore_address = args->ctx_save_restore_address;
  qinfo.ctx_save_restore_area_size = args->ctx_save_restore_size;
  qinfo.ring_size = args->ring_size;
  qinfo.queue_type = args->queue_type;
  qinfo.gpu_id = args->gpu_id;
  // A freshly created queue carries the queue_new exception so a debugger that
  // is (or later becomes) attached learns about it before any wave event on it
  // (kfd_process_queue_manager.c: init_user_queue sets
  // exception_status = KFD_EC_MASK(EC_QUEUE_NEW)). rocm-dbgapi requires a queue
  // to be reported as new before it will accept queue/wave events for it.
  qinfo.exception_status = KFD_EC_MASK(EC_QUEUE_NEW);
  proc.queue_snapshot_map_[queue_id] = qinfo;
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
    proc.queue_snapshot_map_.erase(args->queue_id);
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

  const bool enabling = (args->mode_mask & KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK) != 0;
  {
    std::lock_guard<std::mutex> lock(proc.runtime_mutex_);
    if (enabling) {
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
    } else {
      proc.runtime_state_ = KfdProcess::RuntimeState{};
      args->capabilities_mask = 0;
    }
  }

  // Under a debugger, RUNTIME_ENABLE is a handshake: raise EC_PROCESS_RUNTIME so
  // the debugger reads r_debug and starts tracking code objects, then block
  // until it acknowledges. Done outside runtime_mutex_ so the debugger's
  // concurrent DBG_TRAP ioctls (which read runtime_state_) do not deadlock.
  if (enabling)
    runtime_enable_debugger_handshake(proc.client_pid());

  return 0;
}

std::shared_ptr<KfdProcess> SimulatedDriver::find_process_by_client_pid(pid_t pid) const {
  if (pid == 0)
    return nullptr;
  std::lock_guard<std::mutex> lk(process_mutex_);
  for (auto &[id, proc] : processes_)
    if (proc->client_pid() == pid)
      return proc;
  return nullptr;
}

namespace {

// Build the debugger's serialized view of one stopped wave.
kmd::CwsrWaveState build_cwsr_wave_state(amdgpu::Wavefront &wf) {
  kmd::CwsrWaveState w;
  const uint32_t raw_status = wf.status_raw();
  w.pc = wf.pc;
  w.exec = wf.exec();
  w.vcc = wf.vcc();
  w.m0 = wf.m0();
  w.mode = wf.mode_raw();
  w.trapsts = wf.trapsts();
  // On stop the trap handler sets STATUS.HALT and records the previous value in
  // TTMP6 (see rocm-dbgapi wave_set_state); reproduce that so a later resume can
  // restore the wave's real halt state.
  w.saved_status_halt = (raw_status >> 13) & 1u;
  w.status = raw_status | (1u << 13);
  w.wave_stopped = true;
  w.trap_id = wf.trap_id();
  // TTMP4:5 is rocm-dbgapi's private wave-id slot. A wave the debugger has not
  // seen yet presents the "undefined" id (0, the initial debug_wave_id); dbgapi
  // then creates the wave and writes its own id, which resume-time reload
  // captures back into debug_wave_id so re-serialization preserves it (otherwise
  // dbgapi fails with "wave not found in queue", rocdbgapi queue.cpp).
  w.wave_id = wf.debug_wave_id();
  w.group_ids = {wf.wg_id(), 0u, 0u};
  w.wave_in_group = 0;
  w.queue_packet_id = wf.dispatch_id() & 0x1FFFFFFu;
  w.num_sgprs = wf.num_sgprs();
  w.num_vgprs = wf.num_vgprs();

  const auto &cu = wf.cu();
  const uint32_t sbase = wf.sgpr_alloc().base;
  w.sgprs.resize(w.num_sgprs);
  for (uint32_t s = 0; s < w.num_sgprs; ++s)
    w.sgprs[s] = cu.read_sgpr(sbase + s);
  const uint32_t vbase = wf.vgpr_alloc().base;
  w.vgprs.resize(static_cast<size_t>(w.num_vgprs) * 64);
  for (uint32_t r = 0; r < w.num_vgprs; ++r)
    for (uint32_t lane = 0; lane < 64; ++lane)
      w.vgprs[static_cast<size_t>(r) * 64 + lane] = cu.read_vgpr(vbase + r, lane);
  return w;
}

} // namespace

void SimulatedDriver::notify_debugger(int dbg_fd) {
  // Wake a debugger blocked on its notifier pipe. In local (in-process) mode the
  // fd is invalid; the daemon transport delivers the wake across the RPC
  // boundary. Best effort: an error just means no debugger is listening.
  if (dbg_fd < 0)
    return;
  const char byte = 1;
  ssize_t n = ::write(dbg_fd, &byte, 1);
  (void)n;
}

void SimulatedDriver::raise_debug_event(const std::shared_ptr<KfdProcess> &proc, uint32_t queue_id,
                                        uint32_t gpu_id, uint64_t exception_mask) {
  if (!proc)
    return;
  // Accumulate the raised exception on the queue snapshot (so GET_QUEUE_SNAPSHOT
  // reports it) and fold in any still-pending queue_new. rocm-dbgapi only
  // accepts a queue/wave event once the queue has been reported as new; the
  // kernel reports queue_new together with the subscribed exception that
  // accompanies it (kfd_debug.c: kfd_dbg_ev_query_debug_event). Locks are taken
  // sequentially (never nested) to avoid ordering hazards with the query path.
  uint64_t report_mask = exception_mask;
  {
    std::lock_guard<std::mutex> plk(proc->alloc_mutex_);
    auto qit = proc->queue_snapshot_map_.find(queue_id);
    if (qit != proc->queue_snapshot_map_.end()) {
      qit->second.exception_status |= exception_mask;
      report_mask |= (qit->second.exception_status & KFD_EC_MASK(EC_QUEUE_NEW));
    }
  }
  {
    std::lock_guard<std::mutex> lk(debug_events_mutex_);
    auto &qe = debug_events_[proc->client_pid()][queue_id];
    qe.gpu_id = gpu_id;
    qe.mask |= report_mask;
  }
}

void SimulatedDriver::serialize_queue_debug_waves(uint32_t process_id, uint32_t queue_id,
                                                  uint32_t gpu_id, uint64_t ctx_base,
                                                  uint32_t ctx_size) {
  auto *gpu = find_gpu(gpu_id);
  if (!gpu || !gpu->soc || ctx_base == 0)
    return;

  std::vector<kmd::CwsrWaveState> waves;
  gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
    for (auto *cu : cp->compute_units()) {
      for (uint32_t i = 0; i < cu->num_wf_slots(); ++i) {
        auto *w = cu->wf(i);
        if (w->debug_halted() && w->process_id() == process_id && w->queue_id() == queue_id)
          waves.push_back(build_cwsr_wave_state(*w));
      }
    }
  });
  if (waves.empty())
    return;

  auto *mem = gpu->soc->memory();
  kmd::serialize_queue_cwsr(ctx_base, ctx_size, waves,
                            [&](uint64_t va, uint32_t val) { mem->write32(va, val, process_id); });
}

bool SimulatedDriver::on_wave_trap(amdgpu::Wavefront &wf, uint32_t trap_id) {
  const uint32_t process_id = wf.process_id();
  const uint32_t queue_id = wf.queue_id();

  auto proc = find_process(process_id);
  if (!proc)
    return false;
  const pid_t target_pid = proc->client_pid();

  // Is this process being debugged?
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto sit = debug_sessions_.find(target_pid);
    if (sit == debug_sessions_.end() || !sit->second.enabled)
      return false;
  }

  // Locate the queue's context-save-restore area.
  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto qit = proc->queue_snapshot_map_.find(queue_id);
    if (qit == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = qit->second.ctx_save_restore_address;
    ctx_size = qit->second.ctx_save_restore_area_size;
    gpu_id = qit->second.gpu_id;
  }
  if (ctx_base == 0)
    return false; // no CWSR area advertised: cannot expose the wave

  // Stop the wave (models the trap handler entry). The debugger id in TTMP4:5 is
  // owned by rocm-dbgapi; it stays 0 (undefined) until dbgapi assigns one, which
  // the resume-time reload captures into debug_wave_id.
  wf.debug_trap(trap_id);

  report_wave_stopped(proc, queue_id, gpu_id, ctx_base, ctx_size);
  return true;
}

void SimulatedDriver::report_wave_stopped(const std::shared_ptr<KfdProcess> &proc,
                                          uint32_t queue_id, uint32_t gpu_id, uint64_t ctx_base,
                                          uint32_t ctx_size, uint64_t exception_mask) {
  const pid_t target_pid = proc->client_pid();
  // Serialize this queue's stopped waves, raise the wave exception, and wake the
  // debugger. Shared by the s_trap, single-step, watchpoint and illegal-
  // instruction paths; the default exception (0) is a wave trap.
  if (exception_mask == 0)
    exception_mask = KFD_EC_MASK(EC_QUEUE_WAVE_TRAP);
  serialize_queue_debug_waves(proc->process_id(), queue_id, gpu_id, ctx_base, ctx_size);
  raise_debug_event(proc, queue_id, gpu_id, exception_mask);

  int dbg_fd = -1;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto sit = debug_sessions_.find(target_pid);
    if (sit != debug_sessions_.end())
      dbg_fd = sit->second.dbg_fd;
  }
  notify_debugger(dbg_fd);
}

bool SimulatedDriver::on_wave_single_step_complete(amdgpu::Wavefront &wf) {
  // The engine ran exactly one instruction for a wave rocm-dbgapi put in
  // single-step mode (MODE.debug_en=1). Re-stop it and report, exactly as a
  // trap would: the debugger inspects MODE.debug_en to recognize the step.
  const uint32_t process_id = wf.process_id();
  const uint32_t queue_id = wf.queue_id();
  auto proc = find_process(process_id);
  if (!proc)
    return false;
  const pid_t target_pid = proc->client_pid();
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto sit = debug_sessions_.find(target_pid);
    if (sit == debug_sessions_.end() || !sit->second.enabled)
      return false;
  }
  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto qit = proc->queue_snapshot_map_.find(queue_id);
    if (qit == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = qit->second.ctx_save_restore_address;
    ctx_size = qit->second.ctx_save_restore_area_size;
    gpu_id = qit->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;

  wf.set_debug_single_step(false);
  // Re-stop with trap id 0: a single-step stop is not a breakpoint, so it must
  // not carry the s_trap trap id (1). rocm-dbgapi reads TTMP6's saved trap id to
  // recognize breakpoints and apply the -4 PC adjust; trap id 0 means "no trap",
  // so the stepped PC is reported as-is (rocdbgapi architecture.cpp trap_id()).
  wf.debug_trap(0);
  // Signal single-step completion the way gfx9.4 (MI300/MI350) hardware does:
  // raise TRAPSTS.TRAP_AFTER_INST (bit 25). rocm-dbgapi's wave_get_state maps
  // this exact bit to WAVE_STOP_REASON_SINGLE_STEP (rocdbgapi
  // amdgcn_architecture_t::wave_get_state, architecture.cpp:1544, using
  // gfx9_4_architecture_t::sq_wave_trapsts_trap_after_inst_mask = 1u << 25 at
  // architecture.cpp:3782). It is also what dbgapi itself sets after emulating a
  // stepped instruction (simulate_instruction_fixup, architecture.cpp:4044).
  // Without this bit the debugger sees a stopped wave with no stop reason,
  // concludes the step was spurious, and re-resumes the wave in an unbounded
  // loop that runs the whole kernel. dbgapi clears the bit on the next resume
  // via clear_stop_reasons (architecture.cpp:1418), so it never accumulates.
  constexpr uint32_t kSqWaveTrapstsTrapAfterInstMask = 1u << 25; // gfx9.4 TRAPSTS.TRAP_AFTER_INST
  wf.set_trapsts(wf.trapsts() | kSqWaveTrapstsTrapAfterInstMask);
  report_wave_stopped(proc, queue_id, gpu_id, ctx_base, ctx_size);
  return true;
}

bool SimulatedDriver::on_wave_watchpoint(amdgpu::Wavefront &wf, uint64_t addr, uint32_t bytes,
                                         bool is_write, bool is_atomic) {
  // gfx9 address-watch matching. rocm-dbgapi programs up to four TCP_WATCH slots
  // (SET_NODE_ADDRESS_WATCH) with an address, a compared-bit mask, and an access
  // mode. Hardware traps a wave whose access hits a watched, in-range address of
  // the matching mode and sets TRAPSTS.addr_watch<slot>; rocm-dbgapi reads that
  // bit back (architecture.cpp signaled_exceptions / triggered_watchpoints) to
  // report WAVE_STOP_REASON_WATCHPOINT for the watchpoint bound to that slot.
  const uint32_t process_id = wf.process_id();
  const uint32_t queue_id = wf.queue_id();
  auto proc = find_process(process_id);
  if (!proc)
    return false;
  const pid_t target_pid = proc->client_pid();

  // Find a matching active watchpoint for this session.
  int matched_slot = -1;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto sit = debug_sessions_.find(target_pid);
    if (sit == debug_sessions_.end() || !sit->second.enabled)
      return false;
    for (uint32_t i = 0; i < KfdProcess::DebugSession::kMaxAddressWatches; ++i) {
      const auto &w = sit->second.address_watches[i];
      if (!w.active)
        continue;
      // Access mode gate (kfd_dbg_trap_address_watch_mode): READ=0 matches
      // reads, NONREAD=1 matches non-atomic writes, ATOMIC=2 matches atomics,
      // ALL=3 matches everything.
      bool mode_ok;
      if (w.mode == KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL)
        mode_ok = true;
      else if (is_atomic)
        mode_ok = (w.mode == KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC);
      else if (is_write)
        mode_ok = (w.mode == KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD);
      else
        mode_ok = (w.mode == KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ);
      if (!mode_ok)
        continue;
      // The watch covers the aligned block { X : (X & mask) == (address & mask) }.
      // The low zero bits of the mask span the block; the access [addr, addr+bytes)
      // hits when it overlaps that block. mask == 0 matches every address.
      const uint64_t block_base = w.address & w.mask;
      const uint64_t block_size = ~w.mask + 1; // 0 when mask == 0 (match all)
      const bool hit =
          (block_size == 0) || (addr < block_base + block_size && block_base < addr + bytes);
      if (hit) {
        matched_slot = static_cast<int>(i);
        break;
      }
    }
  }
  if (matched_slot < 0)
    return false;

  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto qit = proc->queue_snapshot_map_.find(queue_id);
    if (qit == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = qit->second.ctx_save_restore_address;
    ctx_size = qit->second.ctx_save_restore_area_size;
    gpu_id = qit->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;

  // Raise TRAPSTS.addr_watch<slot> and enable MODE.excp_en.addr_watch so
  // rocm-dbgapi recognizes the stop as a watchpoint (it gates the exception on
  // both bits: architecture.cpp signaled_exceptions). gfx9 TRAPSTS slot bits:
  // watch0 @7, watch1 @12, watch2 @13, watch3 @14 (architecture.cpp
  // sq_wave_trapsts_excp*_addr_watch*_mask); MODE.EXCP_EN.ADDR_WATCH @19.
  static constexpr uint32_t kAddrWatchTrapstsBits[4] = {1u << 7, 1u << 12, 1u << 13, 1u << 14};
  constexpr uint32_t kModeExcpEnAddrWatchMask = 1u << 19;
  wf.set_trapsts(wf.trapsts() | kAddrWatchTrapstsBits[matched_slot]);
  wf.set_mode_raw(wf.mode_raw() | kModeExcpEnAddrWatchMask);
  // Trap id 0: a watchpoint stop is not an s_trap breakpoint (rocm-dbgapi reads
  // the addr_watch bits, not TTMP6's trap id).
  wf.debug_trap(0);
  report_wave_stopped(proc, queue_id, gpu_id, ctx_base, ctx_size);
  return true;
}

bool SimulatedDriver::on_wave_illegal_instruction(amdgpu::Wavefront &wf) {
  // A wave fetched an undecodable instruction. Under a debugger, report it as an
  // illegal-instruction exception instead of silently retiring the wave.
  // rocm-dbgapi maps TRAPSTS.illegal_inst to WAVE_STOP_REASON_ILLEGAL_INSTRUCTION
  // (architecture.cpp signaled_exceptions) and the queue's
  // EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION exception drives next_pending_event. The PC
  // is left at the faulting instruction (the caller does not advance it) so the
  // debugger reports the illegal instruction's own address.
  const uint32_t process_id = wf.process_id();
  const uint32_t queue_id = wf.queue_id();
  auto proc = find_process(process_id);
  if (!proc)
    return false;
  const pid_t target_pid = proc->client_pid();
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto sit = debug_sessions_.find(target_pid);
    if (sit == debug_sessions_.end() || !sit->second.enabled)
      return false;
  }
  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto qit = proc->queue_snapshot_map_.find(queue_id);
    if (qit == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = qit->second.ctx_save_restore_address;
    ctx_size = qit->second.ctx_save_restore_area_size;
    gpu_id = qit->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;
  // gfx9 TRAPSTS.ILLEGAL_INST is bit 11 (rocdbgapi architecture.cpp
  // sq_wave_trapsts_illegal_inst_mask = 1 << 11).
  constexpr uint32_t kSqWaveTrapstsIllegalInstMask = 1u << 11;
  wf.set_trapsts(wf.trapsts() | kSqWaveTrapstsIllegalInstMask);
  wf.debug_trap(0); // halt at the faulting PC; not an s_trap breakpoint
  report_wave_stopped(proc, queue_id, gpu_id, ctx_base, ctx_size,
                      KFD_EC_MASK(EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION));
  return true;
}

bool SimulatedDriver::on_wave_memory_violation(amdgpu::Wavefront &wf, uint64_t /*addr*/,
                                               bool /*is_write*/) {
  // A wave accessed a global address that is not backed by any mapping. Under a
  // debugger, report it as a memory violation instead of silently servicing it
  // from sparse backing. rocm-dbgapi maps TRAPSTS.xnack_error to
  // WAVE_STOP_REASON_MEMORY_VIOLATION (architecture.cpp signaled_exceptions ->
  // wave_get_state) and the queue's EC_QUEUE_WAVE_MEMORY_VIOLATION drives
  // next_pending_event. The access already completed and the PC advanced, so
  // the stop is reported at the next instruction (imprecise memory reporting).
  const uint32_t process_id = wf.process_id();
  const uint32_t queue_id = wf.queue_id();
  auto proc = find_process(process_id);
  if (!proc)
    return false;
  const pid_t target_pid = proc->client_pid();
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto sit = debug_sessions_.find(target_pid);
    if (sit == debug_sessions_.end() || !sit->second.enabled)
      return false;
  }
  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto qit = proc->queue_snapshot_map_.find(queue_id);
    if (qit == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = qit->second.ctx_save_restore_address;
    ctx_size = qit->second.ctx_save_restore_area_size;
    gpu_id = qit->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;
  // gfx9 TRAPSTS.XNACK_ERROR is bit 28 (rocdbgapi architecture.cpp
  // sq_wave_trapsts_xnack_error_mask = 1 << 28); it is not mode-gated, so it
  // maps straight to WAVE_STOP_REASON_MEMORY_VIOLATION.
  constexpr uint32_t kSqWaveTrapstsXnackErrorMask = 1u << 28;
  wf.set_trapsts(wf.trapsts() | kSqWaveTrapstsXnackErrorMask);
  wf.debug_trap(0); // halt; not an s_trap breakpoint
  report_wave_stopped(proc, queue_id, gpu_id, ctx_base, ctx_size,
                      KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION));
  return true;
}

void SimulatedDriver::set_debug_active_on_all_cus(bool active) {
  // gpus_ is populated once during open()/open_process() and read lock-free
  // afterward, so no additional locking is needed here.
  for (auto &g : gpus_) {
    if (!g.soc)
      continue;
    g.soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      for (auto *cu : cp->compute_units())
        cu->set_debug_active(active);
    });
  }
}

void SimulatedDriver::resume_debug_queues(KfdProcess *proc) {
  // Reload each stopped wave from its queue's CWSR area (applying the debugger's
  // register edits and reading the requested run/single-step state that dbgapi
  // wrote via /proc/<pid>/mem), then unhalt it so the engine runs it again.
  // rocm-dbgapi resumes per queue; with a single compute queue in flight this
  // resumes exactly the waves it asked for. The wave's own MODE.debug_en selects
  // run vs. single-step (rocdbgapi architecture.cpp wave_set_state).
  if (!proc)
    return;
  const uint32_t process_id = proc->process_id();

  // Group stopped waves by queue so each queue's CWSR area is decoded once.
  struct QueueCtx {
    uint64_t ctx_base = 0;
    uint32_t ctx_size = 0;
    uint32_t gpu_id = 0;
    std::vector<amdgpu::Wavefront *> waves;
  };
  std::unordered_map<uint32_t, QueueCtx> by_queue;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    for (auto &[qid, info] : proc->queue_snapshot_map_) {
      QueueCtx qc;
      qc.ctx_base = info.ctx_save_restore_address;
      qc.ctx_size = info.ctx_save_restore_area_size;
      qc.gpu_id = info.gpu_id;
      by_queue[qid] = qc;
    }
  }

  for (auto &[qid, qc] : by_queue) {
    if (qc.ctx_base == 0)
      continue;
    auto *gpu = find_gpu(qc.gpu_id);
    if (!gpu || !gpu->soc)
      continue;

    // Collect this queue's stopped waves and their live geometry, plus the
    // compute units that own them so they can be rescheduled after unhalting.
    std::vector<amdgpu::Wavefront *> stopped;
    std::vector<kmd::CwsrWaveState> states;
    std::vector<amdgpu::ComputeUnitCore *> owning_cus;
    std::unordered_set<amdgpu::ComputeUnitCore *> cus_to_activate;
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      for (auto *cu : cp->compute_units()) {
        for (uint32_t i = 0; i < cu->num_wf_slots(); ++i) {
          auto *w = cu->wf(i);
          if (w->debug_halted() && w->process_id() == process_id && w->queue_id() == qid) {
            stopped.push_back(w);
            states.push_back(build_cwsr_wave_state(*w));
            owning_cus.push_back(cu);
          }
        }
      }
    });
    if (stopped.empty())
      continue;

    auto *mem = gpu->soc->memory();
    if (!kmd::deserialize_queue_cwsr(qc.ctx_base, qc.ctx_size, states,
                                     [&](uint64_t va) { return mem->read32(va, process_id); }))
      continue;

    for (size_t i = 0; i < stopped.size(); ++i) {
      apply_cwsr_to_wave(*stopped[i], states[i]);
      // A wave the debugger left running needs its CU rescheduled.
      if (!stopped[i]->debug_halted())
        cus_to_activate.insert(owning_cus[i]);
    }

    // Reschedule the affected CUs on the engine (thread-safe; also wakes the
    // engine if it had gone idle with all waves halted).
    for (auto *cu : cus_to_activate)
      cu->activate_async();
  }
  // Unhalted (and single-stepped) waves resume when their compute units are next
  // stepped, which the activate_async calls above schedule on the engine thread.
  // The debugger is not blocked here: for a single-step the engine executes one
  // instruction and re-stops the wave, and rocm-dbgapi only reads the wave back
  // after it receives the wave-stop notification the engine writes to dbg_fd
  // once the post-step CWSR is fully serialized (report_wave_stopped), so it
  // never observes a mid-step state.
}

void SimulatedDriver::suspend_debug_queues(KfdProcess *proc) {
  // Stop any still-running waves of the debugged process and serialize their
  // freshly frozen state into the queue's context save area. Waves already
  // stopped at a breakpoint are left untouched: their save area is current and
  // already holds the debugger's register edits and rocm-dbgapi's assigned wave
  // id, so re-serializing the live wave would revert those. Only waves this call
  // transitions from running to stopped need a fresh save.
  if (!proc)
    return;
  const uint32_t process_id = proc->process_id();

  std::unordered_map<uint32_t, std::tuple<uint64_t, uint32_t, uint32_t>>
      queues; // qid -> (ctx,size,gpu)
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    for (auto &[qid, info] : proc->queue_snapshot_map_)
      queues[qid] = {info.ctx_save_restore_address, info.ctx_save_restore_area_size, info.gpu_id};
  }

  for (auto &[qid, geom] : queues) {
    auto [ctx_base, ctx_size, gpu_id] = geom;
    if (ctx_base == 0)
      continue;
    auto *gpu = find_gpu(gpu_id);
    if (!gpu || !gpu->soc)
      continue;
    // Halt any running waves of this queue so their state is frozen for save.
    uint32_t newly_halted = 0;
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      for (auto *cu : cp->compute_units()) {
        for (uint32_t i = 0; i < cu->num_wf_slots(); ++i) {
          auto *w = cu->wf(i);
          if (!w->debug_halted() && w->process_id() == process_id && w->queue_id() == qid) {
            w->set_debug_halted(true);
            ++newly_halted;
          }
        }
      }
    });
    // Only refresh the save area when this suspend actually froze a running
    // wave; otherwise the existing save (with the debugger's edits) stands.
    if (newly_halted > 0)
      serialize_queue_debug_waves(process_id, qid, gpu_id, ctx_base, ctx_size);
  }
}

void SimulatedDriver::apply_cwsr_to_wave(amdgpu::Wavefront &wf, const kmd::CwsrWaveState &w) {
  // Apply the debugger's edits (rocm-dbgapi writes registers straight into the
  // CWSR area) back onto the live wave, then set the run/step state the debugger
  // requested. MODE.debug_en (bit 11) selects single-step; TTMP6.wave_stopped==0
  // means the debugger wants the wave to run (rocdbgapi architecture.cpp).
  constexpr uint32_t kModeDebugEnMask = 1u << 11; // SQ_WAVE_MODE.DEBUG_EN
  wf.pc = w.pc;
  wf.set_exec(w.exec);
  wf.set_vcc(w.vcc);
  wf.set_m0(w.m0);
  wf.set_status_raw(w.status);
  wf.set_mode_raw(w.mode);
  wf.set_trapsts(w.trapsts);
  // Preserve rocm-dbgapi's assigned wave id (TTMP4:5) so later serializations
  // present the same identity.
  wf.set_debug_wave_id(w.wave_id);

  auto &cu = wf.cu();
  const uint32_t sbase = wf.sgpr_alloc().base;
  for (uint32_t s = 0; s < w.num_sgprs && s < w.sgprs.size(); ++s)
    cu.write_sgpr(sbase + s, w.sgprs[s]);
  const uint32_t vbase = wf.vgpr_alloc().base;
  for (uint32_t r = 0; r < w.num_vgprs; ++r)
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      const size_t idx = static_cast<size_t>(r) * 64 + lane;
      if (idx < w.vgprs.size())
        cu.write_vgpr(vbase + r, lane, w.vgprs[idx]);
    }

  const bool single_step = (w.mode & kModeDebugEnMask) != 0;
  wf.set_debug_single_step(single_step);
  // The debugger wants the wave to run when it requests either free-run
  // (MODE.debug_en=0, TTMP6.wave_stopped=0) or single-step (MODE.debug_en=1).
  // Only an explicit stop (debug_en=0 and wave_stopped=1) keeps it halted.
  const bool keep_stopped = w.wave_stopped && !single_step;
  wf.set_debug_halted(keep_stopped);
}

int SimulatedDriver::debug_query_event(pid_t target_pid, KfdProcess *target_proc,
                                       kfd_ioctl_dbg_trap_query_debug_event_args &args) {
  // Mirrors kfd_dbg_ev_query_debug_event(): return one source (queue, or the
  // process itself with queue id 0) that has any raised exception, and clear the
  // bits the caller requested via the IN exception_mask; -EAGAIN when none
  // remain (kfd_ioctl.h). The full raised mask is returned even for bits the
  // caller did not ask to clear (e.g. process_runtime), so the debugger can act
  // on them and clear them via QUERY_EXCEPTION_INFO.
  const uint64_t clear_mask = args.exception_mask;
  std::lock_guard<std::mutex> lk(debug_events_mutex_);
  auto it = debug_events_.find(target_pid);
  if (it == debug_events_.end())
    return -EAGAIN;
  auto &queues = it->second;
  for (auto qit = queues.begin(); qit != queues.end(); ++qit) {
    if (qit->second.mask == 0)
      continue;
    args.exception_mask = qit->second.mask;
    args.queue_id = qit->first;
    args.gpu_id = qit->second.gpu_id;
    qit->second.mask &= ~clear_mask;
    const uint32_t reported_queue = qit->first;
    if (qit->second.mask == 0)
      queues.erase(qit);
    // Keep the queue snapshot's exception_status consistent with what the
    // debugger has now consumed: clear the same bits (queue_new included) so a
    // subsequent GET_QUEUE_SNAPSHOT does not re-report the queue as new and
    // make rocm-dbgapi tear down and recreate it (kfd_debug.c clears the
    // reported exceptions from the queue on query). The process-level source
    // (queue id 0) has no snapshot entry.
    if (target_proc != nullptr && reported_queue != 0) {
      std::lock_guard<std::mutex> plk(target_proc->alloc_mutex_);
      auto sit = target_proc->queue_snapshot_map_.find(reported_queue);
      if (sit != target_proc->queue_snapshot_map_.end())
        sit->second.exception_status &= ~clear_mask;
    }
    return 0;
  }
  return -EAGAIN;
}

void SimulatedDriver::raise_process_debug_event(pid_t target_pid, uint64_t exception_mask) {
  int dbg_fd = -1;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto sit = debug_sessions_.find(target_pid);
    if (sit == debug_sessions_.end() || !sit->second.enabled)
      return;
    dbg_fd = sit->second.dbg_fd;
  }
  {
    std::lock_guard<std::mutex> lk(debug_events_mutex_);
    auto &qe = debug_events_[target_pid][0]; // queue id 0 == process-level source
    qe.gpu_id = 0;
    qe.mask |= exception_mask;
  }
  notify_debugger(dbg_fd);
}

void SimulatedDriver::runtime_enable_debugger_handshake(pid_t target_pid) {
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto sit = debug_sessions_.find(target_pid);
    if (sit == debug_sessions_.end() || !sit->second.enabled)
      return; // not debugged: no handshake
  }

  raise_process_debug_event(target_pid, KFD_EC_MASK(EC_PROCESS_RUNTIME));

  // Block until the debugger acknowledges (SEND_RUNTIME_EVENT), bounded so a
  // detached or non-cooperative debugger cannot wedge the inferior forever. A
  // short bound is enough: the debugger only has to read r_debug and start
  // code-object tracking, and the timeout degrades to "best effort" rather than
  // hanging the workload.
  std::unique_lock<std::mutex> lk(runtime_handshake_mutex_);
  // Best effort: proceed after the bound whether or not the ack arrived.
  runtime_handshake_cv_.wait_for(lk, std::chrono::milliseconds(2000),
                                 [&] { return runtime_acked_.count(target_pid) > 0; });
  runtime_acked_.erase(target_pid);
}

int SimulatedDriver::debug_query_exception_info(
    pid_t target_pid, kfd_ioctl_dbg_trap_query_exception_info_args &args) {
  // Mirrors kfd_dbg_trap_query_exception_info(). Only EC_PROCESS_RUNTIME is
  // queried by rocm-dbgapi in the current flow: it returns the saved
  // kfd_runtime_info (r_debug + runtime_state + ttmp_setup) so the debugger can
  // locate the runtime loader's code-object list, optionally clearing the event.
  if (args.exception_code != EC_PROCESS_RUNTIME)
    return -EINVAL;

  auto proc = find_process_by_client_pid(target_pid);
  kfd_runtime_info info{};
  if (proc) {
    std::lock_guard<std::mutex> rlk(proc->runtime_mutex_);
    info.r_debug = proc->runtime_state_.r_debug;
    info.runtime_state =
        proc->runtime_state_.enabled ? DEBUG_RUNTIME_STATE_ENABLED : DEBUG_RUNTIME_STATE_DISABLED;
    info.ttmp_setup =
        (proc->runtime_state_.mode_mask & KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK) ? 1u : 0u;
  }

  const uint32_t in_size = args.info_size;
  args.info_size = sizeof(info);
  if (args.info_ptr != 0 && in_size >= sizeof(info))
    std::memcpy(reinterpret_cast<void *>(static_cast<uintptr_t>(args.info_ptr)), &info,
                sizeof(info));

  if (args.clear_exception) {
    std::lock_guard<std::mutex> lk(debug_events_mutex_);
    auto it = debug_events_.find(target_pid);
    if (it != debug_events_.end()) {
      auto qit = it->second.find(0);
      if (qit != it->second.end()) {
        qit->second.mask &= ~KFD_EC_MASK(EC_PROCESS_RUNTIME);
        if (qit->second.mask == 0)
          it->second.erase(qit);
      }
    }
  }
  return 0;
}

// AMDKFD_IOC_DBG_TRAP dispatcher. This mirrors the pre-switch validation ladder
// and per-operation routing of the real driver's kfd_ioctl_set_debug_trap()
// (amd/amdkfd/kfd_chardev.c, amdgpu-6.16.13). The individual sub-operations are
// filled in by subsequent changes; reaching the default case means the gate
// ladder admitted an operation that is not implemented yet.
int SimulatedDriver::debug_trap_ioctl(KfdProcess &caller, void *arg) {
  auto *args = static_cast<kfd_ioctl_dbg_trap_args *>(arg);
  util::Logger::cp("DBG_TRAP pid=", args->pid, " op=", args->op);

  // rocjitsu always models hardware scheduling, so the driver's
  // KFD_SCHED_POLICY_NO_HWS -> EINVAL guard is not applicable.

  // Resolve the target process by Linux pid (kernel: find_get_pid() +
  // kfd_lookup_process_by_pid()). The target's KfdProcess may not exist yet:
  // rocgdb enables debug on the inferior right after exec, before its ROCr has
  // opened /dev/kfd. The debug session is keyed by the target pid
  // (debug_sessions_) independently of the KfdProcess, mirroring the kernel
  // creating the target kfd_process in the ENABLE path. Operations that need
  // live GPU state look the process up lazily.
  const auto target_pid = static_cast<pid_t>(args->pid);
  const bool self_debug = caller.client_pid() != 0 && target_pid == caller.client_pid();
  std::shared_ptr<KfdProcess> target_ref =
      self_debug ? nullptr : find_process_by_client_pid(target_pid);
  KfdProcess *target_proc = self_debug ? &caller : target_ref.get();

  // Existence check (kernel: ESRCH from find_get_pid/get_task_mm). A target that
  // has not connected is still valid if it is a live OS process; a pid that maps
  // to no process at all is rejected.
  // Existence check (kernel: ESRCH from find_get_pid/get_task_mm). A target that
  // has not connected is still valid if it is a live OS process; a pid that maps
  // to no process at all is rejected. DISABLE is exempt so a debugger can always
  // tear down its session, even after the inferior has exited (the kernel
  // likewise exempts DISABLE from its liveness checks).
  if (!self_debug && target_proc == nullptr && args->op != KFD_IOC_DBG_TRAP_DISABLE) {
    errno = 0;
    if (::kill(target_pid, 0) != 0 && errno == ESRCH)
      return -ESRCH;
  }

  // PTRACE gate: for any op other than DISABLE, a debugger acting on another
  // process must be that process's ptrace parent. This mirrors the kernel's
  // ptrace_parent(target->lead_thread) == current check in
  // kfd_ioctl_set_debug_trap(); we consult the live /proc TracerPid so the
  // authorization reflects the real OS ptrace relationship established by the
  // debugger (e.g. rocgdb launching the inferior). Self-debug is exempt.
  if (!self_debug && args->op != KFD_IOC_DBG_TRAP_DISABLE &&
      tracer_pid_of(target_pid) != caller.client_pid())
    return -EPERM;

  std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
  auto session_it = debug_sessions_.find(target_pid);
  const bool enabled = session_it != debug_sessions_.end() && session_it->second.enabled;

  // Non-ENABLE ops require an active debug session (kernel: EINVAL).
  if (args->op != KFD_IOC_DBG_TRAP_ENABLE && !enabled)
    return -EINVAL;

  // Live runtime-enable state, set by ROCr's AMDKFD_IOC_RUNTIME_ENABLE on the
  // inferior; false until the inferior connects and enables its runtime.
  bool runtime_enabled = false;
  if (target_proc != nullptr) {
    std::lock_guard<std::mutex> rlk(target_proc->runtime_mutex_);
    runtime_enabled = target_proc->runtime_state_.enabled;
  }

  // DBG_HW_OPs require the runtime to be enabled (kernel: EPERM). Note that the
  // real driver includes SET_FLAGS in this gated set.
  switch (args->op) {
  case KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE:
  case KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE:
  case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:
  case KFD_IOC_DBG_TRAP_RESUME_QUEUES:
  case KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH:
  case KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH:
  case KFD_IOC_DBG_TRAP_SET_FLAGS:
    if (!runtime_enabled)
      return -EPERM;
    break;
  default:
    break;
  }

  // Address-watch ops validate the target gpu (kernel: ENODEV).
  if (args->op == KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH ||
      args->op == KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH) {
    const uint32_t gpu_id = args->op == KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH
                                ? args->set_node_address_watch.gpu_id
                                : args->clear_node_address_watch.gpu_id;
    if (find_gpu(gpu_id) == nullptr)
      return -ENODEV;
  }

  switch (args->op) {
  case KFD_IOC_DBG_TRAP_ENABLE: {
    if (enabled)
      return -EINVAL; // target process is already debug enabled
    KfdProcess::DebugSession sess{};
    sess.enabled = true;
    sess.debugger_pid = caller.client_pid();
    sess.dbg_fd = static_cast<int>(args->enable.dbg_fd);
    sess.exception_enable_mask = args->enable.exception_mask;
    sess.runtime_state =
        runtime_enabled ? DEBUG_RUNTIME_STATE_ENABLED : DEBUG_RUNTIME_STATE_DISABLED;

    // Copy kfd_runtime_info back to the debugger (kernel: kfd_dbg_trap_enable
    // copies the saved runtime info and returns its size). The inferior may not
    // have runtime-enabled yet, in which case the defaults (disabled, r_debug=0)
    // are correct.
    kfd_runtime_info info{};
    info.runtime_state = sess.runtime_state;
    if (target_proc != nullptr) {
      std::lock_guard<std::mutex> rlk(target_proc->runtime_mutex_);
      info.r_debug = target_proc->runtime_state_.r_debug;
      info.ttmp_setup =
          (target_proc->runtime_state_.mode_mask & KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK) ? 1u
                                                                                           : 0u;
    }
    if (args->enable.rinfo_ptr != 0 && args->enable.rinfo_size >= sizeof(info))
      std::memcpy(reinterpret_cast<void *>(static_cast<uintptr_t>(args->enable.rinfo_ptr)), &info,
                  sizeof(info));
    args->enable.rinfo_size = sizeof(info);

    debug_sessions_[target_pid] = sess;
    // Turn on the per-access debugger checks (watchpoints, memory-violation
    // detection) now that a session exists.
    set_debug_active_on_all_cus(true);
    return 0;
  }
  case KFD_IOC_DBG_TRAP_DISABLE:
    // Release the debugger notifier the daemon transport handed us (daemon mode
    // owns the dup'd fd; in local mode dbg_fd is the debugger's own fd or
    // invalid, so it is left untouched).
    if (session_it != debug_sessions_.end()) {
      if (daemon_mode_ && session_it->second.dbg_fd >= 0)
        ::close(session_it->second.dbg_fd);
    }
    debug_sessions_.erase(target_pid);
    // Once the last session is gone, disable the per-access debugger checks so
    // undebugged execution pays no per-access cost.
    if (debug_sessions_.empty())
      set_debug_active_on_all_cus(false);
    return 0;
  case KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED:
    // kfd_dbg_set_enabled_debug_exception_mask(): record the exceptions the
    // debugger wants forwarded. Delivery is wired up with the event channel.
    session_it->second.exception_enable_mask = args->set_exceptions_enabled.exception_mask;
    return 0;
  case KFD_IOC_DBG_TRAP_SET_FLAGS: {
    // kfd_dbg_trap_set_flags(): IN = flags to enable, OUT = previously enabled.
    const uint32_t previous = session_it->second.flags;
    session_it->second.flags = args->set_flags.flags;
    args->set_flags.flags = previous;
    return 0;
  }
  case KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE:
    // Record the launch mode (NORMAL/HALT/DEBUG). Halting waves at launch is
    // wired up with the wave-level scheduler hooks.
    session_it->second.launch_mode = args->launch_mode.launch_mode;
    return 0;
  case KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE: {
    // kfd_dbg_trap_set_wave_launch_override(): validate the request against the
    // device-supported trap-override mask, then record the enabled bits.
    //
    // For gfx9.4 (MI300/MI350) the only overridable exception trap is the
    // address-watch trap: amdgpu kgd_gfx_v9_validate_trap_override_request
    // (amdgpu_amdkfd_gfx_v9.c) masks trap_mask_supported down to
    // KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH and rejects any override mode other
    // than OR (the SPI_GDBG_TRAP_MASK register is global, so REPLACE could
    // disturb other processes). kfd_dbg_validate_trap_override_request then
    // returns -EACCES if the requested support mask is not a subset of the
    // supported mask. Mirror that contract so rocm-dbgapi reads back an
    // accurate supported mask (process.cpp update_agents fatal-errors if its
    // desired wave-trap mask is not a subset of what we report here).
    constexpr uint32_t kSupportedTrapMask = KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH;
    if (args->launch_override.override_mode != KFD_DBG_TRAP_OVERRIDE_OR)
      return -EINVAL;
    if (args->launch_override.support_request_mask & ~kSupportedTrapMask)
      return -EACCES;
    // OUT enable_mask = previously enabled bits; OUT support_request_mask = the
    // actually-supported mask. Only supported bits are recorded as enabled.
    const uint32_t previous = session_it->second.launch_override_enable;
    session_it->second.launch_override_enable =
        args->launch_override.enable_mask & kSupportedTrapMask;
    args->launch_override.enable_mask = previous;
    args->launch_override.support_request_mask = kSupportedTrapMask;
    return 0;
  }
  case KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH: {
    // kfd_dbg_trap_set_dev_address_watch(): allocate a free hardware watch slot
    // (gfx9 has four TCP_WATCH slots) and record the address/mask/mode. The
    // memory pipeline traps any wave whose access matches (see
    // on_wave_watchpoint). ENOMEM when all slots are in use, matching the
    // kernel's IDR-exhaustion path.
    auto &watches = session_it->second.address_watches;
    int slot = -1;
    for (uint32_t i = 0; i < KfdProcess::DebugSession::kMaxAddressWatches; ++i)
      if (!watches[i].active) {
        slot = static_cast<int>(i);
        break;
      }
    if (slot < 0)
      return -ENOMEM;
    watches[slot].active = true;
    watches[slot].address = args->set_node_address_watch.address;
    watches[slot].mask = args->set_node_address_watch.mask;
    watches[slot].mode = args->set_node_address_watch.mode;
    args->set_node_address_watch.id = static_cast<uint32_t>(slot);
    return 0;
  }
  case KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH: {
    // kfd_dbg_trap_clear_dev_address_watch(): free the slot the debugger was
    // handed by SET_NODE_ADDRESS_WATCH.
    const uint32_t id = args->clear_node_address_watch.id;
    if (id >= KfdProcess::DebugSession::kMaxAddressWatches)
      return -EINVAL;
    session_it->second.address_watches[id] = KfdProcess::DebugSession::AddressWatch{};
    return 0;
  }
  case KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT:
    // Drain the next pending wave/queue exception for this target, or EAGAIN.
    return debug_query_event(target_pid, target_proc, args->query_debug_event);
  case KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT:
    // Enumerate the target's compute queues so rocm-dbgapi can locate each
    // queue's CWSR area and walk its waves (kfd_debug.c: get_queue_snapshot).
    return debug_queue_snapshot(target_proc, args->queue_snapshot);
  case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:
    // Waves stop for the debugger by trapping (s_trap breakpoint), which already
    // freezes and serializes them (on_wave_trap). A suspend of an already-stopped
    // wave is therefore a no-op here: re-serializing the live wave would discard
    // the debugger's register edits and rocm-dbgapi's assigned wave id. The
    // return value is the number of queues handled; their ids are left unmodified
    // so rocm-dbgapi decodes them back without error/invalid flags (kfd_debug.c:
    // suspend_queues).
    return static_cast<int>(args->suspend_queues.num_queues);
  case KFD_IOC_DBG_TRAP_RESUME_QUEUES:
    // Reload the target's stopped waves from their context save areas and let
    // them run again (honoring the per-wave run/single-step state the debugger
    // wrote), then report the number of queues handled (kfd_debug.c:
    // resume_queues).
    resume_debug_queues(target_proc);
    return static_cast<int>(args->resume_queues.num_queues);
  case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
    return debug_device_snapshot(args->device_snapshot);
  case KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO:
    // Return the saved kfd_runtime_info for EC_PROCESS_RUNTIME (r_debug etc.).
    return debug_query_exception_info(target_pid, args->query_exception_info);
  case KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT: {
    // The debugger acknowledges the runtime-enable handshake; unblock the
    // inferior's RUNTIME_ENABLE (kfd_dbg_send_exception_to_runtime()).
    std::lock_guard<std::mutex> lk(runtime_handshake_mutex_);
    runtime_acked_.insert(target_pid);
    runtime_handshake_cv_.notify_all();
    return 0;
  }
  default:
    // Any remaining sub-operations are not implemented yet.
    return -ENOSYS;
  }
}

int SimulatedDriver::debug_device_snapshot(kfd_ioctl_dbg_trap_device_snapshot_args &args) {
  // Mirrors kfd_dbg_trap_device_snapshot() (amd/amdkfd/kfd_debug.c): report the
  // total device count, clamp the per-entry size, and fill up to the caller's
  // buffer capacity. The two-call protocol is: pass num_devices=0 to learn the
  // count, then pass a buffer sized for that many entries.
  const uint32_t total = static_cast<uint32_t>(gpus_.size());
  const uint32_t in_entry_size = args.entry_size;
  const uint32_t fill = std::min<uint32_t>(args.num_devices, total);

  args.num_devices = total;
  args.entry_size = std::min<uint32_t>(in_entry_size, sizeof(kfd_dbg_device_info_entry));

  if (fill == 0)
    return 0;
  if (args.snapshot_buf_ptr == 0 || in_entry_size == 0)
    return -EFAULT;

  const Sysfs::GpuInfo empty_info{};
  auto *out = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(args.snapshot_buf_ptr));
  for (uint32_t i = 0; i < fill; ++i) {
    const Sysfs::GpuInfo &info = i < gpu_infos_.size() ? gpu_infos_[i] : empty_info;
    const kfd_process_device_apertures ap = gpu_apertures(i);

    kfd_dbg_device_info_entry e{};
    e.gpu_id = gpus_[i].gpu_id;
    e.lds_base = ap.lds_base;
    e.lds_limit = ap.lds_limit;
    e.scratch_base = ap.scratch_base;
    e.scratch_limit = ap.scratch_limit;
    e.gpuvm_base = ap.gpuvm_base;
    e.gpuvm_limit = ap.gpuvm_limit;
    e.location_id = info.location_id;
    e.vendor_id = info.vendor_id;
    e.device_id = info.device_id;
    e.revision_id = info.pci_revision_id;
    e.fw_version = info.fw_version;
    e.gfx_target_version = info.gfx_target_version;
    e.simd_count = info.simd_count;
    e.max_waves_per_simd = info.max_waves_per_simd;
    // KFD array_count is the total shader-array count (node_props.array_count),
    // which rocjitsu tracks as num_shader_engines.
    e.array_count = info.num_shader_engines;
    e.simd_arrays_per_engine = info.num_shader_arrays_per_engine;
    e.num_xcc = info.num_xcc;
    // Report the same debugger-relevant capability/debug_prop as the sysfs
    // topology (shared kmd::debug_topology_for), including the ASIC revision.
    const kmd::DebugTopology dbg = kmd::debug_topology_for(info.gfx_target_version);
    e.capability = dbg.capability |
                   ((info.revision_id << HSA_CAP_ASIC_REVISION_SHIFT) & HSA_CAP_ASIC_REVISION_MASK);
    e.debug_prop = dbg.debug_prop;

    std::memcpy(out + static_cast<uint64_t>(i) * in_entry_size, &e, args.entry_size);
  }
  return 0;
}

int SimulatedDriver::debug_queue_snapshot(KfdProcess *target,
                                          kfd_ioctl_dbg_trap_queue_snapshot_args &args) {
  // Mirrors kfd_dbg_trap_get_queue_snapshot() (amd/amdkfd/kfd_debug.c). The
  // two-call protocol: pass num_queues=0 to learn the count, then a buffer
  // sized for that many entries. entry_size is clamped to what we populate.
  const uint32_t in_num = args.num_queues;
  const uint32_t in_entry_size = args.entry_size;
  // The debugger clears these exceptions from each reported queue (dbgapi passes
  // exceptions_cleared=QUEUE_NEW here). Applied only when the debugger actually
  // reads entries (in_num > 0), matching the kernel's get_queue_snapshot, which
  // clears the reported exceptions from the queue once copied out.
  const uint64_t clear_mask = args.exception_mask;

  std::vector<kfd_queue_snapshot_entry> entries;
  if (target != nullptr) {
    std::lock_guard<std::mutex> lk(target->alloc_mutex_);
    entries.reserve(target->active_queue_ids_.size());
    // Preserve creation order (active_queue_ids_) so the debugger sees a stable
    // enumeration across calls.
    for (uint32_t qid : target->active_queue_ids_) {
      auto it = target->queue_snapshot_map_.find(qid);
      if (it == target->queue_snapshot_map_.end())
        continue;
      KfdProcess::QueueSnapshotInfo &q = it->second;
      kfd_queue_snapshot_entry e{};
      e.exception_status = q.exception_status;
      e.ring_base_address = q.ring_base_address;
      e.write_pointer_address = q.write_pointer_address;
      e.read_pointer_address = q.read_pointer_address;
      e.ctx_save_restore_address = q.ctx_save_restore_address;
      e.queue_id = qid;
      e.gpu_id = q.gpu_id;
      e.ring_size = q.ring_size;
      e.queue_type = q.queue_type;
      e.ctx_save_restore_area_size = q.ctx_save_restore_area_size;
      entries.push_back(e);
      if (in_num > 0)
        q.exception_status &= ~clear_mask;
    }
  }

  const uint32_t total = static_cast<uint32_t>(entries.size());
  args.num_queues = total;
  args.entry_size = std::min<uint32_t>(in_entry_size, sizeof(kfd_queue_snapshot_entry));

  const uint32_t fill = std::min<uint32_t>(in_num, total);
  if (fill == 0)
    return 0;
  if (args.snapshot_buf_ptr == 0 || in_entry_size == 0)
    return -EFAULT;

  auto *out = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(args.snapshot_buf_ptr));
  for (uint32_t i = 0; i < fill; ++i)
    std::memcpy(out + static_cast<uint64_t>(i) * in_entry_size, &entries[i], args.entry_size);
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
