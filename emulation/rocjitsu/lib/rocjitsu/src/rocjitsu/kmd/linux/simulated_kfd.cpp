// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/cwsr.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/kfd_topology.h"
#include "rocjitsu/kmd/linux/libc_passthrough.h"
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
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <format>
#include <linux/types.h>
#include <poll.h>
#include <sstream>
#include <string_view>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/resource.h>
#include <sys/stat.h>
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
#include <thread>
#include <unistd.h>
#include <vector>

namespace rocjitsu {

namespace {

bool vm_trace_enabled() {
  static const bool enabled = (std::getenv("RJ_VMEM_TRACE") != nullptr);
  return enabled;
}

constexpr uint32_t kTileConfigCount = 32;
constexpr uint32_t kMacroTileConfigCount = 16;

} // namespace

amdgpu::Mtype SimulatedKfd::pte_mtype_for_flags(uint32_t flags) {
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

bool SimulatedKfd::gem_va_map(uint64_t gpu_va, void *host_ptr, size_t size, uint32_t alloc_flags) {
  auto proc = find_process(local_process_id_);
  if (!proc)
    return false;
  map_to_gpu(*proc, gpu_va, host_ptr, size, pte_mtype_for_flags(alloc_flags));
  return true;
}

bool SimulatedKfd::gem_va_unmap(uint64_t gpu_va, size_t size) {
  auto proc = find_process(local_process_id_);
  if (!proc)
    return false;
  unmap_from_gpu(*proc, gpu_va, size);
  return true;
}

namespace {

/// @brief mmap via the real libc, bypassing the interposer.
/// @details Routes through the process-wide libc_passthrough() table so the
/// driver's own mappings never re-enter the interposer's mmap hook. The table is
/// resolved once in the SimulatedKfd constructor.
void *safe_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  return libc_passthrough().mmap(addr, length, prot, flags, fd, offset);
}

/// @brief fstat via the real libc, bypassing the interposer.
/// @details Like safe_mmap: the interposer exports fstat with default visibility,
/// so a bare fstat() from this TU binds to our own hook (which takes fd_mutex_ via
/// is_drm()). Routing through the passthrough table keeps "the driver never
/// re-enters the interposer" total and avoids acquiring fd_mutex_ under a held
/// per-process lock (alloc_mutex_/etc.).
int safe_fstat(int fd, struct stat *st) { return libc_passthrough().fstat_fn(fd, st); }

/// @brief fcntl via the real libc, bypassing the interposer.
/// @details The interposer's fcntl hook takes fd_mutex_ on F_DUPFD paths; calling
/// it from the driver while holding a per-process lock is a latent lock-order
/// inversion. The passthrough table's fcntl is variadic; the int-arg forms
/// (F_DUPFD_CLOEXEC, F_ADD_SEALS, F_GETFL/no-arg) used here forward cleanly.
template <typename... Args> int safe_fcntl(int fd, int cmd, Args... args) {
  return libc_passthrough().fcntl(fd, cmd, args...);
}

int pidfd_is_exited(int pidfd) {
  pollfd pfd{pidfd, POLLIN, 0};
  const int rc = ::poll(&pfd, 1, 0);
  if (rc < 0)
    return -errno;
  return rc == 1 && (pfd.revents & (POLLIN | POLLHUP)) ? 1 : 0;
}

int pin_process_identity(pid_t pid, UniqueFd &pidfd, UniqueFd &procfd) {
  const int raw_pidfd = static_cast<int>(::syscall(SYS_pidfd_open, pid, 0));
  if (raw_pidfd < 0)
    return errno == ESRCH ? -ESRCH : -errno;
  pidfd = UniqueFd(raw_pidfd);

  const std::string proc_path = "/proc/" + std::to_string(pid);
  const int raw_procfd = ::open(proc_path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (raw_procfd < 0) {
    const int open_error = errno;
    const int exited = pidfd_is_exited(pidfd.get());
    return exited == 1 ? -ESRCH : (exited < 0 ? exited : -open_error);
  }
  procfd = UniqueFd(raw_procfd);

  const int exited = pidfd_is_exited(pidfd.get());
  return exited == 0 ? 0 : (exited == 1 ? -ESRCH : exited);
}

/// @brief Read TracerPid through a procfs directory pinned to the pidfd identity.
int tracer_pid_of(const UniqueFd &pidfd, const UniqueFd &procfd,
                  const SimulatedKfd::DebugIdentityValidationHook &validation_hook,
                  pid_t &tracer_pid) {
  int exited = pidfd_is_exited(pidfd.get());
  if (exited != 0)
    return exited == 1 ? -ESRCH : exited;

  UniqueFd status_fd(::openat(procfd.get(), "status", O_RDONLY | O_CLOEXEC));
  if (status_fd.get() < 0) {
    const int open_error = errno;
    const int exited = pidfd_is_exited(pidfd.get());
    return exited == 1 ? -ESRCH : (exited < 0 ? exited : -open_error);
  }

  std::string status;
  char buffer[4096];
  for (;;) {
    const ssize_t count = ::read(status_fd.get(), buffer, sizeof(buffer));
    if (count > 0) {
      status.append(buffer, static_cast<size_t>(count));
      continue;
    }
    if (count == 0)
      break;
    if (errno == EINTR)
      continue;
    return -errno;
  }

  constexpr std::string_view kKey = "TracerPid:";
  tracer_pid = 0;
  size_t offset = 0;
  while (offset < status.size()) {
    const size_t end = status.find('\n', offset);
    const std::string_view line(status.data() + offset,
                                (end == std::string::npos ? status.size() : end) - offset);
    if (line.substr(0, kKey.size()) == kKey) {
      tracer_pid = static_cast<pid_t>(std::strtol(line.data() + kKey.size(), nullptr, 10));
      break;
    }
    if (end == std::string::npos)
      break;
    offset = end + 1;
  }

  if (validation_hook)
    validation_hook();

  exited = pidfd_is_exited(pidfd.get());
  return exited == 0 ? 0 : (exited == 1 ? -ESRCH : exited);
}

} // namespace

std::shared_ptr<KfdProcess> SimulatedKfd::find_process(uint32_t process_id) const {
  std::lock_guard<std::mutex> lk(process_mutex_);
  auto it = processes_.find(process_id);
  return (it != processes_.end()) ? it->second : nullptr;
}

std::shared_ptr<KfdProcess> SimulatedKfd::find_local_process() const {
  return find_process(local_process_id_);
}

uint32_t SimulatedKfd::alloc_flags_for_handle(uint64_t handle) const {
  auto proc = find_process(local_process_id_);
  if (!proc)
    return 0;
  std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
  auto it = proc->allocations_.find(handle);
  return it != proc->allocations_.end() ? it->second.flags : 0;
}

void SimulatedKfd::map_to_gpu(KfdProcess &proc, uint64_t gpu_va, void *host_ptr, size_t size,
                              amdgpu::Mtype mtype) {
  util::Logger::cp("MAP pid=", proc.process_id(), " va=0x", std::hex, gpu_va, " size=0x", size,
                   std::dec, " mtype=", static_cast<int>(mtype));
  proc.map_pages(gpu_va, host_ptr, size, mtype);
}

void SimulatedKfd::unmap_from_gpu(KfdProcess &proc, uint64_t gpu_va, size_t size) {
  util::Logger::cp("UNMAP pid=", proc.process_id(), " va=0x", std::hex, gpu_va, " size=0x", size,
                   std::dec);
  proc.unmap_pages(gpu_va, size);
}

void SimulatedKfd::update_cp_doorbell_base(uint32_t gpu_ordinal, uint32_t process_id, void *base) {
  if (gpu_ordinal >= gpus_.size())
    return;
  auto &g = gpus_[gpu_ordinal];
  if (!g.soc)
    return;
  g.soc->for_each_cp(
      [=](amdgpu::CommandProcessor *cp) { cp->set_doorbell_base(process_id, base); });
}

std::string SimulatedKfd::redirect_sysfs_path(const char *path) const {
  auto result = redirect_sysfs_root_path(path, topology_path(), topology().drm_path());
  if (!result.empty()) {
    util::Logger::vm("sysfs redirect: ", path, " -> ", result);
    return result;
  }
  return {};
}

bool SimulatedKfd::handles_drm_render_minor(uint32_t minor) const {
  if (topology().drm_path().empty())
    return false;
  if (num_gpus() <= 1)
    return true;
  return minor >= 128 && minor < 128 + num_gpus();
}

const Sysfs::GpuInfo *SimulatedKfd::gpu_info_for_render_minor(uint32_t /*minor*/) const {
  if (topology().drm_path().empty())
    return nullptr;
  return &topology().gpu_info();
}

void SimulatedKfd::setup_topology(const config::KfdDeviceConfig &dev, uint32_t num_xcc) {
  if (!dev.present)
    return;

  setup_topology(gpu_info_from_config(dev, num_xcc));
}

SimulatedKfd::SimulatedKfd(SoC &soc, bool daemon_mode,
                           DebugIdentityValidationHook debug_identity_validation_hook)
    : daemon_mode_(daemon_mode),
      debug_identity_validation_hook_(std::move(debug_identity_validation_hook)),
      debug_session_reaper_([this](std::stop_token stop) { reap_exited_debug_sessions(stop); }) {
  // Resolve the real libc entry points once, up front and single-threaded, so no
  // passthrough call site ever triggers a first-time dlsym under a per-process
  // lock. Idempotent: a no-op if the interposer already resolved the table.
  libc_passthrough().resolve();
  gpus_.push_back({&soc, 0, false, {}});
}

SimulatedKfd::SimulatedKfd(std::vector<SoC *> socs, std::vector<uint32_t> gpu_ids, bool daemon_mode,
                           DebugIdentityValidationHook debug_identity_validation_hook)
    : daemon_mode_(daemon_mode),
      debug_identity_validation_hook_(std::move(debug_identity_validation_hook)),
      debug_session_reaper_([this](std::stop_token stop) { reap_exited_debug_sessions(stop); }) {
  libc_passthrough().resolve();
  for (size_t i = 0; i < socs.size(); ++i)
    gpus_.push_back({socs[i], i < gpu_ids.size() ? gpu_ids[i] : socs[i]->gpu_id(), false, {}});
}

SimulatedKfd::GpuDevice *SimulatedKfd::find_gpu(uint32_t gpu_id) {
  for (auto &g : gpus_)
    if (g.gpu_id == gpu_id)
      return &g;
  return nullptr;
}

const SimulatedKfd::GpuDevice *SimulatedKfd::find_gpu(uint32_t gpu_id) const {
  for (auto &g : gpus_)
    if (g.gpu_id == gpu_id)
      return &g;
  return nullptr;
}

SimulatedKfd::~SimulatedKfd() {
  debug_session_reaper_.request_stop();
  debug_session_reaper_.join();

  std::vector<uint32_t> pids;
  {
    std::lock_guard<std::mutex> lk(process_mutex_);
    pids.reserve(processes_.size());
    for (auto &[id, proc] : processes_)
      pids.push_back(id);
  }

  // close() only tears a process down on the LAST open reference (release_open()
  // returns true at zero); a process opened more than once (dup/daemon reuse)
  // would otherwise survive with its allocations, queues, and CP callbacks still
  // live past this driver. Keep closing each snapshotted pid until it is actually
  // removed from the table, so destruction always fully drains every process.
  for (auto pid : pids) {
    while (find_process(pid))
      close(pid);
  }
}

void SimulatedKfd::reap_exited_debug_sessions(std::stop_token stop) {
  std::unique_lock<std::mutex> lock(debug_sessions_mutex_);
  while (!stop.stop_requested()) {
    if (debug_sessions_.empty()) {
      debug_sessions_cv_.wait(lock, stop, [&] { return !debug_sessions_.empty(); });
    } else {
      debug_sessions_cv_.wait_for(lock, stop, std::chrono::milliseconds(10), [] { return false; });
    }
    if (stop.stop_requested())
      break;
    for (auto it = debug_sessions_.begin(); it != debug_sessions_.end();) {
      if (pidfd_is_exited(it->second.target_pidfd.get()) == 1 ||
          pidfd_is_exited(it->second.debugger_pidfd.get()) == 1)
        it = debug_sessions_.erase(it);
      else
        ++it;
    }
  }
}

void SimulatedKfd::setup_topology(const Sysfs::GpuInfo &gpu) {
  if (!gpus_.empty())
    gpus_[0].gpu_id = gpu.gpu_id;
  gpu_infos_ = {gpu};
  topology_.generate(gpu);
  topology_.setup_environment();
}

void SimulatedKfd::setup_topology(const std::vector<config::KfdDeviceConfig> &devs,
                                  uint32_t num_xcc) {
  std::vector<Sysfs::GpuInfo> infos;
  infos.reserve(devs.size());
  for (auto &dev : devs) {
    if (!dev.present)
      continue;
    infos.push_back(gpu_info_from_config(dev, num_xcc));
  }
  if (infos.empty())
    return;
  for (size_t i = 0; i < infos.size() && i < gpus_.size(); ++i)
    gpus_[i].gpu_id = infos[i].gpu_id;
  gpu_infos_ = infos;
  topology_.generate(infos);
  topology_.setup_environment();
}

bool SimulatedKfd::is_doorbell_range(const void *addr, size_t length) const {
  auto p = find_process(local_process_id_);
  if (!p || !addr || length == 0)
    return false;
  // Check every GPU ordinal's doorbell page: dispatch_mmap/dispatch_munmap install
  // and tear down a doorbell page per ordinal, so a multi-GPU process has more than
  // one to guard (checking only ordinal 0 would leave a higher ordinal's page
  // unprotected against a client mprotect). Snapshot each page/size under
  // alloc_mutex_ so a concurrent dispatch_mmap/dispatch_munmap (which mutate these
  // under the same lock) cannot tear the pointer/size read.
  const auto query_base = reinterpret_cast<uintptr_t>(addr);
  const auto query_end = query_base + length;
  std::lock_guard<std::mutex> lock(p->alloc_mutex_);
  for (const auto &gs : p->gpu_state_) {
    if (!gs.doorbell_page || gs.doorbell_page_size == 0)
      continue;
    const auto base = reinterpret_cast<uintptr_t>(gs.doorbell_page);
    const auto end = base + gs.doorbell_page_size;
    if (query_base < end && query_end > base)
      return true;
  }
  return false;
}

bool SimulatedKfd::ensure_fd_created() {
  if (fd_.load(std::memory_order_acquire) >= 0)
    return true;
  int new_fd = memfd_create("rocjitsu_kfd", 0);
  if (new_fd < 0)
    return false;
  int expected = -1;
  // CAS so only one racing opener publishes the backing memfd; a loser closes
  // its own memfd and adopts the winner's, avoiding a double create / fd leak.
  if (!fd_.compare_exchange_strong(expected, new_fd, std::memory_order_acq_rel,
                                   std::memory_order_acquire))
    libc_passthrough().close(new_fd);
  return true;
}

void SimulatedKfd::init_command_processors_locked() {
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
        cu->set_watchpoint_handler([this](amdgpu::Wavefront &wf, uint64_t address, uint32_t bytes,
                                          bool is_write, bool is_atomic) {
          return on_wave_watchpoint(wf, address, bytes, is_write, is_atomic);
        });
        cu->set_illegal_inst_handler(
            [this](amdgpu::Wavefront &wf) { return on_wave_illegal_instruction(wf); });
      }
    });
    g.cps_initialized = true;
  }
}

int SimulatedKfd::open() {
  static std::once_flag raise_nofile_flag;
  std::call_once(raise_nofile_flag, [] {
    struct rlimit rl {};
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < 8192) {
      rl.rlim_cur = std::min<rlim_t>(rl.rlim_max, 65536);
      setrlimit(RLIMIT_NOFILE, &rl);
    }
  });

  // Hold process_mutex_ across fd creation, process selection/retain, and the
  // returned fd load so a racing dup2 (which invalidates fd_ via
  // invalidate_primary_fd, also under process_mutex_) cannot clear fd_ between
  // publishing it and returning it. Either open() completes and returns a valid
  // fd, or invalidation wins first and ensure_fd_created() re-mints one below.
  std::lock_guard<std::mutex> lk(process_mutex_);
  if (!ensure_fd_created())
    return -1;
  if (!daemon_mode_ && local_process_id_ != 0 && processes_.contains(local_process_id_)) {
    processes_[local_process_id_]->retain_open();
    return fd_.load(std::memory_order_acquire);
  }
  uint32_t pid = next_process_id_++;
  auto proc = std::make_shared<KfdProcess>(pid, static_cast<uint32_t>(gpus_.size()));
  // client_pid_ caches getpid() at open() time; DBG_TRAP uses it to resolve a
  // self-debug target, so it must match the caller's live pid. A fork() child
  // inherits this cache stale, but the interposer's reset_after_fork() drops the
  // driver so the child re-open()s (and re-caches here) before any ioctl —
  // DBG_TRAP self-resolution therefore requires a post-fork re-open.
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

  init_command_processors_locked();

  return fd_.load(std::memory_order_acquire);
}

void SimulatedKfd::set_process_client_pid(uint32_t process_id, pid_t client_pid) {
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

uint32_t SimulatedKfd::open_process(pid_t client_pid) {
  uint32_t pid;
  {
    std::lock_guard<std::mutex> lk(process_mutex_);
    // Create the backing fd under process_mutex_ so both entry points
    // (open()/open_process()) serialize fd creation and never publish two
    // different memfds; ensure_fd_created() itself CASes so it is also safe from
    // any lock-free caller.
    if (!ensure_fd_created())
      return 0;
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

    init_command_processors_locked();
  }

  return pid;
}

LinuxKfd::PrimaryInvalidation SimulatedKfd::invalidate_primary_fd(int fd) {
  if (fd < 0)
    return PrimaryInvalidation::kNotPrimary;
  // Serialize with open()/open_process(), which hold process_mutex_ across fd
  // creation and the returned-fd load, so this cannot clear fd_ mid-open.
  std::lock_guard<std::mutex> lk(process_mutex_);
  int expected = fd;
  // The local primary fd holds one counted open reference, so on a successful
  // clear the caller must drop it (kClearedDropRef). Report kNotPrimary if a
  // concurrent overwrite already cleared fd_, so the caller does not double
  // release.
  if (fd_.compare_exchange_strong(expected, -1, std::memory_order_acq_rel))
    return PrimaryInvalidation::kClearedDropRef;
  return PrimaryInvalidation::kNotPrimary;
}

bool SimulatedKfd::retain_local_open() {
  std::lock_guard<std::mutex> lk(process_mutex_);
  if (local_process_id_ == 0)
    return false;
  auto it = processes_.find(local_process_id_);
  if (it == processes_.end())
    return false;
  it->second->retain_open();
  return true;
}

uint32_t SimulatedKfd::local_open_ref_count() const {
  std::lock_guard<std::mutex> lk(process_mutex_);
  if (local_process_id_ == 0)
    return 0;
  auto it = processes_.find(local_process_id_);
  return it != processes_.end() ? it->second->open_ref_count() : 0;
}

int SimulatedKfd::close() { return close(local_process_id_); }

void SimulatedKfd::close_all_processes() {
  // Snapshot the live process ids under process_mutex_, then close each with the lock
  // RELEASED (close() takes process_mutex_ itself). Closing a process fires
  // notify_closing()/signal_page_shutdown(), which wakes any client thread parked in
  // an infinite-timeout WAIT_EVENTS — the daemon teardown path relies on this to
  // unblock such threads so their jthread joins can complete instead of hanging
  // forever. A client that races us to its own rj_vm_device_close() just finds the
  // process already gone and no-ops.
  //
  // Drain each pid to a full teardown rather than a single close(): in daemon mode
  // several client opens of the same client_pid share one KfdProcess and bump
  // open_ref_count_ (open_process()'s retain path), so close() only reaches
  // notify_closing() on the LAST reference. A single decrement would leave a
  // multiply-opened process — exactly the one whose waiters we must wake — parked.
  // Loop close() while the process is still present, mirroring the destructor. The
  // find_process() re-check makes a concurrent client close() benign: whoever drops
  // the last reference tears it down, the other observes it gone and stops.
  std::vector<uint32_t> pids;
  {
    std::lock_guard<std::mutex> lk(process_mutex_);
    pids.reserve(processes_.size());
    for (const auto &[pid, proc] : processes_)
      pids.push_back(pid);
  }
  for (uint32_t pid : pids)
    while (find_process(pid))
      close(pid);
}

int SimulatedKfd::close(uint32_t process_id) {
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

  auto &proc = *extracted;

  // Serialize ALL teardown against any in-flight ioctl on this process. ioctl()
  // only snapshots a shared_ptr via find_process() and does NOT retain an open
  // reference, so an ioctl that started before this close() removed the process
  // from the table can still be running (or about to run) under proc.op_mutex_.
  // Acquire op_mutex_ BEFORE any teardown step — including event_dispatch_ erase
  // and mem->unregister_process() — so those cannot overlap an active
  // op_mutex_-guarded ioctl handler and break CP interrupt routing / memory
  // translation mid-ioctl. notify_closing() (below, still under op_mutex_) sets
  // the closing flag that dispatch_ioctl checks right after it takes op_mutex_,
  // so any ioctl that was blocked on op_mutex_ behind this close() will observe
  // is_closing() and bail instead of operating on a torn-down process.
  //
  // The process was already erased from processes_ above, so no NEW ioctl can
  // find it. Ordering is safe: process_mutex_ was released before taking
  // op_mutex_, so this does not nest against dispatch_ioctl's op_mutex_ ->
  // process_mutex_ order. WAIT_EVENTS does not take op_mutex_, so notify_closing()
  // / signal_page_shutdown() below still wake any parked waiter.
  //
  // NOTE: the mmap/munmap/is_doorbell_range family is NOT dispatched through
  // op_mutex_ — it synchronizes on alloc_mutex_. So the allocation and doorbell
  // teardown below additionally takes alloc_mutex_ to serialize against those
  // paths; op_mutex_ alone does not cover them.
  std::lock_guard<std::mutex> op_lock(proc.op_mutex_);

  // Linux kfd_release() only drops the file's kfd_process reference;
  // kfd_process_notifier_release_internal() disables debug when the process mm
  // actually exits. Preserve live sessions across a /dev/kfd close, but reap
  // targets whose pinned identity has exited.
  {
    std::lock_guard<std::mutex> debug_lock(debug_sessions_mutex_);
    for (auto it = debug_sessions_.begin(); it != debug_sessions_.end();) {
      if (pidfd_is_exited(it->second.target_pidfd.get()) == 1 ||
          pidfd_is_exited(it->second.debugger_pidfd.get()) == 1)
        it = debug_sessions_.erase(it);
      else
        ++it;
    }
  }

  // Set the closing flag first, under op_mutex_, so the dispatch_ioctl guard sees
  // it before any state is dismantled.
  proc.event_state_.notify_closing();
  proc.event_state_.signal_page_shutdown();

  {
    std::lock_guard<std::mutex> ilk(interrupt_mutex_);
    event_dispatch_.erase(process_id);
  }

  for (auto &g : gpus_) {
    if (auto *mem = g.soc ? g.soc->memory() : nullptr)
      mem->unregister_process(process_id);
  }

  const bool trace_enabled = vm_trace_enabled();
  size_t leaked_allocations = 0;
  uint64_t leaked_bytes = 0;
  size_t leaked_queues = 0;
  std::vector<uint64_t> leaked_handles;

  {
    std::lock_guard<std::mutex> alk(proc.alloc_mutex_);
    queue_ids.assign(proc.active_queue_ids_.begin(), proc.active_queue_ids_.end());
    proc.active_queue_ids_.clear();
    proc.queue_snapshot_map_.clear();

    if (trace_enabled)
      leaked_handles.reserve(proc.allocations_.size());
    for (auto &[handle, alloc] : proc.allocations_) {
      ++leaked_allocations;
      leaked_bytes += alloc.size;
      if (trace_enabled)
        leaked_handles.push_back(handle);
      if (alloc.host_ptr && alloc.host_ptr_owned) {
        unmap_from_gpu(proc, alloc.gpu_va, alloc.size);
        libc_passthrough().munmap(alloc.host_ptr, alloc.size);
        alloc.host_ptr = nullptr;
        alloc.host_ptr_owned = false;
      }
      if (alloc.memfd >= 0) {
        {
          std::lock_guard<std::mutex> flk(owned_fds_mutex_);
          owned_fds_.erase(alloc.memfd);
        }
        libc_passthrough().close(alloc.memfd);
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

  // Tear down doorbell pages. The mapped page pointer lives in gpu_state_ (not in
  // allocations_), so the generic host_ptr teardown above does not cover it — hence
  // this separate loop. The doorbell page is always driver-created (dispatch_mmap
  // maps it via safe_mmap in BOTH modes: a memfd MAP_SHARED page in daemon mode, a
  // fresh MAP_ANONYMOUS page in non-daemon mode), so the driver owns it and must
  // reclaim it unconditionally on close. Snapshot and clear the fields under
  // alloc_mutex_ (the lock the doorbell readers use —
  // is_doorbell_range/dispatch_mmap/dispatch_munmap), then munmap outside the lock
  // so the syscall does not run while alloc_mutex_ is held.
  for (auto &gs : proc.gpu_state_) {
    void *doorbell_page;
    size_t doorbell_page_size;
    {
      std::lock_guard<std::mutex> alk(proc.alloc_mutex_);
      doorbell_page = gs.doorbell_page;
      doorbell_page_size = gs.doorbell_page_size;
      gs.doorbell_page = nullptr;
      gs.doorbell_gpu_va = 0;
      gs.doorbell_page_size = 0;
    }
    if (doorbell_page && doorbell_page_size)
      libc_passthrough().munmap(doorbell_page, doorbell_page_size);
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

  // Guard the dmabuf teardown under alloc_mutex_ for consistency with the
  // import_dmabuf_ioctl/get_dmabuf_info_ioctl accessors: although this process was
  // already erased from the table under process_mutex_ and its last open reference
  // released, an ioctl that took a shared_ptr snapshot before the erase could still
  // be touching imported_dmabufs_ under alloc_mutex_.
  {
    std::lock_guard<std::mutex> alk(proc.alloc_mutex_);
    for (auto &[handle, dmabuf] : proc.imported_dmabufs_) {
      [[maybe_unused]] auto &_ = handle;
      if (dmabuf.fd >= 0)
        libc_passthrough().close(dmabuf.fd);
    }
    proc.imported_dmabufs_.clear();
    // Clear the reverse fd->handle map too, so it stays consistent with
    // imported_dmabufs_ (both are maintained together under alloc_mutex_ by
    // import_dmabuf_ioctl/free_memory_ioctl); its fds were just closed above.
    proc.fd_to_import_handle_.clear();
  }

  return 0;
}

int SimulatedKfd::ioctl(unsigned long request, void *arg) {
  return ioctl(local_process_id_, request, arg);
}

int SimulatedKfd::ioctl(uint32_t process_id, unsigned long request, void *arg) {
  auto proc = find_process(process_id);
  if (!proc)
    return -ESRCH;
  return dispatch_ioctl(*proc, request, arg);
}

int SimulatedKfd::dispatch_ioctl(KfdProcess &proc, unsigned long request, void *arg) {
  util::Logger::driver("IOCTL pid=", proc.process_id(), " ", LinuxKfd::ioctl_name(request));

  unsigned long dispatch_request = canonical_ioctl_request(request);

  if (dispatch_request == AMDKFD_IOC_WAIT_EVENTS)
    return wait_events_ioctl(proc, arg);

  std::lock_guard<std::mutex> op_lock(proc.op_mutex_);
  // A concurrent close() may have snapshotted-then-erased this process and be
  // tearing it down under op_mutex_. ioctl() holds only a shared_ptr (no open
  // reference), so an ioctl that raced close() can end up here AFTER teardown
  // ran (allocations/queues cleared, event_dispatch_ removed, memory
  // unregistered). close() sets the closing flag under op_mutex_ before any
  // teardown, so once we hold op_mutex_, is_closing() means the process is
  // logically gone — reject rather than operate on dismantled state. WAIT_EVENTS
  // is handled above and is intentionally exempt (it must observe the closing
  // signal to wake).
  if (proc.event_state_.is_closing())
    return -ESRCH;
  auto dispatch_one = [&]() -> int {
    switch (dispatch_request) {
    case AMDKFD_IOC_GET_VERSION:
      return get_version_ioctl(arg);
    case AMDKFD_IOC_GET_CLOCK_COUNTERS:
      return get_clock_counters_ioctl(arg);
    case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
      return get_process_apertures_ioctl(arg);
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
    // WAIT_EVENTS is handled before op_mutex_ above (it blocks on a condition
    // variable and must not hold the per-process op lock), so it never reaches
    // this switch.
    case AMDKFD_IOC_SET_XNACK_MODE:
      return set_xnack_mode_ioctl(arg);
    case AMDKFD_IOC_SET_MEMORY_POLICY:
      return set_memory_policy_ioctl(proc, arg);
    case AMDKFD_IOC_AVAILABLE_MEMORY:
      return get_available_memory_ioctl(proc, arg);
    case AMDKFD_IOC_RUNTIME_ENABLE:
      return runtime_enable_ioctl(proc, arg);
    case AMDKFD_IOC_DBG_TRAP:
      return debug_trap_ioctl(proc, arg);
    case AMDKFD_IOC_SET_SCRATCH_BACKING_VA: {
      auto *a = static_cast<kfd_ioctl_set_scratch_backing_va_args *>(arg);
      uint32_t ord = gpu_ordinal(a->gpu_id);
      {
        std::lock_guard<std::mutex> plk(process_mutex_);
        proc.gpu(ord).scratch_backing_va = a->va_addr;
      }
      util::Logger::vm([&](auto &os) {
        os << "SET_SCRATCH_BACKING_VA pid=" << proc.process_id() << " gpu_id=" << a->gpu_id
           << " va=" << std::hex << a->va_addr << std::dec;
      });
      return 0;
    }
    case AMDKFD_IOC_SET_TRAP_HANDLER: {
      auto *a = static_cast<kfd_ioctl_set_trap_handler_args *>(arg);
      uint32_t ord = gpu_ordinal(a->gpu_id);
      {
        // Held under process_mutex_ for symmetry with SET_SCRATCH_BACKING_VA and
        // to be race-free once the trap handler is wired into the SoC. NOTE: as of
        // now trap_tba_addr/trap_tma_addr have no reader anywhere (the CP does not
        // yet consume them), so this lock currently guards against a non-existent
        // concurrent access — kept for forward-compatibility.
        std::lock_guard<std::mutex> plk(process_mutex_);
        proc.gpu(ord).trap_tba_addr = a->tba_addr;
        proc.gpu(ord).trap_tma_addr = a->tma_addr;
      }
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
  };
  int ret = dispatch_one();
  if (ret != 0) {
    util::Logger::driver([&](auto &os) {
      os << std::format("IOCTL_ERROR pid={} {} ret={}", proc.process_id(), ioctl_name(request),
                        ret);
    });
  }
  return ret;
}

void *SimulatedKfd::mmap(void *addr, size_t length, int prot, int flags, off_t offset) {
  return mmap(local_process_id_, addr, length, prot, flags, offset);
}

void *SimulatedKfd::mmap(uint32_t process_id, void *addr, size_t length, int prot, int flags,
                         off_t offset) {
  auto p = find_process(process_id);
  if (!p) {
    errno = ESRCH;
    return MAP_FAILED;
  }
  if (daemon_mode_)
    return dispatch_mmap(*p, nullptr, length, prot, flags & ~MAP_FIXED, offset);
  return dispatch_mmap(*p, addr, length, prot, flags, offset);
}

void *SimulatedKfd::dispatch_mmap(KfdProcess &proc, void *addr, size_t length, int prot, int flags,
                                  off_t offset) {
  uint64_t type = static_cast<uint64_t>(offset) & KFD_MMAP_TYPE_MASK;
  util::Logger::vm("SimulatedKfd::mmap type=0x", std::hex, type, " offset=0x", offset,
                   " length=", std::dec, length, " addr=", addr);

  if (type == KFD_MMAP_TYPE_DOORBELL) {
    uint64_t encoded_gpu =
        (static_cast<uint64_t>(offset) & ~KFD_MMAP_TYPE_MASK) >> KFD_MMAP_GPU_ID_SHIFT;
    uint32_t db_gpu_id = static_cast<uint32_t>(encoded_gpu);
    uint32_t ord = gpu_ordinal(db_gpu_id);

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
        if (safe_fstat(doorbell_fd, &st) == 0)
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
        libc_passthrough().munmap(init_ptr, length);
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

      // Hold op_mutex_ across the whole publish -> map_to_gpu -> set_doorbell_base
      // sequence so a concurrent close() (which tears down doorbells under
      // op_mutex_) cannot clear+munmap this page between publishing it and handing
      // it to the CP, which would leave the CP with a dangling doorbell_base.
      // op_mutex_ is the outer lock (op_mutex_ -> alloc_mutex_, matching close());
      // set_doorbell_base takes hw_queue_mutex_, which is never held while taking
      // op_mutex_, so there is no inversion. alloc_mutex_ is taken only for the
      // field publish and released before set_doorbell_base (hw_queue_mutex_).
      std::lock_guard<std::mutex> op_lock(proc.op_mutex_);
      {
        std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
        // If close() has begun tearing this process down, do NOT publish a new
        // doorbell page: it would never be reclaimed (leaked) and would hand the
        // CP a base for a dying process. Fail the mmap instead. is_closing() is
        // set by close() under op_mutex_, which we now hold, so this check is
        // race-free against teardown.
        if (proc.event_state_.is_closing()) {
          libc_passthrough().munmap(ptr, length);
          errno = ENODEV;
          return MAP_FAILED;
        }
        auto &gs = proc.gpu(ord);
        gs.doorbell_page = ptr;
        gs.doorbell_page_size = length;
        gs.doorbell_gpu_va = reinterpret_cast<uint64_t>(ptr);
      }
      // Use the local ptr (== the doorbell_gpu_va just written) rather than
      // re-reading gs.doorbell_gpu_va without alloc_mutex_.
      map_to_gpu(proc, reinterpret_cast<uint64_t>(ptr), ptr, length, amdgpu::Mtype::UC);
      update_cp_doorbell_base(ord, proc.process_id(), ptr);
    }
    return ptr;
  }

  if (type == KFD_MMAP_TYPE_EVENTS) {
    if (proc.event_state_.memfd < 0) {
      auto raw_events_fd = memfd_create("rocjitsu_events", MFD_CLOEXEC | MFD_ALLOW_SEALING);
      if (raw_events_fd < 0)
        return MAP_FAILED;
      proc.event_state_.memfd = safe_fcntl(raw_events_fd, F_DUPFD_CLOEXEC, 4096);
      if (proc.event_state_.memfd < 0)
        proc.event_state_.memfd = raw_events_fd;
      else
        libc_passthrough().close(raw_events_fd);
      {
        std::lock_guard<std::mutex> lk(owned_fds_mutex_);
        owned_fds_.insert(proc.event_state_.memfd);
      }
      if (ftruncate(proc.event_state_.memfd, static_cast<off_t>(length)) != 0) {
        const int ftruncate_errno = errno; // preserve across close() below
        {
          std::lock_guard<std::mutex> lk(owned_fds_mutex_);
          owned_fds_.erase(proc.event_state_.memfd);
        }
        libc_passthrough().close(proc.event_state_.memfd);
        proc.event_state_.memfd = -1;
        errno = ftruncate_errno;
        return MAP_FAILED;
      }
      fallocate(proc.event_state_.memfd, 0, 0, static_cast<off_t>(length));
      {
        auto *init_ptr = static_cast<uint8_t *>(
            safe_mmap(nullptr, length, PROT_WRITE, MAP_SHARED, proc.event_state_.memfd, 0));
        if (init_ptr != MAP_FAILED) {
          libc_passthrough().madvise(init_ptr, length, MADV_POPULATE_WRITE);
          std::memset(init_ptr, 0xFF, length);
          libc_passthrough().munmap(init_ptr, length);
        }
      }
      safe_fcntl(proc.event_state_.memfd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW);
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
  bool host_ptr_owned = true;

  if (alloc.memfd >= 0) {
    if (length > alloc.size) {
      if (ftruncate(alloc.memfd, static_cast<off_t>(length)) != 0) {
        errno = ENOMEM;
        return MAP_FAILED;
      }
    }
    if (alloc.user_va && (flags & MAP_FIXED) && addr != nullptr) {
      auto prot_rc = libc_passthrough().mprotect(addr, length, PROT_READ | PROT_WRITE);
      if (prot_rc == 0) {
        constexpr size_t page_size = 4096;
        size_t num_pages = (length + page_size - 1) / page_size;
        std::vector<uint8_t> page_resident(num_pages);
        auto mc_rc = mincore(addr, length, page_resident.data());

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
          libc_passthrough().munmap(temp_mapping, length);
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
      auto rc = libc_passthrough().mprotect(addr, length, PROT_READ | PROT_WRITE);
      reuse_pages = (rc == 0);
    }
    if (reuse_pages) {
      host_ptr = addr;
      host_ptr_owned = false;
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
  alloc.host_ptr_owned = host_ptr_owned;

  util::Logger::vm([&](auto &os) {
    os << std::format("mmap: gpu_va={:#x} host_ptr={:#x} size={} flags={:#x}"
                      " MAP_FIXED={} user_va={} memfd={}",
                      alloc.gpu_va, reinterpret_cast<uintptr_t>(host_ptr), length, alloc.flags,
                      bool(flags & MAP_FIXED), alloc.user_va, alloc.memfd);
  });

  map_to_gpu(proc, alloc.gpu_va, host_ptr, length, pte_mtype_for_flags(alloc.flags));

  return host_ptr;
}

int SimulatedKfd::munmap(void *addr, size_t length) {
  return munmap(local_process_id_, addr, length);
}

int SimulatedKfd::munmap(uint32_t process_id, void *addr, size_t length) {
  auto p = find_process(process_id);
  if (!p)
    return -ESRCH;
  return dispatch_munmap(*p, addr, length);
}

int SimulatedKfd::dispatch_munmap(KfdProcess &proc, void *addr, size_t length) {
  {
    uint32_t doorbell_ord = 0;
    size_t doorbell_page_size = 0;
    bool is_doorbell = false;
    {
      std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
      for (size_t ord = 0; ord < proc.gpu_state_.size(); ++ord) {
        auto &gs = proc.gpu(ord);
        if (gs.doorbell_page == addr) {
          if (!proc.event_state_.is_closing()) {
            errno = EPERM;
            return -1;
          }
          uint64_t gpu_va = gs.doorbell_gpu_va;
          doorbell_page_size = gs.doorbell_page_size;
          gs.doorbell_page = nullptr;
          gs.doorbell_gpu_va = 0;
          gs.doorbell_page_size = 0;
          if (gpu_va && doorbell_page_size)
            unmap_from_gpu(proc, gpu_va, doorbell_page_size);
          doorbell_ord = static_cast<uint32_t>(ord);
          is_doorbell = true;
          break;
        }
      }
    }
    if (is_doorbell) {
      // Clear the CP's doorbell base for this process BEFORE munmapping the page.
      // The doorbell poll thread reads and dereferences doorbell_base under the CP's
      // hw_queue_mutex_ (scan_doorbells); if we munmapped first, the poll thread
      // could deref the freed page in the window before the base is cleared and
      // SIGSEGV. update_cp_doorbell_base takes hw_queue_mutex_, so once it returns
      // no poll-thread reader can still observe the stale base, and the munmap below
      // is safe.
      //
      // Both steps run AFTER releasing alloc_mutex_: the CP engine thread takes
      // alloc_mutex_ under hw_queue_mutex_ (allocate_scratch_backing), so holding
      // alloc_mutex_ across update_cp_doorbell_base (hw_queue_mutex_) would be an
      // alloc_mutex_->hw_queue_mutex_ inversion that can deadlock.
      update_cp_doorbell_base(doorbell_ord, proc.process_id(), nullptr);
      // Unmap the exact page we mapped: use the recorded doorbell page size, not
      // the caller-provided length. A length that differs from the tracked mapping
      // would otherwise partially unmap the CPU page and leave it inconsistent with
      // the GPU page-table unmap above.
      libc_passthrough().munmap(addr, doorbell_page_size ? doorbell_page_size : length);
      return 0;
    }
  }
  // release_page() clears page/page_size under EventState::mutex_, the same lock
  // the CP interrupt thread holds when reading them in signal_interrupt, so the
  // munmap below cannot race a concurrent signal writing into the mapping.
  if (proc.event_state_.release_page(addr)) {
    libc_passthrough().munmap(addr, length);
    return 0;
  }
  std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
  for (auto &[handle, alloc] : proc.allocations_) {
    if (alloc.host_ptr == addr) {
      unmap_from_gpu(proc, alloc.gpu_va, alloc.size);
      libc_passthrough().munmap(addr, length);
      alloc.host_ptr = nullptr;
      alloc.host_ptr_owned = false;
      return 0;
    }
  }
  return -ENOENT;
}

int SimulatedKfd::get_process_apertures_ioctl(void *arg) {
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

kfd_process_device_apertures SimulatedKfd::gpu_apertures(uint32_t ordinal) const {
  const uint64_t offset = static_cast<uint64_t>(ordinal) * kApertureStride;
  const uint64_t lds_base = 0x1000000000000ULL + offset;
  const uint64_t scratch_base = 0x2000000000000ULL + offset;
  return {
      .lds_base = lds_base,
      .lds_limit = lds_base + 0xFFFFFFFFULL,
      .scratch_base = scratch_base,
      .scratch_limit = scratch_base + 0xFFFFFFFFULL,
      // rocjitsu maps GPU VAs directly to host pointers, so the aperture must
      // cover the host addresses accepted by the runtime.
      .gpuvm_base = 0x10000ULL,
      .gpuvm_limit = 0x7FFFFFFFFFFFULL,
      .gpu_id = ordinal < gpus_.size() ? gpus_[ordinal].gpu_id : 0,
      .pad = 0,
  };
}

int SimulatedKfd::get_available_memory_ioctl(void *arg) {
  auto proc = find_local_process();
  return proc ? get_available_memory_ioctl(*proc, arg) : -ESRCH;
}

int SimulatedKfd::get_available_memory_ioctl(KfdProcess &proc, void *arg) {
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

int SimulatedKfd::get_tile_config_ioctl(void *arg) {
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

int SimulatedKfd::acquire_vm_ioctl([[maybe_unused]] void *arg) {
  (void)arg;
  return 0;
}

int SimulatedKfd::set_memory_policy_ioctl(void *arg) {
  auto proc = find_local_process();
  return proc ? set_memory_policy_ioctl(*proc, arg) : -ESRCH;
}

int SimulatedKfd::alloc_memory_ioctl(void *arg) {
  auto proc = find_local_process();
  return proc ? alloc_memory_ioctl(*proc, arg) : -ESRCH;
}

int SimulatedKfd::free_memory_ioctl(void *arg) {
  auto proc = find_local_process();
  return proc ? free_memory_ioctl(*proc, arg) : -ESRCH;
}

int SimulatedKfd::map_memory_ioctl(void *arg) {
  auto proc = find_local_process();
  return proc ? map_memory_ioctl(*proc, arg) : -ESRCH;
}

int SimulatedKfd::unmap_memory_ioctl(void *arg) {
  auto proc = find_local_process();
  return proc ? unmap_memory_ioctl(*proc, arg) : -ESRCH;
}

int SimulatedKfd::alloc_memory_ioctl(KfdProcess &proc, void *arg) {
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
  bool is_userptr = (args->flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR) != 0;
  bool is_doorbell = (args->flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL) != 0;
  if (is_userptr && !daemon_mode_) {
    alloc.host_ptr = reinterpret_cast<void *>(va);
    map_to_gpu(proc, va, reinterpret_cast<void *>(va), args->size, alloc_mtype);
  } else if (daemon_mode_ || !user_provided_va) {
    auto raw_fd = memfd_create("rocjitsu_alloc", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (raw_fd >= 0) {
      alloc.memfd = safe_fcntl(raw_fd, F_DUPFD_CLOEXEC, 4096);
      if (alloc.memfd < 0)
        alloc.memfd = raw_fd;
      else
        libc_passthrough().close(raw_fd);
      {
        std::lock_guard<std::mutex> lk(owned_fds_mutex_);
        owned_fds_.insert(alloc.memfd);
      }
      if (alloc.memfd >= 0) {
        [[maybe_unused]] auto ft_rc = ftruncate(alloc.memfd, static_cast<off_t>(alloc.size));
        fallocate(alloc.memfd, 0, 0, static_cast<off_t>(alloc.size));
        safe_fcntl(alloc.memfd, F_ADD_SEALS, F_SEAL_SHRINK);

        if (daemon_mode_ && !is_doorbell) {
          auto *mapped =
              safe_mmap(nullptr, alloc.size, PROT_READ | PROT_WRITE, MAP_SHARED, alloc.memfd, 0);
          if (mapped != MAP_FAILED) {
            alloc.host_ptr = mapped;
            alloc.host_ptr_owned = true;
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

  util::Logger::cp([&](auto &os) {
    os << std::format("ALLOC_MEMORY handle={} gpu_va={:#x} size={:#x} flags={:#x}", alloc.handle,
                      va, args->size, args->flags);
  });
  util::Logger::vm([&](auto &os) {
    os << std::format(
        "ALLOC pid={} handle={} gpu_va={:#x} size={} flags={:#x} memfd={} host_ptr={}",
        proc.process_id(), alloc.handle, va, args->size, args->flags, alloc.memfd,
        reinterpret_cast<uintptr_t>(alloc.host_ptr));
  });

  return 0;
}

bool SimulatedKfd::allocate_scratch_backing(uint32_t process_id, uint64_t gpu_va, size_t size) {
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
  auto raw_fd = memfd_create("rocjitsu_scratch", MFD_CLOEXEC);
  if (raw_fd < 0)
    return false;

  int memfd = safe_fcntl(raw_fd, F_DUPFD_CLOEXEC, 4096);
  if (memfd < 0)
    memfd = raw_fd;
  else
    libc_passthrough().close(raw_fd);
  {
    std::lock_guard<std::mutex> lk(owned_fds_mutex_);
    owned_fds_.insert(memfd);
  }

  if (ftruncate(memfd, static_cast<off_t>(aligned_size)) != 0) {
    {
      std::lock_guard<std::mutex> lk(owned_fds_mutex_);
      owned_fds_.erase(memfd);
    }
    libc_passthrough().close(memfd);
    return false;
  }
  auto *host_ptr = safe_mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0);
  if (host_ptr == MAP_FAILED) {
    {
      std::lock_guard<std::mutex> lk(owned_fds_mutex_);
      owned_fds_.erase(memfd);
    }
    libc_passthrough().close(memfd);
    return false;
  }
  {
    std::lock_guard<std::mutex> lk(owned_fds_mutex_);
    owned_fds_.erase(memfd);
  }
  libc_passthrough().close(memfd);
  std::memset(host_ptr, 0, aligned_size);
  proc->map_pages(gpu_va, host_ptr, aligned_size);

  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    KfdProcess::GpuAllocation alloc{};
    alloc.gpu_va = gpu_va;
    alloc.size = aligned_size;
    alloc.host_ptr = host_ptr;
    alloc.host_ptr_owned = true;
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

int SimulatedKfd::free_memory_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_free_memory_of_gpu_args *>(arg);

  std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
  auto it = proc.allocations_.find(args->handle);
  if (it != proc.allocations_.end()) {
    auto &alloc = it->second;
    if (alloc.imported && alloc.dmabuf_fd >= 0) {
      libc_passthrough().close(alloc.dmabuf_fd);
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
      libc_passthrough().close(alloc.memfd);
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
            libc_passthrough().close(ipc_it->second.backing_memfd);
          ipc_it = ipc_store_.erase(ipc_it);
        } else {
          ++ipc_it;
        }
      }
    }
  }
  return 0;
}

int SimulatedKfd::map_memory_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arg);

  std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
  auto it = proc.allocations_.find(args->handle);
  if (it == proc.allocations_.end()) {
    util::Logger::cp(
        [&](auto &os) { os << std::format("MAP_MEMORY_FAIL handle={} not found", args->handle); });
    return -EINVAL;
  }
  auto &alloc = it->second;
  util::Logger::cp([&](auto &os) {
    os << std::format("MAP_MEMORY handle={} gpu_va={:#x} size={:#x} n_devices={} host_ptr={}",
                      alloc.handle, alloc.gpu_va, alloc.size, args->n_devices,
                      alloc.host_ptr != nullptr);
  });
  if (alloc.host_ptr)
    map_to_gpu(proc, alloc.gpu_va, alloc.host_ptr, alloc.size, pte_mtype_for_flags(alloc.flags));
  args->n_success = args->n_devices;
  return 0;
}

int SimulatedKfd::unmap_memory_ioctl(KfdProcess &proc, void *arg) {
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

int SimulatedKfd::create_queue_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_create_queue_args *>(arg);
  auto *gpu = find_gpu(args->gpu_id);
  if (!gpu || !gpu->soc)
    return -EINVAL;

  // Select the target CP before reserving any per-process state, so a null CP
  // cannot leave a doorbell offset / queue-id bookkeeping entry orphaned.
  auto *target_cp = gpu->soc->assign_queue_cp();
  if (!target_cp)
    return -EINVAL;

  // Build the HW queue and reserve all per-process state under alloc_mutex_, then
  // register it with the CommandProcessor with the lock RELEASED. The CP thread
  // takes alloc_mutex_ under hw_queue_mutex_ (allocate_scratch_backing), so holding
  // alloc_mutex_ across register_queue() — which takes hw_queue_mutex_ — would be
  // an alloc_mutex_->hw_queue_mutex_ inversion against that thread and can deadlock.
  // op_mutex_ already serializes all ioctls for this process, so no concurrent
  // ioctl can observe the partially-registered queue in the window between the
  // unlock and register_queue().
  amdgpu::HwQueue hw{};
  uint32_t queue_id = 0;
  {
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

    queue_id = proc.next_queue_id_++;
    uint32_t ord = gpu_ordinal(args->gpu_id);
    auto &gs = proc.gpu(ord);
    uint32_t db_offset;
    bool recycled_offset = false;
    if (!gs.free_doorbell_offsets.empty()) {
      db_offset = gs.free_doorbell_offsets.back();
      gs.free_doorbell_offsets.pop_back();
      recycled_offset = true;
    } else {
      if (gs.doorbell_page_size > 0 &&
          gs.next_doorbell_offset + sizeof(uint64_t) > gs.doorbell_page_size)
        return -ENOSPC;
      db_offset = static_cast<uint32_t>(gs.next_doorbell_offset);
      gs.next_doorbell_offset += sizeof(uint64_t);
    }

    // Reset a recycled doorbell slot to the ~0 sentinel. The mmap-time 0xFF fill
    // only primes freshly-mapped pages; a slot freed by destroy_queue() still
    // holds the prior queue's last-rung write index (typically a small value like
    // 0). The CP starts every queue with last_doorbell==~0, so if the poll thread
    // scans this slot in the window between register_queue() and the host's first
    // ring, it latches that stale value as last_doorbell. When the host then rings
    // the new queue with the same value (write_index 0 for a one-packet queue),
    // val==last_doorbell, no edge is detected, and the submission is never fetched
    // — a lost doorbell that hangs the waiter in hsa_signal_wait. Restoring the
    // sentinel keeps the "first real ring is always an edge" invariant.
    if (recycled_offset && gs.doorbell_page &&
        db_offset + sizeof(uint64_t) <= gs.doorbell_page_size) {
      std::atomic_ref<uint64_t>(
          *reinterpret_cast<uint64_t *>(static_cast<char *>(gs.doorbell_page) + db_offset))
          .store(~uint64_t(0), std::memory_order_release);
    }

    hw.process_id = proc.process_id();
    hw.queue_id = queue_id;
    hw.ring_base_va = args->ring_base_address;
    hw.ring_size = args->ring_size;
    hw.read_ptr_va = args->read_pointer_address;
    hw.write_ptr_va = args->write_pointer_address;
    hw.doorbell_offset = db_offset;
    // doorbell_base is captured here under alloc_mutex_ but register_queue() runs
    // after the lock is released. This is stable because ROCr maps the doorbell
    // page before creating queues, and queue creation for a process is single-
    // threaded (serialized by op_mutex_), so no concurrent dispatch_mmap re-maps
    // the doorbell in the unlock->register window.
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

    args->queue_id = queue_id;
    args->doorbell_offset = KFD_MMAP_TYPE_DOORBELL | kfd_mmap_gpu_id(gpu->gpu_id) | db_offset;
    proc.active_queue_ids_.push_back(queue_id);
    proc.queue_doorbell_map_[queue_id] = {ord, db_offset};
  }

  // Register with the CP OUTSIDE alloc_mutex_ (see note above).
  target_cp->register_queue(std::move(hw));

  // Publish debug metadata only after CP registration. A cross-process debugger
  // does not hold the target's op_mutex_, so publishing it earlier could expose
  // a queue that the command processor cannot service yet.
  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    proc.queue_snapshot_map_[queue_id] = {
        .ring_base_address = args->ring_base_address,
        .write_pointer_address = args->write_pointer_address,
        .read_pointer_address = args->read_pointer_address,
        .ctx_save_restore_address = args->ctx_save_restore_address,
        .ctx_save_restore_area_size = args->ctx_save_restore_size,
        .ring_size = args->ring_size,
        .queue_type = args->queue_type,
        .gpu_id = args->gpu_id,
        .exception_status = KFD_EC_MASK(EC_QUEUE_NEW),
    };
  }
  return 0;
}

int SimulatedKfd::update_queue_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_update_queue_args *>(arg);
  for (auto &g : gpus_)
    if (g.soc)
      g.soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
        cp->update_queue(args->queue_id, proc.process_id(), args->ring_base_address,
                         args->ring_size);
      });
  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    if (auto it = proc.queue_snapshot_map_.find(args->queue_id);
        it != proc.queue_snapshot_map_.end()) {
      it->second.ring_base_address = args->ring_base_address;
      it->second.ring_size = args->ring_size;
    }
  }
  return 0;
}

int SimulatedKfd::destroy_queue_ioctl(KfdProcess &proc, void *arg) {
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

int SimulatedKfd::set_memory_policy_ioctl(KfdProcess &proc, void *arg) {
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

int SimulatedKfd::import_dmabuf_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_import_dmabuf_args *>(arg);
  if (!find_gpu(args->gpu_id))
    return -EINVAL;

  struct stat st {};
  if (safe_fstat(args->dmabuf_fd, &st) != 0)
    return -errno;
  uint64_t size = static_cast<uint64_t>(st.st_size);

  int dupfd = safe_fcntl(args->dmabuf_fd, F_DUPFD_CLOEXEC, 0);
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

    KfdProcess::ImportedDmabuf info{};
    info.handle = handle;
    info.fd = dupfd;
    info.size = size;
    info.va = args->va_addr;
    info.gpu_id = args->gpu_id;
    proc.imported_dmabufs_[handle] = info;
    proc.fd_to_import_handle_[dupfd] = handle;
  }

  if (args->va_addr)
    map_to_gpu(proc, args->va_addr, reinterpret_cast<void *>(args->va_addr), size,
               amdgpu::Mtype::UC);

  args->handle = handle;
  return 0;
}

int SimulatedKfd::export_dmabuf_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_export_dmabuf_args *>(arg);

  std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
  auto it = proc.allocations_.find(args->handle);
  if (it == proc.allocations_.end())
    return -EINVAL;
  const auto &alloc = it->second;
  if (alloc.memfd < 0)
    return -EINVAL;
  int dupfd = safe_fcntl(alloc.memfd, F_DUPFD_CLOEXEC, 0);
  if (dupfd < 0)
    return -errno;
  args->dmabuf_fd = dupfd;
  return 0;
}

int SimulatedKfd::ipc_export_handle_ioctl(KfdProcess &proc, void *arg) {
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
      int promoted_fd = memfd_create("rocjitsu_ipc_promote", MFD_CLOEXEC | MFD_ALLOW_SEALING);
      if (promoted_fd < 0)
        return -errno;
      if (ftruncate(promoted_fd, static_cast<off_t>(alloc.size)) != 0) {
        libc_passthrough().close(promoted_fd);
        return -errno;
      }
      auto *new_host_ptr =
          safe_mmap(nullptr, alloc.size, PROT_READ | PROT_WRITE, MAP_SHARED, promoted_fd, 0);
      if (new_host_ptr == MAP_FAILED) {
        libc_passthrough().close(promoted_fd);
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

      if (alloc.host_ptr_owned)
        libc_passthrough().munmap(alloc.host_ptr, alloc.size);

      alloc.host_ptr = new_host_ptr;
      alloc.host_ptr_owned = true;
      alloc.memfd = promoted_fd;
      {
        std::lock_guard<std::mutex> flk(owned_fds_mutex_);
        owned_fds_.insert(promoted_fd);
      }
    } else if (alloc.memfd < 0) {
      int new_fd = memfd_create("rocjitsu_ipc_lazy", MFD_CLOEXEC | MFD_ALLOW_SEALING);
      if (new_fd < 0)
        return -errno;
      if (ftruncate(new_fd, static_cast<off_t>(alloc.size)) != 0) {
        libc_passthrough().close(new_fd);
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
    dup_fd = safe_fcntl(alloc.memfd, F_DUPFD_CLOEXEC, 0);
  }

  if (dup_fd < 0)
    return -errno;

  IpcHandleKey key{};
  if (getrandom(key.words, sizeof(key.words), 0) != sizeof(key.words)) {
    libc_passthrough().close(dup_fd);
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

int SimulatedKfd::ipc_import_handle_ioctl(KfdProcess &proc, void *arg) {
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
    dup_fd = safe_fcntl(it->second.backing_memfd, F_DUPFD_CLOEXEC, 0);
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
    libc_passthrough().close(dup_fd);
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
    alloc.host_ptr_owned = true;
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

int SimulatedKfd::get_dmabuf_info_ioctl(KfdProcess &proc, void *arg) {
  auto *args = static_cast<kfd_ioctl_get_dmabuf_info_args *>(arg);
  uint64_t size = 0;
  uint32_t gpu_id = gpus_.empty() ? 0 : gpus_[0].gpu_id;

  bool found = false;
  {
    std::lock_guard<std::mutex> lk(proc.alloc_mutex_);
    for (const auto &[handle, info] : proc.imported_dmabufs_) {
      [[maybe_unused]] auto &_ = handle;
      if (info.fd >= 0 && static_cast<uint32_t>(info.fd) == args->dmabuf_fd) {
        size = info.size;
        gpu_id = info.gpu_id;
        found = true;
        break;
      }
    }
  }

  if (!found) {
    struct stat st {};
    if (safe_fstat(args->dmabuf_fd, &st) != 0)
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

int SimulatedKfd::svm_ioctl(KfdProcess &proc, void *arg) {
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

int SimulatedKfd::runtime_enable_ioctl(KfdProcess &proc, void *arg) {
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

  if (enabling)
    runtime_enable_debugger_handshake(proc.client_pid());
  return 0;
}

std::shared_ptr<KfdProcess> SimulatedKfd::find_process_by_client_pid(pid_t pid) const {
  if (pid == 0)
    return nullptr;
  std::lock_guard<std::mutex> lk(process_mutex_);
  for (auto &[id, proc] : processes_)
    if (proc->client_pid() == pid)
      return proc;
  return nullptr;
}

namespace {

kmd::CwsrWaveState build_cwsr_wave_state(amdgpu::Wavefront &wf) {
  kmd::CwsrWaveState state;
  const uint32_t raw_status = wf.status_raw();
  state.pc = wf.pc;
  state.exec = wf.exec();
  state.vcc = wf.vcc();
  state.m0 = wf.m0();
  state.mode = wf.mode_raw();
  state.trapsts = wf.trapsts();
  state.saved_status_halt = (raw_status >> 13) & 1u;
  state.status = raw_status | (1u << 13);
  state.wave_stopped = true;
  state.trap_id = wf.trap_id();
  state.wave_id = wf.debug_wave_id();
  state.group_ids = {wf.wg_id(), 0u, 0u};
  state.wave_in_group = 0;
  state.queue_packet_id = wf.dispatch_id() & 0x1FFFFFFu;
  state.num_sgprs = wf.num_sgprs();
  state.num_vgprs = wf.num_vgprs();

  state.sgprs.resize(state.num_sgprs);
  for (uint32_t s = 0; s < state.num_sgprs; ++s)
    state.sgprs[s] = wf.debug_read_sgpr(s);
  state.vgprs.resize(static_cast<size_t>(state.num_vgprs) * 64);
  for (uint32_t r = 0; r < state.num_vgprs; ++r)
    for (uint32_t lane = 0; lane < 64; ++lane)
      state.vgprs[static_cast<size_t>(r) * 64 + lane] = wf.debug_read_vgpr(r, lane);
  return state;
}

} // namespace

void SimulatedKfd::raise_debug_event(const std::shared_ptr<KfdProcess> &proc, uint32_t queue_id,
                                     uint32_t gpu_id, uint64_t exception_mask) {
  if (!proc)
    return;
  {
    std::lock_guard<std::mutex> lk(debug_events_mutex_);
    auto &queue = debug_events_[proc->client_pid()][queue_id];
    queue.gpu_id = gpu_id;
    queue.mask |= exception_mask;
  }
  std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
  auto queue = proc->queue_snapshot_map_.find(queue_id);
  if (queue != proc->queue_snapshot_map_.end())
    queue->second.exception_status |= exception_mask;
}

void SimulatedKfd::serialize_queue_debug_waves(uint32_t process_id, uint32_t queue_id,
                                               uint32_t gpu_id, uint64_t ctx_base,
                                               uint32_t ctx_size) {
  auto *gpu = find_gpu(gpu_id);
  if (!gpu || !gpu->soc || ctx_base == 0)
    return;

  std::vector<kmd::CwsrWaveState> waves;
  gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
    for (auto *cu : cp->compute_units()) {
      for (uint32_t i = 0; i < cu->num_wf_slots(); ++i) {
        auto *wave = cu->wf(i);
        if (wave->debug_halted() && wave->process_id() == process_id &&
            wave->queue_id() == queue_id)
          waves.push_back(build_cwsr_wave_state(*wave));
      }
    }
  });
  if (waves.empty())
    return;

  auto *memory = gpu->soc->memory();
  kmd::serialize_queue_cwsr(ctx_base, ctx_size, waves, [&](uint64_t address, uint32_t value) {
    memory->write32(address, value, process_id);
  });
}

bool SimulatedKfd::on_wave_trap(amdgpu::Wavefront &wave, uint32_t trap_id) {
  const uint32_t process_id = wave.process_id();
  const uint32_t queue_id = wave.queue_id();
  auto proc = find_process(process_id);
  if (!proc)
    return false;
  const pid_t target_pid = proc->client_pid();

  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session == debug_sessions_.end() || !session->second.enabled)
      return false;
  }

  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(queue_id);
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    ctx_size = queue->second.ctx_save_restore_area_size;
    gpu_id = queue->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;

  if (wave.debug_wave_id() == 0)
    wave.set_debug_wave_id(next_debug_wave_id_.fetch_add(1, std::memory_order_relaxed));
  wave.debug_trap(trap_id);

  report_wave_stopped(proc, queue_id, gpu_id, ctx_base, ctx_size);
  return true;
}

void SimulatedKfd::report_wave_stopped(const std::shared_ptr<KfdProcess> &proc, uint32_t queue_id,
                                       uint32_t gpu_id, uint64_t ctx_base, uint32_t ctx_size,
                                       uint64_t exception_mask) {
  const pid_t target_pid = proc->client_pid();
  serialize_queue_debug_waves(proc->process_id(), queue_id, gpu_id, ctx_base, ctx_size);
  raise_debug_event(proc, queue_id, gpu_id, exception_mask);

  // Duplicate under the session lock so DISABLE/reaping cannot close and reuse
  // the descriptor, then perform notifier I/O without holding a driver lock.
  UniqueFd notifier;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session != debug_sessions_.end() && session->second.dbg_fd >= 0 &&
        (session->second.exception_enable_mask & exception_mask) != 0)
      notifier = UniqueFd(safe_fcntl(session->second.dbg_fd, F_DUPFD_CLOEXEC, 0));
  }
  if (notifier.get() >= 0) {
    const uint64_t one = 1;
    [[maybe_unused]] const ssize_t written = ::write(notifier.get(), &one, sizeof(one));
  }
}

bool SimulatedKfd::on_wave_single_step_complete(amdgpu::Wavefront &wave) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    ctx_size = queue->second.ctx_save_restore_area_size;
    gpu_id = queue->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;
  wave.set_debug_single_step(false);
  wave.debug_trap(0);
  // gfx9.4 reports completed single-step through TRAPSTS.TRAP_AFTER_INST. This
  // is the public stop-reason bit rocm-dbgapi consumes before the next resume.
  constexpr uint32_t kTrapAfterInstMask = 1u << 25;
  wave.set_trapsts(wave.trapsts() | kTrapAfterInstMask);
  report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size);
  return true;
}

bool SimulatedKfd::on_wave_watchpoint(amdgpu::Wavefront &wave, uint64_t address, uint32_t bytes,
                                      bool is_write, bool is_atomic) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  int matched_slot = -1;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    if (session == debug_sessions_.end() || !session->second.enabled)
      return false;
    for (uint32_t slot = 0; slot < KfdProcess::DebugSession::kMaxAddressWatches; ++slot) {
      const auto &watch = session->second.address_watches[slot];
      if (!watch.active)
        continue;
      const bool mode_matches =
          watch.mode == KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL ||
          (is_atomic && watch.mode == KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC) ||
          (!is_atomic && is_write && watch.mode == KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD) ||
          (!is_atomic && !is_write && watch.mode == KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ);
      if (!mode_matches)
        continue;
      const uint64_t block_base = watch.address & watch.mask;
      const uint64_t block_size = ~watch.mask + 1;
      const uint64_t access_end = address > UINT64_MAX - bytes ? UINT64_MAX : address + bytes;
      const uint64_t block_end = block_size == 0 || block_base > UINT64_MAX - block_size
                                     ? UINT64_MAX
                                     : block_base + block_size;
      if (block_size == 0 || (address < block_end && block_base < access_end)) {
        matched_slot = static_cast<int>(slot);
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
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    ctx_size = queue->second.ctx_save_restore_area_size;
    gpu_id = queue->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;

  static constexpr uint32_t kTrapstsBits[] = {1u << 7, 1u << 12, 1u << 13, 1u << 14};
  constexpr uint32_t kModeExcpEnAddrWatch = 1u << 19;
  wave.set_trapsts(wave.trapsts() | kTrapstsBits[matched_slot]);
  wave.set_mode_raw(wave.mode_raw() | kModeExcpEnAddrWatch);
  wave.debug_trap(0);
  report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size);
  return true;
}

bool SimulatedKfd::on_wave_illegal_instruction(amdgpu::Wavefront &wave) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    if (session == debug_sessions_.end() || !session->second.enabled)
      return false;
  }
  uint64_t ctx_base = 0;
  uint32_t ctx_size = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    ctx_size = queue->second.ctx_save_restore_area_size;
    gpu_id = queue->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;
  constexpr uint32_t kTrapstsIllegalInst = 1u << 11;
  wave.set_trapsts(wave.trapsts() | kTrapstsIllegalInst);
  wave.debug_trap(0);
  report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size,
                      KFD_EC_MASK(EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION));
  return true;
}

void SimulatedKfd::apply_cwsr_to_wave(amdgpu::Wavefront &wave, const kmd::CwsrWaveState &state) {
  constexpr uint32_t kModeDebugEnMask = 1u << 11;
  wave.pc = state.pc;
  wave.set_exec(state.exec);
  wave.set_vcc(state.vcc);
  wave.set_m0(state.m0);
  wave.set_status_raw(state.status);
  wave.set_mode_raw(state.mode);
  wave.set_trapsts(state.trapsts);
  wave.set_debug_wave_id(state.wave_id);
  for (uint32_t s = 0; s < state.num_sgprs && s < state.sgprs.size(); ++s)
    wave.debug_write_sgpr(s, state.sgprs[s]);
  for (uint32_t r = 0; r < state.num_vgprs; ++r)
    for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
      const size_t index = static_cast<size_t>(r) * 64 + lane;
      if (index < state.vgprs.size())
        wave.debug_write_vgpr(r, lane, state.vgprs[index]);
    }
  const bool single_step = (state.mode & kModeDebugEnMask) != 0;
  wave.set_debug_single_step(single_step);
  wave.set_debug_halted(state.wave_stopped && !single_step);
}

void SimulatedKfd::resume_debug_queues(KfdProcess *proc) {
  if (!proc)
    return;
  struct QueueContext {
    uint64_t base = 0;
    uint32_t size = 0;
    uint32_t gpu_id = 0;
  };
  std::unordered_map<uint32_t, QueueContext> queues;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    for (const auto &[queue_id, info] : proc->queue_snapshot_map_)
      queues.emplace(queue_id, QueueContext{info.ctx_save_restore_address,
                                            info.ctx_save_restore_area_size, info.gpu_id});
  }
  for (const auto &[queue_id, context] : queues) {
    auto *gpu = find_gpu(context.gpu_id);
    if (!gpu || !gpu->soc || context.base == 0)
      continue;
    std::vector<amdgpu::Wavefront *> stopped;
    std::vector<kmd::CwsrWaveState> states;
    std::vector<amdgpu::ComputeUnitCore *> owners;
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      for (auto *cu : cp->compute_units()) {
        for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
          auto *wave = cu->wf(slot);
          if (wave->debug_halted() && wave->process_id() == proc->process_id() &&
              wave->queue_id() == queue_id) {
            stopped.push_back(wave);
            states.push_back(build_cwsr_wave_state(*wave));
            owners.push_back(cu);
          }
        }
      }
    });
    if (stopped.empty())
      continue;
    auto *memory = gpu->soc->memory();
    if (!kmd::deserialize_queue_cwsr(context.base, context.size, states, [&](uint64_t address) {
          return memory->read32(address, proc->process_id());
        }))
      continue;
    std::unordered_set<amdgpu::ComputeUnitCore *> wake;
    for (size_t index = 0; index < stopped.size(); ++index) {
      apply_cwsr_to_wave(*stopped[index], states[index]);
      if (!stopped[index]->debug_halted())
        wake.insert(owners[index]);
    }
    for (auto *cu : wake)
      cu->schedule_work_async();
  }
}

int SimulatedKfd::debug_query_event(pid_t target_pid, uint64_t enabled_mask,
                                    kfd_ioctl_dbg_trap_query_debug_event_args &args) {
  const uint64_t clear_mask = args.exception_mask;
  std::lock_guard<std::mutex> lk(debug_events_mutex_);
  auto process = debug_events_.find(target_pid);
  if (process == debug_events_.end())
    return -EAGAIN;
  auto &queues = process->second;
  for (auto queue = queues.begin(); queue != queues.end(); ++queue) {
    if ((queue->second.mask & enabled_mask) == 0)
      continue;
    args.exception_mask = queue->second.mask;
    args.queue_id = queue->first;
    args.gpu_id = queue->second.gpu_id;
    queue->second.mask &= ~clear_mask;
    if (queue->second.mask == 0)
      queues.erase(queue);
    return 0;
  }
  return -EAGAIN;
}

void SimulatedKfd::raise_process_debug_event(pid_t target_pid, uint64_t exception_mask) {
  UniqueFd notifier;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session == debug_sessions_.end() || !session->second.enabled)
      return;
    if ((session->second.exception_enable_mask & exception_mask) != 0)
      notifier = UniqueFd(safe_fcntl(session->second.dbg_fd, F_DUPFD_CLOEXEC, 0));
  }
  {
    std::lock_guard<std::mutex> lk(debug_events_mutex_);
    auto &event = debug_events_[target_pid][0];
    event.gpu_id = 0;
    event.mask |= exception_mask;
  }
  if (notifier.get() >= 0) {
    const uint64_t one = 1;
    [[maybe_unused]] const ssize_t written = ::write(notifier.get(), &one, sizeof(one));
  }
}

void SimulatedKfd::runtime_enable_debugger_handshake(pid_t target_pid) {
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session == debug_sessions_.end() || !session->second.enabled)
      return;
  }
  raise_process_debug_event(target_pid, KFD_EC_MASK(EC_PROCESS_RUNTIME));
  std::unique_lock<std::mutex> lk(runtime_handshake_mutex_);
  runtime_handshake_cv_.wait_for(lk, std::chrono::seconds(2),
                                 [&] { return runtime_acked_.contains(target_pid); });
  runtime_acked_.erase(target_pid);
}

int SimulatedKfd::debug_query_exception_info(pid_t target_pid,
                                             kfd_ioctl_dbg_trap_query_exception_info_args &args) {
  if (args.exception_code != EC_PROCESS_RUNTIME)
    return -EINVAL;
  kfd_runtime_info info{};
  if (auto proc = find_process_by_client_pid(target_pid)) {
    std::lock_guard<std::mutex> lk(proc->runtime_mutex_);
    info.r_debug = proc->runtime_state_.r_debug;
    info.runtime_state =
        proc->runtime_state_.enabled ? DEBUG_RUNTIME_STATE_ENABLED : DEBUG_RUNTIME_STATE_DISABLED;
    info.ttmp_setup =
        (proc->runtime_state_.mode_mask & KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK) ? 1u : 0u;
  }
  const uint32_t capacity = args.info_size;
  args.info_size = sizeof(info);
  if (capacity > 0 && args.info_ptr == 0)
    return -EFAULT;
  if (args.info_ptr != 0 && capacity > 0)
    std::memcpy(reinterpret_cast<void *>(static_cast<uintptr_t>(args.info_ptr)), &info,
                std::min(static_cast<size_t>(capacity), sizeof(info)));
  if (args.clear_exception) {
    std::lock_guard<std::mutex> lk(debug_events_mutex_);
    auto process = debug_events_.find(target_pid);
    if (process != debug_events_.end()) {
      auto event = process->second.find(0);
      if (event != process->second.end()) {
        event->second.mask &= ~KFD_EC_MASK(EC_PROCESS_RUNTIME);
        if (event->second.mask == 0)
          process->second.erase(event);
      }
    }
  }
  return 0;
}

// in real kernel, amd/amdkfd/kfd_chardev.c kfd_ioctl_set_debug_trap
int SimulatedKfd::debug_trap_ioctl(KfdProcess &caller, void *arg) {
  auto *args = static_cast<kfd_ioctl_dbg_trap_args *>(arg);
  util::Logger::driver("DBG_TRAP pid=", args->pid, " op=", args->op);

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
  if (target_pid <= 0)
    return -ESRCH;

  const bool self_debug = caller.client_pid() != 0 && target_pid == caller.client_pid();

  std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
  auto session_it = debug_sessions_.find(target_pid);
  if (session_it != debug_sessions_.end()) {
    const int target_exited = pidfd_is_exited(session_it->second.target_pidfd.get());
    if (target_exited < 0)
      return target_exited;
    const int debugger_exited = pidfd_is_exited(session_it->second.debugger_pidfd.get());
    if (debugger_exited < 0)
      return debugger_exited;
    if (target_exited == 1 || debugger_exited == 1) {
      debug_sessions_.erase(session_it);
      session_it = debug_sessions_.end();
      if (target_exited == 1 && args->op != KFD_IOC_DBG_TRAP_ENABLE)
        return -ESRCH;
    }
  }
  const bool enabled = session_it != debug_sessions_.end();

  // Non-ENABLE ops require an active debug session (kernel: EINVAL).
  if (args->op != KFD_IOC_DBG_TRAP_ENABLE && !enabled) {
    UniqueFd probe_pidfd;
    UniqueFd probe_procfd;
    const int probe_result = pin_process_identity(target_pid, probe_pidfd, probe_procfd);
    if (probe_result == -ESRCH)
      return -ESRCH;
    if (probe_result != 0)
      return probe_result;
    return -EINVAL;
  }

  // Pin the exact Linux task before consulting ptrace state. For an existing
  // session, use the identity captured by ENABLE; otherwise capture both a
  // pidfd and the matching procfs directory now. The pidfd liveness checks on
  // both sides of the procfs read ensure a numeric-pid reuse can never authorize
  // a different task.
  UniqueFd new_target_pidfd;
  UniqueFd new_target_procfd;
  UniqueFd *target_pidfd;
  UniqueFd *target_procfd;
  if (enabled) {
    target_pidfd = &session_it->second.target_pidfd;
    target_procfd = &session_it->second.target_procfd;
  } else {
    const int pin_result = pin_process_identity(target_pid, new_target_pidfd, new_target_procfd);
    if (pin_result != 0)
      return pin_result;
    target_pidfd = &new_target_pidfd;
    target_procfd = &new_target_procfd;
  }

  UniqueFd new_debugger_pidfd;
  UniqueFd new_debugger_procfd;
  if (args->op == KFD_IOC_DBG_TRAP_ENABLE) {
    const int debugger_pin_result =
        pin_process_identity(caller.client_pid(), new_debugger_pidfd, new_debugger_procfd);
    if (debugger_pin_result != 0)
      return debugger_pin_result;
  }

  // PTRACE gate: for any op other than DISABLE, a debugger acting on another
  // process must be that exact process's ptrace parent. This mirrors
  // ptrace_parent(target->lead_thread) == current in kfd_ioctl_set_debug_trap().
  if (!self_debug && args->op != KFD_IOC_DBG_TRAP_DISABLE) {
    pid_t tracer_pid = 0;
    const int tracer_result =
        tracer_pid_of(*target_pidfd, *target_procfd, debug_identity_validation_hook_, tracer_pid);
    if (tracer_result != 0) {
      if (tracer_result == -ESRCH && enabled)
        debug_sessions_.erase(target_pid);
      return tracer_result;
    }
    if (tracer_pid != caller.client_pid())
      return -EPERM;
  }

  // Resolve live GPU state only after pinning and authorizing the OS identity,
  // so a KfdProcess associated with a reused numeric pid is never selected.
  std::shared_ptr<KfdProcess> target_ref =
      self_debug ? nullptr : find_process_by_client_pid(target_pid);
  KfdProcess *target_proc = self_debug ? &caller : target_ref.get();

  // Live runtime-enable state, set by ROCr's AMDKFD_IOC_RUNTIME_ENABLE on the
  // inferior; false until the inferior connects and enables its runtime.
  bool runtime_enabled = false;
  if (target_proc != nullptr) {
    std::lock_guard<std::mutex> rlk(target_proc->runtime_mutex_);
    runtime_enabled = target_proc->runtime_state_.enabled;
  }

  // The target may exit after authorization. Revalidate before performing or
  // committing an operation so a reused numeric pid cannot contribute live
  // KfdProcess state to the pinned session.
  const int still_live = pidfd_is_exited(target_pidfd->get());
  if (still_live != 0) {
    if (still_live == 1 && enabled)
      debug_sessions_.erase(target_pid);
    return still_live == 1 ? -ESRCH : still_live;
  }
  if (args->op == KFD_IOC_DBG_TRAP_ENABLE) {
    const int debugger_still_live = pidfd_is_exited(new_debugger_pidfd.get());
    if (debugger_still_live != 0)
      return debugger_still_live == 1 ? -ESRCH : debugger_still_live;
  }

  // https://github.com/torvalds/linux/blob/a635d6748234582ea287c5ffeae28b9b23f91c7e/drivers/gpu/drm/amd/amdkfd/kfd_chardev.c#L3132-L3142
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

  // https://github.com/torvalds/linux/blob/a635d6748234582ea287c5ffeae28b9b23f91c7e/drivers/gpu/drm/amd/amdkfd/kfd_chardev.c#L3144
  if (args->op == KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH ||
      args->op == KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH) {
    const uint32_t gpu_id = args->op == KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH
                                ? args->set_node_address_watch.gpu_id
                                : args->clear_node_address_watch.gpu_id;
    if (find_gpu(gpu_id) == nullptr)
      return -ENODEV;
  }

  // https://github.com/torvalds/linux/blob/a635d6748234582ea287c5ffeae28b9b23f91c7e/drivers/gpu/drm/amd/amdkfd/kfd_chardev.c#L3158
  switch (args->op) {
  // https://github.com/torvalds/linux/blob/a635d6748234582ea287c5ffeae28b9b23f91c7e/drivers/gpu/drm/amd/amdkfd/kfd_debug.c#L788-L847
  case KFD_IOC_DBG_TRAP_ENABLE: {
    if (enabled)
      return -EALREADY; // target process is already debug enabled

    const int dbg_fd = static_cast<int>(args->enable.dbg_fd);
    // Validate the notifier before trusting it. In daemon mode the fd was
    // received via SCM_RIGHTS and already substituted into our fd space; in
    // local mode it is the debugger's own descriptor. Either way the driver
    // *writes* to it to wake the debugger, so it must be a live, writable
    // descriptor. safe_fcntl(F_GETFL) both proves the fd is open (EBADF otherwise)
    // and reports its access mode, so a read-only or otherwise unusable fd —
    // e.g. one a client passed over SCM_RIGHTS that is not a real event target —
    // is rejected instead of being stored on the session.
    const int fl = safe_fcntl(dbg_fd, F_GETFL);
    if (fl == -1 || (fl & O_ACCMODE) == O_RDONLY)
      return -EBADF;

    KfdProcess::DebugSession sess{};
    sess.target_pidfd = std::move(new_target_pidfd);
    sess.target_procfd = std::move(new_target_procfd);
    sess.enabled = true;
    sess.debugger_pid = caller.client_pid();
    sess.debugger_pidfd = std::move(new_debugger_pidfd);
    sess.dbg_fd = dbg_fd;
    sess.exception_enable_mask = args->enable.exception_mask;

    // Snapshot the runtime-enable state under a single lock so the marshaled
    // runtime_state, r_debug and ttmp_setup stay mutually consistent: a
    // concurrent RUNTIME_ENABLE/DISABLE must not change them between reads.
    // Lock order debug_sessions_mutex_ -> runtime_mutex_ is already held that
    // way.
    // Kernel: kfd_dbg_trap_enable copies the saved runtime info and returns its
    // size.
    kfd_runtime_info info{};
    if (target_proc != nullptr) {
      std::lock_guard<std::mutex> rlk(target_proc->runtime_mutex_);
      const auto &rt = target_proc->runtime_state_;
      sess.runtime_state = rt.enabled ? DEBUG_RUNTIME_STATE_ENABLED : DEBUG_RUNTIME_STATE_DISABLED;
      info.r_debug = rt.r_debug;
      info.ttmp_setup = (rt.mode_mask & KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK) ? 1u : 0u;
    }
    info.runtime_state = sess.runtime_state;
    size_t copy_size = std::min(static_cast<size_t>(args->enable.rinfo_size), sizeof(info));
    if (args->enable.rinfo_ptr != 0 && copy_size > 0)
      std::memcpy(reinterpret_cast<void *>(static_cast<uintptr_t>(args->enable.rinfo_ptr)), &info,
                  copy_size);
    args->enable.rinfo_size = sizeof(info);

    const int target_commit_live = pidfd_is_exited(sess.target_pidfd.get());
    if (target_commit_live != 0)
      return target_commit_live == 1 ? -ESRCH : target_commit_live;
    const int debugger_commit_live = pidfd_is_exited(sess.debugger_pidfd.get());
    if (debugger_commit_live != 0)
      return debugger_commit_live == 1 ? -ESRCH : debugger_commit_live;

    // In daemon mode the session owns the transferred fd and releases it via
    // RAII (on DISABLE or process teardown). In local mode dbg_fd is the
    // debugger's own descriptor, left for the debugger to close.
    if (daemon_mode_)
      sess.owned_dbg_fd = UniqueFd(dbg_fd);
    debug_sessions_.emplace(target_pid, std::move(sess));
    debug_sessions_cv_.notify_one();
    return 0;
  }
  case KFD_IOC_DBG_TRAP_DISABLE:
    // Erasing releases the SCM_RIGHTS-transferred notifier in daemon mode. In
    // local mode the session does not own the debugger's descriptor.
    {
      std::lock_guard<std::mutex> event_lock(debug_events_mutex_);
      debug_events_.erase(target_pid);
    }
    if (target_proc != nullptr) {
      std::lock_guard<std::mutex> alloc_lock(target_proc->alloc_mutex_);
      for (auto &[queue_id, queue] : target_proc->queue_snapshot_map_) {
        [[maybe_unused]] const auto id = queue_id;
        queue.exception_status = 0;
      }
    }
    debug_sessions_.erase(target_pid);
    return 0;
  case KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED:
    // kfd_dbg_set_enabled_debug_exception_mask(): record the exceptions the
    // debugger wants forwarded. Delivery is wired up with the event channel.
    session_it->second.exception_enable_mask = args->set_exceptions_enabled.exception_mask;
    return 0;
  case KFD_IOC_DBG_TRAP_SET_FLAGS: {
    const uint32_t previous = session_it->second.flags;
    session_it->second.flags = args->set_flags.flags;
    args->set_flags.flags = previous;
    return 0;
  }
  case KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE:
    if (args->launch_mode.launch_mode != KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL &&
        args->launch_mode.launch_mode != KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT &&
        args->launch_mode.launch_mode != KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG)
      return -EINVAL;
    session_it->second.launch_mode = args->launch_mode.launch_mode;
    return 0;
  case KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE: {
    if (args->launch_override.override_mode != KFD_DBG_TRAP_OVERRIDE_OR)
      return -EINVAL;
    constexpr uint32_t kGfx94SupportedTrapMask = KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH;
    if ((args->launch_override.support_request_mask & ~kGfx94SupportedTrapMask) != 0)
      return -EACCES;
    const uint32_t previous = session_it->second.launch_override_enable;
    session_it->second.launch_override_enable = args->launch_override.enable_mask;
    args->launch_override.enable_mask = previous;
    args->launch_override.support_request_mask = kGfx94SupportedTrapMask;
    return 0;
  }
  case KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH: {
    auto &watches = session_it->second.address_watches;
    uint32_t slot = 0;
    while (slot < KfdProcess::DebugSession::kMaxAddressWatches && watches[slot].active)
      ++slot;
    if (slot == KfdProcess::DebugSession::kMaxAddressWatches)
      return -ENOMEM;
    watches[slot].active = true;
    watches[slot].address = args->set_node_address_watch.address;
    watches[slot].mask = args->set_node_address_watch.mask;
    watches[slot].mode = args->set_node_address_watch.mode;
    args->set_node_address_watch.id = slot;
    return 0;
  }
  case KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH: {
    const uint32_t slot = args->clear_node_address_watch.id;
    if (slot >= KfdProcess::DebugSession::kMaxAddressWatches)
      return -EINVAL;
    session_it->second.address_watches[slot] = {};
    return 0;
  }
  case KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT:
    return debug_query_event(target_pid, session_it->second.exception_enable_mask,
                             args->query_debug_event);
  case KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT:
    return debug_queue_snapshot(target_proc, args->queue_snapshot);
  case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:
    return static_cast<int>(args->suspend_queues.num_queues);
  case KFD_IOC_DBG_TRAP_RESUME_QUEUES:
    resume_debug_queues(target_proc);
    return static_cast<int>(args->resume_queues.num_queues);
  case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
    return debug_device_snapshot(args->device_snapshot);
  case KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO:
    return debug_query_exception_info(target_pid, args->query_exception_info);
  case KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT: {
    std::lock_guard<std::mutex> runtime_lock(runtime_handshake_mutex_);
    runtime_acked_.insert(target_pid);
    runtime_handshake_cv_.notify_all();
    return 0;
  }
  default:
    return -EINVAL;
  }
}

int SimulatedKfd::debug_device_snapshot(kfd_ioctl_dbg_trap_device_snapshot_args &args) {
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

int SimulatedKfd::debug_queue_snapshot(KfdProcess *target,
                                       kfd_ioctl_dbg_trap_queue_snapshot_args &args) {
  // Mirrors pqm_get_queue_snapshot(): report the total queue count, fill only
  // the caller's capacity, and use the input entry size as the output stride.
  const uint32_t in_num = args.num_queues;
  const uint32_t in_entry_size = args.entry_size;

  args.num_queues = 0;
  if (in_entry_size == 0)
    return -EINVAL;
  args.entry_size = std::min<uint32_t>(in_entry_size, sizeof(kfd_queue_snapshot_entry));

  std::vector<kfd_queue_snapshot_entry> entries;
  if (target != nullptr) {
    std::lock_guard<std::mutex> lk(target->alloc_mutex_);
    entries.reserve(std::min<size_t>(in_num, target->active_queue_ids_.size()));
    for (uint32_t qid : target->active_queue_ids_) {
      auto it = target->queue_snapshot_map_.find(qid);
      if (it == target->queue_snapshot_map_.end())
        continue;
      if (args.num_queues < in_num) {
        KfdProcess::QueueSnapshotInfo &q = it->second;
        entries.push_back({
            .exception_status = q.exception_status,
            .ring_base_address = q.ring_base_address,
            .write_pointer_address = q.write_pointer_address,
            .read_pointer_address = q.read_pointer_address,
            .ctx_save_restore_address = q.ctx_save_restore_address,
            .queue_id = qid,
            .gpu_id = q.gpu_id,
            .ring_size = q.ring_size,
            .queue_type = q.queue_type,
            .ctx_save_restore_area_size = q.ctx_save_restore_area_size,
            .reserved = 0,
        });
        q.exception_status &= ~args.exception_mask;
      }
      ++args.num_queues;
    }
  }

  if (entries.empty())
    return 0;
  if (args.snapshot_buf_ptr == 0)
    return -EFAULT;

  auto *out = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(args.snapshot_buf_ptr));
  for (size_t i = 0; i < entries.size(); ++i)
    std::memcpy(out + static_cast<uint64_t>(i) * in_entry_size, &entries[i], args.entry_size);
  return 0;
}

int SimulatedKfd::set_xnack_mode_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_set_xnack_mode_args *>(arg);
  args->xnack_enabled = 0;
  return 0;
}

bool SimulatedKfd::owns_fd(int fd) const {
  if (fd < 0)
    return false;
  std::lock_guard<std::mutex> lock(owned_fds_mutex_);
  return owned_fds_.contains(fd);
}

void SimulatedKfd::init_reserved_fd_range() {
  struct rlimit rl {};
  getrlimit(RLIMIT_NOFILE, &rl);
  reserved_fd_base_ = static_cast<int>(rl.rlim_cur) - kReservedFdCount;
  next_reserved_fd_ = reserved_fd_base_;
}

int SimulatedKfd::claim_fd(int real_fd) {
  if (reserved_fd_base_ == 0)
    init_reserved_fd_range();
  int vfd = next_reserved_fd_++;
  assert(vfd < reserved_fd_base_ + kReservedFdCount && "reserved fd range exhausted");
  libc_passthrough().dup2(real_fd, vfd);
  libc_passthrough().close(real_fd);
  return vfd;
}

bool SimulatedKfd::owns_reserved_fd(int fd) const {
  return reserved_fd_base_ > 0 && fd >= reserved_fd_base_ &&
         fd < reserved_fd_base_ + kReservedFdCount;
}

int SimulatedKfd::get_mmap_memfd(off_t offset) const {
  return get_mmap_memfd(local_process_id_, offset);
}

int SimulatedKfd::get_mmap_memfd(uint32_t process_id, off_t offset) const {
  auto p = find_process(process_id);
  if (!p)
    return -1;
  return dispatch_get_mmap_memfd(*p, offset);
}

int SimulatedKfd::dispatch_get_mmap_memfd(KfdProcess &proc, off_t offset) const {
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
