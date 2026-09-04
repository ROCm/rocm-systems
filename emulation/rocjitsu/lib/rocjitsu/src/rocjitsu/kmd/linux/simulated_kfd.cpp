// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/simulated_kfd.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/cwsr.h"
#include "rocjitsu/kmd/linux/host_mapping_lock.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/kfd_topology.h"
#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/hwreg.h"
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
#include <iterator>
#include <linux/types.h>
#include <poll.h>
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
// pidfd_open(2) landed in Linux 5.3. The build hosts are older than their
// kernels here -- gcc-toolset-13 on the CI base image ships sanitized headers
// that predate it -- so the wrapper and even __NR_pidfd_open can be missing
// while the running kernel implements it fine. 434 is the number on every
// architecture ROCm targets; the syscall was added after new numbers began
// being allocated identically across arches.
#ifndef SYS_pidfd_open
#ifdef __NR_pidfd_open
#define SYS_pidfd_open __NR_pidfd_open
#else
#define SYS_pidfd_open 434
#endif
#endif
#include <thread>
#include <unistd.h>
#include <utility>
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
///
/// Bypassing the interposer also bypasses the interposer's mapping lock, and an
/// ioctl-routed mmap reaches here without ever passing through it. So it takes
/// the lock itself: an emulated atomic holds a raw pointer across a permission
/// check and the store it authorises, and a replacement in between would land
/// that store in whatever took the page's place.
///
/// Held for the syscall alone. The driver's surrounding mapping work re-enters
/// the memory model and takes its VMID lock, which an atomic already holds when
/// it reaches this lock, so widening the region would invert the two orders.
void *safe_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  auto mapping_lock = rocjitsu::host_mapping_lock().lock_exclusive();
  return libc_passthrough().mmap(addr, length, prot, flags, fd, offset);
}

/// @brief munmap via the real libc, with the mapping layout held still.
/// @details Withdrawing a page is the sharper half of the hazard safe_mmap()
/// describes: an atomic that has already validated the page is left storing
/// through a pointer to nothing. Allocation teardown drops the GPU page-table
/// entry first, so a concurrent access can miss the page table, fall through to
/// the identity path, and validate the very host page this call removes.
int safe_munmap(void *addr, size_t length) {
  auto mapping_lock = rocjitsu::host_mapping_lock().lock_exclusive();
  return libc_passthrough().munmap(addr, length);
}

/// @brief mprotect via the real libc, with the mapping layout held still.
/// @details Revoking write permission is as damaging to an in-flight atomic as
/// removing the page, and for the same reason.
int safe_mprotect(void *addr, size_t length, int prot) {
  auto mapping_lock = rocjitsu::host_mapping_lock().lock_exclusive();
  return libc_passthrough().mprotect(addr, length, prot);
}

/// @brief Return whether two non-empty half-open address ranges overlap.
/// @details Uses subtraction instead of computing either end address, avoiding
/// integer overflow for a malformed range near UINTPTR_MAX.
bool ranges_overlap(const void *lhs, size_t lhs_size, const void *rhs, size_t rhs_size) {
  if (lhs_size == 0 || rhs_size == 0)
    return false;
  const auto lhs_base = reinterpret_cast<uintptr_t>(lhs);
  const auto rhs_base = reinterpret_cast<uintptr_t>(rhs);
  return lhs_base <= rhs_base ? rhs_base - lhs_base < lhs_size : lhs_base - rhs_base < rhs_size;
}

/// @brief Return whether one non-empty address range fully contains another.
/// @details Like ranges_overlap(), avoids computing an end address so malformed
/// ranges near UINTPTR_MAX cannot wrap around.
bool range_contains(const void *outer, size_t outer_size, const void *inner, size_t inner_size) {
  if (outer_size == 0 || inner_size == 0)
    return false;
  const auto outer_base = reinterpret_cast<uintptr_t>(outer);
  const auto inner_base = reinterpret_cast<uintptr_t>(inner);
  if (inner_base < outer_base)
    return false;
  const auto offset = inner_base - outer_base;
  return offset <= outer_size && inner_size <= outer_size - offset;
}

/// @brief Move-only owner for an mmap until it is published into driver state.
class UniqueMapping {
public:
  UniqueMapping() = default;
  UniqueMapping(void *addr, size_t size) : addr_(addr), size_(size) {}
  ~UniqueMapping() { reset(); }

  UniqueMapping(const UniqueMapping &) = delete;
  UniqueMapping &operator=(const UniqueMapping &) = delete;

  UniqueMapping(UniqueMapping &&other) noexcept
      : addr_(std::exchange(other.addr_, MAP_FAILED)), size_(std::exchange(other.size_, 0)) {}
  UniqueMapping &operator=(UniqueMapping &&other) noexcept {
    if (this != &other) {
      reset();
      addr_ = std::exchange(other.addr_, MAP_FAILED);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  [[nodiscard]] explicit operator bool() const { return addr_ != MAP_FAILED; }
  [[nodiscard]] void *get() const { return addr_; }
  [[nodiscard]] void *release() {
    size_ = 0;
    return std::exchange(addr_, MAP_FAILED);
  }
  void reset(void *addr = MAP_FAILED, size_t size = 0) {
    if (addr_ != MAP_FAILED)
      safe_munmap(addr_, size_);
    addr_ = addr;
    size_ = size;
  }

private:
  void *addr_ = MAP_FAILED;
  size_t size_ = 0;
};

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

/// @brief Report whether the process behind @p procfd has exited but not been
/// reaped.
///
/// @details A pidfd stays readable for a zombie, so pidfd_is_exited() cannot
/// tell a live debuggee from one that has already run to completion. The state
/// character in /proc/<pid>/stat can. It sits after the last ')' because the
/// comm field is parenthesised and may itself contain spaces and parentheses.
///
/// @retval 1 The target is a zombie (Z) or dead (X).
/// @retval 0 The target is still running.
/// @retval <0 Negative errno; the state could not be determined.
int procfd_is_zombie(int procfd) {
  // Passthrough, not ::openat/::read/::close: these run inside driver ioctls that
  // the interposer dispatched, so a bare ::close() here would re-enter the
  // interposer's own close() hook mid-dispatch. See PassthroughFdTraits.
  UniqueDriverFd stat_fd(libc_passthrough().openat(procfd, "stat", O_RDONLY | O_CLOEXEC, 0));
  if (stat_fd.get() < 0)
    return errno == ENOENT ? 1 : -errno;
  char buffer[4096];
  const ssize_t bytes = libc_passthrough().read(stat_fd.get(), buffer, sizeof(buffer) - 1);
  const int read_error = errno;
  stat_fd.reset();
  if (bytes < 0)
    return -read_error;
  buffer[bytes] = '\0';
  const char *name_end = std::strrchr(buffer, ')');
  if (name_end == nullptr || name_end[1] != ' ' || name_end[2] == '\0')
    return -EIO;
  return name_end[2] == 'Z' || name_end[2] == 'X';
}

int pin_process_identity(pid_t pid, UniqueDriverFd &pidfd, UniqueDriverFd &procfd) {
  const int raw_pidfd = static_cast<int>(::syscall(SYS_pidfd_open, pid, 0));
  if (raw_pidfd < 0)
    return errno == ESRCH ? -ESRCH : -errno;
  pidfd = UniqueDriverFd(raw_pidfd);

  const std::string proc_path = "/proc/" + std::to_string(pid);
  const int raw_procfd =
      libc_passthrough().openat(AT_FDCWD, proc_path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC, 0);
  if (raw_procfd < 0) {
    const int open_error = errno;
    const int exited = pidfd_is_exited(pidfd.get());
    return exited == 1 ? -ESRCH : (exited < 0 ? exited : -open_error);
  }
  procfd = UniqueDriverFd(raw_procfd);

  const int exited = pidfd_is_exited(pidfd.get());
  return exited == 0 ? 0 : (exited == 1 ? -ESRCH : exited);
}

/// @brief Read TracerPid through a procfs directory pinned to the pidfd identity.
int tracer_pid_of(const UniqueDriverFd &pidfd, const UniqueDriverFd &procfd,
                  const SimulatedKfd::DebugIdentityValidationHook &validation_hook,
                  pid_t &tracer_pid) {
  int exited = pidfd_is_exited(pidfd.get());
  if (exited != 0)
    return exited == 1 ? -ESRCH : exited;

  UniqueDriverFd status_fd(
      libc_passthrough().openat(procfd.get(), "status", O_RDONLY | O_CLOEXEC, 0));
  if (status_fd.get() < 0) {
    const int open_error = errno;
    const int exited = pidfd_is_exited(pidfd.get());
    return exited == 1 ? -ESRCH : (exited < 0 ? exited : -open_error);
  }

  std::string status;
  char buffer[4096];
  for (;;) {
    const ssize_t count = libc_passthrough().read(status_fd.get(), buffer, sizeof(buffer));
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
                              amdgpu::Mtype mtype, KfdProcess::HostExtentOwner owner) {
  util::Logger::cp("MAP pid=", proc.process_id(), " va=0x", std::hex, gpu_va, " size=0x", size,
                   std::dec, " mtype=", static_cast<int>(mtype));
  proc.map_pages(gpu_va, host_ptr, size, mtype, owner);
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
  GpuDevice device;
  device.soc = &soc;
  gpus_.push_back(std::move(device));
}

SimulatedKfd::SimulatedKfd(std::vector<SoC *> socs, std::vector<uint32_t> gpu_ids, bool daemon_mode,
                           DebugIdentityValidationHook debug_identity_validation_hook)
    : daemon_mode_(daemon_mode),
      debug_identity_validation_hook_(std::move(debug_identity_validation_hook)),
      debug_session_reaper_([this](std::stop_token stop) { reap_exited_debug_sessions(stop); }) {
  libc_passthrough().resolve();
  for (size_t i = 0; i < socs.size(); ++i) {
    GpuDevice device;
    device.soc = socs[i];
    device.gpu_id = i < gpu_ids.size() ? gpu_ids[i] : socs[i]->gpu_id();
    gpus_.push_back(std::move(device));
  }
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

  // The memory model may outlive this driver, and it holds a bare pointer to
  // us for fault reporting. Retract it while we are still whole.
  for (auto &g : gpus_) {
    if (auto *mem = g.soc ? g.soc->memory() : nullptr)
      mem->set_memory_fault_reporter(nullptr);
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
    std::vector<std::pair<pid_t, std::shared_ptr<KfdProcess>>> released;
    for (auto it = debug_sessions_.begin(); it != debug_sessions_.end();) {
      if (pidfd_is_exited(it->second.debugger_pidfd.get()) == 1) {
        // The debugger is gone and will never ack; release any inferior
        // blocked in the RUNTIME_ENABLE handshake for it.
        cancel_runtime_handshake(it->first);
        // Resolve the debuggee now, while the session's pinned identity still
        // vouches for the pid. After the erase the number alone could name a
        // process that merely reused it.
        released.emplace_back(it->first, find_process_by_client_pid(it->first));
        it = debug_sessions_.erase(it);
      } else if (pidfd_is_exited(it->second.target_pidfd.get()) == 1) {
        if (!it->second.target_exited) {
          if (auto target = find_process_by_client_pid(it->first))
            revoke_target_mem_routing(target->process_id());
          it->second.owned_dbg_fd.reset();
          it->second.target_mem_fd.reset();
          it->second.dbg_fd = -1;
          it->second.enabled = false;
          it->second.target_exited = true;
        }
        ++it;
      } else {
        ++it;
      }
    }
    // A crashed debugger must not strand the inferior. Erasing the session only
    // stops new debug traffic; the waves it left stopped, the closed queue
    // gates and the target-memory routing all have to be undone too, exactly as
    // an explicit detach does. Done outside debug_sessions_mutex_ because the
    // release takes CU wave-state locks in the opposite order to the engine
    // thread.
    if (!released.empty()) {
      lock.unlock();
      // The shared_ptr in `released` is what keeps proc alive across the
      // unlocked release; release_debuggee_state only borrows it.
      for (auto &[pid, proc] : released)
        release_debuggee_state(pid, proc.get());
      lock.lock();
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
  gpu_infos_ = std::move(infos);
  topology_.generate(gpu_infos_);
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
  std::lock_guard<std::mutex> lock(p->alloc_mutex_);
  for (const auto &gs : p->gpu_state_) {
    if (ranges_overlap(addr, length, gs.doorbell_monitor_page, gs.doorbell_page_size))
      return true;
    for (const auto &view : gs.doorbell_views)
      if (ranges_overlap(addr, length, view.page, gs.doorbell_page_size))
        return true;
  }
  return false;
}

void *SimulatedKfd::mmap_replacing_client_doorbell_views(void *addr, size_t length, int prot,
                                                         int flags, int fd, off_t offset) {
  auto proc = find_process(local_process_id_);
  if (!proc)
    return safe_mmap(addr, length, prot, flags, fd, offset);

  // Serialize the replacement with KFD teardown and doorbell mmap. Keep alloc_mutex_ held through
  // the real mmap so mprotect/munmap cannot observe a retired view before MAP_FIXED has replaced
  // it.
  std::lock_guard<std::mutex> op_lock(proc->op_mutex_);
  std::lock_guard<std::mutex> alloc_lock(proc->alloc_mutex_);

  for (const auto &gs : proc->gpu_state_) {
    if (ranges_overlap(addr, length, gs.doorbell_monitor_page, gs.doorbell_page_size)) {
      errno = EINVAL;
      return MAP_FAILED;
    }
    for (const auto &view : gs.doorbell_views) {
      if (ranges_overlap(addr, length, view.page, gs.doorbell_page_size) &&
          !range_contains(addr, length, view.page, gs.doorbell_page_size)) {
        // Doorbell GPU mappings and ownership are tracked for the complete
        // client view. Replacing only part would leave the remaining CPU range
        // live but no longer tracked or GPU-mapped.
        errno = EINVAL;
        return MAP_FAILED;
      }
    }
  }

  struct RetiredView {
    size_t gpu_ordinal;
    size_t page_size;
    KfdProcess::PerGpuState::DoorbellView view;
  };
  std::vector<RetiredView> retired;
  for (size_t ord = 0; ord < proc->gpu_state_.size(); ++ord) {
    auto &gs = proc->gpu_state_[ord];
    for (auto view = gs.doorbell_views.begin(); view != gs.doorbell_views.end();) {
      if (!ranges_overlap(addr, length, view->page, gs.doorbell_page_size)) {
        ++view;
        continue;
      }
      retired.push_back({.gpu_ordinal = ord, .page_size = gs.doorbell_page_size, .view = *view});
      view = gs.doorbell_views.erase(view);
    }
  }

  for (const auto &entry : retired)
    unmap_from_gpu(*proc, entry.view.gpu_va, entry.page_size);

  void *mapped = safe_mmap(addr, length, prot, flags, fd, offset);
  if (mapped != MAP_FAILED)
    return mapped;

  int mmap_errno = errno;
  for (const auto &entry : retired) {
    map_to_gpu(*proc, entry.view.gpu_va, entry.view.page, entry.page_size, amdgpu::Mtype::UC);
    proc->gpu_state_[entry.gpu_ordinal].doorbell_views.push_back(entry.view);
  }
  errno = mmap_errno;
  return MAP_FAILED;
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
    // The WAVES field the CP writes into compute_tmpring_size must be a
    // multiple of the per-XCC shader-engine count, or rocdbgapi disables
    // private memory access outright (architecture.cpp,
    // gfx9_architecture_t::scratch_memory_region). It recovers that count from
    // the topology we publish: os_driver_kfd.cpp derives
    // shader_engine_count = array_count * num_xcc / simd_arrays_per_engine and
    // then divides by xcc_count, which inverts Sysfs::GpuInfo's
    // array_count_per_xcc() exactly back to num_shader_engines. Read that same
    // field here rather than recomputing a quotient, so the value the CP rounds
    // to and the divisor rocdbgapi applies cannot drift apart.
    const uint32_t scratch_wave_divisor =
        i < gpu_infos_.size() ? std::max(1u, gpu_infos_[i].num_shader_engines) : 1u;
    const uint32_t xcc_count = g.soc->num_xcds();
    for (uint32_t xcc_id = 0; xcc_id < xcc_count; ++xcc_id) {
      auto *cp = g.soc->xcd(xcc_id)->command_processor();
      if (!cp)
        continue;
      cp->set_scratch_wave_divisor(scratch_wave_divisor);
      cp->set_scratch_xcc_layout(xcc_id, xcc_count);
    }
    // Same source as the apertures GET_PROCESS_APERTURES_NEW and the DBG_TRAP
    // device snapshot advertise: what the shaders translate LDS/scratch against
    // must be what the runtime and the debugger were told.
    const kfd_process_device_apertures ap = gpu_apertures(static_cast<uint32_t>(i));
    g.soc->set_apertures(ap.lds_base, ap.lds_limit, ap.scratch_base, ap.scratch_limit);
    g.soc->for_each_cp([this, i](amdgpu::CommandProcessor *cp) {
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
        const uint32_t gpu_ordinal = static_cast<uint32_t>(i);
        cu->set_trap_handler_resolver([this, gpu_ordinal](const amdgpu::Wavefront &wf) {
          return resolve_trap_handler(wf, gpu_ordinal);
        });
        cu->set_sendmsg_handler([this](amdgpu::Wavefront &wf, uint32_t message) {
          return on_wave_sendmsg(wf, message);
        });
        cu->set_queue_exception_handler(
            [this, gpu_ordinal](uint32_t queue_id, uint32_t process_id, uint64_t status) {
              if (gpu_ordinal >= gpus_.size())
                return false;
              return signal_runtime_queue_exception(gpus_[gpu_ordinal].gpu_id, queue_id, process_id,
                                                    status);
            });
        cu->set_trap_completion_handler(
            [this](amdgpu::Wavefront &wf) { on_wave_trap_complete(wf); });
        cu->set_single_step_handler(
            [this](amdgpu::Wavefront &wf) { return on_wave_single_step_complete(wf); });
        cu->set_watchpoint_handler([this](amdgpu::Wavefront &wf, uint64_t address, uint32_t bytes,
                                          bool is_write, bool is_atomic) {
          return on_wave_watchpoint(wf, address, bytes, is_write, is_atomic);
        });
        cu->set_illegal_inst_handler(
            [this](amdgpu::Wavefront &wf) { return on_wave_illegal_instruction(wf); });
        cu->set_alu_exception_handler(
            [this](amdgpu::Wavefront &wf) { return on_wave_alu_exception(wf); });
        cu->set_memory_violation_handler(
            [this](amdgpu::Wavefront &wf, uint64_t address, bool is_write) {
              return on_wave_memory_violation(wf, address, is_write);
            });
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
  // inherits this cache stale, but under the fork-then-exec contract that child
  // never reaches the driver at all (owner_pid_ gates every interposed entry
  // point), so the stale value is unobservable until exec replaces the image.
  proc->set_client_pid(static_cast<pid_t>(getpid()));
  proc->event_state_.reset();
  for (auto &g : gpus_) {
    if (auto *mem = g.soc ? g.soc->memory() : nullptr) {
      mem->register_process(pid, &proc->page_table_, &proc->page_table_mutex_,
                            proc->page_table_generation(), proc->page_table_request_mutex());
      mem->set_memory_fault_reporter(fault_reporter_for(g));
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

SimulatedKfd::GpuFaultReporter *SimulatedKfd::fault_reporter_for(GpuDevice &gpu) {
  if (!gpu.fault_reporter)
    gpu.fault_reporter = std::make_unique<GpuFaultReporter>(this, gpu.gpu_id);
  return gpu.fault_reporter.get();
}

void SimulatedKfd::report_memory_fault(uint32_t vmid, uint64_t addr, uint32_t gpu_id,
                                       amdgpu::MemoryFaultCause cause) {
  std::shared_ptr<KfdProcess> proc;
  {
    std::lock_guard<std::mutex> lk(process_mutex_);
    auto it = processes_.find(vmid);
    if (it == processes_.end())
      return;
    proc = it->second;
  }

  MemoryFault fault{};
  fault.va = addr;
  fault.gpu_id = gpu_id;
  // Indeterminate deliberately sets neither failure bit: the violation is real
  // but its cause was never established, and naming a protection the address
  // may not have would send the runtime after the wrong problem. It does not
  // set `imprecise` either -- that reports an inexact faulting address, which
  // is not what is uncertain here.
  fault.not_present = cause == amdgpu::MemoryFaultCause::NotPresent;
  fault.read_only = cause == amdgpu::MemoryFaultCause::ReadOnly;

  // Reporting is best effort by design: a process that never created a
  // memory-exception event has no channel for this, exactly as on hardware,
  // and the warning is the only remaining place the violation can surface.
  if (!proc->event_state_.signal_memory_fault(fault)) {
    util::Logger::warn("GPU memory violation at 0x", std::hex, addr, std::dec, " (process ", vmid,
                       ") has no registered memory-exception event to report it on");
  }
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
        mem->register_process(pid, &proc->page_table_, &proc->page_table_mutex_,
                              proc->page_table_generation(), proc->page_table_request_mutex());
        mem->set_memory_fault_reporter(fault_reporter_for(g));
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

  if (client_pid > 0) {
    std::lock_guard<std::mutex> debug_lock(debug_sessions_mutex_);
    auto session = debug_sessions_.find(client_pid);
    if (session != debug_sessions_.end() && session->second.target_mem_fd.get() >= 0) {
      for (auto &g : gpus_)
        if (auto *mem = g.soc ? g.soc->memory() : nullptr)
          mem->set_process_mem_fd(pid, session->second.target_mem_fd.get());
    }
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

void SimulatedKfd::begin_local_shutdown() {
  // Release an indefinite-timeout WAIT_EVENTS on the local process so it returns
  // and drops the driver snapshot that would otherwise keep this object alive.
  // Snapshot the process, then fire the wake with process_mutex_ RELEASED:
  // begin_wait_cancel() takes the process's own event mutex, and WAIT_EVENTS does
  // not take process_mutex_, so the parked thread can make progress.
  //
  // Deliberately NOT notify_closing()/signal_page_shutdown(): those turn live ioctls
  // into -EBADF/-ESRCH and poison the signal page, which is the caller-visible part
  // of teardown and belongs to close(). What this does instead is release the
  // waiters, so the ordering is "wake, then destroy" rather than two half-teardowns.
  // The wake is one-way for this driver's life: the interposer calls it only after
  // it has committed to teardown, so nothing clears the flags and every subsequent
  // wait reports the same benign timeout.
  if (auto proc = find_local_process()) {
    proc->event_state_.begin_wait_cancel();
    // WAIT_EVENTS is not the only call that parks. RUNTIME_ENABLE waits on the
    // debugger's acknowledgement, and is dispatched outside op_mutex_ precisely
    // because it blocks -- so that thread holds a driver snapshot for the whole
    // wait and teardown would sit behind it until the handshake deadline expired.
    // Cancelling is equally pure: the waiter reports the handshake as cancelled
    // and no event, page or process state is mutated.
    cancel_runtime_handshake(proc->client_pid());
  }
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
  std::vector<std::pair<pid_t, std::shared_ptr<KfdProcess>>> released;
  {
    std::lock_guard<std::mutex> debug_lock(debug_sessions_mutex_);
    for (auto it = debug_sessions_.begin(); it != debug_sessions_.end();) {
      if (pidfd_is_exited(it->second.debugger_pidfd.get()) == 1) {
        // The debugger is gone and will never ack; release any inferior
        // blocked in the RUNTIME_ENABLE handshake for it.
        cancel_runtime_handshake(it->first);
        // Resolve the debuggee now, while the session's pinned identity still
        // vouches for the pid. After the erase the number alone could name a
        // process that merely reused it.
        released.emplace_back(it->first, find_process_by_client_pid(it->first));
        it = debug_sessions_.erase(it);
      } else if (pidfd_is_exited(it->second.target_pidfd.get()) == 1) {
        if (auto target = find_process_by_client_pid(it->first))
          revoke_target_mem_routing(target->process_id());
        it->second.owned_dbg_fd.reset();
        it->second.target_mem_fd.reset();
        it->second.dbg_fd = -1;
        it->second.enabled = false;
        it->second.target_exited = true;
        ++it;
      } else {
        ++it;
      }
    }
  }
  // Same obligation as the background reaper: erasing the session only stops
  // new debug traffic, so a debugger that died before this close still has to
  // have its inferior's waves resumed, queue gates reopened and target-memory
  // routing revoked. Done after the debug_sessions_mutex_ scope above, because
  // this takes CU wave-state locks and the engine thread takes those first.
  for (auto &[pid, proc_ref] : released)
    release_debuggee_state(pid, proc_ref.get());

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
        safe_munmap(alloc.host_ptr, alloc.size);
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

  // Doorbell mappings live in gpu_state_, not allocations_. Snapshot and clear
  // every client view plus the stable monitor alias under alloc_mutex_, then
  // release the GPU and CPU mappings outside the lock.
  for (size_t ord = 0; ord < proc.gpu_state_.size(); ++ord) {
    auto &gs = proc.gpu_state_[ord];
    int doorbell_memfd;
    void *doorbell_monitor_page;
    size_t doorbell_page_size;
    std::vector<KfdProcess::PerGpuState::DoorbellView> doorbell_views;
    {
      std::lock_guard<std::mutex> alk(proc.alloc_mutex_);
      doorbell_memfd = gs.doorbell_memfd;
      doorbell_monitor_page = gs.doorbell_monitor_page;
      doorbell_page_size = gs.doorbell_page_size;
      doorbell_views = std::move(gs.doorbell_views);
      gs.doorbell_memfd = -1;
      gs.doorbell_monitor_page = nullptr;
      gs.doorbell_page_size = 0;
    }
    update_cp_doorbell_base(static_cast<uint32_t>(ord), process_id, nullptr);
    for (const auto &view : doorbell_views) {
      if (view.gpu_va && doorbell_page_size)
        unmap_from_gpu(proc, view.gpu_va, doorbell_page_size);
      if (view.page != MAP_FAILED && doorbell_page_size)
        safe_munmap(view.page, doorbell_page_size);
    }
    if (doorbell_monitor_page && doorbell_page_size)
      safe_munmap(doorbell_monitor_page, doorbell_page_size);
    if (doorbell_memfd >= 0) {
      {
        std::lock_guard<std::mutex> flk(owned_fds_mutex_);
        owned_fds_.erase(doorbell_memfd);
      }
      libc_passthrough().close(doorbell_memfd);
    }
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

int SimulatedKfd::ioctl(uint32_t process_id, unsigned long request, void *arg, int *target_mem_fd,
                        int target_proc_fd) {
  auto proc = find_process(process_id);
  if (!proc)
    return -ESRCH;
  return dispatch_ioctl(*proc, request, arg, target_mem_fd, target_proc_fd);
}

int SimulatedKfd::dispatch_ioctl(KfdProcess &proc, unsigned long request, void *arg,
                                 int *target_mem_fd, int target_proc_fd) {
  util::Logger::driver("IOCTL pid=", proc.process_id(), " ", LinuxKfd::ioctl_name(request));

  unsigned long dispatch_request = canonical_ioctl_request(request);

  if (dispatch_request == AMDKFD_IOC_WAIT_EVENTS)
    return wait_events_ioctl(proc, arg);

  // RUNTIME_ENABLE blocks too: when a debugger is attached it waits for that
  // debugger to acknowledge EC_PROCESS_RUNTIME. Holding op_mutex_ across that
  // wait would serialize the ack behind it whenever the debugger and the target
  // share a KfdProcess (self-debug, and daemon clients that reuse one pid), so
  // it takes the lock itself around the state mutation only.
  if (dispatch_request == AMDKFD_IOC_RUNTIME_ENABLE)
    return runtime_enable_ioctl(proc, arg);

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
    // RUNTIME_ENABLE is handled before op_mutex_ above (it blocks on the
    // debugger handshake), so it never reaches this switch.
    case AMDKFD_IOC_DBG_TRAP:
      return debug_trap_ioctl(proc, arg, target_mem_fd, target_proc_fd);
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
        // Trap entry resolves these fields on the engine thread while ioctls may
        // update them, so publish and consume them under process_mutex_.
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
    if (!find_gpu(db_gpu_id)) {
      errno = EINVAL;
      return MAP_FAILED;
    }
    uint32_t ord = gpu_ordinal(db_gpu_id);

    // Serialize the canonical backing, both mappings, GPU page-table publication,
    // and CP base update against process teardown and another doorbell mmap.
    std::lock_guard<std::mutex> op_lock(proc.op_mutex_);
    if (proc.event_state_.is_closing()) {
      errno = ENODEV;
      return MAP_FAILED;
    }

    int doorbell_fd = -1;
    int source_doorbell_fd = -1;
    void *monitor_ptr = nullptr;
    size_t published_size = 0;
    {
      std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
      auto &gs = proc.gpu(ord);
      doorbell_fd = gs.doorbell_memfd;
      monitor_ptr = gs.doorbell_monitor_page;
      published_size = gs.doorbell_page_size;
      if (doorbell_fd < 0) {
        for (auto &[handle, alloc] : proc.allocations_) {
          if ((alloc.flags & KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL) && alloc.gpu_id == db_gpu_id) {
            source_doorbell_fd = alloc.memfd;
            break;
          }
        }
      }
    }

    UniqueDriverFd new_doorbell_fd;
    if (doorbell_fd < 0) {
      // Retain a private descriptor even when a KFD doorbell allocation supplied
      // the source. The allocation can be freed independently; this duplicate keeps
      // the canonical backing valid until the per-process doorbell state is torn down.
      UniqueDriverFd created_source;
      if (source_doorbell_fd < 0) {
        // Local-mode ROCr can request a doorbell mmap without first allocating a
        // KFD doorbell object. Give that path the same persistent shared backing.
        created_source.reset(memfd_create("rocjitsu_doorbell", MFD_CLOEXEC | MFD_ALLOW_SEALING));
        if (!created_source)
          return MAP_FAILED;
        source_doorbell_fd = created_source.get();
      }
      new_doorbell_fd.reset(safe_fcntl(source_doorbell_fd, F_DUPFD_CLOEXEC, 4096));
      if (!new_doorbell_fd && created_source)
        new_doorbell_fd = std::move(created_source);
      if (!new_doorbell_fd)
        return MAP_FAILED;
      doorbell_fd = new_doorbell_fd.get();
    }

    // A process/GPU owns exactly one monitor view. Doorbell mappings have a fixed
    // KFD page size; rejecting a conflicting remap avoids invalidating live queue
    // offsets or changing the address polled by the CP.
    if (monitor_ptr && published_size != length) {
      new_doorbell_fd.reset();
      errno = EINVAL;
      return MAP_FAILED;
    }
    // MAP_FIXED replaces every existing mapping in its target range. Never let a
    // client-selected address replace the CP's private alias; that would leave
    // live queues polling an invalid address.
    if ((flags & MAP_FIXED) && ranges_overlap(addr, length, monitor_ptr, published_size)) {
      new_doorbell_fd.reset();
      errno = EINVAL;
      return MAP_FAILED;
    }

    off_t cur_size = 0;
    {
      struct stat st {};
      if (safe_fstat(doorbell_fd, &st) == 0)
        cur_size = st.st_size;
    }
    if (static_cast<off_t>(length) > cur_size &&
        ftruncate(doorbell_fd, static_cast<off_t>(length)) != 0) {
      int saved_errno = errno;
      new_doorbell_fd.reset();
      errno = saved_errno;
      return MAP_FAILED;
    }

    int db_mflags = MAP_SHARED;
    if (flags & MAP_FIXED)
      db_mflags |= MAP_FIXED;

    UniqueMapping new_monitor_mapping;
    if (!monitor_ptr) {
      // Initialize doorbell backing to 0xFF via the CP's permanent mapping so
      // every slot matches the HwQueue::last_doorbell sentinel. This also avoids
      // SIGBUS on the final MAP_SHARED mmap on Linux 6.17+ where
      // shmem large folio allocation can fail during a bulk memset on a
      // freshly-mapped region. Writing through a separate PROT_WRITE
      // mapping forces page allocation before the client mapping is published. Keeping this
      // alias gives the simulated CP a stable device-side view even while the
      // runtime is creating or replacing its own mapping of the same memfd.
      void *new_monitor = MAP_FAILED;
      void *forced_monitor_addr =
          next_doorbell_monitor_mmap_addr_.exchange(nullptr, std::memory_order_acq_rel);
      if (fail_next_doorbell_monitor_mmap_.exchange(false, std::memory_order_acq_rel)) {
        errno = ENOMEM;
      } else {
        int monitor_flags = MAP_SHARED;
        if (forced_monitor_addr)
          monitor_flags |= MAP_FIXED_NOREPLACE;
        new_monitor = safe_mmap(forced_monitor_addr, length, PROT_READ | PROT_WRITE, monitor_flags,
                                doorbell_fd, 0);
      }
      new_monitor_mapping.reset(new_monitor, length);
      if (!new_monitor_mapping) {
        int saved_errno = errno;
        new_doorbell_fd.reset();
        errno = saved_errno;
        return MAP_FAILED;
      }

      // Allocate the stable alias before the destructive client MAP_FIXED. If
      // the kernel placed an alias in the requested client range, retain it as
      // a reservation while establishing another. Once an alias lands fully
      // outside the target, the reservations can be released and the final
      // fixed mapping cannot replace the CP's private alias.
      std::vector<UniqueMapping> monitor_reservations;
      while ((flags & MAP_FIXED) &&
             ranges_overlap(addr, length, new_monitor_mapping.get(), length)) {
        monitor_reservations.push_back(std::move(new_monitor_mapping));
        new_monitor_mapping.reset(
            safe_mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, doorbell_fd, 0), length);
        if (!new_monitor_mapping) {
          int saved_errno = errno;
          new_monitor_mapping.reset();
          monitor_reservations.clear();
          new_doorbell_fd.reset();
          errno = saved_errno;
          return MAP_FAILED;
        }
      }
      monitor_ptr = new_monitor_mapping.get();
      std::memset(monitor_ptr, 0xFF, length);
    }

    UniqueMapping client_mapping(
        safe_mmap(addr, length, PROT_READ | PROT_WRITE, db_mflags, doorbell_fd, 0), length);
    if (!client_mapping) {
      int mmap_errno = errno;
      new_monitor_mapping.reset();
      new_doorbell_fd.reset();
      errno = mmap_errno;
      return MAP_FAILED;
    }
    void *ptr = client_mapping.get();
    if (ranges_overlap(ptr, length, monitor_ptr, length)) {
      client_mapping.reset();
      new_monitor_mapping.reset();
      new_doorbell_fd.reset();
      errno = EINVAL;
      return MAP_FAILED;
    }

    {
      std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
      auto &gs = proc.gpu(ord);
      if (new_doorbell_fd) {
        assert(gs.doorbell_memfd < 0);
        gs.doorbell_memfd = new_doorbell_fd.release();
        {
          std::lock_guard<std::mutex> flk(owned_fds_mutex_);
          owned_fds_.insert(gs.doorbell_memfd);
        }
      } else {
        assert(gs.doorbell_memfd == doorbell_fd);
      }
      assert(!gs.doorbell_monitor_page || gs.doorbell_monitor_page == monitor_ptr);
      if (std::ranges::none_of(gs.doorbell_views,
                               [ptr](const auto &view) { return view.page == ptr; })) {
        gs.doorbell_views.push_back({.page = ptr, .gpu_va = reinterpret_cast<uint64_t>(ptr)});
      }
      gs.doorbell_monitor_page = monitor_ptr;
      gs.doorbell_page_size = length;
    }

    static_cast<void>(new_monitor_mapping.release());
    static_cast<void>(client_mapping.release());
    map_to_gpu(proc, reinterpret_cast<uint64_t>(ptr), ptr, length, amdgpu::Mtype::UC);
    update_cp_doorbell_base(ord, proc.process_id(), monitor_ptr);
    return ptr;
  }

  if (type == KFD_MMAP_TYPE_EVENTS) {
    // Create-or-get the backing as ONE locked operation. Two concurrent event-page
    // mmaps would otherwise both observe no backing, each build one, and hand
    // different fds to different callers -- leaving one polling an object that
    // never receives event updates -- besides racing the field itself.
    const int events_fd = proc.event_state_.ensure_backing(length, [&](size_t size) -> int {
      auto raw_events_fd = memfd_create("rocjitsu_events", MFD_CLOEXEC | MFD_ALLOW_SEALING);
      if (raw_events_fd < 0)
        return -1;
      int backing = safe_fcntl(raw_events_fd, F_DUPFD_CLOEXEC, 4096);
      if (backing < 0)
        backing = raw_events_fd;
      else
        libc_passthrough().close(raw_events_fd);
      {
        std::lock_guard<std::mutex> lk(owned_fds_mutex_);
        owned_fds_.insert(backing);
      }
      if (ftruncate(backing, static_cast<off_t>(size)) != 0) {
        const int ftruncate_errno = errno; // preserve across close() below
        {
          std::lock_guard<std::mutex> lk(owned_fds_mutex_);
          owned_fds_.erase(backing);
        }
        libc_passthrough().close(backing);
        errno = ftruncate_errno;
        return -1;
      }
      fallocate(backing, 0, 0, static_cast<off_t>(size));
      {
        auto *init_ptr =
            static_cast<uint8_t *>(safe_mmap(nullptr, size, PROT_WRITE, MAP_SHARED, backing, 0));
        if (init_ptr != MAP_FAILED) {
          libc_passthrough().madvise(init_ptr, size, MADV_POPULATE_WRITE);
          std::memset(init_ptr, 0xFF, size);
          safe_munmap(init_ptr, size);
        }
      }
      safe_fcntl(backing, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW);
      return backing;
    });
    if (events_fd < 0)
      return MAP_FAILED;
    int mflags = MAP_SHARED;
    if (flags & MAP_FIXED)
      mflags |= MAP_FIXED;
    void *ptr = safe_mmap(addr, length, PROT_READ | PROT_WRITE, mflags, events_fd, 0);
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
      auto prot_rc = safe_mprotect(addr, length, PROT_READ | PROT_WRITE);
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
          safe_munmap(temp_mapping, length);
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
      auto rc = safe_mprotect(addr, length, PROT_READ | PROT_WRITE);
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
    uint64_t doorbell_gpu_va = 0;
    int doorbell_memfd = -1;
    void *doorbell_monitor_page = nullptr;
    size_t doorbell_page_size = 0;
    bool is_doorbell = false;
    bool last_doorbell_view = false;
    {
      std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
      for (const auto &gs : proc.gpu_state_) {
        if (ranges_overlap(addr, length, gs.doorbell_monitor_page, gs.doorbell_page_size)) {
          errno = EPERM;
          return -1;
        }
      }
      for (size_t ord = 0; ord < proc.gpu_state_.size(); ++ord) {
        auto &gs = proc.gpu(ord);
        auto view = std::ranges::find_if(
            gs.doorbell_views, [addr](const auto &candidate) { return candidate.page == addr; });
        if (view == gs.doorbell_views.end())
          continue;
        if (!proc.event_state_.is_closing()) {
          errno = EPERM;
          return -1;
        }
        doorbell_gpu_va = view->gpu_va;
        doorbell_page_size = gs.doorbell_page_size;
        gs.doorbell_views.erase(view);
        last_doorbell_view = gs.doorbell_views.empty();
        if (last_doorbell_view) {
          doorbell_memfd = gs.doorbell_memfd;
          doorbell_monitor_page = gs.doorbell_monitor_page;
          gs.doorbell_memfd = -1;
          gs.doorbell_monitor_page = nullptr;
          gs.doorbell_page_size = 0;
        }
        doorbell_ord = static_cast<uint32_t>(ord);
        is_doorbell = true;
        break;
      }
    }
    if (is_doorbell) {
      if (doorbell_gpu_va && doorbell_page_size)
        unmap_from_gpu(proc, doorbell_gpu_va, doorbell_page_size);

      // Clear the CP's doorbell base for this process BEFORE munmapping its alias.
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
      if (last_doorbell_view)
        update_cp_doorbell_base(doorbell_ord, proc.process_id(), nullptr);
      if (doorbell_monitor_page && doorbell_page_size)
        safe_munmap(doorbell_monitor_page, doorbell_page_size);
      // Unmap the exact page we mapped: use the recorded doorbell page size, not
      // the caller-provided length. A length that differs from the tracked mapping
      // would otherwise partially unmap the CPU page and leave it inconsistent with
      // the GPU page-table unmap above.
      if (addr != MAP_FAILED && doorbell_page_size)
        safe_munmap(addr, doorbell_page_size);
      if (doorbell_memfd >= 0) {
        {
          std::lock_guard<std::mutex> flk(owned_fds_mutex_);
          owned_fds_.erase(doorbell_memfd);
        }
        libc_passthrough().close(doorbell_memfd);
      }
      return 0;
    }
  }
  // release_page() clears page_/page_size_ under EventState::mutex_, the same lock
  // the CP interrupt thread holds when reading them in signal_interrupt, so the
  // munmap below cannot race a concurrent signal writing into the mapping.
  if (proc.event_state_.release_page(addr)) {
    safe_munmap(addr, length);
    return 0;
  }
  std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
  for (auto &[handle, alloc] : proc.allocations_) {
    if (alloc.host_ptr == addr) {
      unmap_from_gpu(proc, alloc.gpu_va, alloc.size);
      safe_munmap(addr, length);
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
  const uint32_t filled = std::min(n, args->num_of_nodes);
  for (uint32_t i = 0; i < filled; ++i)
    apertures[i] = gpu_apertures(i);

  // The count written back is how many entries were filled, not how many nodes
  // exist -- kfd_ioctl_get_process_apertures_new() reports the loop index. A
  // caller whose buffer was smaller than the node count would otherwise iterate
  // past its own allocation over entries this call never wrote.
  args->num_of_nodes = filled;
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
            // Driver-owned: this is our memfd, mapped read-write here and held
            // open, so nothing outside can change its protection or unmap it.
            map_to_gpu(proc, va, alloc.host_ptr, alloc.size, alloc_mtype,
                       KfdProcess::HostExtentOwner::Driver);
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

  // Every XCD of a fanned-out dispatch races here for the same process-wide pool
  // VA, each having independently found it unbacked. Only the first may map it:
  // mapping again would repoint the pool underneath waves the first XCD already
  // launched and leak the original. The requested size is grid-scale and therefore
  // identical for every XCD of one grid, so a pool already at least that large
  // satisfies all of them and this covers the whole fan-out race.
  //
  // A later dispatch in the same process needing a LARGER pool still falls through
  // and remaps, as it did before fan-out existed. That path is unchanged here.
  std::lock_guard<std::mutex> scratch_lock(proc->scratch_backing_mutex_);
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    for (const auto &[handle, alloc] : proc->allocations_) {
      if (alloc.gpu_va == gpu_va && alloc.size >= aligned_size)
        return true;
    }
  }

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
  // Driver-owned for the same reason as the allocation path: the backing is a
  // memfd this function created and mapped read-write.
  proc->map_pages(gpu_va, host_ptr, aligned_size, amdgpu::Mtype::RW,
                  KfdProcess::HostExtentOwner::Driver);

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

  // Queue IDs are process-local and start at one. Equivalent runtime queues in
  // different processes therefore share XCD resources while each process still
  // distributes additional queues across the device.
  const uint32_t queue_ordinal = proc.next_queue_id_ - 1;
  auto *target_cp = gpu->soc->assign_queue_owner_cp(queue_ordinal);
  if (!target_cp)
    return -EINVAL;
  const uint32_t target_xcc_id = gpu->soc->queue_xcd_id(queue_ordinal);

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
      map_to_gpu(proc, args->read_pointer_address,
                 reinterpret_cast<void *>(args->read_pointer_address), sizeof(uint64_t),
                 amdgpu::Mtype::UC);
      if (args->write_pointer_address != args->read_pointer_address)
        map_to_gpu(proc, args->write_pointer_address,
                   reinterpret_cast<void *>(args->write_pointer_address), sizeof(uint64_t),
                   amdgpu::Mtype::UC);
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
    if (recycled_offset && gs.doorbell_monitor_page &&
        db_offset + sizeof(uint64_t) <= gs.doorbell_page_size) {
      std::atomic_ref<uint64_t>(
          *reinterpret_cast<uint64_t *>(static_cast<char *>(gs.doorbell_monitor_page) + db_offset))
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
    assert(gs.doorbell_views.empty() || gs.doorbell_monitor_page);
    hw.doorbell_base = gs.doorbell_monitor_page;
    hw.last_doorbell = ~uint64_t(0);
    hw.host_accessible = true;
    hw.is_sdma = (args->queue_type == KFD_IOC_QUEUE_TYPE_SDMA ||
                  args->queue_type == KFD_IOC_QUEUE_TYPE_SDMA_XGMI ||
                  args->queue_type == KFD_IOC_QUEUE_TYPE_SDMA_BY_ENG_ID);
    // The topology advertises every XCD's compute units as one agent, so a
    // compute dispatch must be able to reach all of them. Without this a
    // single-queue application would only ever use the XCD that
    // assign_queue_owner_cp() happened to pick.
    //
    // Named types only, rather than "anything that is not SDMA". SDMA queues are
    // per-engine and are not spread, but an unrecognized queue_type is not
    // thereby a compute queue, and device-wide replication should not be what an
    // unsupported value silently acquires.
    hw.xcd_fanout = (args->queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE ||
                     args->queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE_AQL);
    // amd_queue_t base: write_pointer_address points to write_dispatch_id.
    if (!hw.is_sdma)
      hw.queue_desc_va = args->write_pointer_address - offsetof(amd_queue_t, write_dispatch_id);
    if (!hw.is_sdma && args->ctx_save_restore_address != 0) {
      constexpr uint32_t kErrorReasonOffset = 6 * sizeof(uint32_t);
      constexpr uint32_t kErrorEventIdOffset = kErrorReasonOffset + sizeof(uint64_t);
      hw.exception_status_va = target_cp->read_process_memory64(
          args->ctx_save_restore_address + kErrorReasonOffset, proc.process_id());
      hw.exception_event_id = static_cast<uint32_t>(target_cp->read_process_memory64(
          args->ctx_save_restore_address + kErrorEventIdOffset, proc.process_id()));
    }
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
        // The gfx1250 debugger ABI publishes each queue in its owning XCC
        // save area. Preserve the legacy single-area model for older targets.
        .xcc_id = gpu->soc->arch() == ROCJITSU_CODE_ARCH_CDNA5 ? target_xcc_id : 0,
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
                         args->ring_size, args->queue_percentage);
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

      proc.remap_page_host_ptrs(alloc.gpu_va, alloc.host_ptr, new_host_ptr, alloc.size);

      if (alloc.host_ptr_owned)
        safe_munmap(alloc.host_ptr, alloc.size);

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
    proc.set_page_mtype(alloc.gpu_va, alloc.size, amdgpu::Mtype::CC);

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
  // Whether a debugger was already attached when the runtime came up, decided
  // under the same lock that DBG_TRAP_ENABLE uses to publish a session.
  bool notify_debugger = false;
  // Same decision for the disable direction, which reports the same way.
  bool notify_disable = false;
  {
    // Scoped so neither lock is held across the handshake wait below.
    std::lock_guard<std::mutex> op_lock(proc.op_mutex_);
    // debug_sessions_mutex_ orders this against DBG_TRAP_ENABLE, which holds it
    // while it reads runtime_state_ to fill in the runtime_info it returns.
    // Exactly one of the two has to take responsibility for telling the
    // debugger the runtime is up: either ENABLE observes enabled==true and
    // reports it (rocdbgapi then calls runtime_enable() straight from its
    // attach path, process.cpp), or this side observes the session and raises
    // EC_PROCESS_RUNTIME. Without a common lock both can read the other's
    // pre-state -- ENABLE reports DISABLED while this side sees no session --
    // and NEITHER fires -- the debugger then waits for an event that is never
    // coming and never learns the runtime came up.
    //
    // Closing this did NOT measurably change gdb.rocm/multi-inferior-stress.exp,
    // which fails for a different reason (see below); it is fixed here because
    // the window is real on its own terms, not because it cured that test.
    // Taking it here also cannot double-report: whichever side takes the lock
    // second observes the first, and rocdbgapi treats a runtime_state that does
    // not toggle as a fatal "spurious runtime exception".
    std::lock_guard<std::mutex> session_lock(debug_sessions_mutex_);
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
      auto session = debug_sessions_.find(proc.client_pid());
      notify_debugger = session != debug_sessions_.end() && session->second.enabled;
    } else {
      // Report the !disabled -> disabled transition. rocdbgapi needs it to stop
      // driving the queues of a process whose runtime has gone away: those ops
      // are gated on the runtime being enabled and answer -EPERM, which it
      // turns into a fatal os_driver::resume_queues failure and GDB into an
      // internal error (gdb.rocm/multi-inferior-stress.exp). Only report a real
      // transition -- a second disable is not one, and rocdbgapi rejects an
      // event whose runtime_state has not moved as spurious. Decided against
      // the session table under the same lock the enable branch uses, so an
      // undebugged process exit does not re-take debug_sessions_mutex_ in the
      // callee just to discover there is nobody to tell.
      auto session = debug_sessions_.find(proc.client_pid());
      notify_disable = proc.runtime_state_.enabled && session != debug_sessions_.end() &&
                       session->second.enabled;
      proc.runtime_state_ = KfdProcess::RuntimeState{};
      args->capabilities_mask = 0;
    }
  }

  // Ordered after the state reset above so that the debugger's follow-up
  // QUERY_EXCEPTION_INFO reads the new state, and outside the lock scope
  // because the handshake blocks.
  if (notify_debugger || notify_disable)
    runtime_debugger_handshake(proc.client_pid(), /*enabling=*/notify_debugger);
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

uint32_t gfx1250_status(uint32_t status) {
  // SCC, priorities and HALT moved to STATE_PRIV. The remaining fields used by
  // dbgapi retain their gfx9 positions.
  return status & ~((0x1Fu << 0) | (1u << 7) | (1u << 13) | (1u << 19));
}

uint32_t internal_status_from_gfx1250(const kmd::CwsrWaveState &state) {
  uint32_t status = amdgpu::update_status_from_gfx12_state_priv(state.status, state.state_priv);
  status &= ~amdgpu::Wavefront::kStatusHaltMask;
  if (state.saved_status_halt)
    status |= amdgpu::Wavefront::kStatusHaltMask;
  return status;
}

uint32_t internal_trapsts_from_gfx1250(const kmd::CwsrWaveState &state) {
  uint32_t trapsts = amdgpu::update_trapsts_from_gfx12_excp_flag_priv(0, state.excp_flag_priv);
  return amdgpu::update_trapsts_from_gfx12_excp_flag_user(trapsts, state.excp_flag_user);
}

kmd::CwsrWaveState build_cwsr_wave_state(amdgpu::Wavefront &wf, rj_code_arch_t arch) {
  kmd::CwsrWaveState state;
  const bool gfx12_5 = kmd::cwsr_layout_kind(arch) == kmd::CwsrLayoutKind::Gfx12_5;
  const uint32_t raw_status = wf.status_raw();
  state.pc = wf.pc;
  // A wave frozen inside the trap handler has the HANDLER's EXEC live, not the
  // application's: the handler runs with its own mask and parks 0x80000000 (and
  // then a doorbell id) in EXEC_LO around MSG_GET_DOORBELL. Publishing that
  // makes the debugger report every lane active, which inverts `lane apply
  // -active/-inactive` (gdb.rocm/lane-info.exp). trap_saved_exec_ holds the
  // interrupted value for the whole handler, so un-shadow it here, as the
  // STATUS field below does with trap_saved_status().
  //
  // in_trap_handler() is the predicate for both, and it opens at handler entry
  // rather than at MSG_INTERRUPT because the doorbell park happens on the way
  // there. It closes at s_rfe, which puts the interrupted values back, so from
  // then on the live registers are the application's again.
  //
  // The record is not write-only -- apply_cwsr_to_wave() feeds it back on
  // resume -- so un-shadowing here is only safe because that path shadows in
  // the other direction under the same predicate: a debugger edit to EXEC is
  // written to trap_saved_exec_, which the handler's s_rfe installs, instead of
  // overwriting the mask a mid-flight handler is still running under.
  state.exec = wf.in_trap_handler() ? wf.trap_saved_exec() : wf.exec();
  state.vcc = wf.vcc();
  state.flat_scratch = wf.scratch_base();
  state.m0 = wf.m0();
  state.mode = wf.mode_raw();
  state.trapsts = wf.trapsts();
  // Whether the application was already halted when it trapped outlives the
  // handler: trap_saved_status_ still describes the interrupted wave after
  // s_rfe, and trap_interrupt_sent() is the flag that says a handler produced
  // this stop. in_trap_handler() has to be part of the predicate too, or the
  // two halves of the published STATUS come from different registers between
  // trap entry and MSG_INTERRUPT -- the body from trap_saved_status_ and the
  // HALT bit from the handler's live one. apply_cwsr_to_wave() recombines them
  // into a word that never existed on the wave, so an identity round-trip in
  // that window rewrites the application's saved HALT with the handler's.
  const bool handler_context = wf.in_trap_handler() || wf.trap_interrupt_sent();
  state.saved_status_halt = ((handler_context ? wf.trap_saved_status() : raw_status) >> 13) & 1u;
  state.wave_stopped = wf.debug_halted();
  // The published STATUS shadows on the narrower predicate, the same one EXEC
  // uses: only while the handler is actually running is the live register the
  // handler's rather than the application's. Once s_rfe has run, the live value
  // is the application's again and is what the debugger should see.
  const uint32_t application_status = wf.in_trap_handler() ? wf.trap_saved_status() : raw_status;
  state.status =
      state.wave_stopped ? application_status | (1u << 13) : application_status & ~(1u << 13);
  if (gfx12_5) {
    uint32_t debug_status = application_status;
    if (state.wave_stopped)
      debug_status |= amdgpu::Wavefront::kStatusHaltMask;
    else
      debug_status &= ~amdgpu::Wavefront::kStatusHaltMask;
    state.state_priv = amdgpu::gfx12_state_priv_from_status(debug_status, state.flat_scratch != 0);
    state.status = gfx1250_status(application_status);
    state.excp_flag_priv = amdgpu::gfx12_excp_flag_priv_from_trapsts(state.trapsts);
    state.excp_flag_user = (state.trapsts & 0x7Fu) | wf.gfx12_excp_flag_user_extra_raw();
    state.trap_ctrl = wf.gfx12_trap_ctrl_raw();
    state.xnack_state_priv = wf.gfx1250_xnack_state_priv_raw();
    state.xnack_mask = wf.gfx1250_xnack_mask_raw();
  }
  state.trap_id = wf.trap_id();
  state.wave_id = wf.debug_wave_id();
  state.group_ids = wf.wg_coord();
  state.wave_in_group = wf.wave_in_group();
  state.queue_packet_id = wf.aql_packet_id() & 0x1FFFFFFu;
  state.scratch_scoreboard_id = wf.scratch_scoreboard_id();
  state.shader_engine_id = wf.shader_engine_id();
  state.spi_ttmps_setup = true;
  state.num_sgprs = wf.num_sgprs();
  state.num_vgprs = wf.num_vgprs();

  state.sgprs.resize(state.num_sgprs);
  for (uint32_t s = 0; s < state.num_sgprs; ++s)
    state.sgprs[s] = wf.debug_read_sgpr(s);
  // The CWSR record's VGPR stride is wave64-shaped (cwsr.cpp kVgprLaneBytes)
  // whatever the wave size is, but the physical register only has wf_size()
  // lanes -- a wave32 VectorReg<32> is exactly 32 wide, so reading lanes 32-63
  // runs off it into the neighbouring register. Read the live lanes and leave
  // the rest of the wave64-shaped slot at the zero resize() already wrote; the
  // restore path is bounded by wf_size() too and never looks at them.
  state.vgprs.resize(static_cast<size_t>(state.num_vgprs) * 64);
  const uint32_t live_lanes = wf.wf_size();
  for (uint32_t r = 0; r < state.num_vgprs; ++r)
    for (uint32_t lane = 0; lane < live_lanes; ++lane)
      state.vgprs[static_cast<size_t>(r) * 64 + lane] = wf.debug_read_vgpr(r, lane);
  state.lds.resize(wf.lds_size());
  if (!state.lds.empty())
    static_cast<const amdgpu::Lds &>(wf.lds()).read(wf.lds_base(), state.lds.data(),
                                                    static_cast<uint32_t>(state.lds.size()));
  return state;
}

void prepare_cwsr_wave_states(std::vector<kmd::CwsrWaveState> &waves) {
  std::sort(waves.begin(), waves.end(), [](const auto &lhs, const auto &rhs) {
    if (lhs.queue_packet_id != rhs.queue_packet_id)
      return lhs.queue_packet_id < rhs.queue_packet_id;
    if (lhs.group_ids != rhs.group_ids)
      return lhs.group_ids < rhs.group_ids;
    return lhs.wave_in_group < rhs.wave_in_group;
  });
  auto same_group = [](const auto &lhs, const auto &rhs) {
    return lhs.queue_packet_id == rhs.queue_packet_id && lhs.group_ids == rhs.group_ids;
  };
  for (size_t index = 0; index < waves.size(); ++index) {
    waves[index].is_first_in_group = index == 0 || !same_group(waves[index], waves[index - 1]);
    waves[index].is_last_in_group =
        index + 1 == waves.size() || !same_group(waves[index], waves[index + 1]);
    if (!waves[index].is_first_in_group)
      waves[index].lds.clear();
  }
}

} // namespace

bool SimulatedKfd::serialize_queue_debug_waves(uint32_t process_id, uint32_t queue_id,
                                               uint32_t gpu_id, uint64_t ctx_base,
                                               uint32_t ctx_size) {
  auto *gpu = find_gpu(gpu_id);
  if (!gpu || !gpu->soc || ctx_base == 0)
    return false;

  // Never publish a record shaped for the wrong architecture. Handlers ask
  // debug_stop_publishable() before claiming a stop, so reaching here on an
  // unmodelled part means a caller skipped the gate; refuse rather than hand
  // rocm-dbgapi an image it would decode against its own layout.
  if (!kmd::cwsr_layout_modelled(gpu->soc->arch()))
    return false;

  const bool per_xcc_areas = gpu->soc->arch() == ROCJITSU_CODE_ARCH_CDNA5;
  const uint32_t area_count = per_xcc_areas ? std::max(1u, gpu->soc->num_xcds()) : 1u;
  std::vector<std::vector<kmd::CwsrWaveState>> waves_by_area(area_count);
  auto collect = [&](amdgpu::CommandProcessor *cp, uint32_t area) {
    for (auto *cu : cp->compute_units()) {
      cu->with_wave_state_locked([&] {
        for (uint32_t i = 0; i < cu->num_wf_slots(); ++i) {
          auto *wave = cu->wf(i);
          if (wave->debug_stopped() && wave->process_id() == process_id &&
              wave->queue_id() == queue_id)
            waves_by_area[area].push_back(build_cwsr_wave_state(*wave, gpu->soc->arch()));
        }
      });
    }
  };
  if (per_xcc_areas) {
    for (uint32_t xcc_id = 0; xcc_id < area_count; ++xcc_id)
      collect(gpu->soc->xcd(xcc_id)->command_processor(), xcc_id);
  } else {
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) { collect(cp, 0); });
  }
  size_t wave_count = 0;
  for (const auto &waves : waves_by_area)
    wave_count += waves.size();
  if (wave_count == 0)
    return false;

  auto *memory = gpu->soc->memory();
  auto proc = find_process(process_id);
  if (!proc)
    return false;
  UniqueDriverFd target_mem = duplicate_debug_target_mem(proc->client_pid());
  bool publish_ok = true;
  auto write_block = [&](uint64_t address, std::span<const uint8_t> bytes) {
    if (target_mem.get() < 0) {
      // A refused write publishes nothing, so reporting success would tell the
      // debugger a context is available at an address that holds none of it.
      // The pwrite path below fails publication on a short write for the same
      // reason; this is that failure arriving through the memory model.
      if (memory->write_block(address, bytes, process_id) == amdgpu::AccessOutcome::Faulted) {
        publish_ok = false;
        util::Logger::warn("CWSR target write faulted: addr=0x", std::hex, address, std::dec,
                           " pid=", proc->client_pid());
      }
      return;
    }
    const ssize_t written =
        pwrite(target_mem.get(), bytes.data(), bytes.size(), static_cast<off_t>(address));
    if (written != static_cast<ssize_t>(bytes.size())) {
      publish_ok = false;
      util::Logger::warn("CWSR target write failed: addr=0x", std::hex, address, " pid=", std::dec,
                         proc->client_pid(), " rc=", written, " errno=", errno);
    }
  };
  bool layouts_ok = true;
  constexpr size_t kCwsrHeaderBytes = 10 * sizeof(uint32_t);
  const std::array<uint8_t, kCwsrHeaderBytes> empty_header{};
  for (uint32_t area = 0; area < area_count; ++area) {
    const uint64_t area_base = ctx_base + static_cast<uint64_t>(area) * ctx_size;
    auto &waves = waves_by_area[area];
    if (waves.empty()) {
      write_block(area_base, empty_header);
      continue;
    }
    prepare_cwsr_wave_states(waves);
    layouts_ok &=
        kmd::serialize_queue_cwsr_bulk(area_base, ctx_size, waves, write_block, gpu->soc->arch())
            .ok;
  }
  if (!layouts_ok || !publish_ok)
    return false;

  uint64_t oldest_packet = UINT64_MAX;
  for (const auto &waves : waves_by_area)
    for (const auto &wave : waves)
      oldest_packet = std::min<uint64_t>(oldest_packet, wave.queue_packet_id);
  uint64_t read_pointer = 0;
  uint64_t write_pointer = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(queue_id);
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    read_pointer = queue->second.read_pointer_address;
    write_pointer = queue->second.write_pointer_address;
  }
  if (read_pointer != 0 && write_pointer != 0 &&
      oldest_packet < memory->read64(write_pointer, process_id))
    memory->write64(read_pointer, oldest_packet, process_id);
  return true;
}

void SimulatedKfd::on_wave_trap_complete(amdgpu::Wavefront &wave) {
  const uint32_t process_id = wave.process_id();
  const uint32_t queue_id = wave.queue_id();
  auto proc = find_process(process_id);
  if (!proc)
    return;
  const pid_t target_pid = proc->client_pid();

  const uint64_t queue_exception_status = wave.trap_queue_exception_status();
  const uint64_t debugger_status = kmd::detail::preserve_runtime_queue_exception_owner(
      queue_exception_status, wave.trap_runtime_exception_status(),
      debugger_queue_exception_mask(proc, queue_id, queue_exception_status));
  if (queue_exception_status != 0) {
    const uint64_t unreported =
        queue_exception_status & ~debugger_status & ~wave.trap_runtime_exception_status();
    if (unreported != 0) {
      wave.cu().signal_queue_exception(queue_id, process_id, unreported);
      wave.add_trap_runtime_exception_status(unreported);
    }
  }
  if (queue_exception_status != 0 && debugger_status == 0) {
    // The handler observed an attached debugger and requested HALT, but this
    // exception belongs to ROCr. Do not strand the wave as a debugger stop;
    // the deferred runtime route will apply the queue-wide fatal suspension.
    wave.set_debug_halted(false);
    wave.set_status_halt(false);
    return;
  }

  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session == debug_sessions_.end() || !session->second.enabled)
      return;
  }

  uint64_t ctx_base = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(queue_id);
    if (queue == proc->queue_snapshot_map_.end())
      return;
    ctx_base = queue->second.ctx_save_restore_address;
    gpu_id = queue->second.gpu_id;
  }
  if (ctx_base == 0)
    return;
  // The wave is already halted here by the handler's own STATUS.HALT, so there
  // is no stop to decline -- but waking a debugger that can never be given a
  // record only strands it. resolve_trap_handler() also withholds the debug
  // flag on such a part, so a cooperating handler never gets this far; one that
  // raises STATUS.HALT regardless still would.
  if (!debug_stop_publishable(gpu_id))
    return;

  apply_debug_event_claim_mask_for_testing(target_pid);

  // A trap interrupt is wave-local: hardware reports it without waiting for
  // every peer in the queue to stop. The debugger's ensuing SUSPEND_QUEUES
  // request publishes the authoritative full-queue CWSR snapshot.
  const bool notified = notify_debug_event(
      proc, queue_id, gpu_id,
      queue_exception_status != 0 ? debugger_status : KFD_EC_MASK(EC_QUEUE_WAVE_TRAP));
  if (!notified) {
    const uint64_t unreported = queue_exception_status & ~wave.trap_runtime_exception_status();
    if (unreported != 0 && wave.cu().signal_queue_exception(queue_id, process_id, unreported))
      wave.add_trap_runtime_exception_status(unreported);
    wave.set_debug_halted(false);
    wave.set_status_halt(false);
  }
}

uint64_t SimulatedKfd::debugger_queue_exception_mask(const std::shared_ptr<KfdProcess> &proc,
                                                     uint32_t queue_id, uint64_t exception_mask) {
  uint64_t enabled_mask = 0;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    if (session == debug_sessions_.end() || !session->second.enabled)
      return 0;
    enabled_mask = session->second.exception_enable_mask;
  }
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(queue_id);
    if (queue == proc->queue_snapshot_map_.end())
      return 0;
    gpu_id = queue->second.gpu_id;
  }
  // KFD makes one ownership decision for the complete event: any subscribed
  // bit assigns the full decoded mask to the debugger. Splitting a combined
  // event would let the runtime suspension prevent the handler from reaching
  // trap completion and publishing the debugger-owned portion.
  return debug_stop_publishable(gpu_id) && (exception_mask & enabled_mask) != 0 ? exception_mask
                                                                                : 0;
}

bool SimulatedKfd::signal_runtime_queue_exception(uint32_t gpu_id, uint32_t queue_id,
                                                  uint32_t process_id, uint64_t exception_mask) {
  auto *gpu = find_gpu(gpu_id);
  if (!gpu || !gpu->soc || exception_mask == 0)
    return false;

  // Every XCD owns a replica of a fanned-out queue. Serialize the shared
  // status-word update, stop every replica, and let only the first match
  // publish the one runtime interrupt.
  std::lock_guard<std::mutex> lock(queue_exception_mutex_);
  amdgpu::CommandProcessor *publisher = nullptr;
  gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
    if (cp->signal_queue_exception(queue_id, process_id, exception_mask,
                                   /*publish_interrupt=*/false) &&
        publisher == nullptr)
      publisher = cp;
  });
  return publisher != nullptr &&
         publisher->signal_queue_exception(queue_id, process_id, exception_mask,
                                           /*publish_interrupt=*/true);
}

std::optional<amdgpu::ComputeUnitCore::TrapHandlerConfig>
SimulatedKfd::resolve_trap_handler(const amdgpu::Wavefront &wave, uint32_t gpu_ordinal) {
  std::shared_ptr<KfdProcess> proc;
  uint64_t tba = 0;
  uint64_t tma = 0;
  {
    std::lock_guard<std::mutex> lk(process_mutex_);
    auto proc_it = processes_.find(wave.process_id());
    if (proc_it == processes_.end() || gpu_ordinal >= proc_it->second->gpu_state_.size())
      return std::nullopt;
    proc = proc_it->second;
    tba = proc->gpu(gpu_ordinal).trap_tba_addr;
    tma = proc->gpu(gpu_ordinal).trap_tma_addr;
  }
  if (tba == 0)
    return std::nullopt;

  bool debug_enabled = false;
  {
    std::lock_guard<std::mutex> debug_lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    debug_enabled = session != debug_sessions_.end() && session->second.enabled;
  }
  // Do not tell the trap handler a debugger is attached on a part whose stops
  // could never be published; the handler would raise STATUS.HALT and wait for
  // a debugger that can never read it.
  if (debug_enabled && gpu_ordinal < gpus_.size() &&
      !debug_stop_publishable(gpus_[gpu_ordinal].gpu_id))
    debug_enabled = false;
  return amdgpu::ComputeUnitCore::TrapHandlerConfig{tba, tma, debug_enabled};
}

bool SimulatedKfd::on_wave_sendmsg(amdgpu::Wavefront &wave, uint32_t message) {
  constexpr uint32_t kMessageIdMask = 0xFu;
  constexpr uint32_t kMessageInterrupt = 1;
  constexpr uint32_t kMessageGetDoorbell = 10;
  constexpr uint32_t kDoorbellIdBits = 10;
  constexpr uint32_t kRuntimeQueueExceptionMask = 0x3Fu;
  const uint32_t message_id = message & kMessageIdMask;

  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;

  if (message_id == kMessageGetDoorbell) {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto doorbell = proc->queue_doorbell_map_.find(wave.queue_id());
    if (doorbell == proc->queue_doorbell_map_.end())
      return false;
    const uint32_t doorbell_id = (doorbell->second.doorbell_offset / sizeof(uint64_t)) & 0x3FFu;
    // The handler sets EXEC_LO[31] before issuing MSG_GET_DOORBELL and polls
    // until hardware clears it. Return the 10-bit ID with the pending bit clear.
    wave.set_exec((wave.exec() & 0xFFFFFFFF00000000ULL) | doorbell_id);
    return true;
  }

  if (message_id != kMessageInterrupt || !wave.in_trap_handler())
    return false;

  const auto arch = wave.cu().arch();
  bool profiling_interrupt = false;
  if (kmd::detail::uses_pre_gfx12_trap_interrupt_sites(arch)) {
    // ROCr's pre-GFX12 handler has two MSG_INTERRUPT sites. At the profiling
    // site it loads the event id from TTMP7; at the queue-exception site it
    // loads the packed exception from TTMP3. Both insert an s_nop immediately
    // before the message. Classify the actual control-flow site instead of
    // inspecting the TTMP values: a signal pointer can equal saved M0, and a
    // concurrent trap can legitimately rewrite those registers before the
    // second interrupt.
    if (wave.pc >= 2 * sizeof(uint32_t)) {
      const uint32_t payload_move =
          wave.cu().fetch_instruction_word(wave.pc - 2 * sizeof(uint32_t), wave.process_id());
      const uint32_t delay =
          wave.cu().fetch_instruction_word(wave.pc - sizeof(uint32_t), wave.process_id());
      const auto site =
          kmd::detail::classify_pre_gfx12_trap_interrupt_site(arch, payload_move, delay);
      if (site == kmd::detail::TrapInterruptSite::Profiling)
        profiling_interrupt = true;
      else if (site != kmd::detail::TrapInterruptSite::QueueException)
        return false;
    } else {
      return false;
    }
  } else if (arch == ROCJITSU_CODE_ARCH_CDNA5 || arch == ROCJITSU_CODE_ARCH_RDNA4) {
    // GFX12 exposes the host/performance causes in common TRAPSTS bits 22/26.
    profiling_interrupt = (wave.trapsts() & ((1u << 22) | (1u << 26))) != 0;
  }
  // Profiling completion uses MSG_INTERRUPT too, but puts an event id in M0.
  // Do not reinterpret that id as the queue-exception layout below.
  if (profiling_interrupt)
    return true;

  // The ROCr trap-handler ABI packs the 10-bit doorbell id below six KFD queue
  // exception bits in M0. A debugger owns exception routing while attached; its
  // CWSR publication is deferred until s_rfe applies the handler's STATUS.HALT.
  // Without a debugger, deliver the exception to ROCr's queue error event. The
  // CU defers the CP call until its wave-state lock is released.
  const uint64_t exception_status = (wave.m0() >> kDoorbellIdBits) & kRuntimeQueueExceptionMask;
  if (exception_status != 0) {
    wave.add_trap_queue_exception_status(exception_status);
    const uint64_t debugger_status =
        debugger_queue_exception_mask(proc, wave.queue_id(), exception_status);
    const uint64_t unreported =
        exception_status & ~debugger_status & ~wave.trap_runtime_exception_status();
    if (unreported != 0 &&
        wave.cu().signal_queue_exception(wave.queue_id(), wave.process_id(), unreported))
      wave.add_trap_runtime_exception_status(unreported);
  }

  return true;
}

bool SimulatedKfd::debug_stop_publishable(uint32_t gpu_id) {
  auto *gpu = find_gpu(gpu_id);
  if (gpu == nullptr || gpu->soc == nullptr)
    return false;
  // The codec reproduces the gfx9.4 CWSR layout only (kmd/linux/cwsr.h), and
  // the differences elsewhere are not confined to one field -- the control
  // stack, the COMPUTE_RELAUNCH bits, the wave64 VGPR stride, the SGPR alias
  // slots and the dispatch-identity TTMPs all move. rocm-dbgapi would decode
  // the image against its own layout and act on whatever it read.
  //
  // Asked here, before the stop, rather than refused at DBG_TRAP_ENABLE: every
  // errno that path can return except ESRCH and EALREADY becomes
  // AMD_DBGAPI_STATUS_ERROR, which rocm-dbgapi turns into a [[noreturn]]
  // fatal_error, so refusing there aborts the debugger instead of declining.
  // Declining cleanly is the topology's job (HSA_CAP_TRAP_DEBUG_SUPPORT), and
  // that surface is deliberately kept identical to the real KFD driver.
  if (kmd::cwsr_layout_modelled(gpu->soc->arch()))
    return true;
  std::lock_guard<std::mutex> warning_lock(cwsr_layout_warning_mutex_);
  if (!gpu->cwsr_layout_warned) {
    gpu->cwsr_layout_warned = true;
    util::Logger::warn(
        "wave stops not published on gpu_id=", gpu_id,
        ": no CWSR record layout is modelled for arch=", static_cast<int>(gpu->soc->arch()),
        "; GPU debugging is supported on gfx942/gfx950/gfx1250 only");
  }
  return false;
}

bool SimulatedKfd::report_wave_stopped(const std::shared_ptr<KfdProcess> &proc, uint32_t queue_id,
                                       uint32_t gpu_id, uint64_t ctx_base, uint32_t ctx_size,
                                       uint64_t exception_mask) {
  // Serialization must succeed before the debugger is woken. Event publication
  // latches per-queue exception status and queues an event the debugger will
  // answer with SUSPEND_QUEUES; without a record that request can only come
  // back as a queue error, and the wave stays halted with nothing able to
  // resume it. The stop itself cannot be deferred until after serialization --
  // the serializer selects waves by debug_stopped() -- so the caller undoes it.
  if (!serialize_queue_debug_waves(proc->process_id(), queue_id, gpu_id, ctx_base, ctx_size))
    return false;
  return notify_debug_event(proc, queue_id, gpu_id, exception_mask);
}

bool SimulatedKfd::notify_debug_event(const std::shared_ptr<KfdProcess> &proc, uint32_t queue_id,
                                      uint32_t gpu_id, uint64_t exception_mask) {
  const pid_t target_pid = proc->client_pid();

  // Claim and latch the event under one session snapshot. A concurrent mask
  // update may affect future events, but it cannot revoke an event that has
  // already been assigned to the debugger.
  UniqueDriverFd notifier;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session == debug_sessions_.end() || !session->second.enabled ||
        session->second.dbg_fd < 0 || (session->second.exception_enable_mask & exception_mask) == 0)
      return false;
    notifier = UniqueDriverFd(safe_fcntl(session->second.dbg_fd, F_DUPFD_CLOEXEC, 0));
    if (notifier.get() < 0)
      return false;
    uint64_t report_mask = exception_mask;
    {
      std::lock_guard<std::mutex> alloc_lock(proc->alloc_mutex_);
      auto queue = proc->queue_snapshot_map_.find(queue_id);
      if (queue != proc->queue_snapshot_map_.end()) {
        queue->second.exception_status |= exception_mask;
        report_mask |= queue->second.exception_status & KFD_EC_MASK(EC_QUEUE_NEW);
      }
    }
    std::lock_guard<std::mutex> event_lock(debug_events_mutex_);
    auto &event = debug_events_[target_pid][queue_id];
    event.gpu_id = gpu_id;
    event.mask |= report_mask;
  }
  const uint64_t one = 1;
  [[maybe_unused]] const ssize_t written = ::write(notifier.get(), &one, sizeof(one));
  return true;
}

void SimulatedKfd::set_debug_event_claim_mask_for_testing(uint64_t exception_mask) {
  std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
  debug_event_claim_mask_for_testing_ = exception_mask;
}

void SimulatedKfd::apply_debug_event_claim_mask_for_testing(pid_t target_pid) {
  std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
  if (!debug_event_claim_mask_for_testing_)
    return;
  const uint64_t exception_mask = *debug_event_claim_mask_for_testing_;
  debug_event_claim_mask_for_testing_.reset();
  auto session = debug_sessions_.find(target_pid);
  if (session != debug_sessions_.end())
    session->second.exception_enable_mask = exception_mask;
}

bool SimulatedKfd::on_wave_single_step_complete(amdgpu::Wavefront &wave) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  uint64_t ctx_base = 0;
  uint32_t gpu_id = 0;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    auto queue = proc->queue_snapshot_map_.find(wave.queue_id());
    if (queue == proc->queue_snapshot_map_.end())
      return false;
    ctx_base = queue->second.ctx_save_restore_address;
    gpu_id = queue->second.gpu_id;
  }
  if (ctx_base == 0)
    return false;
  if (!debug_stop_publishable(gpu_id)) {
    // Clear single-step even while declining. The compute unit discards this
    // callback's result and re-enters it for as long as the flag is set, so
    // returning false without clearing it spins on every instruction.
    wave.set_debug_single_step(false);
    return false;
  }
  amdgpu::Wavefront::DebugStopState saved = wave.debug_stop_state();
  // The completed step is consumed even when publication loses its debugger
  // claim. Restoring single_step would make ComputeUnitCore call us after every
  // subsequent instruction because it intentionally ignores our return value.
  saved.single_step = false;
  wave.set_debug_single_step(false);
  wave.debug_trap(0);
  // gfx9.4 reports completed single-step through TRAPSTS.TRAP_AFTER_INST. This
  // is the public stop-reason bit rocm-dbgapi consumes before the next resume.
  constexpr uint32_t kTrapAfterInstMask = 1u << 25;
  wave.set_trapsts(wave.trapsts() | kTrapAfterInstMask);
  // Single-step completion is wave-specific. Report it even if another wave
  // in the queue is running; rocm-dbgapi will suspend the queue before reading
  // its CWSR state. Deferring until every peer stops can lose the event forever
  // when a peer runs to normal completion without entering a debug callback.
  // Like hardware's trap interrupt, notification does not itself save the
  // queue. The debugger's ensuing SUSPEND_QUEUES request publishes one stable,
  // authoritative CWSR snapshot instead of redundantly serializing every
  // resident wave here first.
  apply_debug_event_claim_mask_for_testing(proc->client_pid());
  if (!notify_debug_event(proc, wave.queue_id(), gpu_id)) {
    wave.restore_debug_stop_state(saved);
    return false;
  }
  return true;
}

bool SimulatedKfd::on_wave_watchpoint(amdgpu::Wavefront &wave, uint64_t address, uint32_t bytes,
                                      bool is_write, bool is_atomic) {
  auto proc = find_process(wave.process_id());
  if (!proc)
    return false;
  uint32_t matched_slots = 0;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(proc->client_pid());
    if (session == debug_sessions_.end() || !session->second.enabled)
      return false;
    // The watch modes are not disjoint, so an access can satisfy more than one
    // of them: NONREAD is documented as "write or atomic operations only"
    // (kfd_ioctl.h, and rocdbgapi's os_driver.h maps the STORE_AND_RMW
    // watchpoint kind a plain `(gdb) watch` requests onto it), and ALL matches
    // everything. Hand the set of modes this access triggers to the session
    // rather than a single mode, so a write watch still fires on an atomic
    // read-modify-write. KfdProcess deliberately does not include kfd_ioctl.h,
    // so the mode semantics stay here, on the side of the boundary that owns
    // the KFD ABI.
    const uint32_t matching_modes =
        (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL) |
        (is_atomic ? (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC) |
                         (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD)
         : is_write ? (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD)
                    : (uint32_t{1} << KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ));
    matched_slots = session->second.matching_address_watch_slots(address, bytes, matching_modes);
  }
  if (matched_slots == 0)
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
  if (!debug_stop_publishable(gpu_id))
    return false;

  static constexpr uint32_t kTrapstsBits[] = {1u << 7, 1u << 12, 1u << 13, 1u << 14};
  const auto saved = wave.debug_stop_state();
  uint32_t trapsts = wave.trapsts();
  for (uint32_t slot = 0; slot < KfdProcess::DebugSession::kMaxAddressWatches; ++slot)
    if ((matched_slots & (uint32_t{1} << slot)) != 0)
      trapsts |= kTrapstsBits[slot];
  wave.set_trapsts(trapsts);
  if (wave.uses_separate_trap_ctrl()) {
    constexpr uint32_t kTrapCtrlAddrWatch = 1u << 7;
    wave.set_gfx12_trap_ctrl_raw(wave.gfx12_trap_ctrl_raw() | kTrapCtrlAddrWatch);
  } else {
    constexpr uint32_t kModeExcpEnAddrWatch = 1u << 19;
    wave.set_mode_raw(wave.mode_raw() | kModeExcpEnAddrWatch);
  }
  wave.debug_trap(0);
  if (!report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size)) {
    wave.restore_debug_stop_state(saved);
    return false;
  }
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
  if (!debug_stop_publishable(gpu_id))
    return false;
  constexpr uint32_t kTrapstsIllegalInst = 1u << 11;
  const auto saved = wave.debug_stop_state();
  wave.set_trapsts(wave.trapsts() | kTrapstsIllegalInst);
  wave.set_fatal_exception_pending(true);
  wave.debug_trap(0);
  if (!report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size,
                           KFD_EC_MASK(EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION))) {
    wave.restore_debug_stop_state(saved);
    return false;
  }
  return true;
}

bool SimulatedKfd::on_wave_memory_violation(amdgpu::Wavefront &wave, uint64_t, bool) {
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
  if (!debug_stop_publishable(gpu_id))
    return false;
  constexpr uint32_t kTrapstsXnackError = 1u << 28;
  const auto saved = wave.debug_stop_state();
  wave.set_trapsts(wave.trapsts() | kTrapstsXnackError);
  wave.debug_trap(0);
  if (!report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size,
                           KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION))) {
    wave.restore_debug_stop_state(saved);
    return false;
  }
  return true;
}

bool SimulatedKfd::on_wave_alu_exception(amdgpu::Wavefront &wave) {
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
  if (!debug_stop_publishable(gpu_id))
    return false;
  const auto saved = wave.debug_stop_state();
  wave.set_fatal_exception_pending(true);
  wave.debug_trap(0);
  if (!report_wave_stopped(proc, wave.queue_id(), gpu_id, ctx_base, ctx_size,
                           KFD_EC_MASK(EC_QUEUE_WAVE_MATH_ERROR))) {
    wave.restore_debug_stop_state(saved);
    return false;
  }
  return true;
}

void SimulatedKfd::release_debuggee_state(pid_t target_pid, KfdProcess *target_proc) {
  // Everything a session imposed on its inferior, undone in one place so that
  // explicit detach and debugger death cannot drift apart. Callers have already
  // erased the session and must not hold debug_sessions_mutex_ here: this takes
  // CU wave-state locks, and the engine thread takes those first and then
  // debug_sessions_mutex_ from its trap/watchpoint callbacks, so holding both
  // in this order closes an AB-BA cycle against a wave that is trapping.
  {
    std::lock_guard<std::mutex> event_lock(debug_events_mutex_);
    debug_events_.erase(target_pid);
  }

  // Nothing below has a debuggee to release; the event purge above was the
  // whole job. Returning here keeps the rest free of repeated null tests.
  if (target_proc == nullptr) {
    release_debug_checks_if_last_session();
    return;
  }

  std::vector<std::pair<uint32_t, uint32_t>> queues;
  {
    std::lock_guard<std::mutex> alloc_lock(target_proc->alloc_mutex_);
    for (auto &[queue_id, queue] : target_proc->queue_snapshot_map_) {
      queue.exception_status = 0;
      queues.emplace_back(queue_id, queue.gpu_id);
    }
  }

  for (const auto &[queue_id, gpu_id] : queues) {
    auto *gpu = find_gpu(gpu_id);
    if (!gpu || !gpu->soc)
      continue;
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      cp->set_queue_debug_suspended(queue_id, target_proc->process_id(), false);
      for (auto *cu : cp->compute_units()) {
        bool wake = false;
        cu->with_wave_state_locked([&] {
          for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
            auto *wave = cu->wf(slot);
            if (wave->is_halted() || wave->process_id() != target_proc->process_id() ||
                wave->queue_id() != queue_id || wave->fatal_exception_pending())
              continue;
            wave->set_debug_single_step(false);
            wave->set_debug_halted(false);
            wave->set_debug_suspended(false);
            // The architectural half of the same stop. s_sendmsghalt raises
            // STATUS.HALT and s_rfe consults it on the way out of the handler,
            // so leaving it set outlives the session: the wave would re-halt at
            // the handler return with no debugger left to resume it, and the
            // queue would never drain.
            wave->set_status_halt(false);
            wave->set_self_halted(false);
            wake = true;
          }
        });
        if (wake)
          cu->schedule_work_async();
      }
    });
  }

  revoke_target_mem_routing(target_proc->process_id());
  release_debug_checks_if_last_session();
}

void SimulatedKfd::revoke_target_mem_routing(uint32_t process_id) {
  for (auto &g : gpus_)
    if (auto *mem = g.soc ? g.soc->memory() : nullptr)
      mem->set_process_mem_fd(process_id, -1);
}

void SimulatedKfd::release_debug_checks_if_last_session() {
  // Debug checks stay on only while some session still wants them. Read the
  // answer under the lock, then reach into the CUs after releasing it -- the
  // whole point of this function running unlocked is that debug_sessions_mutex_
  // is never held while touching a CU.
  bool no_sessions_left = false;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    no_sessions_left = debug_sessions_.empty();
  }
  if (no_sessions_left)
    set_debug_active_on_all_cus(false);
}

void SimulatedKfd::set_debug_active_on_all_cus(bool active) {
  for (auto &gpu : gpus_) {
    if (!gpu.soc)
      continue;
    gpu.soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      for (auto *cu : cp->compute_units())
        cu->set_debug_active(active);
    });
  }
}

UniqueDriverFd SimulatedKfd::duplicate_debug_target_mem(pid_t target_pid) const {
  std::lock_guard<std::mutex> lock(debug_sessions_mutex_);
  auto session = debug_sessions_.find(target_pid);
  if (session == debug_sessions_.end() || session->second.target_mem_fd.get() < 0)
    return {};
  return UniqueDriverFd(safe_fcntl(session->second.target_mem_fd.get(), F_DUPFD_CLOEXEC, 0));
}

void SimulatedKfd::apply_cwsr_to_wave(amdgpu::Wavefront &wave, const kmd::CwsrWaveState &state,
                                      rj_code_arch_t arch) {
  constexpr uint32_t kModeDebugEnMask = 1u << 11;
  constexpr uint32_t kStatusHaltMask = amdgpu::Wavefront::kStatusHaltMask;
  const bool gfx12_5 = kmd::cwsr_layout_kind(arch) == kmd::CwsrLayoutKind::Gfx12_5;
  wave.pc = state.pc;
  // Mirror of the un-shadowing in build_cwsr_wave_state(): while a handler is
  // mid-flight the published EXEC is the interrupted application mask, so the
  // value coming back belongs to the shadow, not to the live register the
  // handler is executing under. s_rfe installs it when the handler returns.
  if (wave.in_trap_handler())
    wave.set_trap_saved_exec(state.exec);
  else
    wave.set_exec(state.exec);
  wave.set_vcc_raw(state.vcc);
  wave.set_m0(state.m0);
  // STATUS shadows the same way EXEC does, and the cost of getting it wrong is
  // higher: the ROCr handler raises STATUS.HALT and only then returns, so a
  // record applied in that window used to clear the HALT the handler had just
  // set. s_rfe then saw a running wave, resumed it, and the breakpoint was
  // silently lost -- one inferior in gdb.rocm/multi-inferior-stress.exp running
  // its kernel to completion and exiting without ever stopping. Route the
  // record's HALT to the interrupted state the handler restores and leave the
  // live register to the handler.
  const uint32_t restored_status = gfx12_5 ? internal_status_from_gfx1250(state)
                                           : (state.status & ~kStatusHaltMask) |
                                                 (state.saved_status_halt ? kStatusHaltMask : 0u);
  if (wave.in_trap_handler())
    wave.set_trap_saved_status(restored_status);
  else
    wave.set_status_raw(restored_status);
  // The live STATUS.HALT stays as it is for a handler that raised it with
  // s_setreg: there the bit is the handler's own request to keep the wave
  // stopped, and clearing it on a resume loses the breakpoint --
  // DbgTrapCwsrShadowsTrapHandlerRegistersAndRoutesDebuggerEdits pins that.
  // A wave halted at s_sendmsghalt sits in the same window and wants the
  // opposite: it has already reported and is waiting to be let go, so leaving
  // the bit set strands it at the handler return. The record cannot tell the
  // two apart, which is why the wave records who raised the bit.
  if (!state.wave_stopped && wave.self_halted()) {
    wave.set_status_halt(false);
    wave.set_self_halted(false);
  }
  wave.set_mode_raw(state.mode);
  wave.set_trapsts(gfx12_5 ? internal_trapsts_from_gfx1250(state) : state.trapsts);
  if (gfx12_5) {
    wave.set_gfx12_excp_flag_user_extra_raw(state.excp_flag_user & (0x3u << 30));
    wave.set_gfx12_trap_ctrl_raw(state.trap_ctrl);
    wave.set_gfx1250_xnack_state_priv_raw(state.xnack_state_priv);
    wave.set_gfx1250_xnack_mask_raw(state.xnack_mask);
  }
  wave.set_debug_wave_id(state.wave_id);
  wave.set_aql_packet_id(state.queue_packet_id);
  const uint32_t stack_pointer = wave.debug_read_sgpr(32);
  const uint32_t stack_frame = wave.debug_read_sgpr(33);
  // rocm-dbgapi may submit a lightweight control-state update with no register
  // payload after displaced stepping. Preserve the SGPR/VGPR values produced
  // by the displaced instruction in that case rather than restoring zeros.
  //
  // Skip exactly the slots the codec treats as aliases, not everything above
  // the architected count: s104/s105 sit between the FLAT_SCRATCH and VCC
  // aliases and are ordinary registers the debugger may edit. Sharing
  // cwsr_sgpr_slot_is_aliased() with the codec is what keeps the two ends of
  // the round trip from disagreeing about which slots carry register state.
  if (!state.sgprs.empty())
    for (uint32_t s = 0; s < state.num_sgprs && s < state.sgprs.size(); ++s)
      if (s != 32 && s != 33 && !kmd::cwsr_sgpr_slot_is_aliased(s, arch))
        wave.debug_write_sgpr(s, state.sgprs[s]);
  if (!state.vgprs.empty())
    for (uint32_t r = 0; r < state.num_vgprs; ++r)
      for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
        const size_t index = static_cast<size_t>(r) * 64 + lane;
        if (index < state.vgprs.size())
          wave.debug_write_vgpr(r, lane, state.vgprs[index]);
      }
  // FLAT_SCRATCH travels in its own field because it aliases two SGPR slots the
  // loop above skips. This used to save and re-install wave.scratch_base()
  // around the loop, which the loop cannot reach anyway (scratch_base_ is a
  // standalone member), so the only effect was discarding the debugger's edit.
  wave.set_scratch_base(state.flat_scratch);
  wave.debug_write_sgpr(32, stack_pointer);
  wave.debug_write_sgpr(33, stack_frame);
  const bool single_step = !state.wave_stopped && (gfx12_5 ? (state.trap_ctrl & (1u << 9)) != 0
                                                           : (state.mode & kModeDebugEnMask) != 0);
  wave.set_debug_single_step(single_step);
  wave.set_debug_halted(state.wave_stopped);
  // debug_suspended is deliberately left set here. Clearing it per wave made
  // each wave runnable the moment its own record was applied, so the engine
  // could execute it while its queue-mates -- and their LDS images -- were
  // still being restored. resume_debug_queues() commits the bit for every wave
  // in the queue in one pass once the whole restore has succeeded.
}

int SimulatedKfd::resume_debug_queues(KfdProcess *proc, uint32_t *queue_ids, uint32_t num_queues) {
  constexpr uint32_t kQueueError = uint32_t{1} << KFD_DBG_QUEUE_ERROR_BIT;
  constexpr uint32_t kQueueInvalid = uint32_t{1} << KFD_DBG_QUEUE_INVALID_BIT;
  constexpr uint32_t kQueueStatus = kQueueError | kQueueInvalid;
  struct QueueContext {
    uint32_t request_index = 0;
    uint32_t queue_id = 0;
    uint64_t base = 0;
    uint32_t size = 0;
    uint32_t gpu_id = 0;
  };
  std::vector<QueueContext> queues;
  if (proc) {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    for (uint32_t index = 0; index < num_queues; ++index) {
      const uint32_t queue_id = queue_ids[index] & ~kQueueStatus;
      queue_ids[index] = queue_id;
      auto queue = proc->queue_snapshot_map_.find(queue_id);
      if (queue == proc->queue_snapshot_map_.end()) {
        queue_ids[index] |= kQueueInvalid;
        continue;
      }
      const auto &info = queue->second;
      queues.push_back({index, queue_id, info.ctx_save_restore_address,
                        info.ctx_save_restore_area_size, info.gpu_id});
    }
  } else {
    for (uint32_t index = 0; index < num_queues; ++index)
      queue_ids[index] = (queue_ids[index] & ~kQueueStatus) | kQueueInvalid;
  }
  uint32_t resumed = 0;
  for (const auto &context : queues) {
    auto *gpu = find_gpu(context.gpu_id);
    if (!gpu || !gpu->soc) {
      queue_ids[context.request_index] |= kQueueError;
      continue;
    }
    std::vector<amdgpu::Wavefront *> stopped;
    std::vector<kmd::CwsrWaveState> states;
    std::vector<amdgpu::ComputeUnitCore *> owners;
    const bool per_xcc_areas = gpu->soc->arch() == ROCJITSU_CODE_ARCH_CDNA5;
    const uint32_t area_count = per_xcc_areas ? std::max(1u, gpu->soc->num_xcds()) : 1u;
    std::vector<std::vector<kmd::CwsrWaveState>> states_by_area(area_count);
    auto collect = [&](amdgpu::CommandProcessor *cp, uint32_t area) {
      for (auto *cu : cp->compute_units()) {
        cu->with_wave_state_locked([&] {
          for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
            auto *wave = cu->wf(slot);
            if (wave->debug_stopped() && wave->process_id() == proc->process_id() &&
                wave->queue_id() == context.queue_id) {
              stopped.push_back(wave);
              states_by_area[area].push_back(build_cwsr_wave_state(*wave, gpu->soc->arch()));
              owners.push_back(cu);
            }
          }
        });
      }
    };
    if (per_xcc_areas) {
      for (uint32_t xcc_id = 0; xcc_id < area_count; ++xcc_id)
        collect(gpu->soc->xcd(xcc_id)->command_processor(), xcc_id);
    } else {
      gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) { collect(cp, 0); });
    }
    bool restored = stopped.empty();
    if (!stopped.empty() && context.base != 0) {
      auto *memory = gpu->soc->memory();
      UniqueDriverFd target_mem = duplicate_debug_target_mem(proc->client_pid());
      bool read_ok = true;
      auto read_block = [&](uint64_t address, std::span<uint8_t> bytes) {
        if (target_mem.get() < 0) {
          // A refused read leaves the span zero-filled, which deserializes as a
          // valid-looking context: waves would resume with cleared registers and
          // a zero program counter rather than staying stopped. The pread path
          // below already treats a short read that way, and this is the same
          // failure arriving through the memory model instead of the kernel.
          if (memory->read_block(address, bytes, proc->process_id()) ==
              amdgpu::AccessOutcome::Faulted) {
            read_ok = false;
            util::Logger::warn("CWSR target read faulted: addr=0x", std::hex, address, std::dec,
                               " pid=", proc->client_pid());
          }
          return;
        }
        const ssize_t bytes_read =
            pread(target_mem.get(), bytes.data(), bytes.size(), static_cast<off_t>(address));
        if (bytes_read != static_cast<ssize_t>(bytes.size())) {
          read_ok = false;
          std::fill(bytes.begin(), bytes.end(), 0);
          util::Logger::warn("CWSR target read failed: addr=0x", std::hex, address,
                             " pid=", std::dec, proc->client_pid(), " rc=", bytes_read,
                             " errno=", errno);
        }
      };
      restored = true;
      states.reserve(stopped.size());
      for (uint32_t area = 0; area < area_count; ++area) {
        auto &area_states = states_by_area[area];
        if (area_states.empty())
          continue;
        prepare_cwsr_wave_states(area_states);
        const uint64_t area_base = context.base + static_cast<uint64_t>(area) * context.size;
        restored &= kmd::deserialize_queue_cwsr_bulk(area_base, context.size, area_states,
                                                     read_block, gpu->soc->arch());
        states.insert(states.end(), std::make_move_iterator(area_states.begin()),
                      std::make_move_iterator(area_states.end()));
      }
      restored = restored && read_ok;
    }
    std::unordered_set<amdgpu::ComputeUnitCore *> wake;
    if (restored) {
      std::vector<size_t> matches;
      matches.reserve(states.size());
      std::vector<bool> matched(stopped.size());
      for (const auto &state : states) {
        auto find_match = [&](bool by_wave_id) {
          for (size_t index = 0; index < stopped.size(); ++index) {
            if (matched[index])
              continue;
            const auto *candidate = stopped[index];
            if (by_wave_id ? state.wave_id != 0 && candidate->debug_wave_id() == state.wave_id
                           : candidate->debug_wave_id() == 0 &&
                                 candidate->aql_packet_id() == state.queue_packet_id &&
                                 candidate->wg_coord() == state.group_ids &&
                                 candidate->wave_in_group() == state.wave_in_group)
              return index;
          }
          return stopped.size();
        };
        size_t index = find_match(true);
        if (index == stopped.size())
          index = find_match(false);
        if (index == stopped.size()) {
          restored = false;
          break;
        }
        matched[index] = true;
        matches.push_back(index);
      }
      restored = restored && matches.size() == stopped.size();
      if (restored) {
        for (size_t state_index = 0; state_index < states.size(); ++state_index) {
          const size_t wave_index = matches[state_index];
          if (!states[state_index].lds.empty())
            stopped[wave_index]->lds().write(stopped[wave_index]->lds_base(),
                                             states[state_index].lds.data(),
                                             static_cast<uint32_t>(states[state_index].lds.size()));
          owners[wave_index]->with_wave_state_locked([&] {
            apply_cwsr_to_wave(*stopped[wave_index], states[state_index], gpu->soc->arch());
          });
        }
      }
    }
    // This is the single commit point for debug_suspended: apply_cwsr_to_wave()
    // leaves every wave suspended, so none of them runs until the whole queue's
    // CWSR state and LDS have been restored above.
    //
    // Accumulated inside the locked loop below rather than read afterwards.
    // The loop clears debug_suspended, which makes these waves runnable, so
    // from that point the engine thread can be writing debug_halted_ -- a plain
    // bool -- concurrently. This value decides whether the queue gate reopens,
    // so reading it unlocked would race the answer as well as the byte.
    bool keep_dispatch_suspended = false;
    for (size_t index = 0; index < stopped.size(); ++index) {
      owners[index]->with_wave_state_locked([&] {
        const bool fatal_exception_pending = stopped[index]->fatal_exception_pending();
        // A malformed or stale CWSR image must not strand a temporarily
        // suspended wave after the queue gate is released. Architecturally
        // halted waves remain halted until their CWSR record says otherwise.
        stopped[index]->set_debug_suspended(fatal_exception_pending);
        const bool halted = stopped[index]->debug_halted();
        keep_dispatch_suspended |= halted;
        if (!halted && !stopped[index]->is_halted())
          if (!fatal_exception_pending)
            wake.insert(owners[index]);
      });
    }
    // Keep the queue-level launch gate closed until every resident wave has
    // consumed its CWSR state and become runnable. Reopen it before scheduling
    // those waves so CP completion processing cannot observe a queue as still
    // suspended if a resumed wave completes immediately.
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      cp->set_queue_debug_suspended(context.queue_id, proc->process_id(), keep_dispatch_suspended);
    });
    for (auto *cu : wake)
      cu->schedule_work_async();
    if (restored)
      ++resumed;
    else
      queue_ids[context.request_index] |= kQueueError;
  }
  return static_cast<int>(resumed);
}

int SimulatedKfd::suspend_debug_queues(KfdProcess *proc, uint32_t *queue_ids, uint32_t num_queues,
                                       uint64_t exception_mask) {
  constexpr uint32_t kQueueError = uint32_t{1} << KFD_DBG_QUEUE_ERROR_BIT;
  constexpr uint32_t kQueueInvalid = uint32_t{1} << KFD_DBG_QUEUE_INVALID_BIT;
  constexpr uint32_t kQueueStatus = kQueueError | kQueueInvalid;
  struct RequestedQueue {
    uint32_t request_index = 0;
    uint32_t queue_id = 0;
    KfdProcess::QueueSnapshotInfo info{};
  };
  std::vector<RequestedQueue> queues;
  if (!proc) {
    for (uint32_t index = 0; index < num_queues; ++index)
      queue_ids[index] = (queue_ids[index] & ~kQueueStatus) | kQueueInvalid;
    return 0;
  }
  const uint32_t process_id = proc->process_id();
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    for (uint32_t index = 0; index < num_queues; ++index) {
      const uint32_t queue_id = queue_ids[index] & ~kQueueStatus;
      queue_ids[index] = queue_id;
      auto queue = proc->queue_snapshot_map_.find(queue_id);
      if (queue == proc->queue_snapshot_map_.end() ||
          (queue->second.exception_status & KFD_EC_MASK(EC_QUEUE_NEW)) != 0) {
        queue_ids[index] |= kQueueInvalid;
        continue;
      }
      queue->second.exception_status &= ~exception_mask;
      queues.push_back({index, queue_id, queue->second});
    }
  }
  uint32_t suspended = 0;
  for (const auto &queue : queues) {
    auto *gpu = find_gpu(queue.info.gpu_id);
    if (!gpu || !gpu->soc) {
      queue_ids[queue.request_index] |= kQueueError;
      continue;
    }
    // Ask before suspending anything. This path stops waves first and only then
    // tries to serialize them, so on an unmodelled architecture it would strand
    // every resident wave behind a kQueueError -- the same hole the wave-stop
    // callbacks have, reached through the debugger's own request instead.
    if (!debug_stop_publishable(queue.info.gpu_id)) {
      queue_ids[queue.request_index] |= kQueueError;
      continue;
    }
    std::vector<std::pair<amdgpu::ComputeUnitCore *, amdgpu::Wavefront *>> newly_suspended;
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      cp->set_queue_debug_suspended(queue.queue_id, process_id, true);
      for (auto *cu : cp->compute_units()) {
        cu->with_wave_state_locked([&] {
          for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
            auto *wave = cu->wf(slot);
            if (!wave->is_halted() && !wave->debug_suspended() &&
                wave->process_id() == process_id && wave->queue_id() == queue.queue_id) {
              wave->set_debug_suspended(true);
              newly_suspended.emplace_back(cu, wave);
            }
          }
        });
      }
    });
    bool serialized = newly_suspended.empty();
    if (!newly_suspended.empty() && queue.info.ctx_save_restore_address != 0)
      serialized = serialize_queue_debug_waves(process_id, queue.queue_id, queue.info.gpu_id,
                                               queue.info.ctx_save_restore_address,
                                               queue.info.ctx_save_restore_area_size);
    if (serialized) {
      ++suspended;
      continue;
    }
    // No record was published, so the debugger has nothing to resume from.
    // Undo the suspension rather than leave the queue wedged: reporting the
    // error and stranding the waves is strictly worse than reporting it alone.
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      cp->set_queue_debug_suspended(queue.queue_id, process_id, false);
    });
    amdgpu::ComputeUnitCore *woken = nullptr;
    for (auto &[cu, wave] : newly_suspended) {
      cu->with_wave_state_locked([w = wave] { w->set_debug_suspended(false); });
      // Entries are appended compute unit by compute unit, so comparing against
      // the last one woken collapses a CU's waves into a single wake.
      if (cu != woken) {
        cu->schedule_work_async();
        woken = cu;
      }
    }
    queue_ids[queue.request_index] |= kQueueError;
  }
  return static_cast<int>(suspended);
}

void SimulatedKfd::clear_completed_debug_queues(KfdProcess *proc, const uint32_t *queue_ids,
                                                uint32_t num_queues) {
  if (!proc)
    return;
  constexpr uint32_t kQueueStatus =
      (uint32_t{1} << KFD_DBG_QUEUE_ERROR_BIT) | (uint32_t{1} << KFD_DBG_QUEUE_INVALID_BIT);
  const uint32_t process_id = proc->process_id();
  std::vector<std::pair<uint32_t, KfdProcess::QueueSnapshotInfo>> queues;
  {
    std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
    for (uint32_t index = 0; index < num_queues; ++index) {
      if ((queue_ids[index] & kQueueStatus) != 0)
        continue;
      const uint32_t queue_id = queue_ids[index];
      auto queue = proc->queue_snapshot_map_.find(queue_id);
      if (queue != proc->queue_snapshot_map_.end())
        queues.emplace_back(queue_id, queue->second);
    }
  }
  for (const auto &[queue_id, queue] : queues) {
    auto *gpu = find_gpu(queue.gpu_id);
    if (!gpu || !gpu->soc || queue.ctx_save_restore_address == 0)
      continue;
    bool has_stopped_wave = false;
    gpu->soc->for_each_cp([&](amdgpu::CommandProcessor *cp) {
      for (auto *cu : cp->compute_units()) {
        cu->with_wave_state_locked([&] {
          for (uint32_t slot = 0; slot < cu->num_wf_slots(); ++slot) {
            const auto *wave = cu->wf(slot);
            if (wave->debug_stopped() && wave->process_id() == process_id &&
                wave->queue_id() == queue_id)
              has_stopped_wave = true;
          }
        });
      }
    });
    if (!has_stopped_wave) {
      auto *memory = gpu->soc->memory();
      const uint32_t area_count =
          gpu->soc->arch() == ROCJITSU_CODE_ARCH_CDNA5 ? std::max(1u, gpu->soc->num_xcds()) : 1u;
      for (uint32_t area = 0; area < area_count; ++area) {
        const uint64_t area_base = queue.ctx_save_restore_address +
                                   static_cast<uint64_t>(area) * queue.ctx_save_restore_area_size;
        for (uint64_t offset = 0; offset < 40; offset += sizeof(uint32_t))
          memory->write32(area_base + offset, 0, process_id);
      }
    }
  }
}

int SimulatedKfd::debug_query_event(pid_t target_pid, KfdProcess *target_proc,
                                    kfd_ioctl_dbg_trap_query_debug_event_args &args) {
  const uint64_t clear_mask = args.exception_mask;
  std::lock_guard<std::mutex> lk(debug_events_mutex_);
  auto process = debug_events_.find(target_pid);
  if (process == debug_events_.end())
    return -EAGAIN;
  auto &queues = process->second;
  for (auto queue = queues.begin(); queue != queues.end(); ++queue) {
    args.exception_mask = queue->second.mask;
    args.queue_id = queue->first;
    args.gpu_id = queue->second.gpu_id;
    queue->second.mask &= ~clear_mask;
    const uint32_t queue_id = queue->first;
    if (queue->second.mask == 0)
      queues.erase(queue);
    if (target_proc != nullptr && queue_id != 0) {
      std::lock_guard<std::mutex> alloc_lock(target_proc->alloc_mutex_);
      auto snapshot = target_proc->queue_snapshot_map_.find(queue_id);
      if (snapshot != target_proc->queue_snapshot_map_.end())
        snapshot->second.exception_status &= ~clear_mask;
    }
    return 0;
  }
  return -EAGAIN;
}

void SimulatedKfd::raise_process_debug_event(pid_t target_pid, uint64_t exception_mask) {
  UniqueDriverFd notifier;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session == debug_sessions_.end() || !session->second.enabled ||
        session->second.dbg_fd < 0 || (session->second.exception_enable_mask & exception_mask) == 0)
      return;
    notifier = UniqueDriverFd(safe_fcntl(session->second.dbg_fd, F_DUPFD_CLOEXEC, 0));
    if (notifier.get() < 0)
      return;
    std::lock_guard<std::mutex> event_lock(debug_events_mutex_);
    auto &event = debug_events_[target_pid][0];
    event.gpu_id = 0;
    event.mask |= exception_mask;
  }
  const uint64_t one = 1;
  [[maybe_unused]] const ssize_t written = ::write(notifier.get(), &one, sizeof(one));
}

void SimulatedKfd::cancel_runtime_handshake(pid_t target_pid) {
  {
    std::lock_guard<std::mutex> lk(runtime_handshake_mutex_);
    runtime_handshake_cancelled_.insert(target_pid);
  }
  runtime_handshake_cv_.notify_all();
}

/// @details The inferior must not run a kernel until the debugger attached to
/// it has seen EC_PROCESS_RUNTIME and installed its breakpoints -- that is the
/// whole point of the handshake, and amdkfd's runtime_enable blocks the caller
/// for it. Returning early is not a graceful degradation: the code objects load
/// unobserved, the debugger's breakpoints are never inserted, and the kernel
/// runs to completion. Under a debugger driving many inferiors at once (see
/// gdb.rocm/multi-inferior-stress.exp, 32 of them) a short deadline expires
/// routinely while the debugger is simply busy servicing its queue, which
/// showed up as rare, load-dependent "inferior exited before hitting the
/// kernel" failures.
///
/// So wait for the ack itself, and treat the deadline purely as a liveness
/// backstop against a debugger that has wedged without detaching -- long enough
/// that a merely slow one is never cut off, and noisy when it does fire so the
/// timeout is never again mistaken for correct behaviour. A debugger that
/// detaches or dies cancels the wait explicitly and does not pay it at all.
///
/// The disable direction reports the transition and returns. kfd_chardev.c's
/// runtime_disable does wait there -- it raises EC_PROCESS_RUNTIME and blocks on
/// runtime_enable_sema, the same way runtime_enable does -- and this deliberately
/// does not follow it, because the two are not in the same position. The kernel
/// blocks a task; here the ioctl is served on the daemon's connection thread for
/// that client, and holding it parks the daemon's side of a process that is
/// already exiting. Doing it anyway destabilises the daemon-backed tests
/// (RemoteDriverDbg*) with no measured benefit: the -EPERM storm this was first
/// written for is answered where it is raised, by the queue gate letting a
/// suspend or resume of already-destroyed queues through.
void SimulatedKfd::runtime_debugger_handshake(pid_t target_pid, bool enabling) {
  // The caller already decided, under debug_sessions_mutex_, that a debugger is
  // attached; re-testing here would just re-open the window it closed. Only the
  // self-debug shape still needs distinguishing.
  bool self_debugged = false;
  {
    std::lock_guard<std::mutex> lk(debug_sessions_mutex_);
    auto session = debug_sessions_.find(target_pid);
    if (session == debug_sessions_.end() || !session->second.enabled)
      return;
    self_debugged = session->second.debugger_pid == target_pid;

    // Drop any ack or cancellation left over from an earlier transition for this
    // pid before the event goes out. The disable direction reports without
    // waiting, so nothing consumes the ack it draws; a later enable would
    // otherwise find that stale entry already in runtime_acked_, satisfy its wait
    // immediately, and let the inferior dispatch before the debugger has
    // installed its breakpoints -- the exact failure the deadline below exists to
    // prevent.
    //
    // A cancellation does the same thing and is easier to leave behind, because
    // cancel_runtime_handshake() inserts unconditionally: a detach with no waiter
    // parked -- which every explicit DBG_TRAP_DISABLE performs -- leaves an entry
    // nobody consumes, and the wait predicate below accepts it just as readily as
    // a real ack. Clearing both under the same lock a real cancel would take
    // means a cancel racing this clear necessarily lands on the new wait.
    //
    // That last part only holds while debug_sessions_mutex_ is still held, which
    // is why the clear lives here rather than after the lookup returns. An
    // explicit DBG_TRAP_DISABLE erases the session, drops the mutex, and only
    // then calls cancel_runtime_handshake(). Clearing after releasing the mutex
    // left a window in which the detach's cancellation was recorded after this
    // function had already decided the session was live, and was then erased by
    // this very clear -- so nothing remained to satisfy the wait below, no acker
    // was left to provide one, and the inferior paid the full 60s deadline.
    // Holding the mutex across both means a detach either erases the session
    // before the lookup (which returns early) or blocks until after the clear,
    // in which case its cancellation lands on the wait.
    if (enabling) {
      std::lock_guard<std::mutex> hlk(runtime_handshake_mutex_);
      runtime_acked_.erase(target_pid);
      runtime_handshake_cancelled_.erase(target_pid);
    }
  }
  raise_process_debug_event(target_pid, KFD_EC_MASK(EC_PROCESS_RUNTIME));
  // A process debugging itself cannot answer its own handshake: the ack would
  // have to come from the thread that is, by construction, blocked here. The
  // event is raised so the state is observable; waiting for a reply that can
  // never arrive would just burn the liveness deadline.
  if (self_debugged || !enabling)
    return;

  constexpr auto kHandshakeDeadline = std::chrono::seconds(60);
  std::unique_lock<std::mutex> lk(runtime_handshake_mutex_);
  const bool released = runtime_handshake_cv_.wait_for(lk, kHandshakeDeadline, [&] {
    return runtime_acked_.contains(target_pid) || runtime_handshake_cancelled_.contains(target_pid);
  });
  const bool cancelled = runtime_handshake_cancelled_.erase(target_pid) != 0;
  runtime_acked_.erase(target_pid);
  if (!released) {
    util::Logger::warn("runtime-enable handshake timed out after ", kHandshakeDeadline.count(),
                       "s for pid=", target_pid,
                       "; the debugger never acknowledged EC_PROCESS_RUNTIME, so its "
                       "breakpoints may not be installed before the first dispatch");
  } else if (cancelled) {
    util::Logger::vm("runtime-enable handshake cancelled for pid=", target_pid,
                     " (debugger detached)");
  }
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

namespace {

/// @brief Whether a suspend/resume names only queues the process has destroyed.
/// @details Used to tell a request that cannot touch hardware from one that
/// can, so the runtime-down gate only refuses the latter. Any other op, an
/// absent process, or an empty request answers false, leaving the kernel's
/// behaviour in place.
bool queues_all_dead(KfdProcess *proc, const kfd_ioctl_dbg_trap_args &args) {
  constexpr uint32_t kQueueStatus = KFD_DBG_QUEUE_ERROR_MASK | KFD_DBG_QUEUE_INVALID_MASK;
  if (proc == nullptr)
    return false;
  uint32_t count = 0;
  uint64_t array_ptr = 0;
  if (args.op == KFD_IOC_DBG_TRAP_SUSPEND_QUEUES) {
    count = args.suspend_queues.num_queues;
    array_ptr = args.suspend_queues.queue_array_ptr;
  } else if (args.op == KFD_IOC_DBG_TRAP_RESUME_QUEUES) {
    count = args.resume_queues.num_queues;
    array_ptr = args.resume_queues.queue_array_ptr;
  } else {
    return false;
  }
  // An empty batch names nothing live, so it belongs on the answered side: with
  // the runtime up the handlers return 0 for it, and refusing it here would flip
  // that to the -EPERM rocdbgapi escalates to a fatal purely on runtime state.
  if (count == 0)
    return true;
  // A null array with a non-zero count is malformed; leave it to the handler's
  // -EFAULT rather than inventing an answer for it.
  if (array_ptr == 0)
    return false;
  const auto *queue_ids = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(array_ptr));
  std::lock_guard<std::mutex> lk(proc->alloc_mutex_);
  for (uint32_t index = 0; index < count; ++index)
    if (proc->queue_snapshot_map_.contains(queue_ids[index] & ~kQueueStatus))
      return false;
  return true;
}

} // namespace

// in real kernel, amd/amdkfd/kfd_chardev.c kfd_ioctl_set_debug_trap
int SimulatedKfd::debug_trap_ioctl(KfdProcess &caller, void *arg, int *target_mem_fd,
                                   int target_proc_fd) {
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

  std::unique_lock<std::mutex> lk(debug_sessions_mutex_);
  auto session_it = debug_sessions_.find(target_pid);
  if (session_it != debug_sessions_.end()) {
    if (session_it->second.target_exited) {
      if (args->op != KFD_IOC_DBG_TRAP_ENABLE)
        return -ESRCH;
      debug_sessions_.erase(session_it);
      session_it = debug_sessions_.end();
    }
  }
  if (session_it != debug_sessions_.end()) {
    const int target_exited = pidfd_is_exited(session_it->second.target_pidfd.get());
    if (target_exited < 0)
      return target_exited;
    const int debugger_exited = pidfd_is_exited(session_it->second.debugger_pidfd.get());
    if (debugger_exited < 0)
      return debugger_exited;
    if (target_exited == 1) {
      session_it->second.owned_dbg_fd.reset();
      session_it->second.target_mem_fd.reset();
      session_it->second.dbg_fd = -1;
      session_it->second.enabled = false;
      session_it->second.target_exited = true;
      if (args->op != KFD_IOC_DBG_TRAP_ENABLE)
        return -ESRCH;
      debug_sessions_.erase(session_it);
      session_it = debug_sessions_.end();
    } else if (debugger_exited == 1) {
      debug_sessions_.erase(session_it);
      session_it = debug_sessions_.end();
    }
  }
  const bool enabled = session_it != debug_sessions_.end();

  // Non-ENABLE ops require an active debug session (kernel: EINVAL).
  if (args->op != KFD_IOC_DBG_TRAP_ENABLE && !enabled) {
    UniqueDriverFd probe_pidfd;
    UniqueDriverFd probe_procfd;
    const int probe_result = pin_process_identity(target_pid, probe_pidfd, probe_procfd);
    if (probe_result == -ESRCH)
      return -ESRCH;
    if (probe_result != 0)
      return probe_result;
    const int zombie = procfd_is_zombie(probe_procfd.get());
    if (zombie != 0)
      return zombie == 1 ? -ESRCH : zombie;
    return -EINVAL;
  }

  // Pin the exact Linux task before consulting ptrace state. For an existing
  // session, use the identity captured by ENABLE; otherwise capture both a
  // pidfd and the matching procfs directory now. The pidfd liveness checks on
  // both sides of the procfs read ensure a numeric-pid reuse can never authorize
  // a different task.
  UniqueDriverFd new_target_pidfd;
  UniqueDriverFd new_target_procfd;
  UniqueDriverFd *target_pidfd;
  UniqueDriverFd *target_procfd;
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

  UniqueDriverFd new_debugger_pidfd;
  UniqueDriverFd new_debugger_procfd;
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
  if (target_proc != nullptr && session_it != debug_sessions_.end())
    session_it->second.saw_kfd_process = true;

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
    if (!runtime_enabled) {
      // A target that had a KfdProcess and no longer has one is a process on
      // its way out, not a live one refusing the op. The real driver cannot
      // reach this gate in that state at all: its pid lookup fails first and
      // returns -ESRCH, which rocdbgapi handles (PROCESS_EXITED -> invalidate
      // the queues and move on). -EPERM is what it escalates to a fatal
      // os_driver::resume_queues failure, so getting this distinction wrong
      // crashes GDB during teardown (gdb.rocm/multi-inferior-stress.exp).
      if (target_proc == nullptr && session_it != debug_sessions_.end() &&
          session_it->second.saw_kfd_process)
        return -ESRCH;
      // A suspend or resume naming only queues the process has already
      // destroyed asks nothing of the hardware, and answering it is the only
      // way rocdbgapi learns to drop them: the per-queue INVALID bit the normal
      // path writes back. It cannot learn it any other way once the runtime is
      // down, because its queue-list sweep is gated on the runtime being up
      // (process.cpp update_queues), and it escalates the -EPERM to a fatal
      // rather than retiring the queue.
      //
      // Upstream refuses this unconditionally, and never has to answer it: its
      // teardown does not leave a debugger holding a suspended queue across
      // runtime shutdown, so the request does not arise. Ours does, and the
      // narrow shape -- no runtime, and not one live queue among those named --
      // is exactly the one where refusing costs information and buys nothing.
      // Anything still live keeps the kernel's answer.
      if (!queues_all_dead(target_proc, *args))
        return -EPERM;
      util::Logger::vm("DBG_TRAP op=", args->op, " for pid=", target_pid,
                       " names only destroyed queues and the runtime is down; reporting them "
                       "invalid instead of -EPERM");
    }
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
    if (daemon_mode_) {
      if (target_mem_fd == nullptr || *target_mem_fd < 0 || target_proc_fd < 0)
        return -EBADF;
      const int mem_fl = safe_fcntl(*target_mem_fd, F_GETFL);
      if (mem_fl == -1 || (mem_fl & O_ACCMODE) != O_RDWR)
        return -EBADF;
      struct stat daemon_proc_stat {};
      struct stat client_proc_stat {};
      if (fstat(target_procfd->get(), &daemon_proc_stat) != 0 ||
          fstat(target_proc_fd, &client_proc_stat) != 0)
        return -errno;
      if (daemon_proc_stat.st_dev != client_proc_stat.st_dev ||
          daemon_proc_stat.st_ino != client_proc_stat.st_ino)
        return -ESRCH;
      // The proc directory is pinned to the right process by the check above,
      // but nothing yet ties the transferred mem fd to it -- O_RDWR only says
      // the fd is writable, not whose memory it addresses. This fd becomes
      // authoritative for guest reads and CWSR writes, so a mismatched one
      // silently redirects both at another process. procfs gives each
      // /proc/<pid>/mem its own inode, so comparing it against the pinned
      // directory's own "mem" entry settles the question.
      struct stat expected_mem_stat {};
      struct stat client_mem_stat {};
      if (fstatat(target_procfd->get(), "mem", &expected_mem_stat, 0) != 0 ||
          fstat(*target_mem_fd, &client_mem_stat) != 0)
        return -errno;
      if (expected_mem_stat.st_dev != client_mem_stat.st_dev ||
          expected_mem_stat.st_ino != client_mem_stat.st_ino)
        return -ESRCH;
    }

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
      // The session does not exist yet when the common path above records this,
      // so ENABLE has to do it itself; otherwise an inferior that opens
      // /dev/kfd, runs and exits before the debugger's first suspend/resume
      // never sets the flag and the gate below answers -EPERM after all.
      sess.saw_kfd_process = true;
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

    // Both transferred fds are adopted here, past every failure check, because
    // each only has to survive a successful ENABLE: on any earlier return the
    // caller still owns it and reclaims it. Adopting the notifier sooner would
    // double-close it, because the transport clears cmd->in_handle only when
    // the ioctl succeeds -- a late liveness failure would destroy sess (closing
    // the fd) and then let the transport close the same number again. With
    // daemon requests running concurrently another thread can be handed that
    // number in between, so the second close would land on an unrelated
    // descriptor. In local mode dbg_fd is the debugger's own descriptor and
    // nothing is owned here.
    if (daemon_mode_) {
      sess.owned_dbg_fd = UniqueDriverFd(dbg_fd);
      sess.target_mem_fd = UniqueDriverFd(*target_mem_fd);
      *target_mem_fd = -1;
    }
    auto [inserted, _] = debug_sessions_.emplace(target_pid, std::move(sess));
    if (target_proc != nullptr && inserted->second.target_mem_fd) {
      for (auto &g : gpus_)
        if (auto *mem = g.soc ? g.soc->memory() : nullptr)
          mem->set_process_mem_fd(target_proc->process_id(), inserted->second.target_mem_fd.get());
    }
    set_debug_active_on_all_cus(true);
    debug_sessions_cv_.notify_one();
    return 0;
  }
  case KFD_IOC_DBG_TRAP_DISABLE: {
    // Erasing the session releases the debugger notifier: in daemon mode the
    // session's UniqueHandle closes the SCM_RIGHTS-transferred fd it owns; in
    // local mode nothing is owned, so the debugger's own fd is left untouched.
    //
    // The queued debug events and the per-queue exception status go with it.
    // Both are debugger-visible state that only has meaning inside a session:
    // leaving them behind would hand a stale EC_QUEUE_NEW, or an exception the
    // previous debugger already consumed, to whoever attaches next. Any wave
    // the departing debugger left stopped is resumed below, so detaching never
    // strands the inferior's GPU work.
    //
    // Erase the session first, then release the inferior outside the lock.
    // Dropping debug_sessions_mutex_ before touching any CU is mandatory: the
    // engine thread takes these two the other way round -- ComputeUnitCore::step()
    // runs the issue loop under the CU's wave-state lock and calls back into
    // on_wave_trap_complete()/on_wave_watchpoint()/on_wave_illegal_inst(), each
    // of which acquires debug_sessions_mutex_ -- so holding it across
    // with_wave_state_locked() closes an AB-BA cycle and hangs a detach against
    // a wave that is trapping at that moment. Erasing before the release also
    // closes the window in which the session was still enabled while its events
    // had already been cleared, letting a live callback publish into a session
    // on its way out. This is the same invariant SUSPEND_QUEUES observes.
    debug_sessions_.erase(target_pid);
    lk.unlock();
    // target_proc, not a fresh find_process_by_client_pid(): it was resolved
    // above while the pidfd/procfd pin still vouched for the identity, so a
    // numeric pid reused since then cannot be selected here. It is also the
    // only resolution that is correct for a self-debugging process, where the
    // debuggee is `caller` and need not be reachable by client pid at all --
    // looking it up again would return nullptr and silently skip the whole
    // release, stranding the very waves this path exists to free.
    release_debuggee_state(target_pid, target_proc);
    // An explicit detach also has to release a waiter, or the inferior pays the
    // full liveness deadline for a debugger that is deliberately going away.
    cancel_runtime_handshake(target_pid);
    // Deliberately left unlocked: nothing below touches debug_sessions_, and
    // release_debuggee_state() already re-took and released the mutex for the
    // one guarded question it has to ask.
    return 0;
  }
  case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
    return debug_device_snapshot(args->device_snapshot);
  case KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT:
    return debug_queue_snapshot(target_proc, args->queue_snapshot);
  case KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO:
    return debug_query_exception_info(target_pid, args->query_exception_info);
  case KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT: {
    const auto &event = args->send_runtime_event;
    if (event.exception_mask != 0 &&
        (event.exception_mask & KFD_EC_MASK(EC_PROCESS_RUNTIME)) == 0) {
      // Forwarding a queue exception needs the target's live GPU state, which
      // only exists once it has opened /dev/kfd. A debugger may be attached
      // before that, or still attached after the target closed the device, so
      // this op has to guard the lookup the way its siblings above do.
      if (target_proc == nullptr)
        return -ESRCH;
      auto *gpu = find_gpu(event.gpu_id);
      if (!gpu || !gpu->soc)
        return -ENODEV;
      // Drop debug_sessions_mutex_ first: signal_queue_exception() takes the CU
      // wave-state lock and waits up to a second for the target to observe the
      // exception word, while the engine thread runs its issue loop under that
      // same wave-state lock and calls back into the trap/watchpoint handlers,
      // which take debug_sessions_mutex_. Holding it across the call inverts
      // that order -- the inversion DISABLE and SUSPEND_QUEUES both avoid -- and
      // would additionally stall every trap callback for the duration of the
      // wait. Nothing below this point reads debug_sessions_ or session_it.
      lk.unlock();
      const bool delivered = signal_runtime_queue_exception(
          event.gpu_id, event.queue_id, target_proc->process_id(), event.exception_mask);
      return delivered ? 0 : -ENOENT;
    }
    std::lock_guard<std::mutex> runtime_lock(runtime_handshake_mutex_);
    runtime_acked_.insert(target_pid);
    runtime_handshake_cv_.notify_all();
    return 0;
  }
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
    watches[slot] = KfdProcess::DebugSession::AddressWatch::from_kfd(
        args->set_node_address_watch.address, args->set_node_address_watch.mask,
        args->set_node_address_watch.mode);
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
    return debug_query_event(target_pid, target_proc, args->query_debug_event);
  case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES: {
    if (args->suspend_queues.num_queues != 0 && args->suspend_queues.queue_array_ptr == 0)
      return -EFAULT;
    auto *queue_ids =
        reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(args->suspend_queues.queue_array_ptr));
    // Engine callbacks enter with a CU's wave-state lock and briefly acquire
    // debug_sessions_mutex_. Do not invert that order while freezing CUs.
    lk.unlock();
    const int result = suspend_debug_queues(target_proc, queue_ids, args->suspend_queues.num_queues,
                                            args->suspend_queues.exception_mask);
    clear_completed_debug_queues(target_proc, queue_ids, args->suspend_queues.num_queues);
    return result;
  }
  case KFD_IOC_DBG_TRAP_RESUME_QUEUES: {
    if (args->resume_queues.num_queues != 0 && args->resume_queues.queue_array_ptr == 0)
      return -EFAULT;
    auto *queue_ids =
        reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(args->resume_queues.queue_array_ptr));
    lk.unlock();
    return resume_debug_queues(target_proc, queue_ids, args->resume_queues.num_queues);
  }
  default:
    return -EINVAL;
  }
}

int SimulatedKfd::debug_device_snapshot(kfd_ioctl_dbg_trap_device_snapshot_args &args) {
  // Mirrors kfd_dbg_trap_device_snapshot() (amd/amdkfd/kfd_debug.c): report the
  // total device count, clamp the per-entry size, and fill up to the caller's
  // buffer capacity. The two-call protocol is to call once with a small buffer
  // and read the true total back, then call again sized for it -- rocdbgapi's
  // kfd_snapshots::fetch() probes with a one-entry buffer, not a null pointer.
  //
  // The buffer is validated first, before any output is written: the driver
  // rejects a malformed request outright (-EINVAL) rather than half-answering
  // it, so a caller cannot read a device total off a call that failed this way.
  if (args.snapshot_buf_ptr == 0)
    return -EINVAL;

  // Only devices we can actually describe are enumerable. gpu_infos_ is filled
  // by setup_topology, which every embedder is free to skip or to call with
  // fewer devices than gpus_ holds (a config whose device block is absent, or
  // the single-GpuInfo overload on a multi-SoC driver). Reporting gpus_.size()
  // regardless would hand rocdbgapi entries with simd_count/array_count zero,
  // which its agent_snapshot treats as a fatal error rather than a bad ioctl.
  const uint32_t total = static_cast<uint32_t>(std::min(gpus_.size(), gpu_infos_.size()));
  const uint32_t in_entry_size = args.entry_size;
  const uint32_t fill = std::min<uint32_t>(args.num_devices, total);

  args.num_devices = total;
  args.entry_size = std::min<uint32_t>(in_entry_size, sizeof(kfd_dbg_device_info_entry));

  if (fill == 0)
    return 0;

  // A zero stride is not an error: the driver's per-entry copy_to_user() moves
  // entry_size(OUT) == 0 bytes and succeeds, so the call reports the device
  // total and writes nothing. Falling through reproduces that exactly.
  auto *out = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(args.snapshot_buf_ptr));
  for (uint32_t i = 0; i < fill; ++i) {
    const Sysfs::GpuInfo &info = gpu_infos_[i];
    const kfd_process_device_apertures ap = gpu_apertures(i);

    kfd_dbg_device_info_entry entry{};
    entry.gpu_id = gpus_[i].gpu_id;
    entry.lds_base = ap.lds_base;
    entry.lds_limit = ap.lds_limit;
    entry.scratch_base = ap.scratch_base;
    entry.scratch_limit = ap.scratch_limit;
    entry.gpuvm_base = ap.gpuvm_base;
    entry.gpuvm_limit = ap.gpuvm_limit;
    entry.location_id = info.location_id;
    entry.vendor_id = info.vendor_id;
    entry.device_id = info.device_id;
    entry.revision_id = info.pci_revision_id;
    entry.subsystem_vendor_id = info.vendor_id;
    entry.subsystem_device_id = info.device_id;
    entry.fw_version = info.fw_version;
    entry.gfx_target_version = info.gfx_target_version;
    entry.simd_count = info.simd_count;
    entry.max_waves_per_simd = info.max_waves_per_simd;
    // KFD array_count is the per-XCC shader-array count (node_props.array_count).
    // Unlike sysfs, kfd_debug.c passes it through unscaled and reports num_xcc
    // alongside, so rocdbgapi recovers the SoC's total shader-engine count as
    // array_count * num_xcc / simd_arrays_per_engine. Normalize the XCC count
    // the same way sysfs does so that quotient cannot come out zero.
    entry.array_count = info.array_count_per_xcc();
    entry.simd_arrays_per_engine = info.effective_arrays_per_engine();
    entry.num_xcc = info.effective_num_xcc();
    const kmd::DebugTopology topology =
        kmd::effective_topology_for(info.gfx_target_version, info.capability, info.capability2,
                                    info.debug_prop, info.revision_id);
    entry.capability = topology.capability;
    // debug_prop is __u32 in the snapshot entry but __u64 in the sysfs node
    // property, so a config that captured a debug_prop above 2^32 would have
    // the two paths report different values. The derived bits all fit; make the
    // narrowing the uapi struct imposes explicit rather than incidental.
    entry.debug_prop = static_cast<uint32_t>(topology.debug_prop);

    std::memcpy(out + static_cast<uint64_t>(i) * in_entry_size, &entry, args.entry_size);
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
    return proc.event_state_.backing_fd();

  if (type == KFD_MMAP_TYPE_DOORBELL) {
    uint64_t encoded_gpu =
        (static_cast<uint64_t>(offset) & ~KFD_MMAP_TYPE_MASK) >> KFD_MMAP_GPU_ID_SHIFT;
    uint32_t db_gpu_id = static_cast<uint32_t>(encoded_gpu);
    if (!find_gpu(db_gpu_id))
      return -1;
    std::lock_guard<std::mutex> lock(proc.alloc_mutex_);
    const auto &gs = proc.gpu(gpu_ordinal(db_gpu_id));
    if (gs.doorbell_memfd >= 0) {
      util::Logger::cp("MEMFD_LOOKUP: pid=", proc.process_id(),
                       " DOORBELL canonical gpu_id=", db_gpu_id, " memfd=", gs.doorbell_memfd);
      return gs.doorbell_memfd;
    }
    // Compatibility fallback for callers that query the backing before the
    // mmap path has published the canonical descriptor.
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
