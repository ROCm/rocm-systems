/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/inotify.h>
#include <unistd.h>
#include <poll.h>
#include <dirent.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <array>

#include "rocm_smi/rocm_smi_kfd_data_manager.h"
#include "rocm_smi/kfd_ioctl.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace amd::smi::kfd {

namespace {

constexpr const char* kDevKfd = "/dev/kfd";
constexpr const char* kKfdProcBase = "/sys/class/kfd/kfd/proc/";
constexpr pid_t kInvalidPid = -1;
constexpr auto kMaxGpuCount = std::uint32_t(256);
constexpr auto kDefaultBuffSize = std::uint32_t(128);

using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;
using Milliseconds = std::chrono::milliseconds;
using KfdProcSnapshot = std::set<std::string>;

//=============================================================================
// PID Namespace Detection for Container Support
//=============================================================================
// In containers with PID namespace isolation, fork() returns container-local
// PIDs but KFD uses host PIDs. We detect containers and use baseline
// comparison to find our child's host PID entry.
//
// Shared namespace (baremetal/VM): Direct PID lookup
// Isolated namespace (container):  Baseline diff to find new entry
//=============================================================================
struct InternalConfig {
  bool initialized = false;
  bool pid_namespace_checked = false;
  bool in_pid_namespace = false;
};

struct IpcPayload {
  int32_t status;
  uint64_t data;
};

// Batch IPC payload - results for multiple GPUs
struct BatchIpcPayload {
  uint32_t count;
  struct {
    uint32_t gpu_id;
    int32_t status;
    uint64_t data;
  } results[kMaxGpuCount];  // Max GPUs supported
};

// Updated StoreBatch - also returns target result to avoid second loop
struct StoreBatchResult {
  bool success;           // true if all results were successful
  bool target_found;      // true if target_gpu_id was in the batch
  QueryResult target;     // result for target_gpu_id (if found)
};

//=============================================================================
// Cache Implementation - O(1) lookup using unordered_map
//=============================================================================

struct CachedEntry {
  QueryResult result;
  TimePoint fetched_at;

  bool IsValid(int64_t ttl_ms) const {
    if (ttl_ms <= 0) return false;
    auto age = std::chrono::duration_cast<Milliseconds>(
        SteadyClock::now() - fetched_at);
    return age.count() < ttl_ms;
  }
};

// Per-operation cache: gpu_id -> CachedEntry
// Using array indexed by OpType for fast access
using GpuCacheMap = std::unordered_map<uint32_t, CachedEntry>;

struct CacheState {
  std::shared_mutex mutex;  // Reader-writer lock for better concurrency
  std::array<GpuCacheMap, static_cast<size_t>(OpType::kOpTypeCount)> caches;

  std::optional<QueryResult> TryGet(OpType op, uint32_t gpu_id, int64_t ttl_ms) {
    std::shared_lock lock(mutex);
    auto& cache = caches[static_cast<size_t>(op)];
    auto it = cache.find(gpu_id);
    if (it != cache.end() && it->second.IsValid(ttl_ms)) {
      // Don't return cached errors - force refresh instead
      if (it->second.result.err_code != 0) {
        return std::nullopt;
      }
      return it->second.result;
    }
    return std::nullopt;
  }

  // Combined store + target lookup in single pass
  // Returns: success=true if all GPUs succeeded (cache updated)
  //          success=false if any GPU failed (cache cleared)
  //          target contains result for target_gpu_id regardless of success
  StoreBatchResult StoreBatch(OpType op, const BatchIpcPayload& payload,
                              uint32_t target_gpu_id) {
    std::unique_lock lock(mutex);
    auto& cache = caches[static_cast<size_t>(op)];
    TimePoint now = SteadyClock::now();

    // Typical case: when this could return-
    // 1. Caller passes invalid target_gpu_id or mismatched GPU ID list
    // 2. payload.count == 0, which means the loop just won't run
    StoreBatchResult result{true, false, {EINVAL, 0}};

    for (uint32_t i = 0; i < payload.count; ++i) {
      // Always check if this is our target (even on error path)
      if (payload.results[i].gpu_id == target_gpu_id) {
        result.target_found = true;
        result.target.err_code = payload.results[i].status;
        result.target.value = payload.results[i].data;
      }

      // On first error, clear cache and mark as failed
      if (payload.results[i].status != 0) {
        if (result.success) {  // Only clear once
          cache.clear();
          result.success = false;
        }
        continue;  // Keep iterating to find target
      }

      // Only store successful results if no errors yet
      if (result.success) {
        QueryResult res{payload.results[i].status, payload.results[i].data};
        cache[payload.results[i].gpu_id] = CachedEntry{res, now};
      }
    }

    return result;
  }

  void Purge(OpType op, int32_t gpu_id) {
    std::unique_lock lock(mutex);
    auto& cache = caches[static_cast<size_t>(op)];
    if (gpu_id < 0) {
      cache.clear();
    } else {
      cache.erase(static_cast<uint32_t>(gpu_id));
    }
  }

  void PurgeAll() {
    std::unique_lock lock(mutex);
    for (auto& cache : caches) {
      cache.clear();
    }
  }
};

//=============================================================================
// Global State - Meyers Singleton pattern for safe initialization
//=============================================================================
std::mutex& GetGlobalMutex() {
  static std::mutex instance;
  return instance;
}

InternalConfig& GetInternalConfig() {
  static InternalConfig instance;
  return instance;
}

KFDManagerConfig& GetKFDManagerConfig() {
  static KFDManagerConfig instance;
  return instance;
}

std::atomic<pid_t>& GetHelperPid() {
  static std::atomic<pid_t> instance{kInvalidPid};
  return instance;
}

CacheState& GetGlobalCache() {
  static CacheState instance;
  return instance;
}

//=============================================================================
// Helper Functions
//=============================================================================

inline bool PathExists(const char* path) noexcept {
  struct stat st;
  return stat(path, &st) == 0;
}

inline bool KfdProcEntryExists(const char* full_path) noexcept {
  return access(full_path, F_OK) == 0;
}

inline bool KfdProcEntryExists(const std::string& entry_name) noexcept {
  char path[kDefaultBuffSize];
  snprintf(path, sizeof(path), "%s%s", kKfdProcBase, entry_name.c_str());
  return access(path, F_OK) == 0;
}

void WaitForEntryRemoval(const std::string& entry_name, int poll_ms) {
  const std::string kfd_path = std::string(kKfdProcBase) + entry_name;

  if (!KfdProcEntryExists(kfd_path.c_str())) return;

  auto start_time = SteadyClock::now();
  auto deadline = start_time + Milliseconds(GetKFDManagerConfig().max_cleanup_wait_ms);
  bool use_inotify = false;
  int notify_fd = -1;

  if (!GetKFDManagerConfig().disable_inotify_polling) {
    notify_fd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    use_inotify = (notify_fd >= 0);
  }

  bool timed_out = false;

  if (use_inotify) {
    int watch_fd = inotify_add_watch(notify_fd, kKfdProcBase, IN_DELETE);
    if (watch_fd >= 0) {
      struct pollfd pfd = {notify_fd, POLLIN, 0};
      char buf[kMaxGpuCount];
      while (KfdProcEntryExists(kfd_path.c_str())) {
        if (SteadyClock::now() >= deadline) {
          timed_out = true;
          break;
        }
        if (poll(&pfd, 1, poll_ms) > 0) {
          static_cast<void>(read(notify_fd, buf, sizeof(buf)));
        }
      }
      inotify_rm_watch(notify_fd, watch_fd);
    }
    close(notify_fd);
  } else {
    // Stat-based polling (default, faster for short-lived entries)
    while (KfdProcEntryExists(kfd_path.c_str())) {
      if (SteadyClock::now() >= deadline) {
        timed_out = true;
        break;
      }
      std::this_thread::sleep_for(
          std::chrono::microseconds(GetKFDManagerConfig().cleanup_poll_us));
    }
  }

  auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
      SteadyClock::now() - start_time).count();

  if (timed_out) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | Entry " << entry_name
       << " NOT removed after " << elapsed_us << " us (TIMEOUT)";
    LOG_WARN(ss);
  } else {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | Entry " << entry_name
       << " removed after " << elapsed_us << " us ("
       << (use_inotify ? "inotify" : "stat-poll") << ")";
    LOG_DEBUG(ss);
  }
}

void WaitForKfdProcRemoval(pid_t pid) {
  WaitForEntryRemoval(std::to_string(pid),
                      static_cast<int>(GetKFDManagerConfig().inotify_poll_ms));
}

KfdProcSnapshot CaptureKfdProcEntries() {
  KfdProcSnapshot snapshot;
  DIR* dir = opendir(kKfdProcBase);
  if (!dir) return snapshot;

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_name[0] != '.') {
      snapshot.insert(entry->d_name);
    }
  }

  if (closedir(dir) != 0) {
    // Log but don't fail - data was already read successfully
    // The resource will be cleaned up when the process exits anyway
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | closedir failed: " << strerror(errno);
    LOG_WARN(ss);
  }
  return snapshot;
}

std::string DetectNewKfdProcEntry(const KfdProcSnapshot& baseline) {
  DIR* dir = opendir(kKfdProcBase);
  if (!dir) return "";

  struct dirent* entry;
  std::string new_entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_name[0] == '.') continue;
    if (baseline.count(entry->d_name) == 0) {
      new_entry = entry->d_name;
      break;
    }
  }

  if (closedir(dir) != 0) {
    // Log but don't fail - data was already read successfully
    // The resource will be cleaned up when the process exits anyway
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | closedir failed: " << strerror(errno);
    LOG_WARN(ss);
  }
  return new_entry;
}

bool DetectPidNamespace() {
  if (PathExists("/.dockerenv")) return true;
  if (PathExists("/run/.containerenv")) return true;

  std::ifstream cgroup_file("/proc/1/cgroup");
  if (cgroup_file.is_open()) {
    std::string line;
    while (std::getline(cgroup_file, line)) {
      if (line.find("docker") != std::string::npos ||
          line.find("kubepods") != std::string::npos ||
          line.find("containerd") != std::string::npos ||
          line.find("lxc") != std::string::npos ||
          line.find("podman") != std::string::npos) {
        return true;
      }
    }
  }

  std::ifstream status_file("/proc/self/status");
  if (status_file.is_open()) {
    std::string line;
    while (std::getline(status_file, line)) {
      if (line.compare(0, 6, "NSpid:") == 0) {
        std::istringstream iss(line.substr(6));
        pid_t pid_val;
        int pid_count = 0;
        while (iss >> pid_val) ++pid_count;
        if (pid_count > 1) return true;
        break;
      }
    }
  }
  return false;
}

bool IsInPidNamespace() {
  std::lock_guard lock(GetGlobalMutex());
  if (!GetInternalConfig().pid_namespace_checked) {
    GetInternalConfig().in_pid_namespace = DetectPidNamespace();
    GetInternalConfig().pid_namespace_checked = true;
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | PID namespace: "
       << (GetInternalConfig().in_pid_namespace ? "isolated (container)" : "shared");
    LOG_INFO(ss);
  }
  return GetInternalConfig().in_pid_namespace;
}

[[noreturn]] void ChildExecute(OpType op, uint32_t gpu_id, int write_fd) {
  IpcPayload result{0, 0};

  int fd = open(kDevKfd, O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    result.status = errno;
  } else {
    switch (op) {
      case OpType::kQueryAvailableVram: {
        struct kfd_ioctl_get_available_memory_args args{};
        args.gpu_id = gpu_id;
        result.status = (ioctl(fd, AMDKFD_IOC_AVAILABLE_MEMORY, &args) != 0)
                        ? errno : 0;
        result.data = args.available;
        break;
      }
      default:
        result.status = ENOTSUP;
    }
    close(fd);
  }

  // Note: Cannot log here - we're in forked child, must use only
  // async-signal-safe functions. Write failure will cause parent
  // to detect short read and report EIO.
  static_cast<void>(write(write_fd, &result, sizeof(result)));
  close(write_fd);
  _exit(result.status ? 1 : 0);
}

[[noreturn]] void ChildExecuteBatch(OpType op,
                                    const uint32_t* gpu_ids,
                                    uint32_t count,
                                    int write_fd) {
  BatchIpcPayload payload{};
  payload.count = 0;

  int fd = open(kDevKfd, O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    int err = errno;
    uint32_t limit = (count < kMaxGpuCount) ? count : kMaxGpuCount;
    for (uint32_t i = 0; i < limit; ++i) {
      payload.results[i] = {gpu_ids[i], err, 0};
    }
    payload.count = limit;
    // Note: Cannot log here - we're in forked child, must use only
    // async-signal-safe functions. Write failure will cause parent
    // to detect short read and report EIO.
    static_cast<void>(write(write_fd, &payload, sizeof(payload)));
    close(write_fd);
    _exit(1);
  }

  for (uint32_t i = 0; i < count && i < kMaxGpuCount; ++i) {
    payload.results[i].gpu_id = gpu_ids[i];

    switch (op) {
      case OpType::kQueryAvailableVram: {
        struct kfd_ioctl_get_available_memory_args args{};
        args.gpu_id = gpu_ids[i];
        if (ioctl(fd, AMDKFD_IOC_AVAILABLE_MEMORY, &args) != 0) {
          payload.results[i].status = errno;
          payload.results[i].data = 0;
        } else {
          payload.results[i].status = 0;
          payload.results[i].data = args.available;
        }
        break;
      }
      default:
        payload.results[i].status = ENOTSUP;
        payload.results[i].data = 0;
    }
    payload.count++;
  }

  close(fd);
  // Note: Cannot log here - we're in forked child, must use only
  // async-signal-safe functions. Write failure will cause parent
  // to detect short read and report EIO.
  static_cast<void>(write(write_fd, &payload, sizeof(payload)));
  close(write_fd);
  _exit(0);
}

int64_t ReadEnvInt64(const char* name, int64_t fallback) {
  const char* val = std::getenv(name);
  if (!val) return fallback;
  char* end = nullptr;
  errno = 0;
  long long parsed = std::strtoll(val, &end, 10);
  if (end == val || parsed < 0 || errno == ERANGE) {
    return fallback;
  }
  return static_cast<int64_t>(parsed);
}

void EnsureInitialized() {
  std::lock_guard lock(GetGlobalMutex());
  if (!GetInternalConfig().initialized) {
    GetInternalConfig().initialized = true;
  }
}

// Common fork/exec/wait logic to reduce code duplication
struct ForkResult {
  int err_code = 0;
  BatchIpcPayload payload{};
  bool success = false;
};

ForkResult ExecuteBatchFork(OpType op, const std::vector<uint32_t>& gpu_ids) {
  ForkResult result;

  if (gpu_ids.empty()) {
    result.err_code = EINVAL;
    return result;
  }

  bool in_container = IsInPidNamespace();
  KfdProcSnapshot pre_fork_snapshot;
  if (in_container) {
    pre_fork_snapshot = CaptureKfdProcEntries();
  }

  int pipe_fds[2];
  if (pipe2(pipe_fds, O_CLOEXEC) != 0) {
    result.err_code = errno;
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | pipe2 failed: " << strerror(errno);
    LOG_ERROR(ss);
    return result;
  }

  pid_t child_pid = fork();
  if (child_pid < 0) {
    result.err_code = errno;
    close(pipe_fds[0]);
    close(pipe_fds[1]);
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | fork failed: " << strerror(errno);
    LOG_ERROR(ss);
    return result;
  }

  if (child_pid == 0) {
    close(pipe_fds[0]);
    ChildExecuteBatch(op, gpu_ids.data(),
                      static_cast<uint32_t>(gpu_ids.size()),
                      pipe_fds[1]);
  }

  // Parent
  close(pipe_fds[1]);
  GetHelperPid().store(child_pid, std::memory_order_release);

  ssize_t bytes_read = read(pipe_fds[0], &result.payload, sizeof(result.payload));
  close(pipe_fds[0]);
  // read() returns -1 on error, 0 on EOF, or partial/full byte count.
  // We require exactly sizeof(payload) bytes for a valid response.
  bool read_ok = (bytes_read == static_cast<ssize_t>(sizeof(result.payload)));

  int child_status = 0;
  pid_t wait_result = waitpid(child_pid, &child_status, 0);
  if (wait_result != child_pid) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | waitpid anomaly, sending SIGKILL";
    LOG_WARN(ss);
    kill(child_pid, SIGKILL);
    waitpid(child_pid, nullptr, 0);
  }

  // Wait for KFD proc cleanup
  int poll_ms = static_cast<int>(GetKFDManagerConfig().inotify_poll_ms);
  std::string direct_entry = std::to_string(child_pid);

  if (!in_container) {
    if (KfdProcEntryExists(direct_entry)) {
      WaitForEntryRemoval(direct_entry, poll_ms);
    }
  } else {
    // Container PID namespace handling: We can't directly map the child's
    // container-local PID to its host PID in /sys/class/kfd/kfd/proc/.
    // We use a snapshot-diff approach to find the new entry.
    //
    // See above 'PID Namespace Detection for Container Support' for details.
    //
    // KNOWN RACE: Another process could create a KFD entry between our
    // snapshot and detection, causing us to wait for the wrong entry.
    // This is mitigated by:
    //   1. Narrow window (snapshot taken immediately before fork)
    //   2. Bounded wait (max_cleanup_wait_ms timeout)
    //   3. Kernel cleanup (child's entry disappears when child exits)
    // Worst case: we timeout waiting for wrong entry, but our child's
    // entry will have been cleaned up by the kernel anyway.
    if (KfdProcEntryExists(direct_entry)) {
      WaitForEntryRemoval(direct_entry, poll_ms);
    } else {
      std::string new_entry = DetectNewKfdProcEntry(pre_fork_snapshot);
      if (!new_entry.empty()) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << " | Container PID translation: "
           << child_pid << " -> " << new_entry;
        LOG_DEBUG(ss);
        WaitForEntryRemoval(new_entry, poll_ms);
      }
    }
  }

  GetHelperPid().store(kInvalidPid, std::memory_order_release);

  if (!read_ok) {
    result.err_code = EIO;
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | Pipe read failed";
    LOG_ERROR(ss);
    return result;
  }

  result.success = true;
  return result;
}

}  // anonymous namespace

//=============================================================================
// Public API Implementation
//=============================================================================

KFDManagerConfig GetCurrentConfig() {
  // WARNING: Do not add debug logs in any of initialization/getconfig,
  // since logging is enabled after these functions are called.

  std::lock_guard lock(GetGlobalMutex());
  return GetKFDManagerConfig();
}

void LoadConfigFromEnvironment(KFDManagerConfig& cfg) {  // NOLINT(runtime/references)
  cfg.use_original_vram_fcn = static_cast<bool>(
      ReadEnvInt64("AMDSMI_KFD_USE_ORIG_VRAM",
                   cfg.use_original_vram_fcn ? 1 : 0));
  cfg.disable_inotify_polling = static_cast<bool>(
      ReadEnvInt64("AMDSMI_KFD_DISABLE_INOTIFY_POLLING",
                   cfg.disable_inotify_polling ? 1 : 0));
  cfg.inotify_poll_ms = ReadEnvInt64(
      "AMDSMI_KFD_INOTIFY_POLL_MS", cfg.inotify_poll_ms);
  cfg.cleanup_poll_us = ReadEnvInt64(
      "AMDSMI_KFD_CLEANUP_POLL_US", cfg.cleanup_poll_us);
  cfg.cache_ttl_ms = ReadEnvInt64(
      "AMDSMI_KFD_CACHE_TTL_MS", cfg.cache_ttl_ms);
  cfg.max_cleanup_wait_ms = ReadEnvInt64(
    "AMDSMI_KFD_MAX_CLEANUP_WAIT_MS", cfg.max_cleanup_wait_ms);
}

int InitializeManager(const KFDManagerConfig& cfg) {
  // WARNING: Do not add debug logs in any of initialization/getconfig,
  // since logging is enabled after these functions are called.

  std::lock_guard lock(GetGlobalMutex());
  if (GetInternalConfig().initialized) return 0;
  GetKFDManagerConfig() = cfg;
  GetInternalConfig().initialized = true;
  return 0;
}

QueryResult ExecuteIsolatedQuery(OpType op, uint32_t gpu_id) {
  QueryResult out{};
  std::ostringstream ss;

  EnsureInitialized();

  bool in_container = IsInPidNamespace();
  KfdProcSnapshot pre_fork_snapshot;
  if (in_container) {
    pre_fork_snapshot = CaptureKfdProcEntries();
  }

  int pipe_fds[2];
  if (pipe2(pipe_fds, O_CLOEXEC) != 0) {
    out.err_code = errno;
    ss << __PRETTY_FUNCTION__ << " | pipe2 failed: " << strerror(errno);
    LOG_ERROR(ss);
    return out;
  }

  pid_t child_pid = fork();
  if (child_pid < 0) {
    out.err_code = errno;
    close(pipe_fds[0]);
    close(pipe_fds[1]);
    ss << __PRETTY_FUNCTION__ << " | fork failed: " << strerror(errno);
    LOG_ERROR(ss);
    return out;
  }

  if (child_pid == 0) {
    close(pipe_fds[0]);
    ChildExecute(op, gpu_id, pipe_fds[1]);
  }

  close(pipe_fds[1]);
  GetHelperPid().store(child_pid, std::memory_order_release);

  IpcPayload payload{};
  ssize_t bytes_read = read(pipe_fds[0], &payload, sizeof(payload));
  close(pipe_fds[0]);
  // read() returns -1 on error, 0 on EOF, or partial/full byte count.
  // We require exactly sizeof(payload) bytes for a valid response.
  bool read_ok = (bytes_read == static_cast<ssize_t>(sizeof(payload)));

  int child_status = 0;
  pid_t wait_result = waitpid(child_pid, &child_status, 0);
  if (wait_result != child_pid) {
    ss << __PRETTY_FUNCTION__ << " | waitpid anomaly, sending SIGKILL";
    LOG_WARN(ss);
    kill(child_pid, SIGKILL);
    waitpid(child_pid, nullptr, 0);
  }

  int poll_ms = static_cast<int>(GetKFDManagerConfig().inotify_poll_ms);
  std::string direct_entry = std::to_string(child_pid);

  if (!in_container) {
    if (KfdProcEntryExists(direct_entry)) {
      WaitForEntryRemoval(direct_entry, poll_ms);
    }
  } else {
    // Container PID namespace handling: We can't directly map the child's
    // container-local PID to its host PID in /sys/class/kfd/kfd/proc/.
    // We use a snapshot-diff approach to find the new entry.
    //
    // See above 'PID Namespace Detection for Container Support' for details.
    //
    // KNOWN RACE: Another process could create a KFD entry between our
    // snapshot and detection, causing us to wait for the wrong entry.
    // This is mitigated by:
    //   1. Narrow window (snapshot taken immediately before fork)
    //   2. Bounded wait (max_cleanup_wait_ms timeout)
    //   3. Kernel cleanup (child's entry disappears when child exits)
    // Worst case: we timeout waiting for wrong entry, but our child's
    // entry will have been cleaned up by the kernel anyway.
    if (KfdProcEntryExists(direct_entry)) {
      WaitForEntryRemoval(direct_entry, poll_ms);
    } else {
      std::string new_entry = DetectNewKfdProcEntry(pre_fork_snapshot);
      if (!new_entry.empty()) {
        ss << __PRETTY_FUNCTION__ << " | Container PID translation: "
           << child_pid << " -> " << new_entry;
        LOG_INFO(ss);
        WaitForEntryRemoval(new_entry, poll_ms);
      }
    }
  }

  GetHelperPid().store(kInvalidPid, std::memory_order_release);

  if (!read_ok) {
    out.err_code = EIO;
    ss << __PRETTY_FUNCTION__ << " | Pipe read failed";
    LOG_ERROR(ss);
    return out;
  }

  out.err_code = payload.status;
  out.value = payload.data;
  return out;
}

QueryResult ExecuteBatchQueryCached(OpType op,
                                    const std::vector<uint32_t>& gpu_ids,
                                    uint32_t target_gpu_id) {
  EnsureInitialized();
  int64_t ttl_ms = GetKFDManagerConfig().cache_ttl_ms;

  // O(1) cache lookup - returns nullopt for expired/error entries
  if (auto cached = GetGlobalCache().TryGet(op, target_gpu_id, ttl_ms)) {
    return *cached;
  }

  // TODO(optimization): Consider adding a "single-flight" / "coalescing"
  // pattern. This avoids redundant forks when multiple threads have
  // concurrent cache misses.
  //
  // Current behavior: both threads fork (safe but slightly inefficient).
  // Search: "singleflight pattern", "request coalescing", "call coalescing C++"
  //         or "deduplicate concurrent requests" folly Singleton
  //
  // A few open-source C++ libraries have implementations, try:
  // "folly Singleton" or "folly futures coalescing"
  // or "C++ promise shared future" (to see  underlying mechanism)


  // Cache miss - execute batch fork for all GPUs
  auto fork_result = ExecuteBatchFork(op, gpu_ids);

  // On fork/pipe failure, purge cache (system state unreliable)
  if (!fork_result.success) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | Fork failed (err=" << fork_result.err_code
       << "), purging cache for op=" << static_cast<uint32_t>(op);
    LOG_WARN(ss);
    GetGlobalCache().Purge(op, -1);
    return QueryResult{fork_result.err_code, 0};
  }

  // Store results and find target in single pass
  auto store_result = GetGlobalCache().StoreBatch(op, fork_result.payload, target_gpu_id);

  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | Refreshed " << fork_result.payload.count
     << " GPUs, target_found=" << store_result.target_found
     << ", cache_updated=" << (store_result.success ? "yes" : "no (error)");
  LOG_DEBUG(ss);

  return store_result.target;
}

void PurgeCacheEntries(OpType op, int32_t gpu_id) {
  GetGlobalCache().Purge(op, gpu_id);
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | Purged cache for op="
     << static_cast<uint32_t>(op) << " gpu_id=" << gpu_id;
  LOG_DEBUG(ss);
}

void PurgeAllCacheEntries() {
  GetGlobalCache().PurgeAll();
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | Purged all cache entries";
  LOG_DEBUG(ss);
}

int QueryAvailableVram(uint32_t gpu_id, uint64_t* out_available) {
  if (!out_available) return EINVAL;
  QueryResult res = ExecuteIsolatedQuery(OpType::kQueryAvailableVram, gpu_id);
  if (res.err_code != 0) return res.err_code;
  *out_available = res.value;
  return 0;
}

int QueryAvailableVramBatch(const std::vector<uint32_t>& gpu_ids,
                            uint32_t target_gpu_id,
                            uint64_t* out_available) {
  if (!out_available) return EINVAL;
  QueryResult res = ExecuteBatchQueryCached(OpType::kQueryAvailableVram,
                                            gpu_ids, target_gpu_id);
  if (res.err_code != 0) return res.err_code;
  *out_available = res.value;
  return 0;
}

void TerminateActiveHelpers() {
  pid_t pid = GetHelperPid().exchange(kInvalidPid, std::memory_order_acq_rel);
  if (pid > 0) {
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    WaitForKfdProcRemoval(pid);
  }
}

}  // namespace amd::smi::kfd
