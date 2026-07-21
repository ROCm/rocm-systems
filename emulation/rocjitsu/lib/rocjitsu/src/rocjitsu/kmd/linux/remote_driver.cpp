// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file remote_driver.cpp
/// @brief Client-side RPC stub for the rocjitsu daemon.

#include "rocjitsu/kmd/linux/remote_driver.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <linux/mman.h>
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
#include <mutex>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace rocjitsu {
namespace {

constexpr bool has_embedded_pointers(unsigned long request) {
  switch (canonical_ioctl_request(request)) {
  case AMDKFD_IOC_WAIT_EVENTS:
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
  case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
  case AMDKFD_IOC_DBG_TRAP:
    return true;
  case AMDKFD_IOC_SVM:
    // SVM's variable-length attribute array is part of the ioctl payload, not a
    // client pointer that the daemon has to rewrite.
    return false;
  default:
    return false;
  }
}

/// @brief Safe wrapper around syscall(SYS_mmap, ...) that avoids UB from
/// casting negative return values through uintptr_t/pointer types.
void *safe_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  long rc = syscall(SYS_mmap, addr, length, prot, flags, fd, offset);
  if (rc < 0)
    return MAP_FAILED;
  return reinterpret_cast<void *>(static_cast<uintptr_t>(rc));
}

Sysfs::GpuInfo gpu_info_from_rpc(const RpcGpuInfo &src) {
  Sysfs::GpuInfo gpu{};
  gpu.gpu_id = src.gpu_id;
  gpu.gfx_target_version = src.gfx_target_version;
  gpu.vendor_id = src.vendor_id;
  gpu.device_id = src.device_id;
  gpu.family_id = src.family_id;
  gpu.unique_id = src.unique_id;
  gpu.location_id = src.location_id;
  gpu.domain = src.domain;
  gpu.hive_id = src.hive_id;
  gpu.drm_render_minor = src.drm_render_minor;
  gpu.revision_id = src.revision_id;
  gpu.pci_revision_id = src.pci_revision_id;
  gpu.simd_count = src.simd_count;
  gpu.max_waves_per_simd = src.max_waves_per_simd;
  gpu.num_shader_engines = src.num_shader_engines;
  gpu.num_shader_arrays_per_engine = src.num_shader_arrays_per_engine;
  gpu.num_cu_per_sh = src.num_cu_per_sh;
  gpu.simd_per_cu = src.simd_per_cu;
  gpu.wave_front_size = src.wave_front_size;
  gpu.num_xcc = src.num_xcc;
  gpu.max_slots_scratch_cu = src.max_slots_scratch_cu;
  gpu.local_mem_size = src.local_mem_size;
  gpu.vram_type = src.vram_type;
  gpu.lds_size_kb = src.lds_size_kb;
  gpu.mem_width = src.mem_width;
  gpu.mem_clk_max = src.mem_clk_max;
  gpu.l1_size_kb = src.l1_size_kb;
  gpu.l1_line_size = src.l1_line_size;
  gpu.l1_assoc = src.l1_assoc;
  gpu.l2_size_kb = src.l2_size_kb;
  gpu.l2_line_size = src.l2_line_size;
  gpu.l2_assoc = src.l2_assoc;
  gpu.num_sdma_engines = src.num_sdma_engines;
  gpu.num_sdma_xgmi_engines = src.num_sdma_xgmi_engines;
  gpu.num_cp_queues = src.num_cp_queues;
  gpu.max_engine_clk_fcompute = src.max_engine_clk_fcompute;
  gpu.capability = src.capability;
  gpu.capability2 = src.capability2;
  gpu.debug_prop = src.debug_prop;
  gpu.fw_version = src.fw_version;
  gpu.sdma_fw_version = src.sdma_fw_version;

  auto *name_end =
      static_cast<const char *>(std::memchr(src.marketing_name, '\0', sizeof(src.marketing_name)));
  auto name_len =
      name_end ? static_cast<size_t>(name_end - src.marketing_name) : sizeof(src.marketing_name);
  gpu.marketing_name.assign(src.marketing_name, name_len);
  return gpu;
}

} // namespace

RemoteDriver::MemfdLookup RemoteDriver::find_memfd_for_addr(void *addr, size_t length,
                                                            int *memfd_out, off_t *offset_out) {
  *memfd_out = -1;
  *offset_out = 0;
  auto target = reinterpret_cast<uint64_t>(addr);
  std::lock_guard<std::mutex> lock(rpc_mutex_);
  for (const auto &r : alloc_ranges_) {
    // Overflow-safe containment test for [target, target+length) within
    // [r.va, r.va+r.size). Rearranged to subtraction so neither target+length
    // nor r.va+r.size can wrap uint64_t and admit an out-of-range address.
    if (target >= r.va && length <= r.size && target - r.va <= r.size - length) {
      // Return a DUP of the memfd, not the stored fd, so the caller owns a
      // descriptor whose lifetime is independent of this RemoteDriver. Without
      // this, a concurrent last-close -> RemoteDriver::close() (which closes the
      // stored handle_memfds_ fds) could close the fd between this return and the
      // caller's ftruncate/fallocate/mmap, operating on a closed/reused fd. The
      // dup is taken under rpc_mutex_, the same lock close() holds, so the stored
      // fd is guaranteed still open here. The caller MUST close the returned fd.
      int dup_fd = static_cast<int>(syscall(SYS_fcntl, r.memfd, F_DUPFD_CLOEXEC, 0));
      if (dup_fd < 0) {
        // The range matched but we could not hand out a descriptor. Report this
        // distinctly so the caller fails the mmap rather than silently falling
        // back to an anonymous mapping and breaking the shared-memory invariant.
        // The dup's errno (EMFILE/ENFILE/...) reaches the caller: the only code
        // that runs before this returns is the rpc_mutex_ lock_guard unlock, and
        // pthread_mutex_unlock does not touch errno on success. Do not add any
        // errno-setting call after this point without saving/restoring it.
        return MemfdLookup::kDupFailed;
      }
      *memfd_out = dup_fd;
      *offset_out = static_cast<off_t>(target - r.va);
      return MemfdLookup::kFound;
    }
  }
  return MemfdLookup::kNotFound;
}

RemoteDriver::RemoteDriver(int sock_fd) : sock_(sock_fd) {
  shutdown_efd_ = static_cast<int>(syscall(SYS_eventfd2, 0, EFD_CLOEXEC | EFD_NONBLOCK));
}

RemoteDriver::~RemoteDriver() {
  for (auto &[handle, fd] : handle_memfds_) {
    if (fd >= 0)
      syscall(SYS_close, fd);
  }
  if (sock_ >= 0)
    syscall(SYS_close, sock_);
  if (shutdown_efd_ >= 0)
    syscall(SYS_close, shutdown_efd_);
}

int RemoteDriver::open() {
  assert(sock_ >= 0 && "open called on disconnected RemoteDriver");
  closing_.store(false, std::memory_order_release);
  has_gpu_info_ = false;
  gpu_info_ = {};
  topology_path_.clear();
  drm_path_.clear();
  // Drain the shutdown eventfd so it doesn't immediately wake pollers.
  if (shutdown_efd_ >= 0) {
    uint64_t val = 0;
    syscall(SYS_read, shutdown_efd_, &val, sizeof(val));
  }
  std::lock_guard<std::mutex> lock(rpc_mutex_);

  RpcHeader hdr = {};
  hdr.opcode = RPC_HANDSHAKE;
  hdr.request_id = next_id_++;
  hdr.payload_bytes = 0;

  if (!rpc_send_exact(sock_, &hdr, sizeof(hdr)))
    return -1;

  RpcHeader resp = {};
  if (!rpc_recv_exact(sock_, &resp, sizeof(resp)))
    return -1;

  if (resp.result != 0)
    return resp.result;

  RpcHandshakeResponse hs = {};
  if (!rpc_recv_exact(sock_, &hs, sizeof(hs)))
    return -1;

  if (hs.version != kRpcProtocolVersion)
    return -1;

  has_gpu_info_ = hs.gpu_info.present != 0;
  if (has_gpu_info_)
    gpu_info_ = gpu_info_from_rpc(hs.gpu_info);

  constexpr uint32_t kMaxPathLen = 4096;
  if (hs.topology_path_len > kMaxPathLen)
    return -1;
  if (hs.topology_path_len > 0) {
    topology_path_.resize(hs.topology_path_len);
    if (!rpc_recv_exact(sock_, topology_path_.data(), hs.topology_path_len))
      return -1;
  }

  if (hs.drm_path_len > kMaxPathLen)
    return -1;
  if (hs.drm_path_len > 0) {
    drm_path_.resize(hs.drm_path_len);
    if (!rpc_recv_exact(sock_, drm_path_.data(), hs.drm_path_len))
      return -1;
  }

  return reissue_synthetic_kfd_fd();
}

int RemoteDriver::reissue_synthetic_kfd_fd() {
  // Create a high-numbered synthetic KFD fd to avoid collisions with ROCR's
  // internal fd lifecycle. Use the top of the current rlimit range (same
  // approach as SimulatedKfd::init_reserved_fd_range).
  struct rlimit rl {};
  getrlimit(RLIMIT_NOFILE, &rl);
  int fd_min = static_cast<int>(rl.rlim_cur) - 64;
  if (fd_min < 256)
    fd_min = 256;
  auto raw_fd = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_remote_kfd", MFD_CLOEXEC));
  if (raw_fd < 0)
    return -1;
  // Use the raw syscall, not fcntl(): this shared object exports an interposed
  // fcntl with default visibility, so an unqualified call would re-enter the
  // shim (reserve_dup_backend/untrack_dup, fd_mutex_) for a plain memfd dup.
  int fd = static_cast<int>(syscall(SYS_fcntl, raw_fd, F_DUPFD_CLOEXEC, fd_min));
  syscall(SYS_close, raw_fd);
  return fd;
}

int RemoteDriver::close() {
  closing_.store(true, std::memory_order_release);
  if (shutdown_efd_ >= 0) {
    uint64_t val = 1;
    syscall(SYS_write, shutdown_efd_, &val, sizeof(val));
  }

  {
    std::lock_guard<std::mutex> lock(rpc_mutex_);
    if (sock_ >= 0) {
      RpcHeader hdr{};
      hdr.opcode = RPC_CLOSE;
      hdr.request_id = next_id_++;
      rpc_send_exact(sock_, &hdr, sizeof(hdr));
    }

    if (sock_ >= 0) {
      syscall(SYS_close, sock_);
      sock_ = -1;
    }

    for (auto &[handle, fd] : handle_memfds_) {
      if (fd >= 0)
        syscall(SYS_close, fd);
    }
    handle_memfds_.clear();
    alloc_ranges_.clear();
    // Deliberately do NOT clear the handshake metadata (topology_path_,
    // drm_path_, gpu_info_, has_gpu_info_). It is written once during open() and
    // is immutable for the rest of the object's life. The interposer publishes
    // this RemoteDriver via an atomic shared_ptr and lets lock-free readers take
    // snapshots that call the accessors (topology_path(), drm_path(),
    // gpu_info()); a racing teardown clears the atomic and calls close() while
    // such a snapshot may still be live. Clearing these strings/structs here
    // would be a data race against those readers. Keeping the metadata
    // immutable-after-open makes the reads safe without a lock; the storage is
    // reclaimed when the last shared_ptr drops.
  }

  return 0;
}

int RemoteDriver::ioctl(unsigned long request, void *arg) {
  assert(arg && "ioctl called with null arg");
  // Do NOT assert(sock_ >= 0) here: sock_ is guarded by rpc_mutex_ and a
  // concurrent teardown_remote() -> close() can set it to -1 while another
  // thread holds a live shared_ptr snapshot and is entering this call. Reading
  // sock_ unlocked would be a data race (and would abort in Debug on exactly the
  // teardown-vs-in-flight-ioctl race this design tolerates). The locked send
  // path (send_ioctl) handles a closed socket gracefully by returning -1.

  // WAIT_EVENTS is handled client-side to avoid rpc_mutex_ contention.
  // Multiple ROCR threads poll WAIT_EVENTS concurrently during init. If each
  // poll goes through RPC, the rpc_mutex_ is held for the round-trip duration,
  // starving the main init thread's ioctls and mmaps.
  //
  // The signal page IS shared via memfd (same inode in both processes). The
  // daemon's signal_interrupt writes to the signal page slot. The client polls
  // the slot directly — no RPC round-trip, no mutex contention.
  //
  // For non-signal events (e.g., queue-inactive notifications), a single RPC
  // poll checks the daemon's EventState.
  if (request == AMDKFD_IOC_WAIT_EVENTS) {
    auto *wait_args = static_cast<kfd_ioctl_wait_events_args *>(arg);
    uint32_t original_timeout = wait_args->timeout;
    // timeout=0: return immediately without RPC. Signal values live in shared
    // memory (memfd) — ROCR reads them directly, so the WAIT_EVENTS ioctl
    // only needs to check for non-signal KFD events. Avoid the RPC round-trip
    // and rpc_mutex_ contention that starves the init thread.
    if (original_timeout == 0) {
      wait_args->wait_result = KFD_IOC_WAIT_RESULT_TIMEOUT;
      return 0;
    }

    // Blocking wait: poll the daemon periodically. Between polls, block on
    // the shutdown eventfd via poll(2) instead of sleeping. When close()
    // writes to the eventfd, poll returns immediately and the loop exits.
    // This avoids both the shutdown ordering deadlock (ROCR joins signal
    // threads before calling close) AND arbitrary time-based workarounds.
    auto deadline =
        (original_timeout >= 0xFFFFFFFEu)
            ? std::chrono::steady_clock::time_point::max()
            : std::chrono::steady_clock::now() + std::chrono::milliseconds(original_timeout);

    while (std::chrono::steady_clock::now() < deadline) {
      if (closing_.load(std::memory_order_acquire)) {
        wait_args->timeout = original_timeout;
        wait_args->wait_result = 1;
        return 0;
      }
      wait_args->timeout = 0;
      int rc = send_ioctl(request, arg);
      wait_args->timeout = original_timeout;
      if (rc != 0)
        return rc;
      if (wait_args->wait_result != KFD_IOC_WAIT_RESULT_TIMEOUT)
        return 0;
      // Block on shutdown_efd_ instead of sleeping. poll() returns
      // immediately if close() has written to the eventfd, or after
      // the poll timeout (5ms) for the next daemon poll iteration.
      // rpc_mutex_ is NOT held — other threads can send ioctls.
      if (shutdown_efd_ >= 0) {
        struct pollfd pfd = {shutdown_efd_, POLLIN, 0};
        struct timespec ts = {0, 5'000'000};
        syscall(SYS_ppoll, &pfd, 1, &ts, nullptr, 0);
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }
    wait_args->wait_result = 1;
    return 0;
  }

  return send_ioctl(request, arg);
}

int RemoteDriver::send_ioctl(unsigned long request, void *arg) {
  std::lock_guard<std::mutex> lock(rpc_mutex_);

  size_t arg_size = 0;
  if (!validate_ioctl_arg_size(request, arg, arg_size))
    return -EINVAL;

  // Save original embedded pointers before serialization. The daemon rewrites
  // these to point at its own buffer; we must restore the client-side originals
  // before copying inline response data back.
  uint64_t saved_events_ptr = 0;
  uint64_t saved_apertures_ptr = 0;
  uint64_t saved_device_ids_ptr = 0;
  uint64_t saved_dbg_rinfo_ptr = 0;
  uint32_t saved_dbg_rinfo_size = 0;
  uint64_t saved_dbg_snapshot_ptr = 0;
  size_t saved_dbg_snapshot_cap = 0;
  if (has_embedded_pointers(request)) {
    switch (request) {
    case AMDKFD_IOC_WAIT_EVENTS:
      saved_events_ptr = static_cast<kfd_ioctl_wait_events_args *>(arg)->events_ptr;
      break;
    case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
      saved_apertures_ptr = static_cast<kfd_ioctl_get_process_apertures_new_args *>(arg)
                                ->kfd_process_device_apertures_ptr;
      break;
    case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
    case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
      saved_device_ids_ptr =
          static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arg)->device_ids_array_ptr;
      break;
    case AMDKFD_IOC_DBG_TRAP: {
      auto *dbg = static_cast<kfd_ioctl_dbg_trap_args *>(arg);
      switch (dbg->op) {
      case KFD_IOC_DBG_TRAP_ENABLE:
        saved_dbg_rinfo_ptr = dbg->enable.rinfo_ptr;
        saved_dbg_rinfo_size = dbg->enable.rinfo_size;
        break;
      case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
        saved_dbg_snapshot_ptr = dbg->device_snapshot.snapshot_buf_ptr;
        saved_dbg_snapshot_cap =
            static_cast<size_t>(dbg->device_snapshot.num_devices) * dbg->device_snapshot.entry_size;
        break;
      default:
        break;
      }
      break;
    }
    default:
      break;
    }
  }

  constexpr size_t prefix = sizeof(RpcHeader) + sizeof(RpcIoctlRequest);
  std::vector<uint8_t> buf(prefix + arg_size);

  std::memcpy(buf.data() + prefix, arg, arg_size);

  if (has_embedded_pointers(request)) {
    auto *args_base = buf.data() + prefix;
    switch (request) {
    case AMDKFD_IOC_WAIT_EVENTS: {
      auto *wait_args = reinterpret_cast<kfd_ioctl_wait_events_args *>(args_base);
      size_t inline_size = wait_args->num_events * sizeof(kfd_event_data);
      size_t inline_offset = buf.size();
      buf.resize(inline_offset + inline_size);
      std::memcpy(buf.data() + inline_offset, reinterpret_cast<const void *>(wait_args->events_ptr),
                  inline_size);
      break;
    }
    case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
    case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU: {
      auto *map_args = reinterpret_cast<kfd_ioctl_map_memory_to_gpu_args *>(args_base);
      size_t inline_size = map_args->n_devices * sizeof(uint32_t);
      size_t inline_offset = buf.size();
      buf.resize(inline_offset + inline_size);
      std::memcpy(buf.data() + inline_offset,
                  reinterpret_cast<const void *>(map_args->device_ids_array_ptr), inline_size);
      break;
    }
    case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW: {
      auto *aperture_args = reinterpret_cast<kfd_ioctl_get_process_apertures_new_args *>(args_base);
      buf.resize(buf.size() + aperture_args->num_of_nodes * sizeof(kfd_process_device_apertures));
      break;
    }
    case AMDKFD_IOC_DBG_TRAP: {
      auto *dbg = reinterpret_cast<kfd_ioctl_dbg_trap_args *>(args_base);
      size_t inline_size = 0;
      switch (dbg->op) {
      case KFD_IOC_DBG_TRAP_ENABLE:
        inline_size =
            std::min(static_cast<size_t>(dbg->enable.rinfo_size), sizeof(kfd_runtime_info));
        break;
      case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
        inline_size =
            static_cast<size_t>(dbg->device_snapshot.num_devices) * dbg->device_snapshot.entry_size;
        break;
      default:
        break;
      }
      if (inline_size > 0)
        buf.resize(buf.size() + inline_size);
      break;
    }
    default:
      break;
    }
  }

  auto *hdr = reinterpret_cast<RpcHeader *>(buf.data());
  hdr->opcode = RPC_IOCTL;
  hdr->request_id = next_id_++;
  hdr->payload_bytes = static_cast<uint32_t>(buf.size() - sizeof(RpcHeader));
  hdr->result = 0;

  auto *ireq = reinterpret_cast<RpcIoctlRequest *>(buf.data() + sizeof(RpcHeader));
  ireq->ioctl_cmd = static_cast<uint32_t>(request);
  ireq->args_bytes = static_cast<uint32_t>(buf.size() - prefix);

  // For DBG_TRAP ENABLE, hand the debugger's notifier pipe write-end to the
  // daemon as an SCM_RIGHTS fd. The daemon substitutes it into the ioctl's
  // dbg_fd so the driver can wake the debugger when a wave stops — the same fd
  // the real kernel would receive through the ioctl. KFD_INVALID_FD (0xffffffff)
  // casts to -1 and is not sent.
  int send_fd = -1;
  if (request == AMDKFD_IOC_DBG_TRAP) {
    auto *dbg = static_cast<kfd_ioctl_dbg_trap_args *>(arg);
    if (dbg->op == KFD_IOC_DBG_TRAP_ENABLE && static_cast<int>(dbg->enable.dbg_fd) >= 0)
      send_fd = static_cast<int>(dbg->enable.dbg_fd);
  }
  if (send_fd >= 0) {
    if (rpc_send_msg(sock_, buf.data(), buf.size(), &send_fd, 1) <= 0) {
      // Preserve the transport errno — e.g. EBADF when the client handed us a
      // closed notifier fd for SCM_RIGHTS — instead of a bare -1, which the
      // interposer would surface as EPERM (-EPERM == -1).
      int err = errno;
      return err > 0 ? -err : -1;
    }
  } else if (!rpc_send_exact(sock_, buf.data(), buf.size())) {
    int err = errno;
    return err > 0 ? -err : -1;
  }

  // Receive response — may include a memfd via SCM_RIGHTS for ALLOC_MEMORY.
  uint8_t resp_header_buf[sizeof(RpcHeader)];
  int received_fds[1] = {-1};
  size_t num_fds = 1;
  auto bytes =
      rpc_recv_msg(sock_, resp_header_buf, sizeof(resp_header_buf), received_fds, &num_fds);
  if (bytes <= 0)
    return -1;

  auto *resp = reinterpret_cast<RpcHeader *>(resp_header_buf);

  if (resp->payload_bytes > 0) {
    std::vector<uint8_t> payload(resp->payload_bytes);
    if (!rpc_recv_exact(sock_, payload.data(), resp->payload_bytes))
      return -1;

    size_t copy_size = std::min(arg_size, static_cast<size_t>(resp->payload_bytes));
    std::memcpy(arg, payload.data(), copy_size);

    // Restore original client-side pointers that were overwritten by the
    // daemon's response (daemon rewrites them to point at its own buffer).
    if (has_embedded_pointers(request)) {
      switch (request) {
      case AMDKFD_IOC_WAIT_EVENTS:
        static_cast<kfd_ioctl_wait_events_args *>(arg)->events_ptr = saved_events_ptr;
        break;
      case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
        static_cast<kfd_ioctl_get_process_apertures_new_args *>(arg)
            ->kfd_process_device_apertures_ptr = saved_apertures_ptr;
        break;
      case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
      case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
        static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arg)->device_ids_array_ptr =
            saved_device_ids_ptr;
        break;
      case AMDKFD_IOC_DBG_TRAP: {
        auto *dbg = static_cast<kfd_ioctl_dbg_trap_args *>(arg);
        switch (dbg->op) {
        case KFD_IOC_DBG_TRAP_ENABLE:
          dbg->enable.rinfo_ptr = saved_dbg_rinfo_ptr;
          break;
        case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
          dbg->device_snapshot.snapshot_buf_ptr = saved_dbg_snapshot_ptr;
          break;
        default:
          break;
        }
        break;
      }
      default:
        break;
      }
    }

    if (has_embedded_pointers(request) && resp->payload_bytes > arg_size) {
      size_t extra = resp->payload_bytes - arg_size;
      switch (request) {
      case AMDKFD_IOC_WAIT_EVENTS: {
        auto *wait_args = static_cast<kfd_ioctl_wait_events_args *>(arg);
        std::memcpy(reinterpret_cast<void *>(wait_args->events_ptr), payload.data() + arg_size,
                    std::min(wait_args->num_events * sizeof(kfd_event_data), extra));
        break;
      }
      case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW: {
        auto *aperture_args = static_cast<kfd_ioctl_get_process_apertures_new_args *>(arg);
        std::memcpy(
            reinterpret_cast<void *>(aperture_args->kfd_process_device_apertures_ptr),
            payload.data() + arg_size,
            std::min(aperture_args->num_of_nodes * sizeof(kfd_process_device_apertures), extra));
        break;
      }
      case AMDKFD_IOC_DBG_TRAP: {
        auto *dbg = static_cast<kfd_ioctl_dbg_trap_args *>(arg);
        void *dst = nullptr;
        size_t copy_len = 0;
        switch (dbg->op) {
        case KFD_IOC_DBG_TRAP_ENABLE:
          // Only propagate runtime-info bytes on success; a failed op (e.g.
          // -EBADF from a rejected notifier fd) must not mutate caller memory
          // or dereference the saved output pointer, matching local mode.
          if (resp->result == 0) {
            dst = reinterpret_cast<void *>(saved_dbg_rinfo_ptr);
            copy_len = std::min(static_cast<size_t>(saved_dbg_rinfo_size),
                                std::min(static_cast<size_t>(dbg->enable.rinfo_size), extra));
          }
          break;
        case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
          // Only propagate snapshot bytes on success; a failed op (e.g. -ENOSYS)
          // must not mutate caller memory or dereference the saved output
          // pointer. Clamp any successful copy to the original buffer capacity.
          if (resp->result == 0) {
            dst = reinterpret_cast<void *>(saved_dbg_snapshot_ptr);
            copy_len = std::min(saved_dbg_snapshot_cap, extra);
          }
          break;
        default:
          break;
        }
        if (dst != nullptr && copy_len > 0)
          std::memcpy(dst, payload.data() + arg_size, copy_len);
        break;
      }
      default:
        break;
      }
    }
  }

  auto register_allocation = [&](uint64_t handle, uint64_t va_addr, uint64_t size, int memfd) {
    handle_memfds_[handle] = memfd;
    if (va_addr != 0 && size > 0)
      alloc_ranges_.push_back({va_addr, size, memfd});
  };

  auto promote_userptr = [&](uint64_t va_addr, uint64_t size, int memfd) {
    auto *va = reinterpret_cast<void *>(va_addr);
    auto length = static_cast<size_t>(size);

    [[maybe_unused]] auto ft_rc = ftruncate(memfd, static_cast<off_t>(length));
    fallocate(memfd, 0, 0, static_cast<off_t>(length));

    constexpr size_t page_size = 4096;
    size_t num_pages = (length + page_size - 1) / page_size;
    std::vector<uint8_t> resident(num_pages);
    auto mc_rc = syscall(SYS_mincore, va, length, resident.data());

    auto *temp =
        static_cast<uint8_t *>(safe_mmap(nullptr, length, PROT_WRITE, MAP_SHARED, memfd, 0));
    if (temp != MAP_FAILED) {
      if (mc_rc == 0) {
        auto *src = static_cast<uint8_t *>(va);
        for (size_t i = 0; i < num_pages; ++i) {
          if (resident[i] & 1) {
            size_t off = i * page_size;
            size_t n = std::min(page_size, length - off);
            std::memcpy(temp + off, src + off, n);
          }
        }
      }
      syscall(SYS_munmap, temp, length);
    }

    auto *mapped = safe_mmap(va, length, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, memfd, 0);
    if (mapped != MAP_FAILED)
      syscall(SYS_madvise, mapped, length, MADV_POPULATE_WRITE);
  };

  // Store memfd received from ALLOC_MEMORY for use in the subsequent mmap().
  // Also register the address range for anonymous MAP_FIXED interception.
  if (request == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU && resp->result == 0) {
    auto *alloc_args = static_cast<kfd_ioctl_alloc_memory_of_gpu_args *>(arg);
    if (num_fds > 0 && received_fds[0] >= 0) {
      register_allocation(alloc_args->handle, alloc_args->va_addr, alloc_args->size,
                          received_fds[0]);

      if (alloc_args->flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR)
        promote_userptr(alloc_args->va_addr, alloc_args->size, received_fds[0]);
    }
  }

  if (request == AMDKFD_IOC_IPC_IMPORT_HANDLE && resp->result == 0) {
    auto *import_args = static_cast<kfd_ioctl_ipc_import_handle_args *>(arg);
    if (num_fds > 0 && received_fds[0] >= 0) {
      uint64_t size = 0;
      struct stat st {};
      if (fstat(received_fds[0], &st) == 0)
        size = static_cast<uint64_t>(st.st_size);
      register_allocation(import_args->handle, import_args->va_addr, size, received_fds[0]);
    }
  }

  if (request == AMDKFD_IOC_EXPORT_DMABUF && resp->result == 0) {
    auto *export_args = static_cast<kfd_ioctl_export_dmabuf_args *>(arg);
    if (num_fds > 0 && received_fds[0] >= 0)
      export_args->dmabuf_fd = received_fds[0];
  }

  if (request == AMDKFD_IOC_FREE_MEMORY_OF_GPU && resp->result == 0) {
    auto *free_args = static_cast<kfd_ioctl_free_memory_of_gpu_args *>(arg);
    if (auto it = handle_memfds_.find(free_args->handle); it != handle_memfds_.end()) {
      int freed_memfd = it->second;
      std::erase_if(alloc_ranges_,
                    [freed_memfd](const AllocRange &r) { return r.memfd == freed_memfd; });
      syscall(SYS_close, freed_memfd);
      handle_memfds_.erase(it);
    }
  }

  return resp->result;
}

void *RemoteDriver::mmap(void *addr, size_t length, int prot, int flags, off_t offset) {
  std::lock_guard<std::mutex> lock(rpc_mutex_);

  // Send the mmap RPC so the daemon creates its own mapping for GPU simulation.
  int memfd = -1;
  int rc = send_mmap(addr, length, prot, flags, offset, &memfd);
  if (rc != 0) {
    if (memfd >= 0)
      syscall(SYS_close, memfd);
    return MAP_FAILED;
  }

  // Resolve the memfd for this allocation: prefer the mmap response fd, fall
  // back to the stored ALLOC_MEMORY fd (same underlying file, different fd).
  int mapping_memfd = memfd;
  uint64_t type = static_cast<uint64_t>(offset) & (0x3ULL << 62);
  if (mapping_memfd < 0 && type == 0) {
    uint64_t handle = static_cast<uint64_t>(offset) >> 12;
    if (auto it = handle_memfds_.find(handle); it != handle_memfds_.end())
      mapping_memfd = it->second;
  }
  if (mapping_memfd >= 0) {
    [[maybe_unused]] auto ft_rc2 = ftruncate(mapping_memfd, static_cast<off_t>(length));

    // Pre-copy committed pages (code objects) from the existing anonymous
    // reservation into the memfd before MAP_FIXED replaces them. Uses a temp
    // mapping outside the GPUVM range for the copy target.
    if ((flags & MAP_FIXED) && addr != nullptr) {
      auto prot_rc = syscall(SYS_mprotect, addr, length, PROT_READ | PROT_WRITE);
      if (prot_rc == 0) {
        constexpr size_t page_size = 4096;
        size_t num_pages = (length + page_size - 1) / page_size;
        std::vector<uint8_t> page_resident(num_pages);
        auto mc_rc = syscall(SYS_mincore, addr, length, page_resident.data());
        auto *temp = static_cast<uint8_t *>(
            safe_mmap(nullptr, length, PROT_WRITE, MAP_SHARED, mapping_memfd, 0));
        if (temp != MAP_FAILED) {
          if (mc_rc == 0) {
            auto *src = static_cast<uint8_t *>(addr);
            for (size_t i = 0; i < num_pages; ++i) {
              if (page_resident[i] & 1) {
                size_t off = i * page_size;
                size_t n = std::min(page_size, length - off);
                std::memcpy(temp + off, src + off, n);
              }
            }
          }
          syscall(SYS_munmap, temp, length);
        }
      }
    }

    // QEMU vhost-user pattern: per-allocation memfd with F_SEAL_SHRINK (set at
    // creation in alloc_memory_ioctl), MAP_SHARED|MAP_FIXED, then
    // MADV_POPULATE_WRITE to pre-fault pages. This surfaces any shmem ENOSPC
    // as errno rather than deferred SIGBUS on page fault.
    int mflags = MAP_SHARED;
    if (flags & MAP_FIXED)
      mflags |= MAP_FIXED;
    auto *mapped = safe_mmap(addr, length, PROT_READ | PROT_WRITE, mflags, mapping_memfd, 0);
    if (mapped != MAP_FAILED)
      syscall(SYS_madvise, mapped, length, MADV_POPULATE_WRITE);

    if (memfd >= 0 && memfd != mapping_memfd)
      syscall(SYS_close, memfd);
    return mapped;
  }

  // No memfd — anonymous fallback for doorbell/event pages.
  if (memfd >= 0)
    syscall(SYS_close, memfd);
  int mflags = MAP_ANONYMOUS | MAP_PRIVATE;
  if (flags & MAP_FIXED)
    mflags |= MAP_FIXED;
  return safe_mmap(addr, length, PROT_READ | PROT_WRITE, mflags, -1, 0);
}

int RemoteDriver::munmap(void *addr, size_t length) {
  std::lock_guard<std::mutex> lock(rpc_mutex_);

  RpcMunmapRequest req = {};
  req.addr = reinterpret_cast<uint64_t>(addr);
  req.length = length;

  RpcHeader hdr = {};
  hdr.opcode = RPC_MUNMAP;
  hdr.request_id = next_id_++;
  hdr.payload_bytes = sizeof(req);

  uint8_t send_buffer[sizeof(hdr) + sizeof(req)];
  std::memcpy(send_buffer, &hdr, sizeof(hdr));
  std::memcpy(send_buffer + sizeof(hdr), &req, sizeof(req));

  if (!rpc_send_exact(sock_, send_buffer, sizeof(send_buffer)))
    return -1;

  RpcHeader resp = {};
  if (!rpc_recv_exact(sock_, &resp, sizeof(resp)))
    return -1;

  if (resp.result == 0)
    syscall(SYS_munmap, addr, length);

  return resp.result;
}

int RemoteDriver::send_mmap(void *addr, size_t length, int prot, int flags, off_t offset,
                            int *memfd_out) {
  // rpc_mutex_ is already held by the caller (mmap()).
  RpcMmapRequest req = {};
  req.addr = reinterpret_cast<uint64_t>(addr);
  req.length = length;
  req.prot = prot;
  req.flags = flags;
  req.offset = offset;

  RpcHeader hdr = {};
  hdr.opcode = RPC_MMAP;
  hdr.request_id = next_id_++;
  hdr.payload_bytes = sizeof(req);

  uint8_t send_buffer[sizeof(hdr) + sizeof(req)];
  std::memcpy(send_buffer, &hdr, sizeof(hdr));
  std::memcpy(send_buffer + sizeof(hdr), &req, sizeof(req));

  if (!rpc_send_exact(sock_, send_buffer, sizeof(send_buffer)))
    return -1;

  uint8_t response_buffer[sizeof(RpcHeader) + sizeof(RpcMmapResponse)];
  int received_fds[1] = {-1};
  size_t num_fds = 1;
  auto bytes_received =
      rpc_recv_msg(sock_, response_buffer, sizeof(response_buffer), received_fds, &num_fds);
  if (bytes_received <= 0)
    return -1;

  auto *resp = reinterpret_cast<RpcHeader *>(response_buffer);
  *memfd_out = (num_fds > 0) ? received_fds[0] : -1;
  return resp->result;
}

} // namespace rocjitsu
