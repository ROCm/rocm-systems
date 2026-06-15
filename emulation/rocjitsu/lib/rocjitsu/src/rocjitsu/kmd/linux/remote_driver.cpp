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

} // namespace

bool RemoteDriver::find_memfd_for_addr(void *addr, size_t length, int *memfd_out,
                                       off_t *offset_out) {
  *memfd_out = -1;
  *offset_out = 0;
  auto target = reinterpret_cast<uint64_t>(addr);
  std::lock_guard<std::mutex> lock(rpc_mutex_);
  for (const auto &r : alloc_ranges_) {
    if (target >= r.va && target + length <= r.va + r.size) {
      *memfd_out = r.memfd;
      *offset_out = static_cast<off_t>(target - r.va);
      return true;
    }
  }
  return false;
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

  // Create a high-numbered synthetic KFD fd to avoid collisions with ROCR's
  // internal fd lifecycle. Use the top of the current rlimit range (same
  // approach as SimulatedDriver::init_reserved_fd_range).
  struct rlimit rl {};
  getrlimit(RLIMIT_NOFILE, &rl);
  int fd_min = static_cast<int>(rl.rlim_cur) - 64;
  if (fd_min < 256)
    fd_min = 256;
  auto raw_fd = static_cast<int>(syscall(SYS_memfd_create, "rocjitsu_remote_kfd", MFD_CLOEXEC));
  if (raw_fd < 0)
    return -1;
  int fd = fcntl(raw_fd, F_DUPFD_CLOEXEC, fd_min);
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
  }

  return 0;
}

int RemoteDriver::ioctl(unsigned long request, void *arg) {
  assert(sock_ >= 0 && "ioctl called on disconnected RemoteDriver");
  assert(arg && "ioctl called with null arg");

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

  if (!rpc_send_exact(sock_, buf.data(), buf.size()))
    return -1;

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
