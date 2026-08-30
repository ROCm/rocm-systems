// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/daemon/rj_daemon.h"

#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/vm/rj_vm.h"
#include "util/unique_handle.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stop_token>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

struct rj_daemon_t {
  std::atomic<rj_daemon_status_t> status{RJ_DAEMON_STATUS_STARTING};
  std::atomic<bool> stop_requested{false};
  rj_vm_t *vm = nullptr;
  int listen_fd = -1;
  std::string socket_path;
  dev_t socket_device = 0;
  ino_t socket_inode = 0;
  std::mutex clients_mutex;
  std::vector<int> client_fds;
  std::jthread engine_thread;
  std::jthread accept_thread;
};

namespace {

using namespace rocjitsu;

bool checked_product(size_t lhs, size_t rhs, size_t *product) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    return false;
  *product = lhs * rhs;
  return true;
}

bool validate_ioctl_payload(uint32_t command, const void *buffer, size_t buffer_size) {
  size_t argument_size = 0;
  if (buffer_size < ioctl_arg_size(command))
    return false;
  if (!validate_ioctl_arg_size(command, buffer, argument_size) || buffer_size < argument_size)
    return false;

  size_t inline_size = 0;
  switch (canonical_ioctl_request(command)) {
  case AMDKFD_IOC_WAIT_EVENTS: {
    const auto *args = static_cast<const kfd_ioctl_wait_events_args *>(buffer);
    if (!checked_product(args->num_events, sizeof(kfd_event_data), &inline_size))
      return false;
    break;
  }
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU: {
    const auto *args = static_cast<const kfd_ioctl_map_memory_to_gpu_args *>(buffer);
    if (!checked_product(args->n_devices, sizeof(uint32_t), &inline_size))
      return false;
    break;
  }
  case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW: {
    const auto *args = static_cast<const kfd_ioctl_get_process_apertures_new_args *>(buffer);
    if (!checked_product(args->num_of_nodes, sizeof(kfd_process_device_apertures), &inline_size))
      return false;
    break;
  }
  case AMDKFD_IOC_DBG_TRAP: {
    // Every op the client reserves an inline tail for has to appear here, or
    // the exact-length test below rejects a well-formed request and the
    // connection is dropped. Keep this in step with the tail sizes
    // RemoteDriver::send_ioctl() computes and the pointers
    // reconstruct_embedded_pointers() repoints (rj_vm.cpp) -- all three must
    // agree on the same set of ops.
    const auto *args = static_cast<const kfd_ioctl_dbg_trap_args *>(buffer);
    switch (args->op) {
    case KFD_IOC_DBG_TRAP_ENABLE:
      inline_size =
          std::min(static_cast<size_t>(args->enable.rinfo_size), sizeof(kfd_runtime_info));
      break;
    case KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:
      if (!checked_product(args->device_snapshot.num_devices, args->device_snapshot.entry_size,
                           &inline_size))
        return false;
      break;
    case KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT:
      if (!checked_product(args->queue_snapshot.num_queues, args->queue_snapshot.entry_size,
                           &inline_size))
        return false;
      break;
    case KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO:
      inline_size = args->query_exception_info.info_size;
      break;
    case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:
      if (!checked_product(args->suspend_queues.num_queues, sizeof(uint32_t), &inline_size))
        return false;
      break;
    case KFD_IOC_DBG_TRAP_RESUME_QUEUES:
      if (!checked_product(args->resume_queues.num_queues, sizeof(uint32_t), &inline_size))
        return false;
      break;
    default:
      break;
    }
    break;
  }
  default:
    break;
  }

  return inline_size <= buffer_size - argument_size && argument_size + inline_size == buffer_size;
}

bool send_with_fd_exact(int socket, const void *data, size_t size, int fd) {
  ssize_t sent = 0;
  do {
    sent = rpc_send_msg(socket, data, size, &fd, 1);
  } while (sent < 0 && errno == EINTR);
  if (sent <= 0)
    return false;
  auto bytes_sent = static_cast<size_t>(sent);
  return bytes_sent == size ||
         rpc_send_exact(socket, static_cast<const uint8_t *>(data) + bytes_sent, size - bytes_sent);
}

/// @brief The SCM_RIGHTS descriptors a client may attach to one request.
///
/// @details DBG_TRAP ENABLE sends three: the debugger's notifier, and the
/// authorization the daemon needs to reach the debuggee's address space --
/// /proc/<pid>/mem, plus the /proc/<pid> directory that pins the identity that
/// fd belongs to. Every other request sends at most the notifier. Ownership is
/// RAII, so descriptors the VM declines to adopt are released with the request
/// rather than leaked for the life of the connection.
struct ReceivedFds {
  util::UniqueHandle notifier;
  util::UniqueHandle target_mem;
  util::UniqueHandle target_proc;
};

bool receive_header_with_fds(int socket, RpcHeader *header, ReceivedFds *fds) {
  int received_fds[3] = {-1, -1, -1};
  size_t received_fd_count = std::size(received_fds);
  ssize_t header_bytes = 0;
  do {
    header_bytes = rpc_recv_msg(socket, header, sizeof(*header), received_fds, &received_fd_count);
  } while (header_bytes < 0 && errno == EINTR);
  if (received_fd_count > 0)
    fds->notifier.reset(received_fds[0]);
  if (received_fd_count > 1)
    fds->target_mem.reset(received_fds[1]);
  if (received_fd_count > 2)
    fds->target_proc.reset(received_fds[2]);
  if (header_bytes <= 0)
    return false;
  const size_t bytes_received = static_cast<size_t>(header_bytes);
  return bytes_received == sizeof(*header) ||
         rpc_recv_exact(socket, reinterpret_cast<uint8_t *>(header) + bytes_received,
                        sizeof(*header) - bytes_received);
}

rj_client_pid_t peer_pid_for_socket(int fd) {
  struct ucred credentials {};
  socklen_t length = sizeof(credentials);
  if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &credentials, &length) == 0 &&
      length == sizeof(credentials) && credentials.pid > 0) {
    return static_cast<rj_client_pid_t>(credentials.pid);
  }
  return 0;
}

void close_client(rj_daemon_t *daemon, int client_fd) {
  std::lock_guard lock(daemon->clients_mutex);
  auto position = std::find(daemon->client_fds.begin(), daemon->client_fds.end(), client_fd);
  if (position != daemon->client_fds.end())
    daemon->client_fds.erase(position);
  ::close(client_fd);
}

void interrupt_transport(rj_daemon_t *daemon) {
  daemon->stop_requested.store(true, std::memory_order_release);
  if (daemon->listen_fd >= 0)
    ::shutdown(daemon->listen_fd, SHUT_RDWR);

  std::lock_guard lock(daemon->clients_mutex);
  for (int fd : daemon->client_fds)
    ::shutdown(fd, SHUT_RDWR);
}

void mark_error(rj_daemon_t *daemon) {
  auto state = daemon->status.load(std::memory_order_acquire);
  while (state != RJ_DAEMON_STATUS_STOPPING && state != RJ_DAEMON_STATUS_STOPPED &&
         state != RJ_DAEMON_STATUS_ERROR &&
         !daemon->status.compare_exchange_weak(state, RJ_DAEMON_STATUS_ERROR,
                                               std::memory_order_acq_rel)) {
  }
  if (state == RJ_DAEMON_STATUS_STOPPING || state == RJ_DAEMON_STATUS_STOPPED)
    return;
  interrupt_transport(daemon);
  rj_vm_request_exit(daemon->vm, "daemon transport failure");
}

enum class RequestDisposition { Continue, Disconnect };

class ClientSession {
public:
  ClientSession(int client_fd, rj_daemon_t *daemon)
      : client_fd_(client_fd), daemon_(daemon), client_pid_(peer_pid_for_socket(client_fd)) {}

  ~ClientSession() {
    try {
      close_device();
    } catch (...) {
    }
  }

  ClientSession(const ClientSession &) = delete;
  ClientSession &operator=(const ClientSession &) = delete;

  void run(std::stop_token stop) {
    try {
      while (!stop.stop_requested() && !daemon_->stop_requested.load(std::memory_order_acquire)) {
        RpcHeader header{};
        ReceivedFds input_fds;
        if (!receive_header_with_fds(client_fd_, &header, &input_fds))
          break;

        if (header.reserved != 0 ||
            handle_request(header, input_fds) == RequestDisposition::Disconnect) {
          break;
        }
      }
    } catch (...) {
    }
    close_device();
  }

private:
  RequestDisposition handle_request(const RpcHeader &header, ReceivedFds &input_fds) {
    switch (header.opcode) {
    case RPC_HANDSHAKE:
      return handle_handshake(header, input_fds);
    case RPC_CLOSE:
      return handle_close(header, input_fds);
    case RPC_MMAP:
      return handle_mmap(header, input_fds);
    case RPC_MUNMAP:
      return handle_munmap(header, input_fds);
    case RPC_IOCTL:
      return handle_ioctl(header, input_fds);
    default:
      return RequestDisposition::Disconnect;
    }
  }

  RequestDisposition handle_handshake(const RpcHeader &header, const ReceivedFds &input_fds) {
    if (process_id_ != 0 || header.payload_bytes != 0 || input_fds.notifier)
      return RequestDisposition::Disconnect;

    if (rj_vm_device_open(daemon_->vm, client_pid_, &process_id_) != ROCJITSU_STATUS_SUCCESS) {
      RpcHeader response{};
      response.opcode = header.opcode;
      response.request_id = header.request_id;
      response.result = -1;
      rpc_send_exact(client_fd_, &response, sizeof(response));
      return RequestDisposition::Disconnect;
    }

    RpcHandshakeResponse handshake{};
    handshake.version = kRpcProtocolVersion;
    if (rj_vm_gpu_id(daemon_->vm, &handshake.gpu_id) != ROCJITSU_STATUS_SUCCESS ||
        rj_vm_gpu_info(daemon_->vm, &handshake.gpu_info) != ROCJITSU_STATUS_SUCCESS) {
      return RequestDisposition::Disconnect;
    }

    const char *topology = nullptr;
    const char *drm = nullptr;
    if (rj_vm_topology_path(daemon_->vm, &topology) != ROCJITSU_STATUS_SUCCESS ||
        rj_vm_drm_path(daemon_->vm, &drm) != ROCJITSU_STATUS_SUCCESS) {
      return RequestDisposition::Disconnect;
    }
    const size_t topology_length = topology ? std::strlen(topology) : 0;
    const size_t drm_length = drm ? std::strlen(drm) : 0;
    constexpr size_t max_payload = std::numeric_limits<uint32_t>::max();
    if (topology_length > max_payload - sizeof(handshake) ||
        drm_length > max_payload - sizeof(handshake) - topology_length) {
      return RequestDisposition::Disconnect;
    }
    handshake.topology_path_len = static_cast<uint32_t>(topology_length);
    handshake.drm_path_len = static_cast<uint32_t>(drm_length);

    RpcHeader response{};
    response.opcode = header.opcode;
    response.request_id = header.request_id;
    response.payload_bytes =
        static_cast<uint32_t>(sizeof(handshake) + topology_length + drm_length);
    const bool sent =
        rpc_send_exact(client_fd_, &response, sizeof(response)) &&
        rpc_send_exact(client_fd_, &handshake, sizeof(handshake)) &&
        (topology_length == 0 || rpc_send_exact(client_fd_, topology, topology_length)) &&
        (drm_length == 0 || rpc_send_exact(client_fd_, drm, drm_length));
    return sent ? RequestDisposition::Continue : RequestDisposition::Disconnect;
  }

  RequestDisposition handle_close(const RpcHeader &header, const ReceivedFds &input_fds) {
    if (process_id_ == 0 || header.payload_bytes != 0 || input_fds.notifier)
      return RequestDisposition::Disconnect;

    close_device();
    RpcHeader response{};
    response.opcode = header.opcode;
    response.request_id = header.request_id;
    rpc_send_exact(client_fd_, &response, sizeof(response));
    return RequestDisposition::Disconnect;
  }

  RequestDisposition handle_mmap(const RpcHeader &header, const ReceivedFds &input_fds) {
    if (process_id_ == 0 || header.payload_bytes != sizeof(RpcMmapRequest) || input_fds.notifier) {
      return RequestDisposition::Disconnect;
    }
    RpcMmapRequest request{};
    if (!rpc_recv_exact(client_fd_, &request, sizeof(request)))
      return RequestDisposition::Disconnect;

    rj_vm_map_t map{};
    map.addr = request.addr;
    map.length = request.length;
    map.prot = static_cast<uint32_t>(request.prot);
    map.flags = static_cast<uint32_t>(request.flags);
    map.offset = request.offset;
    if (rj_vm_device_map_as(daemon_->vm, process_id_, &map) != ROCJITSU_STATUS_SUCCESS)
      return RequestDisposition::Disconnect;

    RpcHeader response{};
    response.opcode = header.opcode;
    response.request_id = header.request_id;
    response.result = reinterpret_cast<void *>(map.mapped_addr) == MAP_FAILED ? -map.map_errno : 0;
    response.payload_bytes = sizeof(RpcMmapResponse);
    RpcMmapResponse map_response{.mapped_addr = map.mapped_addr};
    uint8_t response_buffer[sizeof(response) + sizeof(map_response)];
    std::memcpy(response_buffer, &response, sizeof(response));
    std::memcpy(response_buffer + sizeof(response), &map_response, sizeof(map_response));

    rj_handle_t backing_memory = -1;
    if (rj_vm_get_shared_mem_as(daemon_->vm, process_id_, request.offset, &backing_memory) !=
        ROCJITSU_STATUS_SUCCESS) {
      return RequestDisposition::Disconnect;
    }
    const bool sent = backing_memory >= 0
                          ? send_with_fd_exact(client_fd_, response_buffer, sizeof(response_buffer),
                                               backing_memory)
                          : rpc_send_exact(client_fd_, response_buffer, sizeof(response_buffer));
    return sent ? RequestDisposition::Continue : RequestDisposition::Disconnect;
  }

  RequestDisposition handle_munmap(const RpcHeader &header, const ReceivedFds &input_fds) {
    if (process_id_ == 0 || header.payload_bytes != sizeof(RpcMunmapRequest) ||
        input_fds.notifier) {
      return RequestDisposition::Disconnect;
    }
    RpcMunmapRequest request{};
    if (!rpc_recv_exact(client_fd_, &request, sizeof(request)))
      return RequestDisposition::Disconnect;
    rj_vm_unmap_t unmap{.addr = request.addr, .length = request.length};
    if (rj_vm_device_unmap_as(daemon_->vm, process_id_, &unmap) != ROCJITSU_STATUS_SUCCESS)
      return RequestDisposition::Disconnect;
    RpcHeader response{};
    response.opcode = header.opcode;
    response.request_id = header.request_id;
    return rpc_send_exact(client_fd_, &response, sizeof(response)) ? RequestDisposition::Continue
                                                                   : RequestDisposition::Disconnect;
  }

  RequestDisposition handle_ioctl(const RpcHeader &header, ReceivedFds &input_fds) {
    if (process_id_ == 0 || header.payload_bytes > kMaxPayloadBytes ||
        header.payload_bytes < sizeof(RpcIoctlRequest)) {
      return RequestDisposition::Disconnect;
    }
    std::vector<uint8_t> payload(header.payload_bytes);
    if (!rpc_recv_exact(client_fd_, payload.data(), payload.size()))
      return RequestDisposition::Disconnect;
    auto *request = reinterpret_cast<RpcIoctlRequest *>(payload.data());
    const size_t available_arguments = payload.size() - sizeof(RpcIoctlRequest);
    void *arguments = payload.data() + sizeof(RpcIoctlRequest);
    if (request->args_bytes != available_arguments ||
        !validate_ioctl_payload(request->ioctl_cmd, arguments, request->args_bytes)) {
      return RequestDisposition::Disconnect;
    }

    rj_vm_cmd_t command{};
    command.cmd = request->ioctl_cmd;
    command.buf = arguments;
    command.buf_size = request->args_bytes;
    command.shared_handle = -1;
    command.in_handle = input_fds.notifier.get();
    command.in_mem_handle = input_fds.target_mem.get();
    // Borrowed, not adopted: the driver dups it if the session needs to keep it
    // (rj_vm.cpp passes it by value), so this side always closes its own copy.
    command.in_proc_handle = input_fds.target_proc.get();
    if (rj_vm_execute_as(daemon_->vm, process_id_, &command) != ROCJITSU_STATUS_SUCCESS)
      return RequestDisposition::Disconnect;
    // The VM clears each handle it took ownership of; drop ours to match, so
    // RAII does not close a descriptor the debug session now owns.
    if (command.in_handle < 0)
      static_cast<void>(input_fds.notifier.release());
    if (command.in_mem_handle < 0)
      static_cast<void>(input_fds.target_mem.release());
    if (command.buf_size > available_arguments)
      return RequestDisposition::Disconnect;

    RpcHeader response{};
    response.opcode = RPC_IOCTL;
    response.request_id = header.request_id;
    response.result = command.result;
    response.payload_bytes = static_cast<uint32_t>(command.buf_size);

    bool sent = false;
    if (command.shared_handle >= 0) {
      std::vector<uint8_t> response_buffer(sizeof(response) + command.buf_size);
      std::memcpy(response_buffer.data(), &response, sizeof(response));
      if (command.buf_size > 0) {
        std::memcpy(response_buffer.data() + sizeof(response), command.buf, command.buf_size);
      }
      sent = send_with_fd_exact(client_fd_, response_buffer.data(), response_buffer.size(),
                                command.shared_handle);
    } else {
      sent = rpc_send_exact(client_fd_, &response, sizeof(response)) &&
             (command.buf_size == 0 || rpc_send_exact(client_fd_, command.buf, command.buf_size));
    }
    return sent ? RequestDisposition::Continue : RequestDisposition::Disconnect;
  }

  void close_device() {
    if (process_id_ == 0)
      return;
    rj_vm_device_close(daemon_->vm, std::exchange(process_id_, 0));
  }

  const int client_fd_;
  rj_daemon_t *const daemon_;
  const rj_client_pid_t client_pid_;
  uint32_t process_id_ = 0;
};

void handle_client(int client_fd, rj_daemon_t *daemon, std::stop_token stop) {
  ClientSession session(client_fd, daemon);
  session.run(stop);
}

void accept_clients(rj_daemon_t *daemon, std::stop_token stop) {
  struct ClientThread {
    std::jthread thread;
    std::shared_ptr<std::atomic<bool>> complete;
  };
  std::vector<ClientThread> client_threads;
  try {
    while (!stop.stop_requested() && !daemon->stop_requested.load(std::memory_order_acquire)) {
      std::erase_if(client_threads, [](const ClientThread &client) {
        return client.complete->load(std::memory_order_acquire);
      });

      const int client_fd = accept4(daemon->listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
      if (client_fd < 0) {
        if (errno == EINTR)
          continue;
        break;
      }

      {
        std::lock_guard lock(daemon->clients_mutex);
        daemon->client_fds.push_back(client_fd);
      }
      try {
        client_threads.reserve(client_threads.size() + 1);
        auto complete = std::make_shared<std::atomic<bool>>(false);
        std::jthread thread([daemon, client_fd, complete](std::stop_token client_stop) {
          try {
            handle_client(client_fd, daemon, client_stop);
          } catch (...) {
          }
          close_client(daemon, client_fd);
          complete->store(true, std::memory_order_release);
        });
        client_threads.push_back({std::move(thread), std::move(complete)});
      } catch (...) {
        close_client(daemon, client_fd);
        throw;
      }
    }
  } catch (...) {
    mark_error(daemon);
  }

  if (!daemon->stop_requested.load(std::memory_order_acquire))
    mark_error(daemon);

  for (auto &client : client_threads)
    client.thread.request_stop();
  {
    std::lock_guard lock(daemon->clients_mutex);
    for (int fd : daemon->client_fds)
      ::shutdown(fd, SHUT_RDWR);
  }
  client_threads.clear();
}

void remove_owned_socket(const rj_daemon_t *daemon) {
  struct stat current {};
  if (::lstat(daemon->socket_path.c_str(), &current) == 0 &&
      current.st_dev == daemon->socket_device && current.st_ino == daemon->socket_inode) {
    ::unlink(daemon->socket_path.c_str());
  }
}

void teardown(rj_daemon_t *daemon) {
  daemon->status.store(RJ_DAEMON_STATUS_STOPPING, std::memory_order_release);
  interrupt_transport(daemon);
  // Wake clients blocked inside an infinite-timeout device operation before
  // joining their threads. Transport shutdown alone only interrupts socket I/O.
  rj_vm_close_all_devices(daemon->vm);
  daemon->accept_thread.request_stop();
  if (daemon->accept_thread.joinable())
    daemon->accept_thread.join();

  if (daemon->listen_fd >= 0) {
    ::close(daemon->listen_fd);
    daemon->listen_fd = -1;
  }
  remove_owned_socket(daemon);

  rj_vm_request_exit(daemon->vm, "daemon shutdown");
  daemon->engine_thread.request_stop();
  if (daemon->engine_thread.joinable())
    daemon->engine_thread.join();
  rj_vm_destroy(daemon->vm);
  daemon->vm = nullptr;
  daemon->status.store(RJ_DAEMON_STATUS_STOPPED, std::memory_order_release);
}

bool remove_stale_socket(const rj_daemon_t *daemon, const sockaddr_un &address,
                         socklen_t address_length) {
  struct stat before {};
  if (::lstat(daemon->socket_path.c_str(), &before) != 0 || !S_ISSOCK(before.st_mode))
    return false;

  util::UniqueHandle probe(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
  if (probe.get() < 0)
    return false;
  if (::connect(probe.get(), reinterpret_cast<const sockaddr *>(&address), address_length) == 0)
    return false;
  if (errno != ECONNREFUSED && errno != ENOENT)
    return false;

  struct stat current {};
  if (::lstat(daemon->socket_path.c_str(), &current) != 0)
    return errno == ENOENT;
  if (!S_ISSOCK(current.st_mode) || current.st_dev != before.st_dev ||
      current.st_ino != before.st_ino) {
    return false;
  }
  return ::unlink(daemon->socket_path.c_str()) == 0 || errno == ENOENT;
}

rj_status_t bind_socket(rj_daemon_t *daemon) {
  if (daemon->socket_path.empty())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  sockaddr_un address{};
  const auto path_length = daemon->socket_path.size();
  if (path_length >= sizeof(address.sun_path))
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  std::filesystem::path path(daemon->socket_path);
  if (const auto parent = path.parent_path(); !parent.empty()) {
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    if (error)
      return ROCJITSU_STATUS_ERROR;
  }

  util::UniqueHandle socket_fd(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
  if (socket_fd.get() < 0)
    return ROCJITSU_STATUS_ERROR;

  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, daemon->socket_path.data(), path_length);
  const auto address_length =
      static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + path_length + 1);
  if (::bind(socket_fd.get(), reinterpret_cast<sockaddr *>(&address), address_length) != 0) {
    if (errno != EADDRINUSE || !remove_stale_socket(daemon, address, address_length) ||
        ::bind(socket_fd.get(), reinterpret_cast<sockaddr *>(&address), address_length) != 0) {
      return ROCJITSU_STATUS_ERROR;
    }
  }

  struct stat socket_stat {};
  if (::lstat(daemon->socket_path.c_str(), &socket_stat) != 0 || !S_ISSOCK(socket_stat.st_mode)) {
    return ROCJITSU_STATUS_ERROR;
  }
  daemon->socket_device = socket_stat.st_dev;
  daemon->socket_inode = socket_stat.st_ino;

  if (::listen(socket_fd.get(), 16) != 0)
    return ROCJITSU_STATUS_ERROR;
  daemon->listen_fd = socket_fd.release();
  return ROCJITSU_STATUS_SUCCESS;
}

} // namespace

rj_status_t rj_daemon_start(const char *json, const char *socket_path, rj_daemon_t **daemon) {
  if (daemon)
    *daemon = nullptr;
  if (!json || !socket_path || !daemon)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  const size_t socket_path_length = std::strlen(socket_path);
  if (socket_path_length == 0 || socket_path_length >= sizeof(sockaddr_un::sun_path))
    return ROCJITSU_STATUS_INVALID_ARGUMENT;

  std::unique_ptr<rj_daemon_t> created;
  try {
    created = std::make_unique<rj_daemon_t>();
    created->socket_path = socket_path;
  } catch (const std::bad_alloc &) {
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  }

  rj_status_t status = rj_vm_create_from_string(json, RJ_VM_MODE_DAEMON, &created->vm);
  if (status != ROCJITSU_STATUS_SUCCESS)
    return status;

  // Daemon callers supply the same configuration as local-mode callers, so
  // install its plugins before any client can submit work. With no explicit
  // directory, the loader discovers modules relative to librocjitsu or the
  // statically linked rocjitsu CLI.
  status = rj_vm_load_plugins(created->vm, json, nullptr);
  if (status != ROCJITSU_STATUS_SUCCESS) {
    rj_vm_destroy(created->vm);
    created->vm = nullptr;
    return status;
  }

  try {
    status = bind_socket(created.get());
  } catch (const std::bad_alloc &) {
    status = ROCJITSU_STATUS_OUT_OF_RESOURCES;
  } catch (...) {
    status = ROCJITSU_STATUS_ERROR;
  }
  if (status != ROCJITSU_STATUS_SUCCESS) {
    rj_vm_destroy(created->vm);
    created->vm = nullptr;
    remove_owned_socket(created.get());
    return status;
  }

  try {
    created->engine_thread = std::jthread([state = created.get()] {
      const rj_status_t run_status = rj_vm_run(state->vm, nullptr);
      if (run_status != ROCJITSU_STATUS_SUCCESS ||
          !state->stop_requested.load(std::memory_order_acquire)) {
        mark_error(state);
      }
    });
    created->accept_thread = std::jthread(
        [state = created.get()](std::stop_token stop) { accept_clients(state, stop); });
  } catch (const std::system_error &) {
    teardown(created.get());
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  } catch (const std::bad_alloc &) {
    teardown(created.get());
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  }

  rj_daemon_status_t expected = RJ_DAEMON_STATUS_STARTING;
  if (!created->status.compare_exchange_strong(expected, RJ_DAEMON_STATUS_RUNNING,
                                               std::memory_order_acq_rel)) {
    teardown(created.get());
    return ROCJITSU_STATUS_ERROR;
  }

  *daemon = created.release();
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_daemon_stop(rj_daemon_t *daemon) {
  if (!daemon)
    return ROCJITSU_STATUS_SUCCESS;
  teardown(daemon);
  delete daemon;
  return ROCJITSU_STATUS_SUCCESS;
}

rj_daemon_status_t rj_daemon_status(const rj_daemon_t *daemon) {
  return daemon ? daemon->status.load(std::memory_order_acquire) : RJ_DAEMON_STATUS_STOPPED;
}
