// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/linux_kfd.h"
#include "rocjitsu/kmd/linux/remote_driver.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/rocjitsu.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <poll.h>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using namespace rocjitsu;

class TempDirectory {
public:
  TempDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "rj-daemon-test-XXXXXX").string();
    pattern.push_back('\0');
    char *created = mkdtemp(pattern.data());
    if (created)
      path_ = created;
  }

  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  [[nodiscard]] const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

const std::string &daemon_json() {
  static const std::string config = [] {
    std::ifstream input(std::filesystem::path(CONFIG_DIR) / "gfx950_mi355x_kmd.json");
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  }();
  return config;
}

class TestDaemon {
public:
  explicit TestDaemon(const std::filesystem::path &socket_path)
      : TestDaemon(daemon_json().c_str(), socket_path.c_str()) {}

  TestDaemon(const char *json, const char *socket_path)
      : start_status_(rj_daemon_start(json, socket_path, &daemon_)) {}

  TestDaemon(const char *json, const char *socket_path, const rj_daemon_options_t &options)
      : start_status_(rj_daemon_start_with_options(json, socket_path, &options, &daemon_)) {}

  ~TestDaemon() { (void)stop(); }

  TestDaemon(const TestDaemon &) = delete;
  TestDaemon &operator=(const TestDaemon &) = delete;
  TestDaemon(TestDaemon &&) = delete;
  TestDaemon &operator=(TestDaemon &&) = delete;

  [[nodiscard]] rj_status_t start_status() const { return start_status_; }
  [[nodiscard]] rj_daemon_t *get() const { return daemon_; }
  [[nodiscard]] rj_daemon_status_t status() const { return rj_daemon_status(daemon_); }

  rj_status_t stop() {
    const rj_status_t status = rj_daemon_stop(daemon_);
    if (status == ROCJITSU_STATUS_SUCCESS)
      daemon_ = nullptr;
    return status;
  }

private:
  rj_daemon_t *daemon_ = nullptr;
  rj_status_t start_status_ = ROCJITSU_STATUS_ERROR;
};

int connect_to(const std::filesystem::path &socket_path) {
  const std::string path = socket_path.string();
  sockaddr_un address{};
  if (path.size() >= sizeof(address.sun_path))
    return -1;
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);

  const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0)
    return -1;
  const auto length = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + path.size() + 1);
  if (connect(fd, reinterpret_cast<const sockaddr *>(&address), length) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

RpcHandshakeResponse handshake(int fd, uint32_t request_id = 1) {
  RpcHeader request{};
  request.opcode = RPC_HANDSHAKE;
  request.request_id = request_id;
  EXPECT_TRUE(rpc_send_exact(fd, &request, sizeof(request)));

  RpcHeader response{};
  EXPECT_TRUE(rpc_recv_exact(fd, &response, sizeof(response)));
  EXPECT_EQ(response.request_id, request_id);
  EXPECT_EQ(response.result, 0);
  EXPECT_GE(response.payload_bytes, sizeof(RpcHandshakeResponse));

  RpcHandshakeResponse result{};
  EXPECT_TRUE(rpc_recv_exact(fd, &result, sizeof(result)));
  const size_t paths_size = response.payload_bytes - sizeof(result);
  std::string paths(paths_size, '\0');
  if (paths_size > 0) {
    EXPECT_TRUE(rpc_recv_exact(fd, paths.data(), paths.size()));
  }
  return result;
}

void close_session(int fd, uint32_t request_id = 2) {
  RpcHeader request{};
  request.opcode = RPC_CLOSE;
  request.request_id = request_id;
  ASSERT_TRUE(rpc_send_exact(fd, &request, sizeof(request)));
  RpcHeader response{};
  ASSERT_TRUE(rpc_recv_exact(fd, &response, sizeof(response)));
  EXPECT_EQ(response.opcode, RPC_CLOSE);
  EXPECT_EQ(response.request_id, request_id);
  EXPECT_EQ(response.result, 0);
  close(fd);
}

bool send_ioctl_request(int fd, uint32_t request_id, uint32_t command, const void *arguments,
                        size_t argument_size, const void *inline_data = nullptr,
                        size_t inline_size = 0) {
  constexpr size_t max_arguments = UINT32_MAX - sizeof(RpcIoctlRequest);
  if (argument_size > max_arguments || inline_size > max_arguments - argument_size)
    return false;

  const auto args_bytes = static_cast<uint32_t>(argument_size + inline_size);
  RpcHeader header{};
  header.opcode = RPC_IOCTL;
  header.request_id = request_id;
  header.payload_bytes = static_cast<uint32_t>(sizeof(RpcIoctlRequest) + args_bytes);
  RpcIoctlRequest request{.ioctl_cmd = command, .args_bytes = args_bytes};
  return rpc_send_exact(fd, &header, sizeof(header)) &&
         rpc_send_exact(fd, &request, sizeof(request)) &&
         (argument_size == 0 || rpc_send_exact(fd, arguments, argument_size)) &&
         (inline_size == 0 || rpc_send_exact(fd, inline_data, inline_size));
}

void expect_wait_events_payload_rejected(TestDaemon &daemon,
                                         const std::filesystem::path &socket_path,
                                         uint32_t num_events, size_t inline_size) {
  const int client = connect_to(socket_path);
  ASSERT_GE(client, 0);
  ASSERT_EQ(handshake(client).version, kRpcProtocolVersion);

  kfd_ioctl_wait_events_args arguments{};
  arguments.num_events = num_events;
  arguments.timeout = 0;
  std::vector<uint8_t> inline_events(inline_size);
  ASSERT_TRUE(send_ioctl_request(client, 2, AMDKFD_IOC_WAIT_EVENTS, &arguments, sizeof(arguments),
                                 inline_events.data(), inline_events.size()));

  RpcHeader response{};
  EXPECT_FALSE(rpc_recv_exact(client, &response, sizeof(response)));
  EXPECT_EQ(daemon.status(), RJ_DAEMON_STATUS_RUNNING);
  close(client);
}

// Keep the parent and each client on separate control sockets. Every transfer
// has a deadline so a broken RPC path cannot strand the test in a socket read.
class ChildControlSocket {
public:
  ChildControlSocket() { valid_ = socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds_) == 0; }
  ~ChildControlSocket() {
    close_parent();
    close_child();
  }

  [[nodiscard]] bool valid() const { return valid_; }
  [[nodiscard]] int parent() const { return fds_[0]; }
  [[nodiscard]] int child() const { return fds_[1]; }
  void close_parent() { close_end(fds_[0]); }
  void close_child() { close_end(fds_[1]); }

private:
  static void close_end(int &fd) {
    if (fd >= 0) {
      close(fd);
      fd = -1;
    }
  }

  int fds_[2] = {-1, -1};
  bool valid_ = false;
};

bool control_transfer(int fd, void *data, size_t size, bool sending) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  auto *bytes = static_cast<uint8_t *>(data);
  size_t offset = 0;
  while (offset < size) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (remaining.count() <= 0)
      return false;
    pollfd ready{fd, static_cast<short>(sending ? POLLOUT : POLLIN), 0};
    const int polled = poll(&ready, 1, static_cast<int>(remaining.count()));
    if (polled < 0 && errno == EINTR)
      continue;
    if (polled <= 0 || (ready.revents & (POLLERR | POLLNVAL)))
      return false;
    const ssize_t moved = sending ? send(fd, bytes + offset, size - offset, MSG_NOSIGNAL)
                                  : recv(fd, bytes + offset, size - offset, 0);
    if (moved < 0 && errno == EINTR)
      continue;
    if (moved <= 0)
      return false;
    offset += static_cast<size_t>(moved);
  }
  return true;
}

template <typename T> bool send_control(int fd, const T &value) {
  T copy = value;
  return control_transfer(fd, &copy, sizeof(copy), true);
}

template <typename T> bool receive_control(int fd, T &value) {
  return control_transfer(fd, &value, sizeof(value), false);
}

struct PeerRpcReport {
  uint32_t stage = 0;
  int result = 0;
  uint32_t n_success = 0;
  pid_t pid = 0;
  uint64_t va = 0;
  uint32_t share_handle[4]{};
};

constexpr uint32_t kPeerOwnerGpu = 505;
constexpr uint32_t kPeerRequesterGpu = 909;
constexpr size_t kPeerPageBytes = 4096;
constexpr size_t kPeerPayloadOffset = 32;
constexpr size_t kPeerReturnOffset = 128;
constexpr std::array<uint8_t, 16> kOwnerSeed{0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0x17, 0x28,
                                             0x39, 0x4a, 0x5b, 0x6c, 0x7d, 0x8e, 0x9f, 0x10};
constexpr std::array<uint8_t, 16> kPeerPayload{0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
                                               0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xf0, 0x0f};

int run_peer_owner(const std::filesystem::path &socket, int control) {
  PeerRpcReport report{};
  report.pid = getpid();
  auto fail = [&](uint32_t stage, int result, uint32_t n_success = 0) {
    report.stage = stage;
    report.result = result;
    report.n_success = n_success;
    (void)send_control(control, report);
    return 1;
  };

  char start = 0;
  if (!receive_control(control, start) || start != 'S')
    return fail(1, -EIO);
  const int connection = connect_to(socket);
  if (connection < 0)
    return fail(2, -errno);
  RemoteDriver client(connection);
  const int kfd_fd = client.open();
  if (kfd_fd < 0)
    return fail(3, -EIO);

  kfd_ioctl_alloc_memory_of_gpu_args allocation{};
  allocation.size = kPeerPageBytes;
  allocation.gpu_id = kPeerOwnerGpu;
  allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  int result = client.ioctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocation);
  if (result != 0)
    return fail(4, result);
  uint32_t owner_id = kPeerOwnerGpu;
  kfd_ioctl_map_memory_to_gpu_args map{};
  map.handle = allocation.handle;
  map.device_ids_array_ptr = reinterpret_cast<uint64_t>(&owner_id);
  map.n_devices = 1;
  result = client.ioctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map);
  if (result != 0 || map.n_success != 1)
    return fail(5, result, map.n_success);
  auto *owner_bytes = static_cast<uint8_t *>(client.mmap(
      nullptr, kPeerPageBytes, PROT_READ | PROT_WRITE, MAP_SHARED, allocation.mmap_offset));
  if (owner_bytes == MAP_FAILED)
    return fail(10, -errno);
  std::memcpy(owner_bytes + kPeerPayloadOffset, kOwnerSeed.data(), kOwnerSeed.size());
  kfd_ioctl_ipc_export_handle_args exported{};
  exported.handle = allocation.handle;
  exported.gpu_id = kPeerOwnerGpu;
  result = client.ioctl(AMDKFD_IOC_IPC_EXPORT_HANDLE, &exported);
  if (result != 0)
    return fail(6, result);

  report.va = allocation.va_addr;
  report.n_success = map.n_success;
  std::memcpy(report.share_handle, exported.share_handle, sizeof(report.share_handle));
  if (!send_control(control, report))
    return 1;

  char release = 0;
  if (!receive_control(control, release) || release != 'R')
    return fail(7, -EIO);
  if (std::memcmp(owner_bytes + kPeerPayloadOffset, kPeerPayload.data(), kPeerPayload.size()) != 0)
    return fail(11, -EIO);
  kfd_ioctl_free_memory_of_gpu_args free{};
  free.handle = allocation.handle;
  result = client.ioctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free);
  if (result != 0)
    return fail(8, result);
  if (::munmap(owner_bytes, kPeerPageBytes) != 0)
    return fail(12, -errno);
  result = client.close();
  close(kfd_fd);
  if (result != 0)
    return fail(9, result);
  report.stage = 0;
  return send_control(control, report) ? 0 : 1;
}

int run_peer_importer(const std::filesystem::path &socket, int control) {
  PeerRpcReport report{};
  report.pid = getpid();
  auto fail = [&](uint32_t stage, int result, uint32_t n_success = 0) {
    report.stage = stage;
    report.result = result;
    report.n_success = n_success;
    (void)send_control(control, report);
    return 1;
  };

  PeerRpcReport owner{};
  if (!receive_control(control, owner) || owner.stage != 0)
    return fail(1, -EIO);
  const int connection = connect_to(socket);
  if (connection < 0)
    return fail(2, -errno);
  RemoteDriver client(connection);
  const int kfd_fd = client.open();
  if (kfd_fd < 0)
    return fail(3, -EIO);

  kfd_ioctl_ipc_import_handle_args imported{};
  std::memcpy(imported.share_handle, owner.share_handle, sizeof(imported.share_handle));
  imported.gpu_id = kPeerOwnerGpu;
  imported.va_addr = owner.va + 0x20000;
  int result = client.ioctl(AMDKFD_IOC_IPC_IMPORT_HANDLE, &imported);
  if (result != 0)
    return fail(4, result);
  auto *imported_bytes = static_cast<uint8_t *>(client.mmap(
      nullptr, kPeerPageBytes, PROT_READ | PROT_WRITE, MAP_SHARED, imported.mmap_offset));
  if (imported_bytes == MAP_FAILED)
    return fail(13, -errno);
  if (std::memcmp(imported_bytes + kPeerPayloadOffset, kOwnerSeed.data(), kOwnerSeed.size()) != 0)
    return fail(14, -EIO);
  std::array<uint32_t, 2> ids{kPeerOwnerGpu, kPeerRequesterGpu};
  kfd_ioctl_map_memory_to_gpu_args map{};
  map.handle = imported.handle;
  map.device_ids_array_ptr = reinterpret_cast<uint64_t>(ids.data());
  map.n_devices = 1;
  result = client.ioctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map);
  if (result != 0 || map.n_success != 1)
    return fail(5, result, map.n_success);
  map.n_devices = ids.size();
  result = client.ioctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map);
  if (result != 0 || map.n_success != ids.size())
    return fail(6, result, map.n_success);

  struct GttBuffer {
    uint64_t gpu_va = 0;
    uint64_t handle = 0;
    uint8_t *host = nullptr;
  } ring, pointers, local;
  auto prepare_gtt = [&](GttBuffer &buffer, uint32_t stage) {
    kfd_ioctl_alloc_memory_of_gpu_args allocation{};
    allocation.size = kPeerPageBytes;
    allocation.gpu_id = kPeerRequesterGpu;
    allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
    int rc = client.ioctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocation);
    if (rc != 0) {
      (void)fail(stage, rc);
      return false;
    }
    uint32_t gpu_id = kPeerRequesterGpu;
    kfd_ioctl_map_memory_to_gpu_args gtt_map{};
    gtt_map.handle = allocation.handle;
    gtt_map.n_devices = 1;
    gtt_map.device_ids_array_ptr = reinterpret_cast<uint64_t>(&gpu_id);
    rc = client.ioctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU, &gtt_map);
    if (rc != 0 || gtt_map.n_success != 1) {
      (void)fail(stage + 1, rc, gtt_map.n_success);
      return false;
    }
    void *host = client.mmap(nullptr, kPeerPageBytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                             allocation.mmap_offset);
    if (host == MAP_FAILED) {
      (void)fail(stage + 2, -errno);
      return false;
    }
    buffer = {allocation.va_addr, allocation.handle, static_cast<uint8_t *>(host)};
    return true;
  };
  if (!prepare_gtt(ring, 20) || !prepare_gtt(pointers, 23) || !prepare_gtt(local, 26))
    return 1;
  std::memcpy(local.host, kPeerPayload.data(), kPeerPayload.size());

  const off_t doorbell_offset =
      static_cast<off_t>(KFD_MMAP_TYPE_DOORBELL | kfd_mmap_gpu_id(kPeerRequesterGpu));
  void *doorbell_page =
      client.mmap(nullptr, kPeerPageBytes, PROT_READ | PROT_WRITE, MAP_SHARED, doorbell_offset);
  if (doorbell_page == MAP_FAILED)
    return fail(29, -errno);
  kfd_ioctl_create_queue_args queue{};
  queue.gpu_id = kPeerRequesterGpu;
  queue.queue_type = KFD_IOC_QUEUE_TYPE_SDMA;
  queue.ring_base_address = ring.gpu_va;
  queue.ring_size = kPeerPageBytes;
  queue.read_pointer_address = pointers.gpu_va;
  queue.write_pointer_address = pointers.gpu_va + sizeof(uint64_t);
  queue.queue_percentage = 100;
  result = client.ioctl(AMDKFD_IOC_CREATE_QUEUE, &queue);
  if (result != 0)
    return fail(30, result);
  auto *ring_words = reinterpret_cast<uint32_t *>(ring.host);
  auto *queue_pointers = reinterpret_cast<uint64_t *>(pointers.host);
  auto *doorbell = reinterpret_cast<uint64_t *>(static_cast<uint8_t *>(doorbell_page) +
                                                queue.doorbell_offset % kPeerPageBytes);

  auto submit_copy = [&](uint64_t source, uint64_t destination, size_t word_offset) {
    uint32_t *packet = ring_words + word_offset;
    packet[0] = 1; // SDMA_OP_COPY, linear form.
    packet[1] = kPeerPayload.size() - 1;
    packet[2] = 0;
    packet[3] = static_cast<uint32_t>(source);
    packet[4] = static_cast<uint32_t>(source >> 32);
    packet[5] = static_cast<uint32_t>(destination);
    packet[6] = static_cast<uint32_t>(destination >> 32);
    const uint64_t producer = (word_offset + 7) * sizeof(uint32_t);
    std::atomic_ref<uint64_t>(queue_pointers[1]).store(producer, std::memory_order_release);
    std::atomic_ref<uint64_t>(*doorbell).store(producer, std::memory_order_release);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      if (std::atomic_ref<uint64_t>(queue_pointers[0]).load(std::memory_order_acquire) == producer)
        return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
  };

  if (!submit_copy(imported.va_addr + kPeerPayloadOffset, local.gpu_va + kPeerReturnOffset, 0))
    return fail(31, -ETIMEDOUT);
  if (std::memcmp(local.host + kPeerReturnOffset, kOwnerSeed.data(), kOwnerSeed.size()) != 0)
    return fail(32, -EIO);
  if (!submit_copy(local.gpu_va, imported.va_addr + kPeerPayloadOffset, 7))
    return fail(33, -ETIMEDOUT);
  if (std::memcmp(imported_bytes + kPeerPayloadOffset, kPeerPayload.data(), kPeerPayload.size()) !=
      0)
    return fail(34, -EIO);

  kfd_ioctl_destroy_queue_args destroy{};
  destroy.queue_id = queue.queue_id;
  result = client.ioctl(AMDKFD_IOC_DESTROY_QUEUE, &destroy);
  if (result != 0)
    return fail(35, result);

  kfd_ioctl_unmap_memory_from_gpu_args unmap{};
  unmap.handle = imported.handle;
  unmap.device_ids_array_ptr = map.device_ids_array_ptr;
  unmap.n_devices = 1;
  result = client.ioctl(AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmap);
  if (result != 0 || unmap.n_success != 1)
    return fail(7, result, unmap.n_success);
  unmap.n_devices = ids.size();
  result = client.ioctl(AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmap);
  if (result != 0 || unmap.n_success != ids.size())
    return fail(8, result, unmap.n_success);
  kfd_ioctl_free_memory_of_gpu_args free{};
  free.handle = imported.handle;
  result = client.ioctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free);
  if (result != 0)
    return fail(9, result);
  for (const GttBuffer &buffer : {ring, pointers, local}) {
    free.handle = buffer.handle;
    result = client.ioctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free);
    if (result != 0)
      return fail(36, result);
    if (::munmap(buffer.host, kPeerPageBytes) != 0)
      return fail(37, -errno);
  }
  if (::munmap(imported_bytes, kPeerPageBytes) != 0 || ::munmap(doorbell_page, kPeerPageBytes) != 0)
    return fail(38, -errno);
  result = client.close();
  close(kfd_fd);
  if (result != 0)
    return fail(10, result);

  report.va = imported.va_addr;
  report.n_success = unmap.n_success;
  return send_control(control, report) ? 0 : 1;
}

class PeerChildProcesses {
public:
  ~PeerChildProcesses() { stop(); }

  pid_t owner = -1;
  pid_t importer = -1;

  void stop() {
    for (pid_t *child : {&owner, &importer}) {
      if (*child <= 0)
        continue;
      int status = 0;
      pid_t waited;
      do {
        waited = waitpid(*child, &status, WNOHANG);
      } while (waited < 0 && errno == EINTR);
      if (waited == 0) {
        (void)kill(*child, SIGKILL);
        while (waitpid(*child, &status, 0) < 0 && errno == EINTR) {
        }
      }
      *child = -1;
    }
  }
};

bool wait_for_peer_child(pid_t &child, int &status) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t waited = waitpid(child, &status, WNOHANG);
    if (waited == child) {
      child = -1;
      return true;
    }
    if (waited < 0 && errno != EINTR)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

TEST(DaemonApi, RejectsInvalidArguments) {
  TempDirectory directory;
  const std::string socket = (directory.path() / "daemon.sock").string();
  rj_daemon_t *daemon = reinterpret_cast<rj_daemon_t *>(1);

  EXPECT_EQ(rj_daemon_start(nullptr, socket.c_str(), &daemon), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(daemon, nullptr);
  EXPECT_EQ(rj_daemon_start(daemon_json().c_str(), nullptr, &daemon),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(daemon, nullptr);
  EXPECT_EQ(rj_daemon_start(daemon_json().c_str(), socket.c_str(), nullptr),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_daemon_start(daemon_json().c_str(), "", &daemon), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(daemon, nullptr);
  rj_daemon_options_t options{
      .struct_size = sizeof(rj_daemon_options_t), .flags = 0, .fabric_trace_capacity = 0};
  options.struct_size = 0;
  EXPECT_EQ(rj_daemon_start_with_options(daemon_json().c_str(), socket.c_str(), &options, &daemon),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(daemon, nullptr);
  options.struct_size = sizeof(options);
  options.flags = 1u << 1;
  EXPECT_EQ(rj_daemon_start_with_options(daemon_json().c_str(), socket.c_str(), &options, &daemon),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(daemon, nullptr);
  EXPECT_EQ(rj_daemon_status(nullptr), RJ_DAEMON_STATUS_STOPPED);
  EXPECT_EQ(rj_daemon_stop(nullptr), ROCJITSU_STATUS_SUCCESS);
}

TEST(DaemonApi, HandshakePublishesEachSelectedRenderMinor) {
  constexpr const char *json = R"({
    "vm":{"arch":"cdna5","gpu":{"device":{
      "gpu_id":1250,"gfx_target_version":120500,"unique_id":1250,
      "num_sdma_engines":0}}},
    "topology":{"root":{"name":"soc","type":"soc","children":[
      {"name":"vram","type":"gpu_memory"},
      {"name":"xcd0","type":"xcd","children":[
        {"name":"l2","type":"l2_cache"},
        {"name":"cp","type":"command_processor"},
        {"name":"se0","type":"shader_engine","children":[
          {"name":"cu[0:4]","type":"compute_unit"}]}]}]}},
    "system":{"nodes":[{"id":"tray7","gpus":[
      {"id":"g5","gpu_id":505,"unique_id":3005,
       "location_id":1024,"drm_render_minor":129},
      {"id":"g9","gpu_id":909,"unique_id":3009,
       "location_id":1280,"drm_render_minor":130}]}]},
    "local_node":"tray7"
  })";
  TempDirectory directory;
  const std::filesystem::path socket = directory.path() / "daemon.sock";
  TestDaemon daemon(json, socket.c_str());
  ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);

  const int connection = connect_to(socket);
  ASSERT_GE(connection, 0);
  RemoteDriver client(connection);
  const int kfd_fd = client.open();
  ASSERT_GE(kfd_fd, 0);
  EXPECT_FALSE(client.handles_drm_render_minor(128));
  EXPECT_TRUE(client.handles_drm_render_minor(129));
  EXPECT_TRUE(client.handles_drm_render_minor(130));
  const auto *first = client.gpu_info_for_render_minor(129);
  const auto *second = client.gpu_info_for_render_minor(130);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(first->gpu_id, 505u);
  EXPECT_EQ(second->gpu_id, 909u);
  EXPECT_EQ(client.gpu_info_for_render_minor(128), nullptr);
  EXPECT_EQ(client.close(), 0);
  close(kfd_fd);
}

TEST(DaemonApi, LegacyOneGpuHandshakeKeepsRenderMinorCompatibility) {
  std::string json = daemon_json();
  const std::string configured = "\"drm_render_minor\": 128";
  const auto position = json.find(configured);
  ASSERT_NE(position, std::string::npos);
  json.replace(position, configured.size(), "\"drm_render_minor\": 129");

  TempDirectory directory;
  const std::filesystem::path socket = directory.path() / "daemon.sock";
  TestDaemon daemon(json.c_str(), socket.c_str());
  ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);
  const int connection = connect_to(socket);
  ASSERT_GE(connection, 0);
  RemoteDriver client(connection);
  const int kfd_fd = client.open();
  ASSERT_GE(kfd_fd, 0);
  EXPECT_TRUE(client.handles_drm_render_minor(128));
  EXPECT_TRUE(client.handles_drm_render_minor(129));
  ASSERT_NE(client.gpu_info_for_render_minor(128), nullptr);
  EXPECT_EQ(client.gpu_info_for_render_minor(128)->drm_render_minor, 129u);
  EXPECT_EQ(client.close(), 0);
  close(kfd_fd);
}

TEST(DaemonApi, OneGpuManifestHandshakeMatchesOnlySelectedRenderMinor) {
  constexpr const char *json = R"({
    "vm":{"arch":"cdna5","gpu":{"device":{
      "gpu_id":1250,"gfx_target_version":120500,"unique_id":1250,
      "num_sdma_engines":0}}},
    "topology":{"root":{"name":"soc","type":"soc","children":[
      {"name":"vram","type":"gpu_memory"},
      {"name":"xcd0","type":"xcd","children":[
        {"name":"l2","type":"l2_cache"},
        {"name":"cp","type":"command_processor"},
        {"name":"se0","type":"shader_engine","children":[
          {"name":"cu[0:4]","type":"compute_unit"}]}]}]}},
    "system":{"nodes":[{"id":"tray7","gpus":[
      {"id":"g5","gpu_id":505,"unique_id":3005,
       "location_id":1024,"drm_render_minor":129}]}]},
    "local_node":"tray7"
  })";
  TempDirectory directory;
  const std::filesystem::path socket = directory.path() / "daemon.sock";
  TestDaemon daemon(json, socket.c_str());
  ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);

  const int connection = connect_to(socket);
  ASSERT_GE(connection, 0);
  RemoteDriver client(connection);
  const int kfd_fd = client.open();
  ASSERT_GE(kfd_fd, 0);
  EXPECT_FALSE(client.handles_drm_render_minor(128));
  EXPECT_TRUE(client.handles_drm_render_minor(129));
  EXPECT_EQ(client.gpu_info_for_render_minor(128), nullptr);
  const auto *gpu = client.gpu_info_for_render_minor(129);
  ASSERT_NE(gpu, nullptr);
  EXPECT_EQ(gpu->gpu_id, 505u);
  EXPECT_EQ(gpu->drm_render_minor, 129u);
  EXPECT_EQ(client.close(), 0);
  close(kfd_fd);
}

TEST(DaemonApi, PeerIpcMapUnmapAcrossDistinctClientPids) {
  constexpr const char *json = R"({
    "vm":{"arch":"cdna5","gpu":{"device":{
      "gpu_id":1250,"gfx_target_version":120500,"unique_id":1250,
      "num_sdma_engines":1,"num_sdma_queues_per_engine":1}}},
    "topology":{"root":{"name":"soc","type":"soc","children":[
      {"name":"vram","type":"gpu_memory"},
      {"name":"xcd0","type":"xcd","children":[
        {"name":"l2","type":"l2_cache"},
        {"name":"cp","type":"command_processor"},
        {"name":"se0","type":"shader_engine","children":[
          {"name":"cu[0:4]","type":"compute_unit"}]}]}]}},
    "system":{
      "nodes":[{"id":"tray7","gpus":[
        {"id":"g5","gpu_id":505,"unique_id":3005,
         "location_id":1024,"drm_render_minor":129},
        {"id":"g9","gpu_id":909,"unique_id":3009,
         "location_id":1280,"drm_render_minor":130}]}],
      "endpoints":[
        {"id":"e5","kind":"gpu","gpu":"g5"},
        {"id":"e9","kind":"gpu","gpu":"g9"},
        {"id":"sw","kind":"switch"}],
      "links":[
        {"src":"e5","dst":"sw","link_class":"scale_up"},
        {"src":"e9","dst":"sw","link_class":"scale_up"}],
      "coherence_domains":[{"id":"local7","gpus":["g5","g9"]}]},
    "local_node":"tray7"
  })";

  TempDirectory directory;
  ASSERT_FALSE(directory.path().empty());
  const auto socket = directory.path() / "daemon.sock";
  ChildControlSocket owner_control;
  ChildControlSocket importer_control;
  ASSERT_TRUE(owner_control.valid());
  ASSERT_TRUE(importer_control.valid());

  // The daemon owns worker threads. Both clients must fork before it starts.
  PeerChildProcesses children;
  children.owner = fork();
  ASSERT_GE(children.owner, 0);
  if (children.owner == 0) {
    owner_control.close_parent();
    importer_control.close_parent();
    importer_control.close_child();
    _exit(run_peer_owner(socket, owner_control.child()));
  }
  children.importer = fork();
  ASSERT_GE(children.importer, 0);
  if (children.importer == 0) {
    importer_control.close_parent();
    owner_control.close_parent();
    owner_control.close_child();
    _exit(run_peer_importer(socket, importer_control.child()));
  }
  owner_control.close_child();
  importer_control.close_child();

  const rj_daemon_options_t trace_options{.struct_size = sizeof(rj_daemon_options_t),
                                          .flags = RJ_DAEMON_OPTION_TRACE_FABRIC_EFFECTS,
                                          .fabric_trace_capacity = 1};
  TestDaemon daemon(json, socket.c_str(), trace_options);
  struct StopClientsBeforeDaemon {
    PeerChildProcesses &children;
    ~StopClientsBeforeDaemon() { children.stop(); }
  } stop_clients{children};
  ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);

  const char start = 'S';
  ASSERT_TRUE(send_control(owner_control.parent(), start));
  PeerRpcReport owner{};
  ASSERT_TRUE(receive_control(owner_control.parent(), owner));
  ASSERT_EQ(owner.stage, 0u) << "owner result=" << owner.result << " n_success=" << owner.n_success;
  EXPECT_EQ(owner.pid, children.owner);
  EXPECT_EQ(owner.n_success, 1u);

  ASSERT_TRUE(send_control(importer_control.parent(), owner));
  PeerRpcReport importer{};
  ASSERT_TRUE(receive_control(importer_control.parent(), importer));
  ASSERT_EQ(importer.stage, 0u) << "importer result=" << importer.result
                                << " n_success=" << importer.n_success;
  EXPECT_EQ(importer.pid, children.importer);
  EXPECT_NE(importer.pid, owner.pid);
  EXPECT_EQ(importer.va, owner.va + 0x20000);
  EXPECT_EQ(importer.n_success, 2u);

  const char release = 'R';
  ASSERT_TRUE(send_control(owner_control.parent(), release));
  PeerRpcReport owner_cleanup{};
  ASSERT_TRUE(receive_control(owner_control.parent(), owner_cleanup));
  EXPECT_EQ(owner_cleanup.stage, 0u) << "owner cleanup result=" << owner_cleanup.result;

  int status = 0;
  ASSERT_TRUE(wait_for_peer_child(children.importer, status));
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  ASSERT_TRUE(wait_for_peer_child(children.owner, status));
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_EQ(daemon.status(), RJ_DAEMON_STATUS_RUNNING);
  // Two SDMA peer effects overflow the one-record diagnostic trace. Teardown
  // must still release the handle: only failed VM close is retryable.
  testing::internal::CaptureStderr();
  const rj_status_t stop_status = daemon.stop();
  const std::string shutdown_log = testing::internal::GetCapturedStderr();
  EXPECT_EQ(stop_status, ROCJITSU_STATUS_SUCCESS);
  EXPECT_NE(shutdown_log.find("dropped=1"), std::string::npos);
  EXPECT_EQ(daemon.status(), RJ_DAEMON_STATUS_STOPPED);
  EXPECT_EQ(daemon.stop(), ROCJITSU_STATUS_SUCCESS);
}

TEST(DaemonApi, RejectsInvalidJsonAndLongSocketPath) {
  TempDirectory directory;
  const std::string socket = (directory.path() / "daemon.sock").string();

  TestDaemon empty_json("", socket.c_str());
  EXPECT_EQ(empty_json.start_status(), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(empty_json.get(), nullptr);
  EXPECT_FALSE(std::filesystem::exists(socket));

  TestDaemon invalid_json("not json", socket.c_str());
  EXPECT_EQ(invalid_json.start_status(), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(invalid_json.get(), nullptr);
  EXPECT_FALSE(std::filesystem::exists(socket));

  TestDaemon path_as_json("/does/not/exist.json", socket.c_str());
  EXPECT_EQ(path_as_json.start_status(), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(path_as_json.get(), nullptr);
  EXPECT_FALSE(std::filesystem::exists(socket));

  const std::string long_socket(sizeof(sockaddr_un::sun_path), 'x');
  TestDaemon invalid_socket(daemon_json().c_str(), long_socket.c_str());
  EXPECT_EQ(invalid_socket.start_status(), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(invalid_socket.get(), nullptr);
}

TEST(DaemonApi, PreservesExistingSocketAndNonSocketEntries) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "daemon.sock";
  TestDaemon first(socket_path);
  ASSERT_EQ(first.start_status(), ROCJITSU_STATUS_SUCCESS);

  struct stat before {};
  ASSERT_EQ(lstat(socket_path.c_str(), &before), 0);
  TestDaemon second(socket_path);
  EXPECT_EQ(second.start_status(), ROCJITSU_STATUS_ERROR);
  EXPECT_EQ(second.get(), nullptr);
  struct stat after {};
  ASSERT_EQ(lstat(socket_path.c_str(), &after), 0);
  EXPECT_EQ(after.st_dev, before.st_dev);
  EXPECT_EQ(after.st_ino, before.st_ino);
  EXPECT_EQ(first.status(), RJ_DAEMON_STATUS_RUNNING);
  EXPECT_EQ(first.stop(), ROCJITSU_STATUS_SUCCESS);

  std::ofstream(socket_path) << "sentinel";
  TestDaemon non_socket(socket_path);
  EXPECT_EQ(non_socket.start_status(), ROCJITSU_STATUS_ERROR);
  EXPECT_EQ(non_socket.get(), nullptr);
  std::ifstream input(socket_path);
  std::string contents;
  input >> contents;
  EXPECT_EQ(contents, "sentinel");
}

TEST(DaemonApi, RecoversAbandonedSocket) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "daemon.sock";
  const std::string path = socket_path.string();
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
  const auto length = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + path.size() + 1);

  const int stale = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  ASSERT_GE(stale, 0);
  ASSERT_EQ(bind(stale, reinterpret_cast<const sockaddr *>(&address), length), 0);
  close(stale);

  {
    TestDaemon daemon(socket_path);
    ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);
    EXPECT_EQ(daemon.status(), RJ_DAEMON_STATUS_RUNNING);
  }
  EXPECT_FALSE(std::filesystem::exists(socket_path));
}

TEST(DaemonApi, FullListenerQueueDoesNotBlockOrRemoveSocket) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "daemon.sock";
  const std::string path = socket_path.string();
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
  const auto length = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + path.size() + 1);

  const int listener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  ASSERT_GE(listener, 0);
  ASSERT_EQ(bind(listener, reinterpret_cast<const sockaddr *>(&address), length), 0);
  ASSERT_EQ(listen(listener, 1), 0);
  const int first = connect_to(socket_path);
  const int second = connect_to(socket_path);
  ASSERT_GE(first, 0);
  ASSERT_GE(second, 0);

  struct stat before {};
  ASSERT_EQ(lstat(socket_path.c_str(), &before), 0);
  const auto start = std::chrono::steady_clock::now();
  TestDaemon daemon(socket_path);
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
  EXPECT_EQ(daemon.start_status(), ROCJITSU_STATUS_ERROR);
  EXPECT_EQ(daemon.get(), nullptr);
  struct stat after {};
  ASSERT_EQ(lstat(socket_path.c_str(), &after), 0);
  EXPECT_EQ(after.st_dev, before.st_dev);
  EXPECT_EQ(after.st_ino, before.st_ino);

  close(first);
  close(second);
  close(listener);
}

TEST(DaemonApi, ServesMultipleClientsAndCleansUp) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "nested" / "daemon.sock";
  {
    TestDaemon daemon(socket_path);
    ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);
    EXPECT_NE(daemon.get(), nullptr);
    EXPECT_EQ(daemon.status(), RJ_DAEMON_STATUS_RUNNING);
    EXPECT_TRUE(std::filesystem::is_socket(socket_path));

    const int first = connect_to(socket_path);
    const int second = connect_to(socket_path);
    ASSERT_GE(first, 0);
    ASSERT_GE(second, 0);
    const auto first_handshake = handshake(first, 10);
    const auto second_handshake = handshake(second, 20);
    EXPECT_EQ(first_handshake.version, kRpcProtocolVersion);
    EXPECT_EQ(second_handshake.version, kRpcProtocolVersion);
    EXPECT_EQ(first_handshake.gpu_id, second_handshake.gpu_id);
    close_session(first, 11);
    close_session(second, 21);
  }
  EXPECT_FALSE(std::filesystem::exists(socket_path));
}

TEST(DaemonApi, MunmapReturnsDriverFailureAndKeepsConnectionUsable) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "daemon.sock";
  TestDaemon daemon(socket_path);
  ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);

  const int connection = connect_to(socket_path);
  ASSERT_GE(connection, 0);
  RemoteDriver client(connection);
  const int kfd_fd = client.open();
  ASSERT_GE(kfd_fd, 0);

  EXPECT_EQ(client.munmap(reinterpret_cast<void *>(0x12345000), 4096), -ENOENT);
  kfd_ioctl_get_version_args version{};
  EXPECT_EQ(client.ioctl(AMDKFD_IOC_GET_VERSION, &version), 0);
  EXPECT_EQ(version.major_version, KFD_IOCTL_MAJOR_VERSION);
  EXPECT_EQ(client.close(), 0);
  close(kfd_fd);
}

TEST(DaemonApi, ExportDmabufTransfersSizedBackingFd) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "daemon.sock";
  TestDaemon daemon(socket_path);
  ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);

  const int connection = connect_to(socket_path);
  ASSERT_GE(connection, 0);
  RemoteDriver client(connection);
  const int kfd_fd = client.open();
  ASSERT_GE(kfd_fd, 0);
  ASSERT_NE(client.gpu_info(), nullptr);

  constexpr size_t kBytes = 4096;
  kfd_ioctl_alloc_memory_of_gpu_args allocation{};
  allocation.size = kBytes;
  allocation.gpu_id = client.gpu_info()->gpu_id;
  allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(client.ioctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocation), 0);
  auto *source = static_cast<uint8_t *>(
      client.mmap(nullptr, kBytes, PROT_READ | PROT_WRITE, MAP_SHARED, allocation.mmap_offset));
  ASSERT_NE(source, MAP_FAILED);

  kfd_ioctl_export_dmabuf_args exported{};
  exported.handle = allocation.handle;
  ASSERT_EQ(client.ioctl(AMDKFD_IOC_EXPORT_DMABUF, &exported), 0);
  const int export_fd = static_cast<int>(exported.dmabuf_fd);
  ASSERT_GE(export_fd, 0);
  struct stat backing {};
  ASSERT_EQ(fstat(export_fd, &backing), 0);
  EXPECT_EQ(backing.st_size, static_cast<off_t>(kBytes));
  auto *peer = static_cast<uint8_t *>(
      mmap(nullptr, kBytes, PROT_READ | PROT_WRITE, MAP_SHARED, export_fd, 0));
  ASSERT_NE(peer, MAP_FAILED);
  source[17] = 0x5a;
  EXPECT_EQ(peer[17], 0x5a);
  peer[29] = 0xa5;
  EXPECT_EQ(source[29], 0xa5);

  EXPECT_EQ(munmap(peer, kBytes), 0);
  EXPECT_EQ(close(export_fd), 0);
  EXPECT_EQ(munmap(source, kBytes), 0);
  kfd_ioctl_free_memory_of_gpu_args free{};
  free.handle = allocation.handle;
  EXPECT_EQ(client.ioctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free), 0);
  EXPECT_EQ(client.close(), 0);
  close(kfd_fd);
}

TEST(DaemonApi, RemoteCloseReturnsDaemonFailure) {
  int sockets[2] = {-1, -1};
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets), 0);
  std::jthread server([fd = sockets[1]] {
    RpcHeader request{};
    const bool received = rpc_recv_exact(fd, &request, sizeof(request));
    bool sent = false;
    if (received) {
      RpcHeader response{};
      response.opcode = RPC_CLOSE;
      response.request_id = request.request_id;
      response.result = -EIO;
      sent = rpc_send_exact(fd, &response, sizeof(response));
    }
    close(fd);
    EXPECT_TRUE(received);
    EXPECT_EQ(request.opcode, RPC_CLOSE);
    EXPECT_TRUE(sent);
  });

  RemoteDriver client(sockets[0]);
  EXPECT_EQ(client.close(), -EIO);
  EXPECT_EQ(client.close(), 0);
}

TEST(DaemonApi, StopUnblocksActiveAndPartialClients) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "daemon.sock";
  TestDaemon daemon(socket_path);
  ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);

  const int active = connect_to(socket_path);
  const int partial = connect_to(socket_path);
  ASSERT_GE(active, 0);
  ASSERT_GE(partial, 0);
  EXPECT_EQ(handshake(active).version, kRpcProtocolVersion);
  const uint8_t truncated_header[] = {static_cast<uint8_t>(RPC_HANDSHAKE)};
  ASSERT_EQ(send(partial, truncated_header, sizeof(truncated_header), MSG_NOSIGNAL),
            static_cast<ssize_t>(sizeof(truncated_header)));

  const auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(daemon.stop(), ROCJITSU_STATUS_SUCCESS);
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
  EXPECT_FALSE(std::filesystem::exists(socket_path));

  uint8_t byte = 0;
  EXPECT_LE(recv(active, &byte, sizeof(byte), 0), 0);
  EXPECT_LE(recv(partial, &byte, sizeof(byte), 0), 0);
  close(active);
  close(partial);
}

TEST(DaemonApi, RejectsMalformedMessagesWithoutStoppingServer) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "daemon.sock";
  {
    TestDaemon daemon(socket_path);
    ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);

    const int malformed = connect_to(socket_path);
    ASSERT_GE(malformed, 0);
    EXPECT_EQ(handshake(malformed).version, kRpcProtocolVersion);
    RpcHeader bad{};
    bad.opcode = RPC_IOCTL;
    bad.request_id = 2;
    bad.payload_bytes = sizeof(RpcIoctlRequest);
    RpcIoctlRequest bad_request{};
    bad_request.ioctl_cmd = AMDKFD_IOC_GET_VERSION;
    bad_request.args_bytes = UINT32_MAX;
    ASSERT_TRUE(rpc_send_exact(malformed, &bad, sizeof(bad)));
    ASSERT_TRUE(rpc_send_exact(malformed, &bad_request, sizeof(bad_request)));
    RpcHeader response{};
    EXPECT_FALSE(rpc_recv_exact(malformed, &response, sizeof(response)));
    close(malformed);

    const int healthy = connect_to(socket_path);
    ASSERT_GE(healthy, 0);
    EXPECT_EQ(handshake(healthy).version, kRpcProtocolVersion);
    close_session(healthy);
    EXPECT_EQ(daemon.status(), RJ_DAEMON_STATUS_RUNNING);
  }
  EXPECT_FALSE(std::filesystem::exists(socket_path));
}

TEST(DaemonApi, AcceptsValidGetVersionIoctlPayload) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "daemon.sock";
  TestDaemon daemon(socket_path);
  ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);

  const int client = connect_to(socket_path);
  ASSERT_GE(client, 0);
  ASSERT_EQ(handshake(client).version, kRpcProtocolVersion);

  kfd_ioctl_get_version_args arguments{};
  ASSERT_TRUE(send_ioctl_request(client, 2, AMDKFD_IOC_GET_VERSION, &arguments, sizeof(arguments)));

  RpcHeader response{};
  ASSERT_TRUE(rpc_recv_exact(client, &response, sizeof(response)));
  EXPECT_EQ(response.opcode, RPC_IOCTL);
  EXPECT_EQ(response.request_id, 2u);
  EXPECT_EQ(response.result, 0);
  ASSERT_EQ(response.payload_bytes, sizeof(arguments));
  ASSERT_TRUE(rpc_recv_exact(client, &arguments, sizeof(arguments)));
  EXPECT_EQ(arguments.major_version, KFD_IOCTL_MAJOR_VERSION);
  // Metadata-ring queues from KFD 1.19 are not implemented by the simulator.
  EXPECT_EQ(arguments.minor_version, 18u);
  EXPECT_EQ(daemon.status(), RJ_DAEMON_STATUS_RUNNING);
  close_session(client, 3);
}

TEST(DaemonApi, AcceptsCuMaskInlinePayloadAndKeepsConnectionUsable) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "daemon.sock";
  TestDaemon daemon(socket_path);
  ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);
  const int client = connect_to(socket_path);
  ASSERT_GE(client, 0);
  ASSERT_EQ(handshake(client).version, kRpcProtocolVersion);
  kfd_ioctl_set_cu_mask_args args{};
  args.queue_id = 999;
  args.num_cu_mask = 32;
  args.cu_mask_ptr = 0xDEADBEEF;
  const uint32_t mask = 1;
  ASSERT_TRUE(send_ioctl_request(client, 2, AMDKFD_IOC_SET_CU_MASK, &args, sizeof(args), &mask,
                                 sizeof(mask)));
  RpcHeader response{};
  ASSERT_TRUE(rpc_recv_exact(client, &response, sizeof(response)));
  EXPECT_EQ(response.result, -EFAULT); // Valid payload, missing queue.
  std::vector<uint8_t> payload(response.payload_bytes);
  ASSERT_TRUE(rpc_recv_exact(client, payload.data(), payload.size()));
  kfd_ioctl_get_version_args version{};
  ASSERT_TRUE(send_ioctl_request(client, 3, AMDKFD_IOC_GET_VERSION, &version, sizeof(version)));
  ASSERT_TRUE(rpc_recv_exact(client, &response, sizeof(response)));
  ASSERT_EQ(response.result, 0);
  ASSERT_EQ(response.payload_bytes, sizeof(version));
  ASSERT_TRUE(rpc_recv_exact(client, &version, sizeof(version)));
  close_session(client, 4);
}

TEST(DaemonApi, RejectsMalformedCuMaskPayloadWithoutStoppingServer) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "daemon.sock";
  TestDaemon daemon(socket_path);
  ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);
  struct MaskPayloadCase {
    uint32_t bits;
    size_t bytes;
    uint64_t pointer = 0xDEADBEEF;
  };
  for (const MaskPayloadCase &test :
       {MaskPayloadCase{0, 0}, MaskPayloadCase{31, sizeof(uint32_t)},
        MaskPayloadCase{32, sizeof(uint32_t) - 1}, MaskPayloadCase{32, sizeof(uint32_t) + 1},
        MaskPayloadCase{32, sizeof(uint32_t), 0}}) {
    const uint32_t bits = test.bits;
    const size_t bytes = test.bytes;
    SCOPED_TRACE(bits);
    SCOPED_TRACE(bytes);
    const int client = connect_to(socket_path);
    ASSERT_GE(client, 0);
    ASSERT_EQ(handshake(client).version, kRpcProtocolVersion);
    kfd_ioctl_set_cu_mask_args args{};
    args.queue_id = 999;
    args.num_cu_mask = bits;
    args.cu_mask_ptr = test.pointer;
    std::vector<uint8_t> mask(bytes);
    ASSERT_TRUE(send_ioctl_request(client, 2, AMDKFD_IOC_SET_CU_MASK, &args, sizeof(args),
                                   mask.data(), mask.size()));
    RpcHeader response{};
    EXPECT_FALSE(rpc_recv_exact(client, &response, sizeof(response)));
    EXPECT_EQ(daemon.status(), RJ_DAEMON_STATUS_RUNNING);
    close(client);
  }
  // A malformed client's disconnect must not prevent a fresh session.
  const int client = connect_to(socket_path);
  ASSERT_GE(client, 0);
  ASSERT_EQ(handshake(client).version, kRpcProtocolVersion);
  close_session(client, 2);
}

TEST(DaemonApi, RejectsWaitEventsInlineArrayOverflowWithoutStoppingServer) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "daemon.sock";
  TestDaemon daemon(socket_path);
  ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);

  expect_wait_events_payload_rejected(daemon, socket_path, 2, sizeof(kfd_event_data));
}

TEST(DaemonApi, RejectsWaitEventsInlineArrayOneByteShortWithoutStoppingServer) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "daemon.sock";
  TestDaemon daemon(socket_path);
  ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);

  expect_wait_events_payload_rejected(daemon, socket_path, 1, sizeof(kfd_event_data) - 1);
}

TEST(DaemonApi, RejectsWaitEventsInlineArrayWithTrailingByteWithoutStoppingServer) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "daemon.sock";
  TestDaemon daemon(socket_path);
  ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);

  expect_wait_events_payload_rejected(daemon, socket_path, 1, sizeof(kfd_event_data) + 1);
}

TEST(DaemonApi, DoesNotRemoveAReplacementSocketEntry) {
  TempDirectory directory;
  const auto socket_path = directory.path() / "daemon.sock";
  {
    TestDaemon daemon(socket_path);
    ASSERT_EQ(daemon.start_status(), ROCJITSU_STATUS_SUCCESS);

    ASSERT_EQ(unlink(socket_path.c_str()), 0);
    std::ofstream(socket_path) << "replacement";
  }
  EXPECT_TRUE(std::filesystem::is_regular_file(socket_path));
  std::ifstream input(socket_path);
  std::string contents;
  input >> contents;
  EXPECT_EQ(contents, "replacement");
}

} // namespace
