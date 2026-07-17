// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/rocjitsu.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
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
    std::ifstream input(std::filesystem::path(CONFIG_DIR) / "gfx950_cdna4_kmd.json");
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

  ~TestDaemon() { (void)stop(); }

  TestDaemon(const TestDaemon &) = delete;
  TestDaemon &operator=(const TestDaemon &) = delete;
  TestDaemon(TestDaemon &&) = delete;
  TestDaemon &operator=(TestDaemon &&) = delete;

  [[nodiscard]] rj_status_t start_status() const { return start_status_; }
  [[nodiscard]] rj_daemon_t *get() const { return daemon_; }
  [[nodiscard]] rj_daemon_status_t status() const { return rj_daemon_status(daemon_); }

  rj_status_t stop() {
    rj_daemon_t *daemon = std::exchange(daemon_, nullptr);
    return rj_daemon_stop(daemon);
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
  EXPECT_EQ(response.request_id, request_id);
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
  EXPECT_EQ(rj_daemon_status(nullptr), RJ_DAEMON_STATUS_STOPPED);
  EXPECT_EQ(rj_daemon_stop(nullptr), ROCJITSU_STATUS_SUCCESS);
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
  EXPECT_EQ(arguments.minor_version, KFD_IOCTL_MINOR_VERSION);
  EXPECT_EQ(daemon.status(), RJ_DAEMON_STATUS_RUNNING);
  close_session(client, 3);
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
