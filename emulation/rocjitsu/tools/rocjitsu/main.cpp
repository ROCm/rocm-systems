// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file main.cpp
/// @brief rocjitsu CLI — launcher for GPU simulation via LD_PRELOAD interposition.
///
/// @details Supports three usage patterns:
///   rocjitsu --config foo.json -- ./app           (local mode: in-process simulation)
///   rocjitsu --daemon --config foo.json -- ./app  (daemon mode: fork daemon + launch app)
///   rocjitsu --daemon --config foo.json           (daemon-only: run daemon server)

#include "rocjitsu/vm/rj_vm.h"

#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/version.h"

#include <cerrno>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <stop_token>
#include <string_view>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace rocjitsu;

static void handle_client(int client_fd, rj_vm_t *vm, std::stop_token stop) {
  uint32_t process_id = 0;
  bool connected = true;

  while (!stop.stop_requested() && connected) {
    RpcHeader hdr{};
    if (!rpc_recv_exact(client_fd, &hdr, sizeof(hdr)))
      break;

    switch (hdr.opcode) {
    case RPC_HANDSHAKE: {
      auto open_rc = rj_vm_device_open(vm, &process_id);
      if (open_rc != ROCJITSU_STATUS_SUCCESS) {
        RpcHeader resp{};
        resp.request_id = hdr.request_id;
        resp.result = -1;
        rpc_send_exact(client_fd, &resp, sizeof(resp));
        connected = false;
        break;
      }

      uint32_t gpu_id = 0;
      rj_vm_gpu_id(vm, &gpu_id);

      const char *topo = nullptr;
      rj_vm_topology_path(vm, &topo);
      auto topo_len = topo ? std::strlen(topo) : 0;

      const char *drm = nullptr;
      rj_vm_drm_path(vm, &drm);
      auto drm_len = drm ? std::strlen(drm) : 0;

      RpcHeader resp{};
      resp.request_id = hdr.request_id;

      RpcHandshakeResponse hs{};
      hs.version = kRpcProtocolVersion;
      hs.gpu_id = gpu_id;
      hs.topology_path_len = static_cast<uint32_t>(topo_len);
      hs.drm_path_len = static_cast<uint32_t>(drm_len);
      rj_vm_gpu_info(vm, &hs.gpu_info);

      resp.payload_bytes = sizeof(hs) + hs.topology_path_len + hs.drm_path_len;
      rpc_send_exact(client_fd, &resp, sizeof(resp));
      rpc_send_exact(client_fd, &hs, sizeof(hs));
      if (topo_len > 0)
        rpc_send_exact(client_fd, topo, topo_len);
      if (drm_len > 0)
        rpc_send_exact(client_fd, drm, drm_len);
      break;
    }

    case RPC_CLOSE: {
      rj_vm_device_close(vm, process_id);
      process_id = 0;
      RpcHeader resp{};
      resp.request_id = hdr.request_id;
      rpc_send_exact(client_fd, &resp, sizeof(resp));
      connected = false;
      break;
    }

    case RPC_MMAP: {
      RpcMmapRequest mreq{};
      if (!rpc_recv_exact(client_fd, &mreq, sizeof(mreq))) {
        connected = false;
        break;
      }

      rj_vm_map_t map{};
      map.addr = mreq.addr;
      map.length = mreq.length;
      map.prot = static_cast<uint32_t>(mreq.prot);
      map.flags = static_cast<uint32_t>(mreq.flags);
      map.offset = mreq.offset;
      rj_vm_device_map_as(vm, process_id, &map);

      RpcHeader resp{};
      resp.request_id = hdr.request_id;
      resp.result = (reinterpret_cast<void *>(map.mapped_addr) == MAP_FAILED) ? -errno : 0;
      resp.payload_bytes = sizeof(RpcMmapResponse);

      RpcMmapResponse mresp{.mapped_addr = map.mapped_addr};

      uint8_t response_buffer[sizeof(resp) + sizeof(mresp)];
      std::memcpy(response_buffer, &resp, sizeof(resp));
      std::memcpy(response_buffer + sizeof(resp), &mresp, sizeof(mresp));

      rj_handle_t backing_memfd = -1;
      rj_vm_get_shared_mem_as(vm, process_id, mreq.offset, &backing_memfd);
      if (backing_memfd >= 0)
        rpc_send_msg(client_fd, response_buffer, sizeof(response_buffer), &backing_memfd, 1);
      else
        rpc_send_exact(client_fd, response_buffer, sizeof(response_buffer));
      break;
    }

    case RPC_MUNMAP: {
      RpcMunmapRequest mreq{};
      if (!rpc_recv_exact(client_fd, &mreq, sizeof(mreq))) {
        connected = false;
        break;
      }

      rj_vm_unmap_t unmap{.addr = mreq.addr, .length = mreq.length};
      rj_vm_device_unmap_as(vm, process_id, &unmap);

      RpcHeader resp{};
      resp.request_id = hdr.request_id;
      rpc_send_exact(client_fd, &resp, sizeof(resp));
      break;
    }

    case RPC_IOCTL: {
      constexpr uint32_t kMaxPayloadBytes = 16 * 1024 * 1024;
      if (hdr.payload_bytes > kMaxPayloadBytes || hdr.payload_bytes < sizeof(RpcIoctlRequest)) {
        connected = false;
        break;
      }
      std::vector<uint8_t> payload(hdr.payload_bytes);
      if (!rpc_recv_exact(client_fd, payload.data(), hdr.payload_bytes)) {
        connected = false;
        break;
      }
      auto *ioctl_request = reinterpret_cast<RpcIoctlRequest *>(payload.data());

      rj_vm_cmd_t cmd{};
      cmd.cmd = ioctl_request->ioctl_cmd;
      cmd.buf = payload.data() + sizeof(RpcIoctlRequest);
      cmd.buf_size = ioctl_request->args_bytes;
      cmd.shared_handle = -1;
      rj_vm_execute_as(vm, process_id, &cmd);

      RpcHeader resp{};
      resp.opcode = RPC_IOCTL;
      resp.request_id = hdr.request_id;
      resp.result = cmd.result;
      resp.payload_bytes = static_cast<uint32_t>(cmd.buf_size);

      if (cmd.shared_handle >= 0) {
        std::vector<uint8_t> response_buffer(sizeof(resp) + cmd.buf_size);
        std::memcpy(response_buffer.data(), &resp, sizeof(resp));
        if (cmd.buf_size > 0)
          std::memcpy(response_buffer.data() + sizeof(resp), cmd.buf, cmd.buf_size);
        rpc_send_msg(client_fd, response_buffer.data(), response_buffer.size(), &cmd.shared_handle,
                     1);
      } else {
        rpc_send_exact(client_fd, &resp, sizeof(resp));
        if (cmd.buf_size > 0)
          rpc_send_exact(client_fd, cmd.buf, cmd.buf_size);
      }
      break;
    }

    default:
      connected = false;
      break;
    }
  }

  if (process_id != 0)
    rj_vm_device_close(vm, process_id);
  ::close(client_fd);
}

static volatile sig_atomic_t g_listen_fd = -1;

static int run_daemon_server(const char *config_path) {
  rj_vm_t *vm = nullptr;
  if (rj_vm_create(config_path, RJ_VM_MODE_DAEMON, &vm) != ROCJITSU_STATUS_SUCCESS) {
    std::cerr << std::format("rocjitsu: failed to create VM from {}\n", config_path);
    return 1;
  }

  std::jthread engine_thread([vm]() { rj_vm_run(vm, nullptr); });

  auto sock_path = rpc_default_socket_path();
  std::filesystem::create_directories(std::filesystem::path(sock_path).parent_path());
  unlink(sock_path.c_str());

  int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (listen_fd < 0)
    return 1;
  g_listen_fd = listen_fd;

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  sock_path.copy(addr.sun_path, sizeof(addr.sun_path) - 1);

  if (bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
      listen(listen_fd, 16) != 0) {
    ::close(listen_fd);
    return 1;
  }

  std::stop_source stop_source;
  std::vector<std::jthread> client_threads;
  std::vector<int> active_client_fds;
  std::mutex client_threads_mutex;

  std::signal(SIGINT, [](int) {
    int fd = g_listen_fd;
    g_listen_fd = -1;
    if (fd >= 0)
      shutdown(fd, SHUT_RDWR);
  });
  std::signal(SIGTERM, [](int) {
    int fd = g_listen_fd;
    g_listen_fd = -1;
    if (fd >= 0)
      shutdown(fd, SHUT_RDWR);
  });

  while (g_listen_fd >= 0) {
    int client = accept(listen_fd, nullptr, nullptr);
    if (client < 0)
      break;

    std::lock_guard<std::mutex> lock(client_threads_mutex);
    active_client_fds.push_back(client);
    client_threads.emplace_back(handle_client, client, vm, stop_source.get_token());
  }

  stop_source.request_stop();
  {
    std::lock_guard<std::mutex> lock(client_threads_mutex);
    for (int fd : active_client_fds)
      shutdown(fd, SHUT_RDWR);
    client_threads.clear();
  }
  ::close(listen_fd);
  unlink(sock_path.c_str());

  rj_vm_request_exit(vm, "daemon shutdown");
  engine_thread.join();
  rj_vm_destroy(vm);
  return 0;
}

static std::string find_interposer_lib() {
  char self[4096];
  auto n = readlink("/proc/self/exe", self, sizeof(self) - 1);
  if (n <= 0)
    return {};
  self[n] = '\0';
  auto bin_dir = std::filesystem::path(self).parent_path();
  // Installed layout: <prefix>/bin/rocjitsu → <prefix>/lib/librocjitsu_kmd.so
  //                   or <prefix>/bin/rocjitsu → <prefix>/lib64/librocjitsu_kmd.so
  // Build layout: build/tools/rocjitsu/rocjitsu → build/lib/.../librocjitsu_kmd.so
  //               or build/tools/rocjitsu/rocjitsu → build/lib64/.../librocjitsu_kmd.so
  for (auto &candidate : {
           bin_dir / ".." / "lib" / "librocjitsu_kmd.so",
           bin_dir / ".." / "lib64" / "librocjitsu_kmd.so",
           bin_dir / ".." / ".." / "lib" / "rocjitsu" / "src" / "rocjitsu" / "kmd" / "linux" /
               "librocjitsu_kmd.so",
       }) {
    if (std::filesystem::exists(candidate))
      return std::filesystem::canonical(candidate).string();
  }
  return {};
}

static bool write_config_file(const std::string &config_path) {
  auto cfg_file = rpc_default_config_file_path();
  std::filesystem::create_directories(std::filesystem::path(cfg_file).parent_path());
  std::ofstream ofs(cfg_file);
  if (!ofs)
    return false;
  ofs << config_path << '\n';
  return ofs.good();
}

static void cleanup_runtime_files() {
  auto cfg_file = rpc_default_config_file_path();
  unlink(cfg_file.c_str());
  auto sock_file = rpc_default_socket_path();
  unlink(sock_file.c_str());
}

static void print_usage() {
  std::cerr
      << "Usage: rocjitsu --config <config.json> [--daemon|--attach] -- <app> [args...]\n"
         "\n"
         "Modes:\n"
         "  rocjitsu --config foo.json -- ./app          Local mode (in-process simulation)\n"
         "  rocjitsu --daemon --config foo.json -- ./app Daemon mode (fork daemon + launch app)\n"
         "  rocjitsu --daemon --config foo.json          Daemon-only (run server)\n"
         "  rocjitsu --attach --config foo.json -- ./app Attach to running daemon\n"
         "\n"
         "Options:\n"
         "  --config <path>   Simulation config JSON (required)\n"
         "  --version, -v     Print version and exit\n"
         "  --help, -h        Print this help and exit\n";
}

int main(int argc, char *argv[]) {
  std::signal(SIGPIPE, SIG_IGN);

  const char *config_path = nullptr;
  bool daemon_mode = false;
  bool attach_mode = false;
  int separator_idx = -1;

  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--") {
      separator_idx = i;
      break;
    }
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else if (arg == "--daemon") {
      daemon_mode = true;
    } else if (arg == "--attach") {
      attach_mode = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    } else if (arg == "--version" || arg == "-v") {
      std::cout << "rocjitsu " << ROCJITSU_VERSION << "\n";
      return 0;
    } else {
      std::cerr << std::format("rocjitsu: unknown option: {}\n", arg);
      print_usage();
      return 1;
    }
  }

  if (!config_path) {
    std::cerr << "rocjitsu: --config is required\n";
    print_usage();
    return 1;
  }

  auto abs_config = std::filesystem::absolute(config_path).string();
  if (!std::filesystem::exists(abs_config)) {
    std::cerr << std::format("rocjitsu: config file not found: {}\n", abs_config);
    return 1;
  }

  bool has_app = (separator_idx >= 0 && separator_idx + 1 < argc);

  if (daemon_mode && !has_app)
    return run_daemon_server(abs_config.c_str());

  if (!has_app) {
    std::cerr << "rocjitsu: no application specified after --\n";
    print_usage();
    return 1;
  }

  char **app_argv = &argv[separator_idx + 1];

  auto lib_path = find_interposer_lib();
  if (lib_path.empty()) {
    std::cerr << "rocjitsu: could not find librocjitsu_kmd.so\n";
    return 1;
  }

  if (attach_mode) {
    auto sock_path = rpc_default_socket_path();
    if (!std::filesystem::exists(sock_path)) {
      std::cerr << std::format("rocjitsu: no daemon socket at {}\n", sock_path);
      return 1;
    }
  } else if (daemon_mode) {
    pid_t daemon_pid = fork();
    if (daemon_pid < 0) {
      std::cerr << std::format("rocjitsu: fork failed: {}\n", strerror(errno));
      return 1;
    }

    if (daemon_pid == 0) {
      prctl(PR_SET_PDEATHSIG, SIGTERM);
      return run_daemon_server(abs_config.c_str());
    }

    auto sock_path = rpc_default_socket_path();
    for (int i = 0; i < 300; ++i) {
      if (std::filesystem::exists(sock_path))
        break;
      usleep(10000);
    }
    if (!std::filesystem::exists(sock_path)) {
      std::cerr << "rocjitsu: daemon socket did not appear\n";
      kill(daemon_pid, SIGTERM);
      waitpid(daemon_pid, nullptr, 0);
      return 1;
    }
  } else {
    if (!write_config_file(abs_config)) {
      std::cerr << "rocjitsu: failed to write config file\n";
      return 1;
    }
  }

  setenv("LD_PRELOAD", lib_path.c_str(), 1);
  execvp(app_argv[0], app_argv);

  std::cerr << std::format("rocjitsu: execvp failed: {}\n", strerror(errno));
  cleanup_runtime_files();
  return 1;
}
