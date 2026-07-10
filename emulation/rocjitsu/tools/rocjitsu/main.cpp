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

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/config/dbt_guest_config.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/version.h"

#include "embedded_schema.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string_view>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace rocjitsu;

namespace {

pid_t peer_pid_for_socket(int fd) {
  struct ucred cred {};
  socklen_t len = sizeof(cred);
  if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &len) == 0 && cred.pid > 0)
    return cred.pid;
  return 0;
}

void handle_client(int client_fd, rj_vm_t *vm, pid_t client_pid, std::stop_token stop) {
  uint32_t process_id = 0;
  bool connected = true;

  while (!stop.stop_requested() && connected) {
    RpcHeader hdr{};
    if (!rpc_recv_exact(client_fd, &hdr, sizeof(hdr)))
      break;

    switch (hdr.opcode) {
    case RPC_HANDSHAKE: {
      auto open_rc = rj_vm_device_open(vm, client_pid, &process_id);
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

volatile sig_atomic_t g_listen_fd = -1;

int run_daemon_server(const char *config_path) {
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

    pid_t peer_pid = peer_pid_for_socket(client);
    std::lock_guard<std::mutex> lock(client_threads_mutex);
    active_client_fds.push_back(client);
    client_threads.emplace_back(handle_client, client, vm, peer_pid, stop_source.get_token());
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

std::optional<std::filesystem::path> current_executable_path() {
  std::vector<char> buffer(256);
  for (;;) {
    ssize_t n = readlink("/proc/self/exe", buffer.data(), buffer.size());
    if (n < 0)
      return std::nullopt;
    if (static_cast<size_t>(n) < buffer.size())
      return std::filesystem::path(std::string(buffer.data(), static_cast<size_t>(n)));
    buffer.resize(buffer.size() * 2);
  }
}

std::string canonical_existing_path(const std::filesystem::path &candidate) {
  std::error_code ec;
  if (!std::filesystem::exists(candidate, ec) || ec)
    return {};

  // The launcher should keep probing candidate layouts when a path disappears
  // or cannot be canonicalized, rather than letting a filesystem exception
  // terminate the process before exec.
  std::filesystem::path canonical = std::filesystem::canonical(candidate, ec);
  return ec ? std::string{} : canonical.string();
}

std::string find_runtime_lib(std::string_view lib_name) {
  auto self = current_executable_path();
  if (!self)
    return {};
  auto bin_dir = self->parent_path();
  const std::filesystem::path library_name{std::string(lib_name)};

  // Installed layouts use <prefix>/lib or <prefix>/lib64. CMake build trees may
  // place shared libraries either at the build root or under target directories,
  // so use the same ordered probe list for both the interposer and HSA hooks.
  for (const auto &candidate : {
           bin_dir / ".." / "lib" / library_name,
           bin_dir / ".." / "lib64" / library_name,
           bin_dir / ".." / ".." / library_name,
           bin_dir / ".." / ".." / "lib" / "rocjitsu" / "src" / "rocjitsu" / "hooks" / library_name,
           bin_dir / ".." / ".." / "lib64" / "rocjitsu" / "src" / "rocjitsu" / "hooks" /
               library_name,
       }) {
    if (std::string path = canonical_existing_path(candidate); !path.empty())
      return path;
  }
  return {};
}

std::string find_interposer_lib() { return find_runtime_lib("librocjitsu.so"); }

std::string find_hooks_lib() { return find_runtime_lib("librocjitsu_hooks.so"); }

void prepend_env_path(const char *name, const std::string &value) {
  if (const char *old_value = std::getenv(name); old_value && *old_value) {
    std::string combined = value + ":" + old_value;
    setenv(name, combined.c_str(), 1);
    return;
  }
  setenv(name, value.c_str(), 1);
}

bool write_config_file(const std::string &config_path) {
  auto cfg_file = rpc_default_config_file_path();
  std::filesystem::create_directories(std::filesystem::path(cfg_file).parent_path());
  std::ofstream ofs(cfg_file);
  if (!ofs)
    return false;
  ofs << config_path << '\n';
  return ofs.good();
}

void cleanup_runtime_files() {
  auto cfg_file = rpc_default_config_file_path();
  unlink(cfg_file.c_str());
  auto sock_file = rpc_default_socket_path();
  unlink(sock_file.c_str());
}

struct KfdGpuOrdinal {
  uint32_t ordinal = 0;
  uint32_t node_id = 0;
  uint32_t gpu_id = 0;
  uint32_t gfx_target_version = 0;
};

std::optional<uint32_t> parse_u32(std::string_view text) {
  uint32_t value = 0;
  auto *begin = text.data();
  auto *end = text.data() + text.size();
  auto [ptr, err] = std::from_chars(begin, end, value);
  if (err != std::errc{} || ptr != end)
    return std::nullopt;
  return value;
}

std::string_view trim(std::string_view text) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
    text.remove_prefix(1);
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
    text.remove_suffix(1);
  return text;
}

std::optional<uint32_t> read_u32_file(const std::filesystem::path &path) {
  std::ifstream in(path);
  uint32_t value = 0;
  if (!(in >> value))
    return std::nullopt;
  return value;
}

std::optional<uint32_t> read_u32_property(const std::filesystem::path &path, std::string_view key) {
  std::ifstream in(path);
  std::string name;
  uint64_t value = 0;
  while (in >> name >> value) {
    if (name == key)
      return static_cast<uint32_t>(value);
  }
  return std::nullopt;
}

std::vector<KfdGpuOrdinal> real_kfd_gpu_ordinals() {
  std::filesystem::path nodes_dir = "/sys/devices/virtual/kfd/kfd/topology/nodes";
  if (!std::filesystem::exists(nodes_dir))
    nodes_dir = "/sys/class/kfd/kfd/topology/nodes";

  struct KfdNodeInfo {
    uint32_t node_id = 0;
    uint32_t gpu_id = 0;
    uint32_t gfx_target_version = 0;
  };

  std::vector<KfdNodeInfo> nodes;
  std::error_code ec;
  for (const auto &entry : std::filesystem::directory_iterator(nodes_dir, ec)) {
    if (!entry.is_directory(ec))
      continue;

    std::string name = entry.path().filename().string();
    auto node_id = parse_u32(name);
    if (!node_id)
      continue;

    auto gpu_id = read_u32_file(entry.path() / "gpu_id");
    if (gpu_id && *gpu_id != 0) {
      uint32_t gfx_target_version =
          read_u32_property(entry.path() / "properties", "gfx_target_version").value_or(0);
      nodes.push_back({*node_id, *gpu_id, gfx_target_version});
    }
  }

  std::sort(nodes.begin(), nodes.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.node_id < rhs.node_id; });

  std::vector<KfdGpuOrdinal> gpus;
  gpus.reserve(nodes.size());
  for (uint32_t ordinal = 0; ordinal < nodes.size(); ++ordinal)
    gpus.push_back({ordinal, nodes[ordinal].node_id, nodes[ordinal].gpu_id,
                    nodes[ordinal].gfx_target_version});
  return gpus;
}

/// @brief Reject implicit host selection when more than one GPU has the host ISA.
///
/// @details The launcher, GuestKfd, and HSA tools hook discover the host through
/// separate views of KFD and ROCR. Selecting the first ISA match independently
/// is only well-defined when that match is unique. On a multi-GPU host, require
/// the config to name the shared KFD gpu_id so every layer routes to the same
/// physical GPU.
bool has_unambiguous_host_gpu(const rocjitsu::config::DbtGuestConfig &dbt_guest) {
  if (dbt_guest.host_gpu_id != 0)
    return true;

  std::optional<uint32_t> target_version =
      rocjitsu::kmd::gfx_target_version_from_name(dbt_guest.host_isa);
  if (!target_version)
    return true;

  std::vector<uint32_t> matching_gpu_ids;
  for (const KfdGpuOrdinal &gpu : real_kfd_gpu_ordinals()) {
    if (gpu.gfx_target_version == *target_version)
      matching_gpu_ids.push_back(gpu.gpu_id);
  }
  if (matching_gpu_ids.size() <= 1)
    return true;

  std::cerr << std::format(
      "rocjitsu: dbt_guest.host_isa '{}' matches {} host GPUs; set host_gpu_id to one of:",
      dbt_guest.host_isa, matching_gpu_ids.size());
  for (uint32_t gpu_id : matching_gpu_ids)
    std::cerr << ' ' << gpu_id;
  std::cerr << '\n';
  return false;
}

bool append_unique(std::vector<std::string> *tokens, std::string token) {
  if (token.empty())
    return false;
  if (std::find(tokens->begin(), tokens->end(), token) != tokens->end())
    return false;
  tokens->push_back(std::move(token));
  return true;
}

std::string join_comma(const std::vector<std::string> &tokens) {
  std::string result;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i != 0)
      result += ',';
    result += tokens[i];
  }
  return result;
}

void maybe_expand_rocr_visible_devices(const rocjitsu::config::DbtGuestConfig &dbt_guest) {
  const char *visible = std::getenv("ROCR_VISIBLE_DEVICES");
  if (visible == nullptr || *visible == '\0')
    return;

  std::vector<KfdGpuOrdinal> gpus = real_kfd_gpu_ordinals();
  if (gpus.empty())
    return;

  uint32_t host_ordinal = 0;
  if (dbt_guest.host_gpu_id != 0) {
    auto match = std::find_if(gpus.begin(), gpus.end(), [&](const KfdGpuOrdinal &gpu) {
      return gpu.gpu_id == dbt_guest.host_gpu_id;
    });
    if (match == gpus.end())
      return;
    host_ordinal = match->ordinal;
  } else {
    std::optional<uint32_t> target_version =
        rocjitsu::kmd::gfx_target_version_from_name(dbt_guest.host_isa);
    if (!target_version)
      return;

    auto match = std::find_if(gpus.begin(), gpus.end(), [&](const KfdGpuOrdinal &gpu) {
      return gpu.gfx_target_version == *target_version;
    });
    if (match == gpus.end())
      return;
    host_ordinal = match->ordinal;
  }

  const uint32_t guest_ordinal = static_cast<uint32_t>(gpus.size());
  std::vector<std::string> expanded;
  std::string_view rest = visible;
  bool changed = false;

  while (true) {
    size_t comma = rest.find(',');
    std::string_view raw = comma == std::string_view::npos ? rest : rest.substr(0, comma);
    std::string_view token = trim(raw);
    if (!token.empty()) {
      std::optional<uint32_t> ordinal = parse_u32(token);
      if (ordinal && (*ordinal == host_ordinal || *ordinal == guest_ordinal)) {
        // ROCR filters topology before HSA tools callbacks. Include both the
        // hidden host and appended guest internally so our HSA iteration hook
        // can present one public replacement agent.
        changed = append_unique(&expanded, std::to_string(host_ordinal)) || changed;
        changed = append_unique(&expanded, std::to_string(guest_ordinal)) || changed;
      } else {
        changed = append_unique(&expanded, std::string(token)) || changed;
      }
    }

    if (comma == std::string_view::npos)
      break;
    rest.remove_prefix(comma + 1);
  }

  std::string rewritten = join_comma(expanded);
  if (!rewritten.empty() && rewritten != visible) {
    setenv("ROCR_VISIBLE_DEVICES", rewritten.c_str(), 1);
  }
}

void print_usage() {
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

} // namespace

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

  rocjitsu::config::DbtGuestConfig dbt_guest_config;
  try {
    dbt_guest_config = rocjitsu::config::load_dbt_guest_config_from_file(abs_config);
    if (!dbt_guest_config.enabled)
      (void)rocjitsu::config::load_config(abs_config, rocjitsu::kEmbeddedSchema);
  } catch (const std::exception &e) {
    std::cerr << std::format("rocjitsu: failed to parse config: {}\n", e.what());
    return 1;
  }
  const bool dbt_guest_mode = dbt_guest_config.enabled;
  if (dbt_guest_mode && (daemon_mode || attach_mode)) {
    std::cerr << "rocjitsu: dbt_guest mode currently supports local launch only\n";
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
    std::cerr << "rocjitsu: could not find librocjitsu.so\n";
    return 1;
  }

  std::string hooks_path;
  if (dbt_guest_mode) {
    if (dbt_guest_config.guest_isa.empty() || dbt_guest_config.host_isa.empty()) {
      std::cerr << "rocjitsu: dbt_guest requires guest_isa and host_isa\n";
      return 1;
    }
    if (!has_unambiguous_host_gpu(dbt_guest_config))
      return 1;
    hooks_path = find_hooks_lib();
    if (hooks_path.empty()) {
      std::cerr << "rocjitsu: could not find librocjitsu_hooks.so\n";
      return 1;
    }
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

  prepend_env_path("LD_PRELOAD", lib_path);
  if (dbt_guest_mode) {
    maybe_expand_rocr_visible_devices(dbt_guest_config);
    // The HSA hook still uses the legacy tools callback path. Disable only the
    // rocprofiler-register table-delivery path so it cannot validate an
    // unshadowed table before rocjitsu installs guest-agent wrappers.
    setenv("HSA_TOOLS_DISABLE_REGISTER", "1", 1);
    setenv("HSA_TOOLS_LIB", hooks_path.c_str(), 1);
  }
  execvp(app_argv[0], app_argv);

  std::cerr << std::format("rocjitsu: execvp failed: {}\n", strerror(errno));
  cleanup_runtime_files();
  return 1;
}
