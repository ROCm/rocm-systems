// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file main.cpp
/// @brief rocjitsu CLI — launcher for GPU simulation via LD_PRELOAD interposition.
///
/// @details Supports three usage patterns:
///   rocjitsu --config foo.json -- ./app           (local mode: in-process simulation)
///   rocjitsu --daemon --config foo.json -- ./app  (daemon mode: fork daemon + launch app)
///   rocjitsu --daemon --config foo.json           (daemon-only: run daemon server)
///
/// Builds configured with ROCJITSU_ENABLE_VFIO additionally support
///   rocjitsu --config foo.json --vfio-socket <path>  (serve a PCI device to a VMM)

#include "rocjitsu/daemon/rj_daemon.h"
#if defined(RJ_ENABLE_VFIO_USER)
#include "rocjitsu/vmm/vfu/vfio_server.h"
#endif

#include "rocjitsu/base/rj_version.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/config/dbt_guest_config.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include "embedded_schema.h"
#include "launch_preload.h"
#include "rocm_visibility.h"

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
#include <iterator>
#include <optional>
#include <poll.h>
#include <string_view>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace rocjitsu;

namespace {

constexpr int kDaemonReadyTimeoutMs = 30'000;

int run_daemon_server(const char *config_path, const std::string &socket_path = {},
                      int ready_fd = -1) {
  sigset_t daemon_signals;
  sigemptyset(&daemon_signals);
  sigaddset(&daemon_signals, SIGINT);
  sigaddset(&daemon_signals, SIGTERM);
  sigset_t previous_signals;
  if (sigprocmask(SIG_BLOCK, &daemon_signals, &previous_signals) != 0) {
    std::cerr << std::format("rocjitsu: failed to block daemon signals: {}\n", strerror(errno));
    if (ready_fd >= 0)
      close(ready_fd);
    return 1;
  }

  rj_daemon_t *daemon = nullptr;
  const std::string resolved_socket_path =
      socket_path.empty() ? rpc_default_socket_path() : socket_path;
  std::ifstream config_stream(config_path);
  const std::string json((std::istreambuf_iterator<char>(config_stream)),
                         std::istreambuf_iterator<char>());
  if (!config_stream) {
    std::cerr << std::format("rocjitsu: failed to read daemon configuration from {}\n",
                             config_path);
    if (ready_fd >= 0)
      close(ready_fd);
    sigprocmask(SIG_SETMASK, &previous_signals, nullptr);
    return 1;
  }
  if (rj_daemon_start(json.c_str(), resolved_socket_path.c_str(), &daemon) !=
      ROCJITSU_STATUS_SUCCESS) {
    std::cerr << std::format("rocjitsu: failed to start daemon from {} at {}\n", config_path,
                             resolved_socket_path);
    if (ready_fd >= 0)
      close(ready_fd);
    sigprocmask(SIG_SETMASK, &previous_signals, nullptr);
    return 1;
  }

  if (ready_fd >= 0) {
    const uint8_t ready = 1;
    ssize_t written = 0;
    do {
      written = write(ready_fd, &ready, sizeof(ready));
    } while (written < 0 && errno == EINTR);
    close(ready_fd);
    if (written != static_cast<ssize_t>(sizeof(ready))) {
      rj_daemon_stop(daemon);
      sigprocmask(SIG_SETMASK, &previous_signals, nullptr);
      return 1;
    }
  }

  while (rj_daemon_status(daemon) == RJ_DAEMON_STATUS_RUNNING) {
    const timespec timeout{.tv_sec = 0, .tv_nsec = 100'000'000};
    const int signal = sigtimedwait(&daemon_signals, nullptr, &timeout);
    if (signal == SIGINT || signal == SIGTERM)
      break;
    if (signal < 0 && errno != EAGAIN && errno != EINTR)
      break;
  }

  const bool failed = rj_daemon_status(daemon) == RJ_DAEMON_STATUS_ERROR;
  rj_daemon_stop(daemon);
  if (!socket_path.empty()) {
    std::error_code remove_error;
    std::filesystem::remove(std::filesystem::path(socket_path).parent_path(), remove_error);
  }
  sigprocmask(SIG_SETMASK, &previous_signals, nullptr);
  return failed ? 1 : 0;
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

void cleanup_runtime_files(pid_t pid) {
  std::error_code error;
  std::filesystem::remove_all(rpc_invocation_runtime_dir(pid), error);
}

// Best-effort reap of per-PID runtime dirs left behind by prior invocations that
// exited via execvp (which never returns, so cleanup_runtime_files does not run).
// Each numeric <pid> subdir of the runtime root is removed if that PID is no
// longer alive, so a recycled PID cannot inherit a stale config_path/daemon.sock.
void reap_stale_runtime_dirs() {
  // Never iterate an empty root: directory_iterator("") scans the CWD, which would
  // let this reaper remove_all unrelated numeric directories. rpc_default_runtime_dir()
  // already treats a set-but-empty $ROCJITSU_RUNTIME_DIR as unset, but guard here too
  // since the loop body deletes.
  const std::string root = rpc_default_runtime_dir();
  if (root.empty())
    return;
  // Advance the iterator with an error_code (not the throwing operator++): another
  // launcher may remove_all an entry concurrently, and a throw here would abort the
  // launcher before exec. Best-effort — any filesystem error just ends the scan.
  std::error_code error;
  std::filesystem::directory_iterator it(root, error);
  const std::filesystem::directory_iterator end;
  for (; !error && it != end; it.increment(error)) {
    // Only real per-PID directories are reapable. Use symlink_status() (which does
    // NOT follow the link) and require a plain directory: is_directory() follows
    // symlinks, so a numeric symlink pointing at a directory would otherwise pass
    // and have its target remove_all'd — never chase a symlink out of the runtime
    // root.
    std::error_code status_error;
    auto status = std::filesystem::symlink_status(it->path(), status_error);
    if (status_error || status.type() != std::filesystem::file_type::directory)
      continue;
    const std::string name = it->path().filename().string();
    if (!std::all_of(name.begin(), name.end(), [](unsigned char c) { return std::isdigit(c); }))
      continue;
    pid_t pid = 0;
    auto [ptr, parse_error] = std::from_chars(name.data(), name.data() + name.size(), pid);
    if (parse_error != std::errc{} || ptr != name.data() + name.size() || pid <= 0)
      continue;
    // kill(pid, 0) probes existence without signalling: ESRCH means the process
    // is gone and its runtime dir is safe to reclaim. EPERM/success mean it is
    // still alive (possibly another user's PID), so leave it alone.
    if (kill(pid, 0) != 0 && errno == ESRCH) {
      std::error_code remove_error;
      std::filesystem::remove_all(it->path(), remove_error);
    }
  }
}

using KfdGpuOrdinal = rocjitsu::cli::VisibleGpu;

std::optional<uint32_t> parse_u32(std::string_view text) {
  uint32_t value = 0;
  auto *begin = text.data();
  auto *end = text.data() + text.size();
  auto [ptr, err] = std::from_chars(begin, end, value);
  if (err != std::errc{} || ptr != end)
    return std::nullopt;
  return value;
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

std::optional<uint64_t> read_u64_property(const std::filesystem::path &path, std::string_view key) {
  std::ifstream in(path);
  std::string name;
  uint64_t value = 0;
  while (in >> name >> value) {
    if (name == key)
      return value;
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
    uint64_t unique_id = 0;
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
      const std::filesystem::path properties = entry.path() / "properties";
      uint32_t gfx_target_version = read_u32_property(properties, "gfx_target_version").value_or(0);
      uint64_t unique_id = read_u64_property(properties, "unique_id").value_or(0);
      nodes.push_back({*node_id, *gpu_id, gfx_target_version, unique_id});
    }
  }

  std::sort(nodes.begin(), nodes.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.node_id < rhs.node_id; });

  std::vector<KfdGpuOrdinal> gpus;
  gpus.reserve(nodes.size());
  for (uint32_t ordinal = 0; ordinal < nodes.size(); ++ordinal)
    gpus.push_back({ordinal, nodes[ordinal].gpu_id, nodes[ordinal].gfx_target_version,
                    nodes[ordinal].unique_id});
  return rocjitsu::cli::enumerate_kfd_gpus(gpus);
}

std::optional<std::string_view> environment_value(const char *name) {
  const char *value = std::getenv(name);
  return value == nullptr ? std::nullopt : std::optional<std::string_view>(value);
}

/// @brief Resolve automatic host selection once for every DBT runtime layer.
bool resolve_host_gpu(rocjitsu::config::DbtGuestConfig *dbt_guest,
                      const std::vector<KfdGpuOrdinal> &topology) {
  const std::vector<KfdGpuOrdinal> visible_gpus = rocjitsu::cli::effective_visible_gpus(
      topology, environment_value("ROCR_VISIBLE_DEVICES"), environment_value("HIP_VISIBLE_DEVICES"),
      environment_value("CUDA_VISIBLE_DEVICES"));

  std::optional<uint32_t> target_version =
      rocjitsu::kmd::gfx_target_version_from_name(dbt_guest->host.isa);
  if (!target_version) {
    std::cerr << std::format("rocjitsu: unrecognized dbt_guest.host_isa '{}'\n",
                             dbt_guest->host.isa);
    return false;
  }

  const rocjitsu::cli::HostSelection selection =
      rocjitsu::cli::select_host_gpu(visible_gpus, dbt_guest->host.gpu_id, *target_version);
  if (selection.status == rocjitsu::cli::HostSelectionStatus::Selected) {
    dbt_guest->host.gpu_id = selection.gpu_id;
    return true;
  }
  if (selection.status == rocjitsu::cli::HostSelectionStatus::ExplicitGpuHidden) {
    std::cerr << std::format(
        "rocjitsu: dbt_guest.host_gpu_id {} is hidden by ROCm device visibility settings\n",
        dbt_guest->host.gpu_id);
    return false;
  }
  if (selection.status == rocjitsu::cli::HostSelectionStatus::ExplicitGpuIsaMismatch) {
    std::cerr << std::format("rocjitsu: dbt_guest.host_gpu_id {} does not match host_isa '{}'\n",
                             dbt_guest->host.gpu_id, dbt_guest->host.isa);
    return false;
  }

  std::cerr << std::format("rocjitsu: no host GPU matches dbt_guest.host_isa '{}'\n",
                           dbt_guest->host.isa);
  return false;
}

std::vector<KfdGpuOrdinal>
dbt_execution_gpu_ordinals(const std::string &dbt_config_path,
                           const rocjitsu::config::DbtGuestConfig &dbt_guest) {
  if (dbt_guest.host.backend == rocjitsu::config::DbtExecutionBackend::Hardware)
    return real_kfd_gpu_ordinals();

  const std::string host_config_path = rocjitsu::config::resolve_dbt_host_config_path(
      dbt_config_path, dbt_guest.host.simulator_config_path);
  rocjitsu::config::LoadedConfig loaded =
      rocjitsu::config::load_config(host_config_path, rocjitsu::kEmbeddedSchema);
  std::vector<rocjitsu::config::KfdDeviceConfig> devices = loaded.devices;
  if (devices.empty() && loaded.device.present)
    devices.push_back(loaded.device);

  std::vector<KfdGpuOrdinal> gpus;
  for (uint32_t ordinal = 0; ordinal < devices.size(); ++ordinal) {
    const auto &device = devices[ordinal];
    gpus.push_back({ordinal, device.gpu_id, device.gfx_target_version, device.unique_id});
  }
  return rocjitsu::cli::enumerate_kfd_gpus(gpus);
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
         "  rocjitsu --config foo.json --vfio-socket <path>\n"
         "                                               Serve a PCI device to a VMM\n"
         "\n"
         "Options:\n"
         "  --config <path>   Simulation config JSON (required)\n"
         "  --vfio-socket <path>\n"
         "                    Serve the configured GPU as a PCI function to a VMM over\n"
         "                    the vfio-user protocol on this AF_UNIX socket, instead of\n"
         "                    launching an application. Requires a build with\n"
         "                    ROCJITSU_ENABLE_VFIO.\n"
         "                    The VMM must share guest RAM through an mmap-able\n"
         "                    descriptor or the device cannot reach it; with QEMU that\n"
         "                    means -object\n"
         "                    memory-backend-memfd,id=mem,size=<N>,share=on together\n"
         "                    with -machine memory-backend=mem.\n"
         "  --version, -v     Print version and exit\n"
         "  --help, -h        Print this help and exit\n";
}

} // namespace

int main(int argc, char *argv[]) {
  std::signal(SIGPIPE, SIG_IGN);

  const char *config_path = nullptr;
  const char *vfio_socket = nullptr;
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
    } else if (arg == "--vfio-socket" && i + 1 < argc) {
      vfio_socket = argv[++i];
    } else if (arg == "--daemon") {
      daemon_mode = true;
    } else if (arg == "--attach") {
      attach_mode = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    } else if (arg == "--version" || arg == "-v") {
      std::cout << rj_get_version_string() << "\n";
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

  // Serving a VMM needs the device's identity, not a simulated machine, so this
  // is dispatched before the parse that builds one.
  if (vfio_socket != nullptr) {
    if (daemon_mode || attach_mode || (separator_idx >= 0 && separator_idx + 1 < argc)) {
      std::cerr << "rocjitsu: --vfio-socket serves a VMM and cannot be combined with "
                   "--daemon, --attach, or an application\n";
      return 1;
    }
#if defined(RJ_ENABLE_VFIO_USER)
    return rocjitsu::run_vfio_server(abs_config, vfio_socket);
#else
    std::cerr << "rocjitsu: this build has no vfio-user support; reconfigure with "
                 "-DROCJITSU_ENABLE_VFIO=ON\n";
    return 1;
#endif
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

  // Reclaim per-PID runtime dirs orphaned by prior runs (execvp never returns, so
  // those invocations could not clean up after themselves). Done for every mode,
  // including daemon-only, before this invocation creates its own directory.
  reap_stale_runtime_dirs();

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
  std::vector<KfdGpuOrdinal> dbt_execution_gpus;
  if (dbt_guest_mode) {
    if (dbt_guest_config.guest_isa.empty() || dbt_guest_config.host.isa.empty()) {
      std::cerr << "rocjitsu: dbt_guest requires guest_isa and host_isa\n";
      return 1;
    }
    try {
      dbt_execution_gpus = dbt_execution_gpu_ordinals(abs_config, dbt_guest_config);
    } catch (const std::exception &e) {
      std::cerr << std::format("rocjitsu: failed to load DBT execution topology: {}\n", e.what());
      return 1;
    }
    if (!resolve_host_gpu(&dbt_guest_config, dbt_execution_gpus))
      return 1;
    hooks_path = find_hooks_lib();
    if (hooks_path.empty()) {
      std::cerr << "rocjitsu: could not find librocjitsu_hooks.so\n";
      return 1;
    }
  }

  pid_t my_pid = getpid();

  // Set only where this process forked the daemon itself. A client authorizes
  // that process to reach into its address space, so which PID it is has to
  // come from whoever started it rather than from the socket, and this exec
  // is the last point that still knows.
  std::optional<pid_t> launched_daemon_pid;

  if (attach_mode) {
    auto sock_path = rpc_default_socket_path();
    if (!std::filesystem::exists(sock_path)) {
      std::cerr << std::format("rocjitsu: no daemon socket at {}\n", sock_path);
      return 1;
    }
  } else if (daemon_mode) {
    auto socket_path = rpc_invocation_socket_path(my_pid);
    std::error_code directory_error;
    std::filesystem::create_directories(rpc_invocation_runtime_dir(my_pid), directory_error);
    if (directory_error) {
      std::cerr << std::format("rocjitsu: failed to create runtime directory: {}\n",
                               directory_error.message());
      return 1;
    }

    int ready_pipe[2];
    if (pipe(ready_pipe) != 0) {
      std::cerr << std::format("rocjitsu: pipe failed: {}\n", strerror(errno));
      cleanup_runtime_files(my_pid);
      return 1;
    }

    pid_t daemon_pid = fork();
    if (daemon_pid < 0) {
      std::cerr << std::format("rocjitsu: fork failed: {}\n", strerror(errno));
      close(ready_pipe[0]);
      close(ready_pipe[1]);
      cleanup_runtime_files(my_pid);
      return 1;
    }

    if (daemon_pid == 0) {
      close(ready_pipe[0]);
      prctl(PR_SET_PDEATHSIG, SIGTERM);
      return run_daemon_server(abs_config.c_str(), socket_path, ready_pipe[1]);
    }

    close(ready_pipe[1]);
    pollfd ready_poll{.fd = ready_pipe[0], .events = POLLIN | POLLHUP, .revents = 0};
    int poll_result = 0;
    do {
      poll_result = poll(&ready_poll, 1, kDaemonReadyTimeoutMs);
    } while (poll_result < 0 && errno == EINTR);
    uint8_t ready = 0;
    const ssize_t ready_bytes = poll_result > 0 ? read(ready_pipe[0], &ready, sizeof(ready)) : -1;
    close(ready_pipe[0]);
    if (ready_bytes != static_cast<ssize_t>(sizeof(ready)) || ready != 1) {
      std::cerr << "rocjitsu: daemon did not become ready\n";
      kill(daemon_pid, SIGKILL);
      waitpid(daemon_pid, nullptr, 0);
      cleanup_runtime_files(my_pid);
      return 1;
    }
    launched_daemon_pid = daemon_pid;
  } else {
    if (!rocjitsu::config::write_dbt_runtime_config_handoff(abs_config, dbt_guest_config, my_pid)) {
      std::cerr << "rocjitsu: failed to write config file\n";
      cleanup_runtime_files(my_pid);
      return 1;
    }
  }

  rocjitsu::cli::LaunchEnvironment launch_environment;
  rocjitsu::cli::prepend_launch_preloads(launch_environment, lib_path);
  if (dbt_guest_mode) {
    std::optional<std::string_view> child_rocr_visible = environment_value("ROCR_VISIBLE_DEVICES");
    std::optional<std::string> expanded_rocr_visible =
        rocjitsu::cli::expanded_rocr_visible_devices(dbt_execution_gpus, child_rocr_visible);
    if (expanded_rocr_visible) {
      launch_environment.set("ROCR_VISIBLE_DEVICES", *expanded_rocr_visible);
      child_rocr_visible = *expanded_rocr_visible;
    }
    if (std::optional<rocjitsu::cli::VisibilityOverride> client_visible =
            rocjitsu::cli::normalized_client_visible_devices(
                dbt_execution_gpus, child_rocr_visible, environment_value("HIP_VISIBLE_DEVICES"),
                environment_value("CUDA_VISIBLE_DEVICES"), dbt_guest_config.host.gpu_id))
      launch_environment.set(client_visible->name, client_visible->value);
    // The HSA hook still uses the legacy tools callback path. Disable only the
    // rocprofiler-register table-delivery path so it cannot validate an
    // unshadowed table before rocjitsu installs guest-agent wrappers.
    launch_environment.set("HSA_TOOLS_DISABLE_REGISTER", "1");
    launch_environment.set("HSA_TOOLS_LIB", hooks_path);
  }
  // Export the invocation runtime dir so every descendant (including grandchild
  // processes spawned through wrappers like ctest) inherits the exact directory
  // holding config_path/daemon.sock. Attach mode creates no such dir.
  if (!attach_mode)
    launch_environment.set(rocjitsu::kRpcInvocationDirEnv, rpc_invocation_runtime_dir(my_pid));
  if (launched_daemon_pid)
    launch_environment.set(rocjitsu::kRpcDaemonPidEnv, std::to_string(*launched_daemon_pid));
  rocjitsu::cli::execvp_with_environment(app_argv[0], app_argv, launch_environment);

  std::cerr << std::format("rocjitsu: execvp failed: {}\n", strerror(errno));
  cleanup_runtime_files(my_pid);
  return 1;
}
