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
#include "rocjitsu/config/config_common.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/config/dbt_guest_config.h"
#include "rocjitsu/config/effective_config.h"
#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "rocjitsu/vm/amdgpu/partitioning.h"
#include "util/diagnostic.h"

#include "embedded_schema.h"
#include "launch_preload.h"
#include "rocm_visibility.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <csignal>
#include <cstdint>
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
#include <string>
#include <string_view>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#ifndef RJ_SOURCE_CONFIG_DIR
#define RJ_SOURCE_CONFIG_DIR ""
#endif

#ifndef RJ_INSTALL_CONFIG_DIR
#define RJ_INSTALL_CONFIG_DIR ""
#endif

#ifndef RJ_INSTALL_CONFIG_RELDIR
#define RJ_INSTALL_CONFIG_RELDIR ""
#endif

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

std::string ascii_lower(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (unsigned char c : text)
    out.push_back(static_cast<char>(std::tolower(c)));
  return out;
}

constexpr std::array<std::pair<std::string_view, std::string_view>, 8> kBuiltinProfiles = {{
    {"gfx1100_w7900", "gfx1100_w7900"},
    {"gfx1151", "gfx1151"},
    {"gfx1201_r9700", "gfx1201_r9700"},
    {"gfx1251_synthetic", "gfx1251_synthetic"},
    {"gfx90a_mi210_kmd", "gfx90a_mi210_kmd"},
    {"mi300x", "gfx942_cdna3"},
    {"mi350x", "gfx950_mi355x"},
    {"mi450x", "gfx1250_mi455x"},
}};

std::optional<std::string_view> builtin_profile_config_stem(std::string_view profile) {
  const std::string normalized = ascii_lower(profile);
  auto found = std::ranges::find_if(kBuiltinProfiles,
                                    [&](const auto &entry) { return entry.first == normalized; });
  if (found == kBuiltinProfiles.end())
    return std::nullopt;
  return found->second;
}

std::string known_profile_names() {
  std::string names;
  for (const auto &[profile, _] : kBuiltinProfiles) {
    if (!names.empty())
      names += ", ";
    names += profile;
  }
  return names;
}

void append_unique_path(std::vector<std::filesystem::path> &paths, std::filesystem::path path) {
  if (path.empty())
    return;
  std::error_code error;
  std::filesystem::path normalized = std::filesystem::weakly_canonical(path, error);
  if (!error)
    path = std::move(normalized);
  if (std::ranges::none_of(paths, [&](const auto &existing) { return existing == path; }))
    paths.push_back(std::move(path));
}

std::vector<std::filesystem::path> profile_config_search_dirs() {
  std::vector<std::filesystem::path> dirs;
  if (auto self = current_executable_path()) {
    const std::filesystem::path bin_dir = self->parent_path();
    append_unique_path(dirs, bin_dir / ".." / RJ_INSTALL_CONFIG_RELDIR);
    append_unique_path(dirs, bin_dir / ".." / "share" / "rocjitsu" / "configs");
    append_unique_path(dirs, bin_dir / ".." / ".." / "configs");
  }
  append_unique_path(dirs, RJ_INSTALL_CONFIG_DIR);
  append_unique_path(dirs, RJ_SOURCE_CONFIG_DIR);
  std::error_code cwd_error;
  if (std::filesystem::path cwd = std::filesystem::current_path(cwd_error); !cwd_error) {
    append_unique_path(dirs, cwd / "emulation" / "rocjitsu" / "configs");
    append_unique_path(dirs, cwd / "configs");
  }
  return dirs;
}

std::string path_list_for_error(const std::vector<std::filesystem::path> &paths) {
  std::string out;
  for (const auto &path : paths) {
    if (!out.empty())
      out += ", ";
    out += path.string();
  }
  return out;
}

std::optional<std::filesystem::path> resolve_profile_config(std::string_view profile,
                                                            std::string &error) {
  const auto stem = builtin_profile_config_stem(profile);
  if (!stem) {
    error = std::format("unknown profile '{}'; known profiles: {}", profile, known_profile_names());
    return std::nullopt;
  }

  const std::filesystem::path filename = std::string(*stem) + ".json";
  const std::vector<std::filesystem::path> dirs = profile_config_search_dirs();
  for (const auto &dir : dirs) {
    const std::filesystem::path candidate = dir / filename;
    std::error_code exists_error;
    if (!std::filesystem::exists(candidate, exists_error) || exists_error)
      continue;
    if (std::string canonical = canonical_existing_path(candidate); !canonical.empty())
      return std::filesystem::path(std::move(canonical));
    return candidate;
  }

  error = std::format("profile '{}' maps to {}, but that config was not found", profile,
                      filename.string());
  if (!dirs.empty())
    error += std::format("; searched: {}", path_list_for_error(dirs));
  return std::nullopt;
}

std::string json_escape(std::string_view text) {
  std::string out;
  out.reserve(text.size() + 2);
  for (unsigned char c : text) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20) {
        constexpr char kHex[] = "0123456789abcdef";
        out += "\\u00";
        out.push_back(kHex[c >> 4]);
        out.push_back(kHex[c & 0x0f]);
      } else {
        out.push_back(static_cast<char>(c));
      }
      break;
    }
  }
  return out;
}

void print_json_string(std::string_view key, std::string_view value, bool comma = true) {
  std::cout << "  \"" << key << "\": \"" << json_escape(value) << "\"";
  if (comma)
    std::cout << ",";
  std::cout << "\n";
}

void print_paths(bool json) {
  const std::vector<std::filesystem::path> config_dirs = profile_config_search_dirs();
  const std::string runtime_dir = rpc_default_runtime_dir();
  const std::string daemon_socket = rpc_default_socket_path();
  const std::string default_config_path = rpc_default_config_file_path();
  const pid_t pid = getpid();
  const std::string invocation_runtime_dir = rpc_invocation_runtime_dir(pid);
  const std::string invocation_socket = rpc_invocation_socket_path(pid);
  const std::string invocation_config_path = rpc_invocation_config_file_path(pid);

  if (json) {
    std::cout << "{\n";
    std::cout << "  \"config_search_dirs\": [\n";
    for (size_t i = 0; i < config_dirs.size(); ++i) {
      std::cout << "    \"" << json_escape(config_dirs[i].string()) << "\"";
      if (i + 1 != config_dirs.size())
        std::cout << ",";
      std::cout << "\n";
    }
    std::cout << "  ],\n";
    print_json_string("runtime_dir", runtime_dir);
    print_json_string("daemon_socket", daemon_socket);
    print_json_string("default_config_path", default_config_path);
    print_json_string("invocation_runtime_dir", invocation_runtime_dir);
    print_json_string("invocation_socket", invocation_socket);
    print_json_string("invocation_config_path", invocation_config_path, false);
    std::cout << "}\n";
    return;
  }

  std::cout << "config_search_dirs:\n";
  for (const auto &dir : config_dirs)
    std::cout << "  " << dir.string() << "\n";
  std::cout << "runtime_dir: " << runtime_dir << "\n";
  std::cout << "daemon_socket: " << daemon_socket << "\n";
  std::cout << "default_config_path: " << default_config_path << "\n";
  std::cout << "invocation_runtime_dir: " << invocation_runtime_dir << "\n";
  std::cout << "invocation_socket: " << invocation_socket << "\n";
  std::cout << "invocation_config_path: " << invocation_config_path << "\n";
}

void print_paths_usage() {
  std::cerr << "Usage: rocjitsu paths [--json]\n"
               "\n"
               "Print the config search roots and runtime paths the rocjitsu CLI uses.\n";
}

void cleanup_runtime_files(const std::filesystem::path &invocation_runtime_dir) {
  std::error_code error;
  std::filesystem::remove_all(invocation_runtime_dir, error);
}

// Best-effort reap of per-PID runtime dirs left behind by prior invocations that
// exited via execvp (which never returns, so cleanup_runtime_files does not run).
// Each numeric <pid> subdir of the runtime root is removed if that PID is no
// longer alive, so a recycled PID cannot inherit a stale config_path/daemon.sock.
void reap_stale_runtime_dirs(const std::filesystem::path &root) {
  // Never iterate an empty root: directory_iterator("") scans the CWD, which would
  // let this reaper remove_all unrelated numeric directories.
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
    if (!std::ranges::all_of(name, [](unsigned char c) { return std::isdigit(c); }))
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
    const std::optional<uint32_t> node_id = config::parse_uint32(name);
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

  std::ranges::sort(nodes, {}, &KfdNodeInfo::node_id);

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

bool append_env_override(std::string_view assignment,
                         std::vector<std::pair<std::string, std::string>> &overrides) {
  const size_t equals = assignment.find('=');
  if (equals == std::string_view::npos) {
    std::cerr << std::format("rocjitsu: --env expects KEY=VALUE, got: {}\n", assignment);
    return false;
  }
  if (equals == 0) {
    std::cerr << std::format("rocjitsu: --env key is empty in: {}\n", assignment);
    return false;
  }
  overrides.emplace_back(std::string(assignment.substr(0, equals)),
                         std::string(assignment.substr(equals + 1)));
  return true;
}

bool apply_config_option(std::string_view assignment, config::LaunchOverrides &overrides) {
  const size_t equals = assignment.find('=');
  if (equals == std::string_view::npos) {
    std::cerr << std::format("rocjitsu: --option expects KEY=VALUE, got: {}\n", assignment);
    return false;
  }
  if (equals == 0) {
    std::cerr << std::format("rocjitsu: --option key is empty in: {}\n", assignment);
    return false;
  }

  const std::string_view key = assignment.substr(0, equals);
  const std::string_view value = assignment.substr(equals + 1);
  if (key == "cpu_thread_budget") {
    overrides.cpu_thread_budget = config::parse_uint32(value);
    if (!overrides.cpu_thread_budget) {
      std::cerr << "rocjitsu: --option cpu_thread_budget requires an unsigned integer\n";
      return false;
    }
    return true;
  }
  if (key == "wait_checking") {
    overrides.wait_checking = config::parse_wait_checking(value);
    if (!overrides.wait_checking) {
      std::cerr << "rocjitsu: --option wait_checking requires on, off, or all\n";
      return false;
    }
    return true;
  }

  std::cerr << std::format(
      "rocjitsu: unknown --option key '{}'; known options: cpu_thread_budget, wait_checking\n",
      key);
  return false;
}

bool looks_like_option(std::string_view arg) {
  if (arg.starts_with("--"))
    return arg.size() > 2;
  if (!arg.starts_with("-") || arg.size() <= 1)
    return false;
  return std::isalpha(static_cast<unsigned char>(arg[1]));
}

std::string_view trim_ascii_whitespace(std::string_view text) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
    text.remove_prefix(1);
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
    text.remove_suffix(1);
  return text;
}

bool is_valid_plugin_name(std::string_view name) {
  if (name.empty())
    return false;
  for (unsigned char c : name)
    if (!std::isalnum(c) && c != '_' && c != '-')
      return false;
  return true;
}

bool validate_workdir(const std::filesystem::path &workdir) {
  std::error_code status_error;
  const std::filesystem::file_status status = std::filesystem::status(workdir, status_error);
  if (status_error) {
    if (status_error == std::errc::no_such_file_or_directory ||
        status_error == std::errc::not_a_directory) {
      std::cerr << std::format("rocjitsu: --workdir {}: directory does not exist\n",
                               workdir.string());
      return false;
    }
    std::cerr << std::format("rocjitsu: --workdir {}: {}\n", workdir.string(),
                             status_error.message());
    return false;
  }
  if (!std::filesystem::exists(status)) {
    std::cerr << std::format("rocjitsu: --workdir {}: directory does not exist\n",
                             workdir.string());
    return false;
  }
  if (!std::filesystem::is_directory(status)) {
    std::cerr << std::format("rocjitsu: --workdir {}: not a directory\n", workdir.string());
    return false;
  }
  return true;
}

void preserve_clear_environment_defaults(rocjitsu::cli::LaunchEnvironment &environment) {
  constexpr std::array<std::string_view, 8> kPreservedNames = {
      "PATH", "HOME", "TERM", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL",
  };

  for (std::string_view name : kPreservedNames) {
    const std::string key(name);
    if (std::optional<std::string_view> value = environment_value(key.c_str()))
      environment.set(key, std::string(*value));
  }
}

std::filesystem::path absolute_runtime_path(std::filesystem::path path) {
  if (path.is_relative())
    path = std::filesystem::absolute(path);
  return path;
}

std::filesystem::path runtime_dir_from_values(std::optional<std::string_view> rocjitsu_runtime_dir,
                                              std::optional<std::string_view> xdg_runtime_dir) {
  if (rocjitsu_runtime_dir && !rocjitsu_runtime_dir->empty())
    return absolute_runtime_path(std::filesystem::path(std::string(*rocjitsu_runtime_dir)));
  if (xdg_runtime_dir && !xdg_runtime_dir->empty())
    return absolute_runtime_path(std::filesystem::path(std::string(*xdg_runtime_dir)) / "rocjitsu");
  return std::filesystem::path("/tmp") / ("rocjitsu-" + std::to_string(getuid()));
}

std::filesystem::path runtime_dir_from_process_environment() {
  return runtime_dir_from_values(environment_value("ROCJITSU_RUNTIME_DIR"),
                                 environment_value("XDG_RUNTIME_DIR"));
}

std::optional<std::string_view>
launch_environment_value(const rocjitsu::cli::LaunchEnvironment &environment, const char *name) {
  const char *value = environment.get(name);
  return value == nullptr ? std::nullopt : std::optional<std::string_view>(value);
}

std::filesystem::path
runtime_dir_from_launch_environment(const rocjitsu::cli::LaunchEnvironment &environment,
                                    const std::filesystem::path &fallback_runtime_dir) {
  const std::optional<std::string_view> rocjitsu_runtime_dir =
      launch_environment_value(environment, "ROCJITSU_RUNTIME_DIR");
  const std::optional<std::string_view> xdg_runtime_dir =
      launch_environment_value(environment, "XDG_RUNTIME_DIR");
  if ((rocjitsu_runtime_dir && !rocjitsu_runtime_dir->empty()) ||
      (xdg_runtime_dir && !xdg_runtime_dir->empty()))
    return runtime_dir_from_values(rocjitsu_runtime_dir, xdg_runtime_dir);
  return fallback_runtime_dir;
}

bool set_process_runtime_dir(const std::filesystem::path &runtime_dir) {
  const std::string value = runtime_dir.string();
  if (setenv("ROCJITSU_RUNTIME_DIR", value.c_str(), 1) == 0)
    return true;
  std::cerr << std::format("rocjitsu: failed to set runtime directory: {}\n", strerror(errno));
  return false;
}

std::vector<std::string> gdb_wrapped_arguments(char *const app_argv[],
                                               const std::vector<std::string> &commands) {
  std::vector<std::string> args;
  args.emplace_back("rocgdb");
  args.emplace_back("-ex");
  args.emplace_back("set breakpoint pending on");
  for (const std::string &command : commands) {
    args.emplace_back("-ex");
    args.push_back(command);
  }
  args.emplace_back("--args");
  for (char *const *arg = app_argv; *arg != nullptr; ++arg)
    args.emplace_back(*arg);
  return args;
}

std::vector<char *> argv_pointers(std::vector<std::string> &args) {
  std::vector<char *> pointers;
  pointers.reserve(args.size() + 1);
  for (std::string &arg : args)
    pointers.push_back(arg.data());
  pointers.push_back(nullptr);
  return pointers;
}

/// @brief Resolve automatic host selection once for every DBT runtime layer.
bool resolve_host_gpu(rocjitsu::config::DbtGuestConfig *dbt_guest,
                      const std::vector<KfdGpuOrdinal> &topology,
                      std::optional<std::string_view> rocr_visible,
                      std::optional<std::string_view> hip_visible,
                      std::optional<std::string_view> cuda_visible) {
  const std::vector<KfdGpuOrdinal> visible_gpus =
      rocjitsu::cli::effective_visible_gpus(topology, rocr_visible, hip_visible, cuda_visible);

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
      << "Usage: rocjitsu [--json] [--config <config.json>|--profile <name>] "
         "[--env KEY=VALUE] [--workdir <dir>] "
         "[--exec-mode functional|clocked] [--plugin <name>]... [--gpus-per-node <n>] "
         "[--daemon|--attach] [--] <app> [args...]\n"
         "\n"
         "Modes:\n"
         "  rocjitsu --config foo.json -- ./app          Local mode (in-process simulation)\n"
         "  rocjitsu --config foo.json ./app             Same launch without a separator\n"
         "  rocjitsu --profile mi350x -- ./app          Run with a bundled profile config\n"
         "  rocjitsu --exec-mode clocked -- ./app        Override the config execution mode\n"
         "  rocjitsu --gpus-per-node 2 -- ./app         Override the emulated GPU count\n"
         "  rocjitsu --daemon --config foo.json -- ./app Daemon mode (fork daemon + launch app)\n"
         "  rocjitsu --daemon --config foo.json          Daemon-only (run server)\n"
         "  rocjitsu --attach --config foo.json -- ./app Attach to running daemon\n"
         "  rocjitsu --in-process --config foo.json -- ./app\n"
         "                                               Explicit local mode\n"
         "  rocjitsu --config foo.json --vfio-socket <path>\n"
         "                                               Serve a PCI device to a VMM\n"
         "  rocjitsu [--json] paths                      Print config/runtime paths\n"
         "\n"
         "Options:\n"
         "  --config <path>   Simulation config JSON. Cannot be combined with --profile.\n"
         "  --profile <name>  Bundled profile config to run. Defaults to mi350x.\n"
         "  --exec-mode <mode>\n"
         "                    Override the top-level config exec_mode: functional or clocked.\n"
         "  --plugin <name>   Enable an execution plugin with its default config. Repeatable.\n"
         "  --gpus-per-node <n>\n"
         "                    Override the emulated GPU count for this launch.\n"
         "  --env KEY=VALUE  Add or replace an environment variable for the launched app.\n"
         "  --clear-env-vars Start the launched app with a clean environment, preserving\n"
         "                    only basic shell/session variables before applying --env.\n"
         "  --workdir <dir>  Run the launched app from this existing directory.\n"
         "  --gdb           Run the launched app under rocgdb.\n"
         "  --gdb-ex <cmd>  Extra rocgdb command to run before the program. May be repeated\n"
         "                    and implies --gdb.\n"
         "  --option KEY=VALUE, -o KEY=VALUE\n"
         "                    Override a supported top-level config value for this launch.\n"
         "  --cpu-thread-budget <n>, --cpu-thread-budget=<n>\n"
         "                    Replace cpu_thread_budget for this launch only; the\n"
         "                    config file is left unchanged. Zero selects automatic\n"
         "                    sizing. Not available with --attach.\n"
         "  --in-process     Run in local in-process mode. This is the default and\n"
         "                    cannot be combined with --daemon or --attach.\n"
         "  --wait-checking <mode>, --wait-checking=<mode>\n"
         "                    Override every CU and GPU for this launch:\n"
         "                    on = ordinary checks, off = no wait checks,\n"
         "                    all = ordinary plus gfx1250 XCNT.\n"
         "                    Omission keeps JSON settings and defaults. The source\n"
         "                    config is unchanged. Not available with --attach.\n"
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
         "  --thread-budget-table\n"
         "                    Print engine / dispatch allocations without running a VM\n"
         "  --check-vfio-user Report whether this binary includes VFIO-user support\n"
         "  --json            Emit machine-readable JSON where available\n"
         "  --version, -v     Print version and exit\n"
         "  --help, -h        Print this help and exit\n";
}

} // namespace

int main(int argc, char *argv[]) {
  std::signal(SIGPIPE, SIG_IGN);

  bool json_output = false;
  int command_arg = 1;
  while (command_arg < argc && std::string_view(argv[command_arg]) == "--json") {
    json_output = true;
    ++command_arg;
  }

  if (command_arg < argc && std::string_view(argv[command_arg]) == "paths") {
    bool json = json_output;
    for (int i = command_arg + 1; i < argc; ++i) {
      std::string_view arg(argv[i]);
      if (arg == "--json") {
        json = true;
      } else if (arg == "--help" || arg == "-h") {
        print_paths_usage();
        return 0;
      } else {
        std::cerr << std::format("rocjitsu: unknown paths option: {}\n", arg);
        print_paths_usage();
        return 1;
      }
    }
    print_paths(json);
    return 0;
  }

  const char *config_path = nullptr;
  const char *profile_name = nullptr;
  const char *vfio_socket = nullptr;
  std::vector<std::pair<std::string, std::string>> env_overrides;
  std::vector<std::string> gdb_commands;
  std::optional<std::filesystem::path> workdir;
  bool thread_budget_table = false;
  config::LaunchOverrides overrides;
  config::LaunchOverrides flag_overrides;
  int vfio_ready_fd = -1;
  bool daemon_mode = false;
  bool attach_mode = false;
  bool in_process_mode = false;
  bool clear_env_vars = false;
  bool gdb = false;
  int separator_idx = -1;
  for (int i = command_arg; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--") {
      separator_idx = i;
      break;
    }
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else if (arg == "--profile" && i + 1 < argc) {
      profile_name = argv[++i];
    } else if (arg == "--exec-mode" || arg.starts_with("--exec-mode=")) {
      std::string_view mode;
      if (arg == "--exec-mode") {
        if (i + 1 >= argc) {
          std::cerr << "rocjitsu: --exec-mode requires functional or clocked\n";
          return 1;
        }
        mode = argv[++i];
      } else {
        mode = arg.substr(std::string_view("--exec-mode=").size());
      }
      if (mode != "functional" && mode != "clocked") {
        std::cerr << "rocjitsu: --exec-mode requires functional or clocked\n";
        return 1;
      }
      overrides.exec_mode = std::string(mode);
    } else if (arg == "--gpus-per-node" || arg.starts_with("--gpus-per-node=")) {
      std::string_view count_text;
      if (arg == "--gpus-per-node") {
        if (i + 1 >= argc) {
          std::cerr << "rocjitsu: --gpus-per-node requires a positive integer\n";
          return 1;
        }
        count_text = argv[++i];
      } else {
        count_text = arg.substr(std::string_view("--gpus-per-node=").size());
      }
      std::optional<uint32_t> count = config::parse_uint32(count_text);
      if (!count || *count == 0) {
        std::cerr << "rocjitsu: --gpus-per-node requires a positive integer\n";
        return 1;
      }
      overrides.gpus_per_node = *count;
    } else if (arg == "--env" || arg.starts_with("--env=")) {
      std::string_view assignment;
      if (arg == "--env") {
        if (i + 1 >= argc) {
          std::cerr << "rocjitsu: --env requires KEY=VALUE\n";
          return 1;
        }
        assignment = argv[++i];
      } else {
        assignment = arg.substr(std::string_view("--env=").size());
      }
      if (!append_env_override(assignment, env_overrides))
        return 1;
    } else if (arg == "--option" || arg == "-o" || arg.starts_with("--option=") ||
               (arg.starts_with("-o") && arg.size() > 2)) {
      std::string_view assignment;
      if (arg == "--option" || arg == "-o") {
        if (i + 1 >= argc) {
          std::cerr << "rocjitsu: --option requires KEY=VALUE\n";
          return 1;
        }
        assignment = argv[++i];
      } else if (arg.starts_with("--option=")) {
        assignment = arg.substr(std::string_view("--option=").size());
      } else {
        assignment = arg.substr(2);
      }
      if (!apply_config_option(assignment, overrides))
        return 1;
    } else if (arg == "--plugin" || arg.starts_with("--plugin=")) {
      std::string_view plugin;
      if (arg == "--plugin") {
        if (i + 1 >= argc) {
          std::cerr << "rocjitsu: --plugin requires a plugin name\n";
          return 1;
        }
        plugin = argv[++i];
      } else {
        plugin = arg.substr(std::string_view("--plugin=").size());
      }
      plugin = trim_ascii_whitespace(plugin);
      if (!is_valid_plugin_name(plugin)) {
        std::cerr << std::format("rocjitsu: invalid plugin name '{}'; allowed: letters, digits, "
                                 "'_', '-'\n",
                                 plugin);
        return 1;
      }
      overrides.plugins.emplace_back(plugin);
    } else if (arg == "--clear-env-vars") {
      clear_env_vars = true;
    } else if (arg == "--gdb") {
      gdb = true;
    } else if (arg == "--gdb-ex" || arg.starts_with("--gdb-ex=")) {
      if (arg == "--gdb-ex") {
        if (i + 1 >= argc) {
          std::cerr << "rocjitsu: --gdb-ex requires a command\n";
          return 1;
        }
        gdb_commands.emplace_back(argv[++i]);
      } else {
        gdb_commands.emplace_back(arg.substr(std::string_view("--gdb-ex=").size()));
      }
    } else if (arg == "--workdir" || arg.starts_with("--workdir=")) {
      if (arg == "--workdir") {
        if (i + 1 >= argc) {
          std::cerr << "rocjitsu: --workdir requires a directory\n";
          return 1;
        }
        workdir = argv[++i];
      } else {
        workdir =
            std::filesystem::path(std::string(arg.substr(std::string_view("--workdir=").size())));
      }
    } else if (arg == "--cpu-thread-budget" || arg.starts_with("--cpu-thread-budget=")) {
      // Both spellings are accepted because wrappers that forward a single opaque
      // token cannot pass a flag and its value as two arguments.
      const size_t equals = arg.find('=');
      std::string_view value;
      if (equals != std::string_view::npos) {
        value = arg.substr(equals + 1);
      } else if (i + 1 < argc) {
        value = argv[++i];
      } else {
        std::cerr << "rocjitsu: --cpu-thread-budget requires a value\n";
        return 1;
      }
      flag_overrides.cpu_thread_budget = config::parse_uint32(value);
      if (!flag_overrides.cpu_thread_budget) {
        std::cerr << "rocjitsu: --cpu-thread-budget requires an unsigned integer\n";
        return 1;
      }
    } else if (arg == "--wait-checking" || arg.starts_with("--wait-checking=")) {
      const size_t equals = arg.find('=');
      std::string_view value;
      if (equals != std::string_view::npos)
        value = arg.substr(equals + 1);
      else if (i + 1 < argc)
        value = argv[++i];
      flag_overrides.wait_checking = config::parse_wait_checking(value);
      if (!flag_overrides.wait_checking) {
        std::cerr << "rocjitsu: --wait-checking requires on, off, or all\n";
        return 1;
      }
    } else if (arg == "--vfio-socket" && i + 1 < argc) {
      vfio_socket = argv[++i];
    } else if (arg == "--thread-budget-table") {
      thread_budget_table = true;
    } else if (arg == "--vfio-ready-fd" && i + 1 < argc) {
      std::string_view value(argv[++i]);
      auto [ptr, error] = std::from_chars(value.data(), value.data() + value.size(), vfio_ready_fd);
      if (error != std::errc{} || ptr != value.data() + value.size() || vfio_ready_fd < 0) {
        std::cerr << "rocjitsu: --vfio-ready-fd requires a nonnegative descriptor\n";
        return 1;
      }
    } else if (arg == "--daemon") {
      daemon_mode = true;
    } else if (arg == "--attach") {
      attach_mode = true;
    } else if (arg == "--in-process") {
      in_process_mode = true;
    } else if (arg == "--check-vfio-user") {
#if defined(RJ_ENABLE_VFIO_USER)
      std::cout << "vfio-user support enabled\n";
      return 0;
#else
      std::cerr << "rocjitsu: this build has no vfio-user support; reconfigure with "
                   "-DROCJITSU_ENABLE_VFIO=ON\n";
      return 1;
#endif
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    } else if (arg == "--version" || arg == "-v") {
      std::cout << rj_get_version_string() << "\n";
      return 0;
    } else if (!looks_like_option(arg)) {
      separator_idx = i - 1;
      break;
    } else {
      std::cerr << std::format("rocjitsu: unknown option: {}\n", arg);
      print_usage();
      return 1;
    }
  }

  if (flag_overrides.cpu_thread_budget)
    overrides.cpu_thread_budget = flag_overrides.cpu_thread_budget;
  if (flag_overrides.wait_checking)
    overrides.wait_checking = flag_overrides.wait_checking;

  if (config_path && profile_name) {
    std::cerr << "rocjitsu: --config and --profile cannot be combined\n";
    return 1;
  }
  if (in_process_mode && (daemon_mode || attach_mode)) {
    std::cerr << "rocjitsu: --in-process cannot be combined with --daemon or --attach\n";
    return 1;
  }

  const bool has_app = (separator_idx >= 0 && separator_idx + 1 < argc);
  const bool launch_environment_options =
      clear_env_vars || !env_overrides.empty() || workdir.has_value();
  const bool gdb_requested = gdb || !gdb_commands.empty();
  if (thread_budget_table && (launch_environment_options || gdb_requested)) {
    std::cerr << "rocjitsu: --thread-budget-table cannot be combined with launch-only options\n";
    return 1;
  }
  if (!has_app && launch_environment_options) {
    std::cerr << "rocjitsu: launch environment options require an application after --\n";
    return 1;
  }
  if (!has_app && gdb_requested) {
    std::cerr << "rocjitsu: rocgdb options require an application after --\n";
    return 1;
  }
  if (gdb_requested && !attach_mode)
    daemon_mode = true;
  if (workdir && !validate_workdir(*workdir))
    return 1;

  std::string abs_config;
  if (config_path) {
    abs_config = std::filesystem::absolute(config_path).string();
  } else {
    const std::string_view selected_profile =
        profile_name ? std::string_view(profile_name) : "mi350x";
    std::string profile_error;
    std::optional<std::filesystem::path> profile_config =
        resolve_profile_config(selected_profile, profile_error);
    if (!profile_config) {
      std::cerr << std::format("rocjitsu: {}\n", profile_error);
      return 1;
    }
    abs_config = std::filesystem::absolute(*profile_config).string();
  }
  if (!std::filesystem::exists(abs_config)) {
    std::cerr << std::format("rocjitsu: config file not found: {}\n", abs_config);
    return 1;
  }

  // Attaching joins a daemon that already built its machine from its own config,
  // so these overrides cannot change it.
  if (overrides.cpu_thread_budget && attach_mode) {
    std::cerr << "rocjitsu: --cpu-thread-budget cannot be combined with --attach\n";
    return 1;
  }
  if (overrides.exec_mode && attach_mode) {
    std::cerr << "rocjitsu: --exec-mode cannot be combined with --attach\n";
    return 1;
  }
  if (overrides.gpus_per_node && attach_mode) {
    std::cerr << "rocjitsu: --gpus-per-node cannot be combined with --attach\n";
    return 1;
  }
  if (!overrides.plugins.empty() && attach_mode) {
    std::cerr << "rocjitsu: --plugin cannot be combined with --attach\n";
    return 1;
  }
  if (overrides.wait_checking && attach_mode) {
    std::cerr << "rocjitsu: --wait-checking cannot be combined with --attach\n";
    return 1;
  }

  // Applying overrides to a copy keeps them inside this launch: the
  // config the user named is never rewritten, and simulations built from other
  // configs in the same process tree keep their own settings.
  util::StringDiagnostic launch_config_diagnostic;
  auto config_json_for_this_launch = [&]() -> rocjitsu::FailureOr<std::string> {
    std::string json = rocjitsu::config::read_config_file(abs_config);
    if (overrides.empty())
      return json;
    return rocjitsu::config::json_with_launch_overrides(json, overrides,
                                                        launch_config_diagnostic.emitter());
  };

  if (thread_budget_table) {
    if (vfio_socket || vfio_ready_fd >= 0 || daemon_mode || attach_mode || separator_idx >= 0) {
      std::cerr << "rocjitsu: --thread-budget-table cannot be combined with a launch mode\n";
      return 1;
    }
    try {
      rocjitsu::FailureOr<std::string> json = config_json_for_this_launch();
      if (json.failed()) {
        std::cerr << std::format("rocjitsu: thread allocation failed: {}\n",
                                 launch_config_diagnostic.message());
        return 1;
      }
      auto settings = rocjitsu::config::load_execution_thread_settings_from_string(
          json.value(), rocjitsu::kEmbeddedSchema);
      const uint32_t host = rocjitsu::amdgpu::available_host_threads();
      std::cout
          << "Budget | num_threads | cpu_dispatch_threads per GPU | async_helper_threads | Total\n";
      auto print = [&](const std::string &label) {
        const auto plan = settings.resolve(host);
        uint64_t total = uint64_t{plan.engines} + plan.helpers;
        std::string dispatch;
        for (uint32_t width : plan.dispatch) {
          if (!dispatch.empty())
            dispatch += ",";
          dispatch += std::to_string(width);
          total += width - 1;
        }
        std::cout << std::format("{} | {} | {} | {} | {}\n", label, plan.engines, dispatch,
                                 plan.helpers, total);
      };
      print("Configured");
      for (uint32_t budget : {1u, 2u, 4u, 8u, 12u, 16u, 24u, 32u, 34u, 36u, 40u, 48u, 64u}) {
        settings.request.budget = budget;
        print(std::to_string(budget));
      }
      return 0;
    } catch (const std::exception &e) {
      std::cerr << std::format("rocjitsu: thread allocation failed: {}\n", e.what());
      return 1;
    }
  }

  rocjitsu::cli::LaunchEnvironment launch_environment(!clear_env_vars);
  if (clear_env_vars)
    preserve_clear_environment_defaults(launch_environment);
  for (const auto &[name, value] : env_overrides)
    launch_environment.set(name, value);

  const std::filesystem::path launcher_runtime_dir = runtime_dir_from_process_environment();
  if (!set_process_runtime_dir(launcher_runtime_dir))
    return 1;

  const pid_t my_pid = getpid();
  const std::filesystem::path invocation_runtime_dir =
      launcher_runtime_dir / std::to_string(my_pid);

  // Reclaim per-PID runtime dirs orphaned by prior runs (execvp never returns, so
  // those invocations could not clean up after themselves). Done for every launch
  // mode, before this invocation creates its own directory.
  reap_stale_runtime_dirs(launcher_runtime_dir);

  // The copy lives in this invocation's runtime directory, beside the config-path
  // handoff an exec'd workload already reads, so the same cleanup reclaims it.
  auto write_config_for_this_launch = [&]() {
    if (overrides.empty())
      return true;
    util::StringDiagnostic diagnostic;
    rocjitsu::FailureOr<std::string> written = rocjitsu::config::write_effective_config(
        abs_config, overrides, my_pid, diagnostic.emitter());
    if (written.failed()) {
      std::cerr << std::format("rocjitsu: {}\n", diagnostic.message());
      cleanup_runtime_files(invocation_runtime_dir);
      return false;
    }
    abs_config = std::move(written.value());
    return true;
  };

  // Serving a VMM builds its own machine, with the PCI function inside it, so
  // this is dispatched before the parse that builds one here -- not because no
  // machine is needed, but because the one it needs is assembled differently.
  if (vfio_socket != nullptr) {
    if (daemon_mode || attach_mode || (separator_idx >= 0 && separator_idx + 1 < argc)) {
      std::cerr << "rocjitsu: --vfio-socket serves a VMM and cannot be combined with "
                   "--daemon, --attach, or an application\n";
      return 1;
    }
    if (!write_config_for_this_launch())
      return 1;
#if defined(RJ_ENABLE_VFIO_USER)
    return rocjitsu::run_vfio_server(abs_config, vfio_socket, vfio_ready_fd);
#else
    std::cerr << "rocjitsu: this build has no vfio-user support; reconfigure with "
                 "-DROCJITSU_ENABLE_VFIO=ON\n";
    return 1;
#endif
  }

  if (vfio_ready_fd >= 0) {
    std::cerr << "rocjitsu: --vfio-ready-fd requires --vfio-socket\n";
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
  // A dbt_guest simulator_config is resolved relative to the config naming it, and
  // it, not this config, is what the host VM is built from. Moving overrides onto
  // a copy would therefore both break that path and miss the config they must reach.
  const bool dbt_guest_delegates_to_simulator =
      dbt_guest_mode && !dbt_guest_config.host.simulator_config_path.empty();
  if (dbt_guest_delegates_to_simulator && overrides.exec_mode) {
    std::cerr << std::format("rocjitsu: --exec-mode cannot be combined with a dbt_guest "
                             "simulator_config; set exec_mode in {} instead\n",
                             dbt_guest_config.host.simulator_config_path);
    return 1;
  }
  if (dbt_guest_delegates_to_simulator && overrides.gpus_per_node) {
    std::cerr << std::format("rocjitsu: --gpus-per-node cannot be combined with a dbt_guest "
                             "simulator_config; set vm.gpu.num_gpus in {} instead\n",
                             dbt_guest_config.host.simulator_config_path);
    return 1;
  }
  if (dbt_guest_delegates_to_simulator && !overrides.plugins.empty()) {
    std::cerr << std::format("rocjitsu: --plugin cannot be combined with a dbt_guest "
                             "simulator_config; set plugins in {} instead\n",
                             dbt_guest_config.host.simulator_config_path);
    return 1;
  }
  if (dbt_guest_delegates_to_simulator && overrides.cpu_thread_budget) {
    std::cerr << std::format("rocjitsu: --cpu-thread-budget cannot be combined with a dbt_guest "
                             "simulator_config; set cpu_thread_budget in {} instead\n",
                             dbt_guest_config.host.simulator_config_path);
    return 1;
  }
  if (dbt_guest_delegates_to_simulator && overrides.wait_checking) {
    std::cerr << std::format("rocjitsu: --wait-checking cannot be combined with a dbt_guest "
                             "simulator_config; set wait_checking in {} instead\n",
                             dbt_guest_config.host.simulator_config_path);
    return 1;
  }

  if (!write_config_for_this_launch())
    return 1;

  if (daemon_mode && !has_app)
    return run_daemon_server(abs_config.c_str());

  if (!has_app) {
    std::cerr << "rocjitsu: no application specified after --\n";
    print_usage();
    return 1;
  }

  char **app_argv = &argv[separator_idx + 1];
  std::vector<std::string> gdb_argv_storage;
  std::vector<char *> gdb_argv;
  if (gdb_requested) {
    gdb_argv_storage = gdb_wrapped_arguments(app_argv, gdb_commands);
    gdb_argv = argv_pointers(gdb_argv_storage);
    app_argv = gdb_argv.data();
  }

  auto lib_path = find_interposer_lib();
  if (lib_path.empty()) {
    std::cerr << "rocjitsu: could not find librocjitsu.so\n";
    return 1;
  }
  rocjitsu::cli::prepend_launch_preloads(launch_environment, lib_path);

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
    std::optional<std::string_view> child_rocr_visible =
        launch_environment_value(launch_environment, "ROCR_VISIBLE_DEVICES");
    std::optional<std::string> expanded_rocr_visible =
        rocjitsu::cli::expanded_rocr_visible_devices(dbt_execution_gpus, child_rocr_visible);
    if (expanded_rocr_visible) {
      launch_environment.set("ROCR_VISIBLE_DEVICES", *expanded_rocr_visible);
      child_rocr_visible = *expanded_rocr_visible;
    }
    const std::optional<std::string_view> child_hip_visible =
        launch_environment_value(launch_environment, "HIP_VISIBLE_DEVICES");
    const std::optional<std::string_view> child_cuda_visible =
        launch_environment_value(launch_environment, "CUDA_VISIBLE_DEVICES");

    if (!resolve_host_gpu(&dbt_guest_config, dbt_execution_gpus, child_rocr_visible,
                          child_hip_visible, child_cuda_visible))
      return 1;
    if (std::optional<rocjitsu::cli::VisibilityOverride> client_visible =
            rocjitsu::cli::normalized_client_visible_devices(dbt_execution_gpus, child_rocr_visible,
                                                             child_hip_visible, child_cuda_visible,
                                                             dbt_guest_config.host.gpu_id))
      launch_environment.set(client_visible->name, client_visible->value);

    hooks_path = find_hooks_lib();
    if (hooks_path.empty()) {
      std::cerr << "rocjitsu: could not find librocjitsu_hooks.so\n";
      return 1;
    }
    // The HSA hook still uses the legacy tools callback path. Disable the
    // rocprofiler-register table-delivery path so it cannot validate an
    // unshadowed table before rocjitsu installs guest-agent wrappers. Also
    // exclude the overlapping automatic HotSwap hook across ROCr generations.
    rocjitsu::cli::configure_dbt_guest_tool_environment(launch_environment, hooks_path);
  }

  // Set only where this process forked the daemon itself. A client authorizes
  // that process to reach into its address space, so which PID it is has to
  // come from whoever started it rather than from the socket, and this exec
  // is the last point that still knows.
  std::optional<pid_t> launched_daemon_pid;
  std::optional<std::filesystem::path> attach_runtime_dir;

  if (attach_mode) {
    attach_runtime_dir =
        runtime_dir_from_launch_environment(launch_environment, launcher_runtime_dir);
    const std::filesystem::path sock_path = *attach_runtime_dir / "daemon.sock";
    if (!std::filesystem::exists(sock_path)) {
      std::cerr << std::format("rocjitsu: no daemon socket at {}\n", sock_path.string());
      return 1;
    }
  } else if (daemon_mode) {
    const std::string socket_path = (invocation_runtime_dir / "daemon.sock").string();
    std::error_code directory_error;
    std::filesystem::create_directories(invocation_runtime_dir, directory_error);
    if (directory_error) {
      std::cerr << std::format("rocjitsu: failed to create runtime directory: {}\n",
                               directory_error.message());
      return 1;
    }

    int ready_pipe[2];
    if (pipe(ready_pipe) != 0) {
      std::cerr << std::format("rocjitsu: pipe failed: {}\n", strerror(errno));
      cleanup_runtime_files(invocation_runtime_dir);
      return 1;
    }

    pid_t daemon_pid = fork();
    if (daemon_pid < 0) {
      std::cerr << std::format("rocjitsu: fork failed: {}\n", strerror(errno));
      close(ready_pipe[0]);
      close(ready_pipe[1]);
      cleanup_runtime_files(invocation_runtime_dir);
      return 1;
    }

    if (daemon_pid == 0) {
      close(ready_pipe[0]);
      prctl(PR_SET_PDEATHSIG, SIGTERM);
      if (gdb_requested && setpgid(0, 0) != 0) {
        std::cerr << std::format("rocjitsu: failed to detach daemon process group: {}\n",
                                 strerror(errno));
        close(ready_pipe[1]);
        return 1;
      }
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
      cleanup_runtime_files(invocation_runtime_dir);
      return 1;
    }
    launched_daemon_pid = daemon_pid;
  } else {
    if (!rocjitsu::config::write_dbt_runtime_config_handoff(abs_config, dbt_guest_config, my_pid)) {
      std::cerr << "rocjitsu: failed to write config file\n";
      cleanup_runtime_files(invocation_runtime_dir);
      return 1;
    }
  }

  // Export the invocation runtime dir so every descendant (including grandchild
  // processes spawned through wrappers like ctest) inherits the exact directory
  // holding config_path/daemon.sock. Attach mode exports the resolved daemon
  // runtime directory so a cleared or changed child environment still reaches
  // the socket the launcher validated.
  if (attach_mode)
    launch_environment.set(rocjitsu::kRpcInvocationDirEnv, attach_runtime_dir->string());
  else
    launch_environment.set(rocjitsu::kRpcInvocationDirEnv, invocation_runtime_dir.string());
  if (launched_daemon_pid)
    launch_environment.set(rocjitsu::kRpcDaemonPidEnv, std::to_string(*launched_daemon_pid));
  if (workdir && chdir(workdir->c_str()) != 0) {
    std::cerr << std::format("rocjitsu: failed to enter --workdir {}: {}\n", workdir->string(),
                             strerror(errno));
    cleanup_runtime_files(invocation_runtime_dir);
    return 1;
  }
  rocjitsu::cli::execvp_with_environment(app_argv[0], app_argv, launch_environment);

  std::cerr << std::format("rocjitsu: execvp failed: {}\n", strerror(errno));
  cleanup_runtime_files(invocation_runtime_dir);
  return 1;
}
