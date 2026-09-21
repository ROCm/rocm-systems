// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/guest_kfd.h"

#include "rocjitsu/kmd/linux/amdgpu_properties.h"
#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"
#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include "util/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <dirent.h>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <new>
#include <optional>
#include <sstream>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

namespace fs = std::filesystem;

constexpr std::string_view kRealTopologyPaths[] = {
    "/sys/devices/virtual/kfd/kfd/topology",
    "/sys/class/kfd/kfd/topology",
};
constexpr uint64_t kGuestSyntheticMmapType = 0x1ULL << KFD_MMAP_TYPE_SHIFT;
constexpr uint64_t kGuestSyntheticMmapPayloadMask = (1ULL << KFD_MMAP_TYPE_SHIFT) - 1;

std::string read_text_file(const fs::path &path) {
  const std::string path_str = path.string();
  auto &real = libc_passthrough();
  int fd = real.openat(AT_FDCWD, path_str.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return {};

  std::string out;
  std::array<char, 4096> buffer{};
  for (;;) {
    ssize_t n = real.read(fd, buffer.data(), buffer.size());
    if (n == 0)
      break;
    if (n < 0) {
      real.close(fd);
      return {};
    }
    out.append(buffer.data(), static_cast<size_t>(n));
  }
  real.close(fd);
  return out;
}

void write_text_file(const fs::path &path, const std::string &text) {
  const std::string path_str = path.string();
  auto &real = libc_passthrough();
  int fd = real.openat(AT_FDCWD, path_str.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0)
    return;

  const char *cursor = text.data();
  size_t remaining = text.size();
  while (remaining > 0) {
    ssize_t written = real.write(fd, cursor, remaining);
    if (written <= 0)
      break;
    cursor += written;
    remaining -= static_cast<size_t>(written);
  }
  real.close(fd);
}

uint32_t read_u32_property(const fs::path &path, std::string_view key, uint32_t fallback) {
  std::istringstream input(read_text_file(path));
  std::string name;
  uint64_t value = 0;
  while (input >> name >> value) {
    if (name == key)
      return static_cast<uint32_t>(value);
  }
  return fallback;
}

std::string set_property(std::string text, std::string_view key, uint64_t value) {
  std::istringstream input(text);
  std::ostringstream output;
  std::string line;
  bool replaced = false;
  while (std::getline(input, line)) {
    if (line.starts_with(key) && line.size() > key.size() && line[key.size()] == ' ') {
      output << key << " " << value << "\n";
      replaced = true;
    } else {
      output << line << "\n";
    }
  }
  if (!replaced)
    output << key << " " << value << "\n";
  return output.str();
}

bool make_temp_dir(const char *prefix, std::string *out) {
  std::error_code ec;
  fs::path base = rpc_default_runtime_dir();
  fs::create_directories(base, ec);
  if (ec)
    return false;

  // Keep generated overlay trees under the rocjitsu runtime directory instead
  // of raw /tmp. That gives test/daemon callers one directory to clean and
  // avoids process-private synthetic sysfs trees accumulating at /tmp top-level.
  std::string tmpl = (base / (std::string(prefix) + "_XXXXXX")).string();
  std::vector<char> tmpl_buffer(tmpl.begin(), tmpl.end());
  tmpl_buffer.push_back('\0');
  char *dir = mkdtemp(tmpl_buffer.data());
  if (!dir)
    return false;
  *out = dir;
  return true;
}

uint32_t count_numeric_dirs(const fs::path &dir) {
  uint32_t count = 0;
  std::error_code ec;
  for (const auto &entry : fs::directory_iterator(dir, ec)) {
    if (!entry.is_directory(ec))
      continue;
    auto name = entry.path().filename().string();
    uint32_t ignored = 0;
    auto [ptr, err] = std::from_chars(name.data(), name.data() + name.size(), ignored);
    if (err == std::errc{} && ptr == name.data() + name.size())
      ++count;
  }
  return count;
}

std::optional<uint32_t> read_u32_file(const fs::path &path) {
  std::istringstream in(read_text_file(path));
  uint32_t value = 0;
  if (!(in >> value))
    return std::nullopt;
  return value;
}

void append_unique_gpu_id(std::vector<uint32_t> *ids, uint32_t gpu_id) {
  if (std::find(ids->begin(), ids->end(), gpu_id) == ids->end())
    ids->push_back(gpu_id);
}

uint32_t max_numeric_dir(const fs::path &dir) {
  uint32_t max_id = 0;
  const std::string dir_str = dir.string();
  auto &real = libc_passthrough();
  // Host validation can run with GuestKfd::mutex_ held. Going through the
  // interposed directory functions for real sysfs would re-enter
  // redirect_sysfs_path(), which takes the same mutex.
  DIR *stream = real.opendir(dir_str.c_str());
  if (!stream)
    return max_id;

  while (struct dirent *entry = real.readdir(stream)) {
    std::string_view name(entry->d_name);
    const std::string child = (dir / name).string();
    struct stat st {};
    if (real.stat(child.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))
      continue;
    uint32_t id = 0;
    auto [ptr, err] = std::from_chars(name.data(), name.data() + name.size(), id);
    if (err == std::errc{} && ptr == name.data() + name.size())
      max_id = std::max(max_id, id);
  }
  real.closedir(stream);
  return max_id;
}

bool gpu_id_matches_isa_in_topology(const fs::path &topology_root, uint32_t gpu_id,
                                    std::string_view host_isa) {
  std::optional<uint32_t> target_version = kmd::gfx_target_version_from_name(host_isa);
  if (!target_version)
    return false;

  fs::path nodes_dir = topology_root / "nodes";
  const uint32_t max_node = max_numeric_dir(nodes_dir);
  for (uint32_t node_id = 1; node_id <= max_node; ++node_id) {
    fs::path node_dir = nodes_dir / std::to_string(node_id);
    if (read_u32_file(node_dir / "gpu_id") != gpu_id)
      continue;
    return read_u32_property(node_dir / "properties", "gfx_target_version", 0) == *target_version;
  }
  return false;
}

bool real_gpu_id_matches_isa(uint32_t gpu_id, std::string_view host_isa) {
  for (std::string_view topology_path : kRealTopologyPaths) {
    if (gpu_id_matches_isa_in_topology(fs::path(topology_path), gpu_id, host_isa))
      return true;
  }
  return false;
}

std::string io_link_to_cpu(uint32_t node_from) {
  std::ostringstream link;
  link << "type 2\n"
       << "version_major 0\n"
       << "version_minor 0\n"
       << "node_from " << node_from << "\n"
       << "node_to 0\n"
       << "weight 20\n"
       << "min_latency 0\n"
       << "max_latency 0\n"
       << "min_bandwidth 0\n"
       << "max_bandwidth 0\n"
       << "recommended_transfer_size 0\n"
       << "num_hops 1\n"
       << "flags 1\n";
  return link.str();
}

std::string io_link_from_cpu(uint32_t node_to) {
  std::ostringstream link;
  link << "type 2\n"
       << "version_major 0\n"
       << "version_minor 0\n"
       << "node_from 0\n"
       << "node_to " << node_to << "\n"
       << "weight 20\n"
       << "min_latency 0\n"
       << "max_latency 0\n"
       << "min_bandwidth 0\n"
       << "max_bandwidth 0\n"
       << "recommended_transfer_size 0\n"
       << "num_hops 1\n"
       << "flags 1\n";
  return link.str();
}

uint32_t choose_render_minor(uint32_t requested) {
  constexpr uint32_t kRenderMinorBegin = 128;
  constexpr uint32_t kRenderMinorEnd = 192;
  auto available = [](uint32_t minor) {
    return !fs::exists("/sys/class/drm/renderD" + std::to_string(minor)) &&
           !fs::exists("/dev/dri/renderD" + std::to_string(minor));
  };

  if (requested >= kRenderMinorBegin && requested < kRenderMinorEnd && available(requested))
    return requested;
  for (uint32_t minor = kRenderMinorBegin; minor < kRenderMinorEnd; ++minor) {
    if (available(minor))
      return minor;
  }
  return (requested >= kRenderMinorBegin && requested < kRenderMinorEnd) ? requested
                                                                         : kRenderMinorEnd - 1;
}

uint64_t synthetic_mmap_offset_for_handle(uint64_t handle, uint64_t handle_base) {
  // KFD allocation mmap offsets normally use type 0. Use the otherwise unused
  // type-1 top-bit pattern for guest-only allocations, and keep a private set
  // of emitted offsets so GuestKfd::mmap() can reject only offsets it created.
  const uint64_t ordinal = handle - handle_base + 1;
  return kGuestSyntheticMmapType | ((ordinal << 12) & kGuestSyntheticMmapPayloadMask);
}

bool passthrough_lstat(const std::string &path, struct stat *st) {
  return libc_passthrough().lstat(path.c_str(), st) == 0;
}

bool passthrough_copy_file(const std::string &src, const std::string &dst) {
  auto &real = libc_passthrough();
  int in = real.openat(AT_FDCWD, src.c_str(), O_RDONLY | O_CLOEXEC);
  if (in < 0)
    return false;

  std::error_code ec;
  fs::create_directories(fs::path(dst).parent_path(), ec);
  if (ec) {
    real.close(in);
    return false;
  }

  int out = real.openat(AT_FDCWD, dst.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (out < 0) {
    real.close(in);
    return false;
  }

  std::array<char, 4096> buffer{};
  for (;;) {
    ssize_t n = real.read(in, buffer.data(), buffer.size());
    if (n == 0)
      break;
    if (n < 0) {
      real.close(out);
      real.close(in);
      return false;
    }
    char *p = buffer.data();
    ssize_t remaining = n;
    while (remaining > 0) {
      ssize_t written = real.write(out, p, static_cast<size_t>(remaining));
      if (written <= 0) {
        real.close(out);
        real.close(in);
        return false;
      }
      p += written;
      remaining -= written;
    }
  }

  real.close(out);
  real.close(in);
  return true;
}

} // namespace

/// @brief Generated topology overlay containing host KFD nodes plus one guest node.
///
/// @details The overlay is private to GuestKfd because it is an implementation
/// detail of guest discovery. It copies the real host topology from sysfs, then
/// appends a guest GPU node generated from the configured KFD identity. Only KFD
/// topology paths and the guest DRM render node are redirected to this overlay;
/// host DRM paths remain real so ROCR can still create a host agent and execute
/// on it.
class GuestKfd::TopologyOverlay {
public:
  /// @brief Construct an empty overlay.
  TopologyOverlay() = default;

  /// @brief Remove any generated overlay directories owned by this instance.
  ~TopologyOverlay();

  /// @brief Overlays own temporary filesystem state and cannot be copied.
  TopologyOverlay(const TopologyOverlay &) = delete;

  /// @brief Overlays own temporary filesystem state and cannot be copied.
  TopologyOverlay &operator=(const TopologyOverlay &) = delete;

  /// @brief Build the overlay from the current host sysfs topology.
  /// @param guest Synthetic guest GPU properties to append.
  /// @param host_topology_path Optional path to a generated simulated-host topology root.
  /// @returns true when topology and guest DRM paths are ready.
  bool generate(const Sysfs::GpuInfo &guest, const std::string &host_topology_path = {});

  /// @brief Remove generated overlay directories.
  void cleanup();

  /// @brief Generated KFD topology root.
  [[nodiscard]] const std::string &topology_path() const { return topology_dir_; }

  /// @brief Generated DRM root containing the guest render node.
  [[nodiscard]] const std::string &guest_drm_path() const { return guest_drm_dir_; }

  /// @brief Synthetic topology node ID assigned to the guest GPU.
  [[nodiscard]] uint32_t guest_node_id() const { return guest_node_id_; }

private:
  /// @brief Copy a host sysfs subtree into the generated overlay.
  bool copy_tree(const std::string &src, const std::string &dst);

  /// @brief Copy a simulated topology path or the first available real KFD topology.
  bool copy_host_topology(const std::string &host_topology_path);

  /// @brief Generate the appended guest KFD node and guest DRM metadata.
  bool copy_guest_node(const Sysfs::GpuInfo &guest);

  /// @brief Patch aggregate topology files after appending the guest node.
  bool patch_topology_files();

  std::string topology_dir_;
  std::string guest_drm_dir_;
  Sysfs guest_sysfs_;
  uint32_t guest_node_id_ = 0;
};

GuestKfd::TopologyOverlay::~TopologyOverlay() { cleanup(); }

bool GuestKfd::TopologyOverlay::copy_tree(const std::string &src, const std::string &dst) {
  std::error_code ec;
  fs::create_directories(dst, ec);
  if (ec)
    return false;

  auto &real = libc_passthrough();
  DIR *dir = real.opendir(src.c_str());
  if (!dir)
    return false;

  bool ok = true;
  for (;;) {
    errno = 0;
    struct dirent *entry = real.readdir(dir);
    if (!entry) {
      ok = errno == 0;
      break;
    }

    std::string name = entry->d_name;
    if (name == "." || name == "..")
      continue;

    std::string child_src = src + "/" + name;
    std::string child_dst = dst + "/" + name;
    struct stat st {};
    if (!passthrough_lstat(child_src, &st))
      continue;

    if (S_ISDIR(st.st_mode)) {
      if (!copy_tree(child_src, child_dst)) {
        ok = false;
        break;
      }
      continue;
    }
    if (S_ISLNK(st.st_mode)) {
      std::array<char, 4096> link_target{};
      ssize_t n = real.readlink_fn(child_src.c_str(), link_target.data(), link_target.size() - 1);
      if (n > 0) {
        link_target[static_cast<size_t>(n)] = '\0';
        fs::create_directories(fs::path(child_dst).parent_path(), ec);
        fs::create_symlink(link_target.data(), child_dst, ec);
        ec.clear();
      }
      continue;
    }

    (void)passthrough_copy_file(child_src, child_dst);
  }
  if (real.closedir(dir) != 0)
    ok = false;
  return ok;
}

bool GuestKfd::TopologyOverlay::copy_host_topology(const std::string &host_topology_path) {
  if (!host_topology_path.empty())
    return copy_tree(host_topology_path, topology_dir_);

  for (std::string_view topology_path : kRealTopologyPaths) {
    if (copy_tree(std::string(topology_path), topology_dir_))
      return true;

    // If the first topology root is absent or only partially copyable, reset
    // the destination before trying the alternate root. This mirrors the
    // launcher/HSA-hook discovery policy without mixing trees from two roots.
    std::error_code ec;
    fs::remove_all(topology_dir_, ec);
    ec.clear();
    fs::create_directories(topology_dir_, ec);
  }
  return false;
}

bool GuestKfd::TopologyOverlay::copy_guest_node(const Sysfs::GpuInfo &guest) {
  if (guest_sysfs_.generate(guest).empty())
    return false;
  guest_drm_dir_ = guest_sysfs_.drm_path();

  fs::path nodes_dir = fs::path(topology_dir_) / "nodes";
  guest_node_id_ = max_numeric_dir(nodes_dir) + 1;
  return copy_tree(fs::path(guest_sysfs_.path()) / "nodes" / "1",
                   nodes_dir / std::to_string(guest_node_id_));
}

bool GuestKfd::TopologyOverlay::patch_topology_files() {
  fs::path root = topology_dir_;
  fs::path nodes_dir = root / "nodes";
  fs::path cpu_props = nodes_dir / "0" / "properties";
  fs::path system_props = root / "system_properties";
  fs::path guest_props = nodes_dir / std::to_string(guest_node_id_) / "properties";
  fs::path guest_link =
      nodes_dir / std::to_string(guest_node_id_) / "io_links" / "0" / "properties";

  uint32_t existing_links = read_u32_property(cpu_props, "io_links_count",
                                              count_numeric_dirs(nodes_dir / "0" / "io_links"));
  fs::path new_cpu_link = nodes_dir / "0" / "io_links" / std::to_string(existing_links);
  fs::create_directories(new_cpu_link);

  write_text_file(system_props, set_property(read_text_file(system_props), "num_devices",
                                             count_numeric_dirs(nodes_dir)));
  write_text_file(nodes_dir / "0" / "gpu_id", "0\n");
  write_text_file(cpu_props,
                  set_property(read_text_file(cpu_props), "io_links_count", existing_links + 1));
  write_text_file(new_cpu_link / "properties", io_link_from_cpu(guest_node_id_));
  write_text_file(guest_props, set_property(read_text_file(guest_props), "io_links_count", 1));
  write_text_file(guest_link, io_link_to_cpu(guest_node_id_));
  return true;
}

bool GuestKfd::TopologyOverlay::generate(const Sysfs::GpuInfo &guest,
                                         const std::string &host_topology_path) {
  cleanup();
  if (!make_temp_dir("rocjitsu_guest_topology", &topology_dir_))
    return false;
  if (!copy_host_topology(host_topology_path)) {
    cleanup();
    return false;
  }
  if (!copy_guest_node(guest) || !patch_topology_files()) {
    cleanup();
    return false;
  }
  return true;
}

void GuestKfd::TopologyOverlay::cleanup() {
  if (!topology_dir_.empty()) {
    std::error_code ec;
    fs::remove_all(topology_dir_, ec);
    topology_dir_.clear();
  }
  guest_drm_dir_.clear();
  guest_node_id_ = 0;
  guest_sysfs_.cleanup();
}

GuestKfd::GuestKfd(config::DbtGuestConfig config, LinuxKfd *execution_driver)
    : config_(std::move(config)), execution_driver_(execution_driver),
      overlay_(std::make_unique<TopologyOverlay>()),
      owns_execution_driver_open_(execution_driver && execution_driver->fd() >= 0) {
  libc_passthrough().resolve();
  // The guest overlay describes a single synthetic node with no XCD topology
  // behind it, so its XCC count is 1. (The argument is the XCC count, not a
  // geometry field: passing num_shader_engines here made sysfs publish
  // array_count * num_shader_engines and num_xcc == num_shader_engines.)
  guest_ = gpu_info_from_config(config_.guest_device, 1u);
  guest_.drm_render_minor = choose_render_minor(guest_.drm_render_minor);
  host_gpu_id_ = config_.host.gpu_id;
}

GuestKfd::~GuestKfd() {
  // The synthetic descriptor is this object's own; nothing else closes it once the
  // driver is gone.
  const int synthetic = app_fd_.exchange(-1, std::memory_order_acq_rel);
  if (synthetic >= 0)
    libc_passthrough().close(synthetic);
  int kfd_fd = real_kfd_fd_.load(std::memory_order_acquire);
  if (execution_driver_ && owns_execution_driver_open_)
    execution_driver_->close();
  else if (kfd_fd >= 0 && !execution_driver_)
    libc_passthrough().close(kfd_fd);
}

bool GuestKfd::ensure_real_kfd_locked() {
  if (real_kfd_fd_.load(std::memory_order_acquire) >= 0)
    return true;
  if (execution_driver_) {
    int fd = execution_driver_->fd();
    if (fd < 0) {
      const bool already_owns_open = owns_execution_driver_open_;
      fd = execution_driver_->open();
      if (fd >= 0) {
        if (already_owns_open) {
          // Re-minting an overwritten simulator primary retains the existing
          // process once. GuestKfd already pins that process for the lifetime
          // of its app-facing opens, so balance the temporary retain and keep
          // the original owned reference for final teardown.
          execution_driver_->close();
        } else {
          owns_execution_driver_open_ = true;
        }
      }
    }
    if (fd < 0) {
      errno = ENODEV;
      return false;
    }
    real_kfd_fd_.store(fd, std::memory_order_release);
    return true;
  }
  int fd = libc_passthrough().openat(AT_FDCWD, "/dev/kfd", O_RDWR | O_CLOEXEC, 0);
  if (fd < 0)
    return false;
  // Remember which device this is. The fd number alone is not identity: it can be
  // closed and recycled onto an unrelated file, so anything that holds the
  // descriptor across a window re-checks against this before using it.
  record_kfd_identity(fd);
  real_kfd_fd_.store(fd, std::memory_order_release);
  return true;
}

bool GuestKfd::ensure_ready() {
  if (ready_.load(std::memory_order_acquire))
    return true;

  std::lock_guard lock(mutex_);
  return ensure_ready_locked();
}

bool GuestKfd::ensure_ready_locked() {
  if (ready_.load(std::memory_order_acquire))
    return true;
  if (!config_.enabled || !config_.guest_device.present) {
    errno = EINVAL;
    return false;
  }
  if (!ensure_real_kfd_locked())
    return false;
  if (host_gpu_id_ == 0) {
    // The launcher resolves the host GPU once and publishes it in the runtime handoff, and
    // load_dbt_guest_config_from_handoff() refuses an enabled DBT config that arrives here
    // unresolved. Re-deriving it by first-ISA-match would reintroduce the per-layer
    // divergence that lets this driver and the hook layer pick different GPUs on a
    // multi-GPU host, so refuse rather than guess.
    util::Logger::warn("rocjitsu: DBT config carries no resolved host KFD gpu_id; refusing to "
                       "guess a host GPU");
    errno = ENODEV;
    return false;
  }
  const bool host_matches = execution_driver_
                                ? gpu_id_matches_isa_in_topology(execution_driver_->topology_path(),
                                                                 host_gpu_id_, config_.host.isa)
                                : real_gpu_id_matches_isa(host_gpu_id_, config_.host.isa);
  if (!host_matches) {
    util::Logger::warn("rocjitsu: host KFD gpu_id=", host_gpu_id_,
                       " does not match the configured host ISA '", config_.host.isa,
                       "' in the KFD topology");
    errno = ENODEV;
    return false;
  }
  const std::string host_topology = execution_driver_ ? execution_driver_->topology_path() : "";
  if (!overlay_->generate(guest_, host_topology)) {
    errno = ENODEV;
    return false;
  }
  ready_.store(true, std::memory_order_release);
  return true;
}

bool GuestKfd::prepare_for_discovery() { return ensure_ready(); }

bool GuestKfd::ensure_app_fd_locked() {
  if (app_fd_.load(std::memory_order_acquire) >= 0)
    return true;
  const int minted = memfd_create("rocjitsu_guest_kfd", MFD_CLOEXEC);
  if (minted < 0)
    return false;
  app_fd_.store(minted, std::memory_order_release);
  return true;
}

int GuestKfd::open() {
  std::lock_guard lock(mutex_);
  if (!ensure_ready_locked())
    return -1;

  bool opened_execution_driver = false;
  if (execution_driver_ && !owns_execution_driver_open_) {
    const int execution_fd = execution_driver_->open();
    if (execution_fd < 0)
      return -1;
    real_kfd_fd_.store(execution_fd, std::memory_order_release);
    owns_execution_driver_open_ = true;
    opened_execution_driver = true;
  }

  const int kfd_fd = real_kfd_fd_.load(std::memory_order_acquire);
  if (kfd_fd < 0) {
    if (opened_execution_driver) {
      execution_driver_->close();
      owns_execution_driver_open_ = false;
    }
    errno = ENODEV;
    return -1;
  }

  // Applications receive a DISTINCT descriptor per open, duplicated from a
  // synthetic memfd -- never from the real /dev/kfd. Two properties matter and both
  // are load-bearing:
  //
  //   * Synthetic. A real duplicate would let a forked child -- whose calls the
  //     interposer passes straight to libc under the fork-then-exec contract --
  //     operate real hardware through the inherited fd, and dup2() it without
  //     FD_CLOEXEC to carry that authority across exec. A memfd dup is inert.
  //   * Distinct. Real KFD returns a fresh descriptor per open, and callers close
  //     them independently; handing the same number to concurrent openers would let
  //     one thread's close() invalidate another's live fd.
  //
  // The real fd stays in real_kfd_fd_, private to this object, and every forwarded
  // operation uses it from there.
  if (!ensure_app_fd_locked()) {
    if (opened_execution_driver) {
      execution_driver_->close();
      owns_execution_driver_open_ = false;
    }
    return -1;
  }
  const int app_fd =
      libc_passthrough().fcntl(app_fd_.load(std::memory_order_acquire), F_DUPFD_CLOEXEC, 0);
  if (app_fd < 0) {
    if (opened_execution_driver) {
      execution_driver_->close();
      owns_execution_driver_open_ = false;
    }
    return -1;
  }
  ++open_refs_;
  return app_fd;
}

bool GuestKfd::retain_local_open() {
  std::lock_guard lock(mutex_);
  if (ready_.load(std::memory_order_acquire) && real_kfd_fd_.load(std::memory_order_acquire) >= 0) {
    ++open_refs_;
    return true;
  }
  return false;
}

uint32_t GuestKfd::local_open_ref_count() const {
  std::lock_guard lock(mutex_);
  return open_refs_;
}

void GuestKfd::begin_local_shutdown() {
  // Release any parked WAIT_EVENTS so it returns and drops the driver snapshot that
  // would otherwise keep this object alive. Snapshot execution_driver_ under
  // mutex_, then call with the lock released (its begin_local_shutdown() takes the
  // simulator's own process mutex; holding GuestKfd::mutex_ across it is
  // unnecessary). A simulator-backed guest forwards to that driver. A hardware-
  // backed guest has no execution_driver_ and forwards WAIT_EVENTS to the real
  // kernel as bounded polls (forward_wait_events_bounded), so set hw_closing_ to
  // cancel an in-flight poll loop instead.
  LinuxKfd *execution_driver = nullptr;
  {
    std::lock_guard lock(mutex_);
    execution_driver = execution_driver_;
  }
  if (execution_driver)
    execution_driver->begin_local_shutdown();
  else
    hw_closing_.store(true, std::memory_order_release);
}

LinuxKfd::PrimaryInvalidation GuestKfd::invalidate_primary_fd(int fd) {
  if (fd < 0)
    return PrimaryInvalidation::kNotPrimary;
  std::lock_guard lock(mutex_);
  // The synthetic backing memfd is internal, like the real fd: app-facing dups
  // carry the counted references, so an overwrite of this number must NOT release
  // one. A later open() re-mints the backing.
  int expected_app = fd;
  if (app_fd_.compare_exchange_strong(expected_app, -1, std::memory_order_acq_rel))
    return PrimaryInvalidation::kClearedKeepRefs;

  // Compare-and-clear the hidden real fd number. Do NOT close it: dup2/dup3
  // already atomically replaced whatever this number named, so closing it here
  // would close the app's replacement. A concurrent overwrite that already
  // cleared it is reported as kNotPrimary.
  int expected = fd;
  if (!real_kfd_fd_.compare_exchange_strong(expected, -1, std::memory_order_acq_rel))
    return PrimaryInvalidation::kNotPrimary;
  if (execution_driver_)
    // Clear the simulator's stale primary-fd classification, but do not honor
    // kClearedDropRef here. That counted reference is the execution-backend
    // open GuestKfd owns and must keep pinned while app-facing dups are live.
    // ensure_real_kfd_locked() balances the extra retain if a later open has to
    // re-mint the simulator primary.
    (void)execution_driver_->invalidate_primary_fd(fd);
  // Also drop out of the ready state (mirroring close()'s teardown, minus the
  // fd close / overlay cleanup / ref decrement). This matters because GuestKfd
  // forwards every ioctl/mmap through real_kfd_fd_: with the number now cleared,
  // forward paths fail safe (ENODEV) instead of routing to whatever the app's
  // dup2 installed, and the next ensure_ready() lazily reopens a fresh real
  // /dev/kfd fd. KFD binds one process context per mm, so the reopened fd rejoins
  // the same context and existing app-facing dup fds keep working transparently.
  ready_.store(false, std::memory_order_release);
  // The hidden real fd is kept alive by app-facing dup references, not by a
  // counted "primary" reference, so the caller must NOT drop an open reference.
  return PrimaryInvalidation::kClearedKeepRefs;
}

GuestKfd::FinalCloseWork
GuestKfd::begin_final_close_locked(std::unique_ptr<TopologyOverlay> fresh_overlay) {
  FinalCloseWork work;
  close_deferred_ = false;
  work.kfd_fd = real_kfd_fd_.exchange(-1, std::memory_order_acq_rel);
  if (execution_driver_ && owns_execution_driver_open_) {
    work.execution_driver = execution_driver_;
    owns_execution_driver_open_ = false;
  }
  ready_.store(false, std::memory_order_release);
  synthetic_handles_.clear();
  synthetic_mmap_offsets_.clear();
  next_synthetic_handle_ = kSyntheticHandleBase;
  // Removing copied sysfs trees can block on filesystem work. Swap the overlay object
  // while serialized, then perform the actual remove_all after releasing mutex_ so
  // concurrent ioctl/redirect callers are not stalled.
  work.overlay = std::move(overlay_);
  overlay_ = std::move(fresh_overlay);
  return work;
}

void GuestKfd::finish_final_close(FinalCloseWork work) {
  work.overlay.reset();
  if (work.execution_driver)
    work.execution_driver->close();
  else if (work.kfd_fd >= 0 && !execution_driver_)
    libc_passthrough().close(work.kfd_fd);
}

int GuestKfd::close() {
  FinalCloseWork work;
  auto fresh_overlay = std::make_unique<TopologyOverlay>();
  {
    std::lock_guard lock(mutex_);
    if (open_refs_ == 0)
      return 0;
    --open_refs_;
    if (open_refs_ != 0)
      return 0;

    // The app sees only the synthetic descriptor; the interposer owns close
    // ordering so app-facing references keep the host KFD connection alive. Only
    // the final open reference closes the private real fd and tears down discovery
    // state; the close hook separately closes the app fd that triggered this.
    //
    // The synthetic BACKING memfd is deliberately retained across close/reopen
    // epochs and released only in the destructor. Clearing it here without closing
    // leaked one descriptor per epoch; closing it here instead would need the close
    // to happen outside mutex_, and there is nothing to gain -- the backing is
    // inert, private, and reused by the next open().
    //
    // Unless a caller-visible WAIT_EVENTS is being served right now. That one call is
    // many bounded ioctls, so clearing the connection here would fail it ENODEV on its
    // next slice -- an operation the kernel had already started, revoked partway, which
    // a native blocking ioctl is never subject to. Hand the close to whichever lease
    // drains last instead. close() itself must not wait for that: an indefinite wait
    // would then block close() for as long as it runs. The deferred branch drops
    // readiness like any other close, but the DRAIN is what runs the transition for
    // real, because readiness can be republished while the close is pending.
    if (wait_leases_ > 0) {
      close_deferred_ = true;
      // Hand the replacement overlay over NOW. The drain runs from a destructor, which
      // is noexcept, so an allocation there would turn a bad_alloc into a terminate on
      // the way out of an ioctl -- and it would put an allocate/free on every wait.
      deferred_overlay_ = std::move(fresh_overlay);
      ready_.store(false, std::memory_order_release);
    } else {
      deferred_overlay_.reset();
      work = begin_final_close_locked(std::move(fresh_overlay));
    }
  }
  finish_final_close(std::move(work));
  return 0;
}

bool GuestKfd::acquire_wait_lease() {
  std::lock_guard lock(mutex_);
  // Validating the reference, establishing readiness and taking the lease is ONE critical
  // section on purpose. Split across two, a final close lands in the gap: arriving after
  // readiness it revokes a wait that was already cleared to run, and arriving before it
  // lets a wait with no application reference behind it republish a connection that then
  // outlives every open. Requiring a live reference here is what closes the second: a
  // wait is always issued on an application descriptor, so no reference means the wait
  // has already lost its claim on the connection.
  if (open_refs_ == 0) {
    errno = ENODEV;
    return false;
  }
  if (!ensure_ready_locked()) {
    if (errno == 0)
      errno = ENODEV;
    return false;
  }
  if (real_kfd_fd_.load(std::memory_order_acquire) < 0) {
    errno = ENODEV;
    return false;
  }
  ++wait_leases_;
  return true;
}

int GuestKfd::wait_events(unsigned long request, void *arg) {
  // The lease covers the whole caller-visible wait, for BOTH backends: a final close
  // releases the simulator's backend open exactly as it retires a host connection, and
  // the simulated wait that was already running would then return EBADF.
  WaitLease lease(*this);
  if (!lease.held())
    return -(errno != 0 ? errno : ENODEV);

  int rc = 0;
  std::exception_ptr failure;
  try {
    // Forward a null arg unchanged rather than dereferencing it for the timeout: the
    // kernel answers it with EFAULT/EINVAL, and the interposer must reproduce that
    // failure instead of segfaulting inside the caller's ioctl().
    //
    // The simulator serves the wait itself and begin_local_shutdown() can wake it, so it
    // takes ONE ordinary blocking forward -- slicing it would add repolling that buys
    // nothing. Only the kernel, which nothing in this process can interrupt, needs the
    // wait broken into cancellable slices.
    if (!arg || execution_driver_)
      rc = forward_ioctl(request, arg);
    else
      rc = forward_wait_events_bounded(request, arg);
  } catch (...) {
    failure = std::current_exception();
  }

  // EVERY exit drops the lease here, never on the way out through ~WaitLease -- including
  // the exceptional one, which is why a throw is caught and held rather than allowed to
  // unwind through it. Dropping the last lease can run a deferred final close, and tearing
  // a backend down allocates; a destructor is implicitly noexcept, so a bad_alloc there
  // terminates the process rather than propagating, and under unwinding it would do so
  // with a first exception already in flight. The result is captured by now, so nothing
  // this does can disturb it.
  lease.release();
  if (failure)
    std::rethrow_exception(failure);
  return rc;
}

void GuestKfd::release_wait_lease() {
  FinalCloseWork work;
  {
    std::lock_guard lock(mutex_);
    assert(wait_leases_ > 0);
    // Only the LAST lease to drain executes a deferred close, and only while there is
    // still no open reference: a reopen in the meantime ADOPTS the connection this was
    // holding open (KFD binds one process context per mm, so it is the same connection
    // a reopen would have produced), and closing it then would turn the deferral into
    // the very revocation it exists to prevent.
    //
    // This runs the WHOLE final-close transition, not just the fd close. Anything that
    // calls ensure_ready() during the deferral -- an ioctl arriving at zero references,
    // or an open that publishes readiness and then fails to hand out its descriptor --
    // republishes readiness against the connection being held open. Closing the fd and
    // leaving readiness set would wedge a hardware guest permanently: ensure_ready()
    // short-circuits on the stale flag, so it would never reopen, and every later call
    // would find no host fd and fail ENODEV. The simulator backend hides this because
    // its own open() re-mints a connection; hardware has no such second path.
    //
    // The exchange inside is what makes this safe against a dup2 that landed on the
    // hidden number: invalidate_primary_fd() already cleared it WITHOUT closing, so this
    // finds -1 and closes nothing rather than closing the app's file.
    if (--wait_leases_ == 0 && close_deferred_ && open_refs_ == 0)
      work = begin_final_close_locked(std::move(deferred_overlay_));
  }
  finish_final_close(std::move(work));
}

int GuestKfd::forward_ioctl(unsigned long request, void *arg) {
  if (execution_driver_)
    return execution_driver_->ioctl(request, arg);

  int kfd_fd = host_fd();
  if (kfd_fd < 0) {
    errno = ENODEV;
    return -1;
  }
  int ret = libc_passthrough().ioctl(kfd_fd, request, arg);
  if (ret < 0)
    // A Driver returns the KERNEL convention, -errno, and the interposer's single
    // conversion turns that into libc's -1/errno for the application. libc's ioctl gives
    // this path the libc convention instead, so translate rather than pass it through:
    // returning -1 would be read as -EPERM, which is how an interrupted wait reached the
    // runtime as EPERM and defeated its EINTR retry.
    return -(errno != 0 ? errno : EIO);
  return ret;
}

int GuestKfd::forward_wait_events_bounded(unsigned long request, void *arg) {
  if (host_fd() < 0)
    return -ENODEV;
  auto *wait_args = static_cast<kfd_ioctl_wait_events_args *>(arg);

  // Each slice re-loads the hidden descriptor and re-checks its identity rather than
  // holding a private duplicate across the wait.
  //
  // A duplicate looks like the obvious answer -- a native blocking ioctl holds its file
  // description for the whole call, and slicing reopens a window between slices. It is
  // the wrong answer here, because the duplicate would be an ordinary numeric fd that
  // the interposer does not track. A dup2 onto its number would redirect a later slice
  // unnoticed, and, worse, OWNING that number means the RAII close at the end of the
  // wait would close whatever now sits there -- the application's own descriptor.
  // Detecting the swap does not help: detection does not relinquish numeric ownership.
  // Holding an uncoordinated descriptor across an indefinite wait trades a slice that
  // fails ENODEV for closing a file we do not own, which is strictly worse.
  //
  // So: own nothing. Re-load per slice, confirm the number is still the device we
  // opened before issuing at it, and let the wait end if it is not. That leaves the
  // load/use split every other forwarded path here already has, and no new ownership.
  // Closing it for real needs a descriptor-lifecycle protocol this driver does not have
  // yet: a private fd that the interposer REGISTERS, and that the dup2/dup3 transaction
  // relocates before it mutates the fd table, rather than one validated after the fact.
  // That is a change to the interposer/LinuxKfd boundary and belongs in its own review.
  //
  // Zero-timeout requests come through here too: "non-blocking" removes the wait, not
  // the load/use race, and wait_events_poll_loop issues them unsliced anyway.
  return wait_events_poll_loop(*wait_args, hw_closing_, [&] {
    const int slice_fd = host_fd();
    if (slice_fd < 0 || !fd_is_expected_kfd(slice_fd))
      return -ENODEV;
    int ret = libc_passthrough().ioctl(slice_fd, request, arg);
    if (ret < 0)
      return -(errno != 0 ? errno : EIO);
    return ret;
  });
}

void GuestKfd::record_kfd_identity(int kfd_fd) {
  auto &real = libc_passthrough();
  struct stat st {};
  if (real.fstat_fn && real.fstat_fn(kfd_fd, &st) == 0 && S_ISCHR(st.st_mode))
    real_kfd_rdev_.store(st.st_rdev, std::memory_order_release);
  else
    real_kfd_rdev_.store(0, std::memory_order_release);
}

bool GuestKfd::fd_is_expected_kfd(int fd) const {
  const dev_t expected = real_kfd_rdev_.load(std::memory_order_acquire);
  if (expected == 0)
    return false; // identity was never established: fail closed rather than guess.
  auto &real = libc_passthrough();
  if (!real.fstat_fn)
    return false;
  struct stat st {};
  if (real.fstat_fn(fd, &st) != 0)
    return false;
  return S_ISCHR(st.st_mode) && st.st_rdev == expected;
}

int GuestKfd::wait_events_poll_loop(kfd_ioctl_wait_events_args &args,
                                    const std::atomic<bool> &cancelled,
                                    const std::function<int()> &poll) {
  const uint32_t original_timeout = args.timeout;

  // A zero-timeout poll already returns promptly; forward it unchanged.
  if (original_timeout == 0)
    return poll();

  // Break a long/indefinite wait into short kernel polls, re-checking the
  // cancellation flag between iterations so begin_local_shutdown() can cancel it
  // and the caller's driver snapshot drops. The per-poll timeout bounds how long a
  // single blocking syscall can hold it. Mirrors RemoteDriver's client-side
  // WAIT_EVENTS loop.
  //
  // Deliberately NOT how the real stack waits: the thunk issues one blocking
  // WAIT_EVENTS and re-enters only for EINTR/EAGAIN, because a kernel wait is
  // interruptible by a signal and needs no cancellation flag. Nothing here can
  // interrupt a poll already inside the driver, so the wait is sliced instead.
  // Collapsing this back to a single blocking call would match the thunk and
  // reintroduce the teardown stall it exists to avoid.
  constexpr uint32_t kPollMs = 5;
  const bool indefinite = original_timeout == kWaitEventsInfiniteMs;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(original_timeout);

  for (;;) {
    if (cancelled.load(std::memory_order_acquire)) {
      // Report a benign timeout so the caller re-polls (and observes the driver
      // going away) rather than treating the cancellation as an event completion.
      args.timeout = original_timeout;
      args.wait_result = KFD_IOC_WAIT_RESULT_TIMEOUT;
      return 0;
    }
    // Clamp the slice to what the caller actually has left, so a short wait is not
    // rounded UP to kPollMs (WAIT_EVENTS(timeout=1) must not block 5 ms). The
    // deadline is only re-tested after a poll returns, so without this the first
    // poll alone can overshoot the whole budget.
    args.timeout = kPollMs;
    if (!indefinite) {
      // Round the remainder UP, and never to zero. A zero timeout is
      // KFD_EVENT_TIMEOUT_IMMEDIATE, which is a poll that returns at once, so a
      // sub-millisecond remainder truncated to zero would busy-spin issuing
      // immediate polls until the deadline passed. Rounding up is also what the
      // driver does with the millisecond timeout it is handed -- it ceilings the
      // conversion to jiffies and then adds one -- so a nonzero request never
      // becomes a zero-length sleep there either. Overshoot stays under a
      // millisecond because the loop still stops at the deadline below.
      const auto remaining_ms =
          std::chrono::ceil<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now())
              .count();
      args.timeout = static_cast<uint32_t>(std::clamp<int64_t>(remaining_ms, 1, kPollMs));
    }
    int rc = poll();
    if (rc != 0) {
      // Report what is LEFT of the caller's budget, not what they asked for. This is
      // the ioctl-restart path: the driver reduces the timeout in place so a caller
      // that re-issues on EINTR continues the same wait instead of starting a fresh
      // one. Restoring the original here would let a stream of signals extend a
      // finite wait without bound, since each retry recomputes the deadline. Zero is
      // a legal remainder -- it means poll once -- and an indefinite wait has no
      // remainder to report.
      if (!indefinite) {
        const auto left_ms = std::chrono::ceil<std::chrono::milliseconds>(
                                 deadline - std::chrono::steady_clock::now())
                                 .count();
        args.timeout = static_cast<uint32_t>(std::clamp<int64_t>(left_ms, 0, original_timeout));
      } else {
        args.timeout = original_timeout;
      }
      return rc;
    }
    args.timeout = original_timeout;
    if (args.wait_result != KFD_IOC_WAIT_RESULT_TIMEOUT)
      return 0;
    if (!indefinite && std::chrono::steady_clock::now() >= deadline)
      return 0; // real timeout: leave wait_result == TIMEOUT
  }
}

int GuestKfd::get_process_apertures_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_process_apertures_new_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }

  kfd_ioctl_get_process_apertures_new_args count_args{};
  int ret = forward_ioctl(AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &count_args);
  if (ret != 0)
    return ret;
  const uint32_t host_count = count_args.num_of_nodes;

  if (args->num_of_nodes == 0 || args->kfd_process_device_apertures_ptr == 0) {
    args->num_of_nodes = count_args.num_of_nodes + 1;
    return 0;
  }

  auto *out = reinterpret_cast<kfd_process_device_apertures *>(
      static_cast<uintptr_t>(args->kfd_process_device_apertures_ptr));
  const uint32_t requested = args->num_of_nodes;
  // The caller controls requested capacity, so allocate only enough temporary
  // space for the host nodes KFD reported. The guest aperture is appended
  // directly into the caller buffer only when the caller supplied a spare slot.
  const uint32_t host_capacity = std::min(requested, host_count);
  std::vector<kfd_process_device_apertures> host(host_capacity);
  if (!host.empty()) {
    kfd_ioctl_get_process_apertures_new_args host_args{};
    host_args.kfd_process_device_apertures_ptr =
        reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(host.data()));
    host_args.num_of_nodes = host_capacity;
    ret = forward_ioctl(AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &host_args);
    if (ret != 0)
      return ret;
    const uint32_t host_to_copy = std::min(host_capacity, host_args.num_of_nodes);
    for (uint32_t i = 0; i < host_to_copy; ++i)
      out[i] = host[i];
    if (requested > host_to_copy) {
      out[host_to_copy] = guest_apertures();
      args->num_of_nodes = host_to_copy + 1;
    } else {
      args->num_of_nodes = host_to_copy;
    }
    return 0;
  }
  if (requested > 0) {
    out[0] = guest_apertures();
    args->num_of_nodes = 1;
  }
  return 0;
}

int GuestKfd::get_clock_counters_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_clock_counters_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }
  if (args->gpu_id != guest_.gpu_id)
    return forward_ioctl(AMDKFD_IOC_GET_CLOCK_COUNTERS, arg);

  return LinuxKfd::get_clock_counters_ioctl(arg);
}

int GuestKfd::acquire_vm_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_acquire_vm_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }
  if (args->gpu_id == guest_.gpu_id)
    return 0;
  return forward_ioctl(AMDKFD_IOC_ACQUIRE_VM, arg);
}

int GuestKfd::get_available_memory_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_available_memory_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }
  if (args->gpu_id == guest_.gpu_id) {
    args->available = guest_.local_mem_size;
    return 0;
  }
  return forward_ioctl(AMDKFD_IOC_AVAILABLE_MEMORY, arg);
}

int GuestKfd::set_memory_policy_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_set_memory_policy_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }
  if (args->gpu_id == guest_.gpu_id)
    return 0;
  return forward_ioctl(AMDKFD_IOC_SET_MEMORY_POLICY, arg);
}

int GuestKfd::alloc_memory_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_alloc_memory_of_gpu_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }
  if (args->gpu_id != guest_.gpu_id) {
    int ret = forward_ioctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, arg);
    if (ret == 0)
      assert(args->handle < kSyntheticHandleBase);
    return ret;
  }

  std::lock_guard lock(mutex_);
  args->handle = next_synthetic_handle_++;
  args->mmap_offset = synthetic_mmap_offset_for_handle(args->handle, kSyntheticHandleBase);
  synthetic_handles_.insert(args->handle);
  synthetic_mmap_offsets_.insert(args->mmap_offset);
  return 0;
}

int GuestKfd::free_memory_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_free_memory_of_gpu_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }
  {
    std::lock_guard lock(mutex_);
    if (synthetic_handles_.erase(args->handle) != 0) {
      synthetic_mmap_offsets_.erase(
          synthetic_mmap_offset_for_handle(args->handle, kSyntheticHandleBase));
      return 0;
    }
  }
  assert(args->handle < kSyntheticHandleBase);
  return forward_ioctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, arg);
}

template <typename Args>
int GuestKfd::map_or_unmap_memory_ioctl(Args *args, unsigned long request) {
  if (!args) {
    errno = EINVAL;
    return -1;
  }

  uint32_t host_gpu_id = 0;
  {
    std::lock_guard lock(mutex_);
    host_gpu_id = host_gpu_id_;
    if (synthetic_handles_.count(args->handle) != 0) {
      args->n_success = args->n_devices;
      return 0;
    }
  }

  auto *ids =
      reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(args->device_ids_array_ptr));
  if (!ids) {
    assert(args->handle < kSyntheticHandleBase);
    return forward_ioctl(request, args);
  }

  std::vector<uint32_t> host_ids;
  host_ids.reserve(args->n_devices);
  bool has_guest = false;
  for (uint32_t i = 0; i < args->n_devices; ++i) {
    if (ids[i] == guest_.gpu_id) {
      has_guest = true;
      append_unique_gpu_id(&host_ids, host_gpu_id);
    } else {
      append_unique_gpu_id(&host_ids, ids[i]);
    }
  }
  if (!has_guest) {
    assert(args->handle < kSyntheticHandleBase);
    return forward_ioctl(request, args);
  }
  if (host_ids.empty()) {
    args->n_success = args->n_devices;
    return 0;
  }

  assert(args->handle < kSyntheticHandleBase);
  auto host_args = *args;
  host_args.device_ids_array_ptr =
      reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(host_ids.data()));
  host_args.n_devices = static_cast<uint32_t>(host_ids.size());
  host_args.n_success = 0;
  int ret = forward_ioctl(request, &host_args);
  if (ret != 0)
    return ret;
  args->n_success = args->n_devices;
  return 0;
}

int GuestKfd::map_memory_ioctl(void *arg) {
  return map_or_unmap_memory_ioctl(static_cast<kfd_ioctl_map_memory_to_gpu_args *>(arg),
                                   AMDKFD_IOC_MAP_MEMORY_TO_GPU);
}

int GuestKfd::unmap_memory_ioctl(void *arg) {
  return map_or_unmap_memory_ioctl(static_cast<kfd_ioctl_unmap_memory_from_gpu_args *>(arg),
                                   AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU);
}

int GuestKfd::ioctl(unsigned long request, void *arg) {
  // WAIT_EVENTS establishes readiness and takes its operation lease as one locked step,
  // so it must not come through the generic ensure_ready() first -- that would reopen
  // the very gap the single step exists to close.
  if (canonical_ioctl_request(request) == AMDKFD_IOC_WAIT_EVENTS)
    return wait_events(request, arg);
  if (!ensure_ready())
    return -(errno != 0 ? errno : ENODEV);

  switch (canonical_ioctl_request(request)) {
  case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
    return get_process_apertures_ioctl(arg);
  case AMDKFD_IOC_GET_CLOCK_COUNTERS:
    return get_clock_counters_ioctl(arg);
  case AMDKFD_IOC_ACQUIRE_VM:
    return acquire_vm_ioctl(arg);
  case AMDKFD_IOC_AVAILABLE_MEMORY:
    return get_available_memory_ioctl(arg);
  case AMDKFD_IOC_SET_MEMORY_POLICY:
    return set_memory_policy_ioctl(arg);
  case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU:
    return alloc_memory_ioctl(arg);
  case AMDKFD_IOC_FREE_MEMORY_OF_GPU:
    return free_memory_ioctl(arg);
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
    return map_memory_ioctl(arg);
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
    return unmap_memory_ioctl(arg);
  case AMDKFD_IOC_GET_VERSION:
    return get_version_ioctl(arg);
  case AMDKFD_IOC_SET_XNACK_MODE:
  case AMDKFD_IOC_RUNTIME_ENABLE:
    return forward_ioctl(request, arg);
  default:
    if (request_targets_guest(request, arg))
      return reject_guest_execution_ioctl(request, arg);
    return forward_ioctl(request, arg);
  }
}

void *GuestKfd::mmap(void *addr, size_t length, int prot, int flags, off_t offset) {
  if (!ensure_ready())
    return MAP_FAILED;
  int kfd_fd = host_fd();
  if (kfd_fd < 0) {
    errno = ENODEV;
    return MAP_FAILED;
  }
  uint64_t encoded_gpu =
      (static_cast<uint64_t>(offset) & ~KFD_MMAP_TYPE_MASK) >> KFD_MMAP_GPU_ID_SHIFT;
  {
    std::lock_guard lock(mutex_);
    if (synthetic_mmap_offsets_.count(static_cast<uint64_t>(offset)) != 0) {
      errno = ENODEV;
      return MAP_FAILED;
    }
  }
  if ((static_cast<uint64_t>(offset) & KFD_MMAP_TYPE_MASK) == KFD_MMAP_TYPE_DOORBELL &&
      encoded_gpu == guest_.gpu_id) {
    errno = ENODEV;
    return MAP_FAILED;
  }
  if (execution_driver_)
    return execution_driver_->mmap(addr, length, prot, flags, offset);
  return libc_passthrough().mmap(addr, length, prot, flags, kfd_fd, offset);
}

int GuestKfd::munmap(void *addr, size_t length) {
  return execution_driver_ ? execution_driver_->munmap(addr, length) : -ENOENT;
}

bool GuestKfd::owns_fd(int fd) const {
  if (fd < 0)
    return false;
  if (fd == app_fd_.load(std::memory_order_acquire))
    return true;
  if (fd == real_kfd_fd_.load(std::memory_order_acquire))
    return true;
  return execution_driver_ && execution_driver_->owns_fd(fd);
}

int GuestKfd::fd() const { return real_kfd_fd_.load(std::memory_order_acquire); }

int GuestKfd::host_fd() const { return real_kfd_fd_.load(std::memory_order_acquire); }

std::string GuestKfd::redirect_sysfs_path(const char *path) const {
  if (!path)
    return {};

  std::string topology_path;
  std::string guest_drm_path;
  {
    std::lock_guard lock(mutex_);
    if (!ready_.load(std::memory_order_acquire))
      return {};
    // Snapshot mutable overlay strings while serialized with close() teardown.
    // The returned path must not reference buffers that cleanup() may clear.
    topology_path = overlay_->topology_path();
    guest_drm_path = overlay_->guest_drm_path();
  }

  auto redirected = redirect_sysfs_root_path(path, topology_path, {});
  if (!redirected.empty())
    return redirected;
  if (guest_drm_path.empty())
    return {};

  std::string_view sv(path);
  uint32_t minor = 0;
  std::string_view suffix;
  if (parse_drm_render_path(sv, &minor, &suffix) && minor == guest_.drm_render_minor)
    return guest_drm_path + "/renderD" + std::to_string(minor) + std::string(suffix);
  if (parse_sys_dev_drm_render_path(sv, &minor, &suffix) && minor == guest_.drm_render_minor)
    return guest_drm_path + "/renderD" + std::to_string(minor) + std::string(suffix);

  constexpr std::string_view kDevDriRenderPrefix = "/dev/dri/renderD";
  if (sv.starts_with(kDevDriRenderPrefix)) {
    auto rest = sv.substr(kDevDriRenderPrefix.size());
    auto slash = rest.find('/');
    auto number = slash == std::string_view::npos ? rest : rest.substr(0, slash);
    uint32_t parsed = 0;
    auto [ptr, err] = std::from_chars(number.data(), number.data() + number.size(), parsed);
    if (err == std::errc{} && ptr == number.data() + number.size() &&
        parsed == guest_.drm_render_minor) {
      auto dev_suffix = slash == std::string_view::npos ? std::string_view{} : rest.substr(slash);
      return guest_drm_path + "/dev_dri/renderD" + std::to_string(parsed) + std::string(dev_suffix);
    }
  }

  return execution_driver_ ? execution_driver_->redirect_sysfs_path(path) : std::string{};
}

bool GuestKfd::is_doorbell_range(const void *addr, size_t length) const {
  return execution_driver_ && execution_driver_->is_doorbell_range(addr, length);
}

void *GuestKfd::mmap_replacing_client_doorbell_views(void *addr, size_t length, int prot, int flags,
                                                     int fd, off_t offset) {
  if (execution_driver_)
    return execution_driver_->mmap_replacing_client_doorbell_views(addr, length, prot, flags, fd,
                                                                   offset);
  return libc_passthrough().mmap(addr, length, prot, flags, fd, offset);
}

bool GuestKfd::handles_drm_render_minor(uint32_t minor) const {
  if (ready_.load(std::memory_order_acquire) && minor == guest_.drm_render_minor)
    return true;
  return execution_driver_ && execution_driver_->handles_drm_render_minor(minor);
}

const Sysfs::GpuInfo *GuestKfd::gpu_info_for_render_minor(uint32_t minor) const {
  if (ready_.load(std::memory_order_acquire) && minor == guest_.drm_render_minor)
    return &guest_;
  return execution_driver_ ? execution_driver_->gpu_info_for_render_minor(minor) : nullptr;
}

std::string GuestKfd::topology_path() const {
  std::lock_guard lock(mutex_);
  return ready_.load(std::memory_order_acquire) ? overlay_->topology_path() : std::string{};
}

std::string GuestKfd::drm_path() const {
  return execution_driver_ ? execution_driver_->drm_path() : std::string{};
}

int GuestKfd::reject_guest_execution_ioctl(unsigned long request, void *) const {
  util::Logger::driver("rocjitsu guest gpu: rejecting ", LinuxKfd::ioctl_name(request),
                       " for guest gpu_id=", guest_.gpu_id);
  errno = ENODEV;
  return -1;
}

bool GuestKfd::request_targets_guest(unsigned long request, void *arg) const {
  if (!arg)
    return false;
  switch (canonical_ioctl_request(request)) {
  case AMDKFD_IOC_CREATE_QUEUE:
    return static_cast<kfd_ioctl_create_queue_args *>(arg)->gpu_id == guest_.gpu_id;
  case AMDKFD_IOC_IMPORT_DMABUF:
    return static_cast<kfd_ioctl_import_dmabuf_args *>(arg)->gpu_id == guest_.gpu_id;
  case AMDKFD_IOC_SMI_EVENTS:
    return static_cast<kfd_ioctl_smi_events_args *>(arg)->gpuid == guest_.gpu_id;
  case AMDKFD_IOC_SET_SCRATCH_BACKING_VA:
    return static_cast<kfd_ioctl_set_scratch_backing_va_args *>(arg)->gpu_id == guest_.gpu_id;
  case AMDKFD_IOC_GET_TILE_CONFIG:
    return static_cast<kfd_ioctl_get_tile_config_args *>(arg)->gpu_id == guest_.gpu_id;
  case AMDKFD_IOC_SET_TRAP_HANDLER:
    return static_cast<kfd_ioctl_set_trap_handler_args *>(arg)->gpu_id == guest_.gpu_id;
  case AMDKFD_IOC_SVM: {
    auto *a = static_cast<kfd_ioctl_svm_args *>(arg);
    for (uint32_t i = 0; i < a->nattr; ++i) {
      auto &attr = a->attrs[i];
      if ((attr.type == KFD_IOCTL_SVM_ATTR_PREFERRED_LOC ||
           attr.type == KFD_IOCTL_SVM_ATTR_PREFETCH_LOC || attr.type == KFD_IOCTL_SVM_ATTR_ACCESS ||
           attr.type == KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE ||
           attr.type == KFD_IOCTL_SVM_ATTR_NO_ACCESS) &&
          attr.value == guest_.gpu_id)
        return true;
    }
    return false;
  }
  default:
    return false;
  }
}

kfd_process_device_apertures GuestKfd::guest_apertures() const {
  uint32_t guest_node_id = 0;
  {
    std::lock_guard lock(mutex_);
    guest_node_id = overlay_->guest_node_id();
  }
  const uint64_t ordinal = std::max<uint32_t>(1, guest_node_id);
  const uint64_t aperture_stride = 0x10000000000ULL;
  kfd_process_device_apertures apertures{};
  apertures.lds_base = 0x1000000000000ULL + ordinal * aperture_stride;
  apertures.lds_limit = apertures.lds_base + 0xFFFFFFFFULL;
  apertures.scratch_base = 0x2000000000000ULL + ordinal * aperture_stride;
  apertures.scratch_limit = apertures.scratch_base + 0xFFFFFFFFULL;
  apertures.gpuvm_base = 0x1000000000ULL;
  apertures.gpuvm_limit = 0x3FFFFFFFFFFFULL;
  apertures.gpu_id = guest_.gpu_id;
  return apertures;
}

} // namespace rocjitsu
