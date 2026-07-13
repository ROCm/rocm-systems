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
#include <cassert>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <dirent.h>
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
  std::error_code ec;
  for (const auto &entry : fs::directory_iterator(dir, ec)) {
    if (!entry.is_directory(ec))
      continue;
    auto name = entry.path().filename().string();
    uint32_t id = 0;
    auto [ptr, err] = std::from_chars(name.data(), name.data() + name.size(), id);
    if (err == std::errc{} && ptr == name.data() + name.size())
      max_id = std::max(max_id, id);
  }
  return max_id;
}

std::optional<uint32_t> first_real_gpu_id_matching_isa(std::string_view host_isa) {
  std::optional<uint32_t> target_version = kmd::gfx_target_version_from_name(host_isa);
  if (!target_version)
    return std::nullopt;

  for (std::string_view topology_path : kRealTopologyPaths) {
    fs::path nodes_dir = fs::path(std::string(topology_path)) / "nodes";
    const uint32_t max_node = max_numeric_dir(nodes_dir);
    for (uint32_t node_id = 1; node_id <= max_node; ++node_id) {
      fs::path node_dir = nodes_dir / std::to_string(node_id);
      const uint32_t gfx_target_version =
          read_u32_property(node_dir / "properties", "gfx_target_version", 0);
      if (gfx_target_version != *target_version)
        continue;

      std::optional<uint32_t> gpu_id = read_u32_file(node_dir / "gpu_id");
      if (gpu_id && *gpu_id != 0)
        return gpu_id;
    }
  }
  return std::nullopt;
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
  /// @returns true when topology and guest DRM paths are ready.
  bool generate(const Sysfs::GpuInfo &guest);

  /// @brief Remove generated overlay directories.
  void cleanup();

  /// @brief Drop inherited overlay ownership without removing parent-owned paths.
  void release_after_fork();

  /// @brief Generated KFD topology root.
  [[nodiscard]] const std::string &topology_path() const { return topology_dir_; }

  /// @brief Generated DRM root containing the guest render node.
  [[nodiscard]] const std::string &guest_drm_path() const { return guest_drm_dir_; }

  /// @brief Synthetic topology node ID assigned to the guest GPU.
  [[nodiscard]] uint32_t guest_node_id() const { return guest_node_id_; }

private:
  /// @brief Copy a host sysfs subtree into the generated overlay.
  bool copy_tree(const std::string &src, const std::string &dst);

  /// @brief Copy the first available real KFD topology root into the overlay.
  bool copy_host_topology();

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

bool GuestKfd::TopologyOverlay::copy_host_topology() {
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

bool GuestKfd::TopologyOverlay::generate(const Sysfs::GpuInfo &guest) {
  cleanup();
  if (!make_temp_dir("rocjitsu_guest_topology", &topology_dir_))
    return false;
  if (!copy_host_topology()) {
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

void GuestKfd::TopologyOverlay::release_after_fork() {
  topology_dir_.clear();
  guest_drm_dir_.clear();
  guest_node_id_ = 0;
  guest_sysfs_.release_after_fork();
}

GuestKfd::GuestKfd(config::DbtGuestConfig config)
    : config_(std::move(config)), overlay_(std::make_unique<TopologyOverlay>()) {
  libc_passthrough().resolve();
  guest_ = gpu_info_from_config(config_.guest_device,
                                std::max(1u, config_.guest_device.num_shader_engines));
  guest_.drm_render_minor = choose_render_minor(guest_.drm_render_minor);
  host_gpu_id_ = config_.host_gpu_id;
}

GuestKfd::~GuestKfd() {
  int kfd_fd = real_kfd_fd_.load(std::memory_order_acquire);
  if (kfd_fd >= 0)
    libc_passthrough().close(kfd_fd);
}

void GuestKfd::reset_after_fork() {
  // After fork() only the calling thread survives. If another parent thread was
  // in an ioctl or close path, mutex_ may be inherited as permanently locked in
  // the child. Reinitialize it in-place before touching ordinary members; the
  // child is single-threaded here and the interposer destroys this copy next.
  new (&mutex_) std::mutex();
  ready_.store(false, std::memory_order_release);
  int kfd_fd = real_kfd_fd_.exchange(-1, std::memory_order_acq_rel);
  if (kfd_fd >= 0)
    libc_passthrough().close(kfd_fd);
  open_refs_ = 0;
  overlay_->release_after_fork();
  synthetic_handles_.clear();
  synthetic_mmap_offsets_.clear();
  next_synthetic_handle_ = kSyntheticHandleBase;
}

bool GuestKfd::ensure_real_kfd_locked() {
  if (real_kfd_fd_.load(std::memory_order_acquire) >= 0)
    return true;
  int fd = libc_passthrough().openat(AT_FDCWD, "/dev/kfd", O_RDWR | O_CLOEXEC, 0);
  if (fd < 0)
    return false;
  real_kfd_fd_.store(fd, std::memory_order_release);
  return true;
}

bool GuestKfd::ensure_ready() {
  if (ready_.load(std::memory_order_acquire))
    return true;

  std::lock_guard lock(mutex_);
  if (ready_.load(std::memory_order_acquire))
    return true;
  if (!config_.enabled || !config_.guest_device.present) {
    errno = EINVAL;
    return false;
  }
  if (!ensure_real_kfd_locked())
    return false;
  if (host_gpu_id_ == 0) {
    std::optional<uint32_t> host_gpu_id = first_real_gpu_id_matching_isa(config_.host_isa);
    if (!host_gpu_id) {
      errno = ENODEV;
      return false;
    }
    host_gpu_id_ = *host_gpu_id;
  }
  if (!overlay_->generate(guest_)) {
    errno = ENODEV;
    return false;
  }
  ready_.store(true, std::memory_order_release);
  return true;
}

bool GuestKfd::prepare_for_discovery() { return ensure_ready(); }

int GuestKfd::open() {
  for (;;) {
    if (!ensure_ready())
      return -1;
    std::lock_guard lock(mutex_);
    int kfd_fd = real_kfd_fd_.load(std::memory_order_acquire);
    if (ready_.load(std::memory_order_acquire) && kfd_fd >= 0) {
      // Keep the real /dev/kfd fd internal to the driver. Applications receive
      // ordinary dup fds, so close(fd) releases that fd number immediately while
      // the hidden real fd keeps host-KFD forwarding alive until the last
      // app-facing open/dup reference is closed.
      int app_fd = libc_passthrough().fcntl(kfd_fd, F_DUPFD_CLOEXEC, 0);
      if (app_fd < 0)
        return -1;
      ++open_refs_;
      return app_fd;
    }
  }
}

void GuestKfd::retain_local_open() {
  std::lock_guard lock(mutex_);
  if (ready_.load(std::memory_order_acquire) && real_kfd_fd_.load(std::memory_order_acquire) >= 0)
    ++open_refs_;
}

int GuestKfd::close() {
  int kfd_fd = -1;
  std::unique_ptr<TopologyOverlay> overlay_to_cleanup;
  auto fresh_overlay = std::make_unique<TopologyOverlay>();
  {
    std::lock_guard lock(mutex_);
    if (open_refs_ == 0)
      return 0;
    --open_refs_;
    if (open_refs_ != 0)
      return 0;

    // The app sees a real KFD fd, but the interposer owns close ordering so
    // dup'd descriptors can keep the process KFD connection alive. Only the
    // final open reference closes the primary real fd and tears down discovery
    // state; the close hook separately closes any dup fd that triggered this.
    kfd_fd = real_kfd_fd_.exchange(-1, std::memory_order_acq_rel);
    ready_.store(false, std::memory_order_release);
    synthetic_handles_.clear();
    synthetic_mmap_offsets_.clear();
    next_synthetic_handle_ = kSyntheticHandleBase;
    // Removing copied sysfs trees can block on filesystem work. Swap the
    // overlay object while serialized, then perform the actual remove_all after
    // releasing mutex_ so concurrent ioctl/redirect callers are not stalled.
    overlay_to_cleanup = std::move(overlay_);
    overlay_ = std::move(fresh_overlay);
  }
  overlay_to_cleanup.reset();
  if (kfd_fd >= 0)
    libc_passthrough().close(kfd_fd);
  return 0;
}

int GuestKfd::forward_ioctl(unsigned long request, void *arg) {
  int kfd_fd = fd();
  if (kfd_fd < 0) {
    errno = ENODEV;
    return -1;
  }
  int ret = libc_passthrough().ioctl(kfd_fd, request, arg);
  if (ret < 0) {
    // GuestKfd forwards to the real kernel ABI here. Preserve errno and return
    // -1 so callers that check for libc ioctl failure semantics behave the same
    // whether an ioctl was handled locally or forwarded to /dev/kfd.
    const int saved_errno = errno != 0 ? errno : EIO;
    errno = saved_errno;
    return -1;
  }
  return ret;
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
  if (!ensure_ready())
    return -1;

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
  int kfd_fd = fd();
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
  return libc_passthrough().mmap(addr, length, prot, flags, kfd_fd, offset);
}

int GuestKfd::munmap(void *, size_t) { return -ENOENT; }

bool GuestKfd::owns_fd(int fd) const {
  return fd >= 0 && fd == real_kfd_fd_.load(std::memory_order_acquire);
}

int GuestKfd::fd() const { return real_kfd_fd_.load(std::memory_order_acquire); }

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

  return {};
}

bool GuestKfd::is_doorbell_range(const void *, size_t) const { return false; }

bool GuestKfd::handles_drm_render_minor(uint32_t minor) const {
  return ready_.load(std::memory_order_acquire) && minor == guest_.drm_render_minor;
}

const Sysfs::GpuInfo *GuestKfd::gpu_info_for_render_minor(uint32_t minor) const {
  return handles_drm_render_minor(minor) ? &guest_ : nullptr;
}

std::string GuestKfd::topology_path() const {
  std::lock_guard lock(mutex_);
  return ready_.load(std::memory_order_acquire) ? overlay_->topology_path() : std::string{};
}

int GuestKfd::reject_guest_execution_ioctl(unsigned long request, void *) const {
  util::Logger::debug_print("rocjitsu guest gpu: rejecting ", LinuxKfd::ioctl_name(request),
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
