// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <dirent.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <climits>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <vector>

#include "include/amd_cuid.h"
#include "rocm/sha2/sha256.h"
#include "src/cuid_cpu.h"
#include "src/cuid_device.h"
#include "src/cuid_device_manager.h"
#include "src/cuid_gpu.h"
#include "src/cuid_internal.h"
#include "src/cuid_nic.h"
#include "src/cuid_npu.h"
#include "src/cuid_platform.h"
#include "src/cuid_util.h"
#include "src/hmac.h"

namespace {

// Static instance for API
cuid_hmac global_hmac = cuid_hmac();
CuidDeviceManager& mgr = CuidDeviceManager::instance();

// Share global_hmac with mgr so build_cuid_index() derives CUIDs with the
// key reload_key() last read.
struct HmacWiring {
  HmacWiring() { mgr.set_hmac(&global_hmac); }
} hmac_wiring;

// Root with a node key gets it. Anyone else gets nullptr, which puts a
// key-gated component (CPU, NIC, NPU, Platform) on its temporary CUID.
cuid_hmac* derivation_key() {
  return (geteuid() == 0 && global_hmac.is_valid()) ? &global_hmac : nullptr;
}

amdcuid_status_t current_handle(const DevicePtr& device, amdcuid_id_t* handle) {
  amdcuid_derived_id derived{};
  const auto status = device->get_derived_cuid(derived, derivation_key());
  if (status != AMDCUID_STATUS_SUCCESS) return status;
  const auto indexed = mgr.index_handle(device, derived.UUIDv8_representation);
  if (indexed != AMDCUID_STATUS_SUCCESS) return indexed;
  *handle = derived.UUIDv8_representation;
  return AMDCUID_STATUS_SUCCESS;
}

// The driver persists the seed to AmdCuidKey and re-keys every component.
// cuid_seed accepts exactly 32 raw bytes in one write, with no trailing newline.
// Retrying a short write would submit an invalid length.
//
// Returns SUCCESS when the seed was accepted, UNSUPPORTED when the attribute is
// absent, so there is no kernel-published value to go stale, and
// PERMISSION_DENIED or FILE_ERROR when it exists and the write did not land.
amdcuid_status_t write_driver_seed(const std::string& bdf, const uint8_t key[key_length]) {
  const std::string path = "/sys/bus/pci/devices/" + bdf + "/cuid_seed";

  const int fd = open(path.c_str(), O_WRONLY | O_CLOEXEC);
  if (fd < 0) {
    if (errno == ENOENT) return AMDCUID_STATUS_UNSUPPORTED;
    const int err = errno;
    LOG(ERROR,
        "amdcuid_set_hash_key: cannot open " << path << ": " << CuidUtilities::errno_string(err));
    return (err == EACCES || err == EPERM) ? AMDCUID_STATUS_PERMISSION_DENIED
                                           : AMDCUID_STATUS_FILE_ERROR;
  }

  const ssize_t written = write(fd, key, key_length);
  const int err = errno;
  close(fd);

  if (written == static_cast<ssize_t>(key_length)) return AMDCUID_STATUS_SUCCESS;

  LOG(ERROR,
      "amdcuid_set_hash_key: cannot write " << path << ": " << CuidUtilities::errno_string(err));
  return (err == EACCES || err == EPERM) ? AMDCUID_STATUS_PERMISSION_DENIED
                                         : AMDCUID_STATUS_FILE_ERROR;
}

// efivarfs marks the variable immutable, which refuses both opening it for
// writing and chmod. Sets `cleared` when the flag was set and is now clear.
bool clear_immutable(int fd, bool& cleared) {
  cleared = false;
  int flags = 0;
  if (ioctl(fd, FS_IOC_GETFLAGS, &flags) != 0) return false;
  if (!(flags & FS_IMMUTABLE_FL)) return true;
  flags &= ~FS_IMMUTABLE_FL;
  if (ioctl(fd, FS_IOC_SETFLAGS, &flags) != 0) return false;
  cleared = true;
  return true;
}

// Put back what clear_immutable() took off, so the variable is not left
// deletable.
void restore_immutable(int fd) {
  int flags = 0;
  if (ioctl(fd, FS_IOC_GETFLAGS, &flags) == 0) {
    flags |= FS_IMMUTABLE_FL;
    if (ioctl(fd, FS_IOC_SETFLAGS, &flags) == 0) return;
  }
  const int err = errno;
  LOG(WARN, "amdcuid_set_hash_key: cannot restore the immutable flag on "
                << kKeyVariablePath << ": " << CuidUtilities::errno_string(err));
}

// Without amdgpu the key goes straight to AmdCuidKey, flagged as set by an
// administrator. UNSUPPORTED when there is no efivarfs to write it to.
amdcuid_status_t write_key_variable(const uint8_t key[key_length]) {
  struct stat st{};
  if (stat(kEfivarsDir, &st) != 0) return AMDCUID_STATUS_UNSUPPORTED;

  uint8_t buf[4 + kKeyVariablePayloadLen];
  CuidUtilities::build_key_variable(key, buf);

  bool cleared = false;
  bool clear_failed = false;
  int fd = -1;
  const int ro_fd = open(kKeyVariablePath, O_RDONLY | O_CLOEXEC);
  if (ro_fd >= 0) {
    if (!clear_immutable(ro_fd, cleared)) {
      const int err = errno;
      clear_failed = true;
      LOG(WARN, "amdcuid_set_hash_key: cannot clear the immutable flag on "
                    << kKeyVariablePath << ": " << CuidUtilities::errno_string(err));
    }
    fd = open(kKeyVariablePath, O_WRONLY | O_CLOEXEC);
  } else if (errno == ENOENT) {
    fd = open(kKeyVariablePath, O_WRONLY | O_CREAT | O_CLOEXEC, 0600);
  }
  const int open_err = errno;

  ssize_t written = -1;
  int err = open_err;
  if (fd >= 0) {
    written = write(fd, buf, sizeof(buf));
    err = errno;
    close(fd);
  }
  rocm::sha2::secure_zero(buf, sizeof(buf));
  if (ro_fd >= 0) {
    if (cleared) restore_immutable(ro_fd);
    close(ro_fd);
  }

  if (fd < 0) {
    LOG(ERROR, "amdcuid_set_hash_key: cannot open " << kKeyVariablePath << ": "
                                                    << CuidUtilities::errno_string(open_err));
    // With the immutable flag still set, EPERM says nothing about privilege.
    if (clear_failed) return AMDCUID_STATUS_FILE_ERROR;
    return (open_err == EACCES || open_err == EPERM) ? AMDCUID_STATUS_PERMISSION_DENIED
                                                     : AMDCUID_STATUS_FILE_ERROR;
  }
  if (written == static_cast<ssize_t>(sizeof(buf))) return AMDCUID_STATUS_SUCCESS;
  LOG(ERROR, "amdcuid_set_hash_key: cannot write " << kKeyVariablePath << ": "
                                                   << CuidUtilities::errno_string(err));
  return AMDCUID_STATUS_FILE_ERROR;
}

// efivarfs lists a variable only as it was at mount time, so one the driver
// created this boot is absent until the next boot or a resume from
// hibernation, and listed 0644 until the tmpfiles.d rule runs at boot.
void restrict_key_variable() {
  const int fd = open(kKeyVariablePath, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return;
  bool cleared = false;
  clear_immutable(fd, cleared);
  if (fchmod(fd, 0600) != 0)
    LOG(WARN, "amdcuid_set_hash_key: cannot make " << kKeyVariablePath << " mode 0600 ("
                                                   << CuidUtilities::errno_string(errno)
                                                   << "); every local user can read the key");
  if (cleared) restore_immutable(fd);
  close(fd);
}

bool find_cuid_seed_device(std::string& bdf) {
  DIR* dir = opendir("/sys/bus/pci/devices");
  if (!dir) return false;
  bool found = false;
  struct dirent* entry;
  // This call site owns its DIR*, which is all POSIX requires; readdir_r is
  // deprecated and must not be adopted.
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  while (!found && (entry = readdir(dir)) != nullptr) {
    if (entry->d_name[0] == '.') continue;
    struct stat st{};
    const std::string path = std::string("/sys/bus/pci/devices/") + entry->d_name + "/cuid_seed";
    if (stat(path.c_str(), &st) == 0) {
      bdf = entry->d_name;
      found = true;
    }
  }
  closedir(dir);
  return found;
}

}  // namespace

void amdcuid_get_library_version(uint32_t* major, uint32_t* minor, uint32_t* patch) {
  if (major) *major = AMDCUID_LIB_VERSION_MAJOR;
  if (minor) *minor = AMDCUID_LIB_VERSION_MINOR;
  if (patch) *patch = AMDCUID_LIB_VERSION_PATCH;
}

const char* amdcuid_library_version_to_string() {
  static std::string version_str = std::to_string(AMDCUID_LIB_VERSION_MAJOR) + "." +
                                   std::to_string(AMDCUID_LIB_VERSION_MINOR) + "." +
                                   std::to_string(AMDCUID_LIB_VERSION_PATCH);
  return version_str.c_str();
}

const char* amdcuid_status_to_string(amdcuid_status_t status) {
  switch (status) {
    case AMDCUID_STATUS_SUCCESS:
      return "SUCCESS";
    case AMDCUID_STATUS_FILE_NOT_FOUND:
      return "FILE_NOT_FOUND";
    case AMDCUID_STATUS_DEVICE_NOT_FOUND:
      return "DEVICE_NOT_FOUND";
    case AMDCUID_STATUS_INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case AMDCUID_STATUS_PERMISSION_DENIED:
      return "PERMISSION_DENIED";
    case AMDCUID_STATUS_UNSUPPORTED:
      return "UNSUPPORTED";
    case AMDCUID_STATUS_WRONG_DEVICE_TYPE:
      return "WRONG_DEVICE_TYPE";
    case AMDCUID_STATUS_INSUFFICIENT_SIZE:
      return "INSUFFICIENT_SIZE";
    case AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND:
      return "HARDWARE_FINGERPRINT_NOT_FOUND";
    case AMDCUID_STATUS_KEY_ERROR:
      return "KEY_ERROR";
    case AMDCUID_STATUS_HMAC_ERROR:
      return "HMAC_ERROR";
    case AMDCUID_STATUS_FILE_ERROR:
      return "FILE_ERROR";
    case AMDCUID_STATUS_INVALID_FORMAT:
      return "INVALID_FORMAT";
    case AMDCUID_STATUS_PCI_ERROR:
      return "PCI_ERROR";
    case AMDCUID_STATUS_SMBIOS_ERROR:
      return "SMBIOS_ERROR";
    case AMDCUID_STATUS_ACPI_ERROR:
      return "ACPI_ERROR";
    case AMDCUID_STATUS_CPUINFO_ERROR:
      return "CPUINFO_ERROR";
    case AMDCUID_STATUS_IPC_ERROR:
      return "IPC_ERROR";
    default:
      return "UNKNOWN_ERROR";
  }
}

const char* amdcuid_id_to_string(amdcuid_id_t cuid_value) {
  // Use thread_local static buffer to avoid returning dangling pointer from
  // temporary string
  thread_local static char uuid_str[37];  // 36 chars + null terminator
  std::string result = CuidUtilities::get_cuid_as_string(&cuid_value);
  std::strncpy(uuid_str, result.c_str(), sizeof(uuid_str) - 1);
  uuid_str[sizeof(uuid_str) - 1] = '\0';
  return uuid_str;
}

amdcuid_status_t amdcuid_get_all_handles(amdcuid_id_t* handles, uint32_t* count) {
  std::lock_guard<std::recursive_mutex> operation(cuid_operation_mutex());
  if (!count) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }

  amdcuid_status_t status;
  if (geteuid() == 0) global_hmac.reload_key();
  // get all the devices on the system first
  if (mgr.devices().empty()) {
    status = mgr.discover_devices();
    if (status != AMDCUID_STATUS_SUCCESS) {
      return status;
    }
  }

  status = mgr.build_cuid_index();
  if (status != AMDCUID_STATUS_SUCCESS) {
    *count = 0;
    return status;
  }
  auto handle_list = mgr.get_all_handles();
  auto handle_count = static_cast<uint32_t>(handle_list.size());
  if (handle_count == 0) {
    *count = 0;
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  if (*count < handle_count) {
    *count = handle_count;
    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
  }

  if (handles != nullptr) {
    for (uint32_t i = 0; i < handle_count; ++i) {
      std::memcpy(handles[i].bytes, handle_list[i].bytes, 16);
    }
  }

  *count = handle_count;
  return AMDCUID_STATUS_SUCCESS;
}

namespace {

// Discover devices into the manager if it has none cached yet.
amdcuid_status_t enumerate_into_manager() {
  if (!mgr.devices().empty()) {
    return AMDCUID_STATUS_SUCCESS;
  }
  return mgr.discover_devices();
}

// The physical package of the logical CPU at `path`, or -1. A package is
// recorded once, under its lowest logical CPU, and any of its logical CPUs
// names it.
int cpu_package_of(const std::string& path) {
  char buf[PATH_MAX];
  const std::string resolved = realpath(path.c_str(), buf) ? std::string(buf) : path;
  const std::string text =
      CuidUtilities::read_sysfs_file(resolved + "/topology/physical_package_id");
  if (text.empty() || text.size() > 5 || text.find_first_not_of("0123456789") != std::string::npos)
    return -1;
  const long package = std::strtol(text.c_str(), nullptr, 10);
  return package <= 0xFFFF ? static_cast<int>(package) : -1;
}

// helper function to discover device given dev_path
DevicePtr discover_device_by_path(const char* dev_path, amdcuid_device_type_t device_type) {
  DevicePtr device = nullptr;
  amdcuid_status_t status;

  // CPU sysfs paths (e.g., /sys/devices/system/cpu/cpu0) are directories,
  // not character/block devices, so real_dev_path_from_fd() cannot resolve
  // them via /sys/dev/char or /sys/dev/block. Use the path directly,
  // resolving symlinks with realpath but without appending "/device".
  if (device_type == AMDCUID_DEVICE_TYPE_CPU) {
    char buf[PATH_MAX];
    std::string resolved_path;
    if (realpath(dev_path, buf) != nullptr) {
      resolved_path = std::string(buf);
    } else {
      resolved_path = dev_path;
    }
    amdcuid_cpu_info cpu_info = {};
    status = CuidCpu::discover_single(&cpu_info, resolved_path);
    if (status != AMDCUID_STATUS_SUCCESS) {
      return nullptr;
    }
    return std::make_shared<CuidCpu>(cpu_info);
  }

  // NPU sysfs paths may be PCI device directories (e.g.,
  // /sys/bus/pci/devices/0000:c6:00.1) when /sys/class/accel/ is not
  // populated. These are directories, not char/block devices, so the
  // fd-based real path resolution below would fail. Pass directly to
  // discover_single() which knows how to read PCI sysfs attributes.
  if (device_type == AMDCUID_DEVICE_TYPE_NPU) {
    char buf[PATH_MAX];
    std::string resolved_path;
    if (realpath(dev_path, buf) != nullptr) {
      resolved_path = std::string(buf);
    } else {
      resolved_path = dev_path;
    }
    amdcuid_npu_info npu_info = {};
    status = CuidNpu::discover_single(&npu_info, resolved_path);
    if (status != AMDCUID_STATUS_SUCCESS) {
      return nullptr;
    }
    return std::make_shared<CuidNpu>(npu_info);
  }

  // discover_single() takes a sysfs path. Resolve only character/block nodes
  // through their device number: a directory has st_rdev == 0, which would
  // incorrectly resolve through /sys/dev/char/0:0.
  std::string real_dev_path;
  struct stat path_stat = {};
  if (stat(dev_path, &path_stat) != 0) {
    return nullptr;
  }
  if (S_ISCHR(path_stat.st_mode) || S_ISBLK(path_stat.st_mode)) {
    int fd = open(dev_path, O_RDONLY);
    if (fd < 0) {
      // unable to open device path
      return nullptr;
    }
    real_dev_path = CuidUtilities::real_dev_path_from_fd(fd);
    close(fd);
  } else {
    // Preserve the class path: discover_single() extracts its DRM node name.
    // Class directories need "/device" appended to reach the PCI attributes.
    real_dev_path = dev_path;
    struct stat vendor_stat = {};
    struct stat child_stat = {};
    if (stat((real_dev_path + "/vendor").c_str(), &vendor_stat) != 0 &&
        stat((real_dev_path + "/device/vendor").c_str(), &child_stat) == 0) {
      real_dev_path += "/device";
    }
  }
  if (real_dev_path.empty()) {
    return nullptr;
  }
  switch (device_type) {
    case AMDCUID_DEVICE_TYPE_GPU: {
      amdcuid_gpu_info gpu_info = {};
      status = CuidGpu::discover_single(&gpu_info, real_dev_path);
      if (status != AMDCUID_STATUS_SUCCESS) {
        return nullptr;
      }
      device = std::make_shared<CuidGpu>(gpu_info);
      break;
    }
    case AMDCUID_DEVICE_TYPE_NIC: {
      amdcuid_nic_info nic_info = {};
      status = CuidNic::discover_single(&nic_info, real_dev_path);
      if (status != AMDCUID_STATUS_SUCCESS) {
        return nullptr;
      }
      device = std::make_shared<CuidNic>(nic_info);
      break;
    }
    case AMDCUID_DEVICE_TYPE_NPU: {
      amdcuid_npu_info npu_info = {};
      status = CuidNpu::discover_single(&npu_info, real_dev_path);
      if (status != AMDCUID_STATUS_SUCCESS) {
        return nullptr;
      }
      device = std::make_shared<CuidNpu>(npu_info);
      break;
    }
    default:
      return nullptr;
  }
  return device;
}

}  // namespace

amdcuid_status_t amdcuid_get_handle_by_dev_path(const char* dev_path,
                                                amdcuid_device_type_t device_type,
                                                amdcuid_id_t* handle) {
  std::lock_guard<std::recursive_mutex> operation(cuid_operation_mutex());
  if (!dev_path || !handle) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }
  if (geteuid() == 0) global_hmac.reload_key();

  std::string input_dev_path(dev_path);
  CuidGpuRoute gpu_route;
  bool gpu_resolved = false;
  if (device_type == AMDCUID_DEVICE_TYPE_GPU) {
    const auto resolved = CuidGpu::resolve_path(input_dev_path, gpu_route);
    gpu_resolved = resolved == AMDCUID_STATUS_SUCCESS;
    if (!gpu_resolved && (gpu_route.partition || resolved != AMDCUID_STATUS_FILE_NOT_FOUND))
      return resolved;
    if (gpu_resolved) input_dev_path = gpu_route.node;
  }
  std::string real_dev_path;
  // For NIC paths (e.g., /sys/class/net/eth0), GPU paths
  // (e.g., /sys/class/drm/renderD128), and CPU paths
  // (e.g., /sys/devices/system/cpu/cpu0), use the path as-is since
  // get_real_path resolves symlinks and appends "/device" which does not
  // match the device paths discovery stores.
  // CPU and Platform sysfs paths are directories without a /device
  // subdirectory, so the /device suffix would produce an invalid path.
  std::string dev_path_str(input_dev_path);
  if (device_type == AMDCUID_DEVICE_TYPE_NIC || device_type == AMDCUID_DEVICE_TYPE_GPU ||
      device_type == AMDCUID_DEVICE_TYPE_CPU || device_type == AMDCUID_DEVICE_TYPE_PLATFORM ||
      device_type == AMDCUID_DEVICE_TYPE_NPU ||
      dev_path_str.find("/sys/class/net/") != std::string::npos ||
      dev_path_str.find("/sys/class/drm/") != std::string::npos ||
      dev_path_str.find("/sys/class/accel/") != std::string::npos ||
      dev_path_str.find("/sys/bus/pci/devices/") != std::string::npos ||
      dev_path_str.find("/sys/devices/system/cpu/") != std::string::npos) {
    real_dev_path = dev_path_str;
  } else {
    real_dev_path = CuidUtilities::get_real_path(dev_path);
  }

  const int cpu_package =
      device_type == AMDCUID_DEVICE_TYPE_CPU ? cpu_package_of(input_dev_path) : -1;

  const auto matches = [&](const DevicePtr& device) {
    if (device->type() != device_type) return false;
    uint16_t package = 0;
    if (cpu_package >= 0 && device->get_physical_id(package) == AMDCUID_STATUS_SUCCESS &&
        package == cpu_package)
      return true;
    std::string path;
    if (device->get_device_path(path) != AMDCUID_STATUS_SUCCESS) return false;
    if (device_type == AMDCUID_DEVICE_TYPE_GPU) {
      const auto gpu = std::dynamic_pointer_cast<CuidGpu>(device);
      return gpu && (gpu_resolved ? gpu->matches_route(gpu_route) : path == input_dev_path);
    }
    const auto real = CuidUtilities::get_real_path(path);
    return path == input_dev_path || path == real_dev_path ||
           (!real.empty() && real == real_dev_path);
  };
  // check mgr first to see if device is already known
  for (const auto& device : mgr.devices()) {
    if (matches(device)) return current_handle(device, handle);
  }

  const auto enumeration_status = enumerate_into_manager();
  if (enumeration_status == AMDCUID_STATUS_SUCCESS) {
    for (const auto& enumerated : mgr.devices()) {
      if (matches(enumerated)) return current_handle(enumerated, handle);
    }
  } else if (enumeration_status != AMDCUID_STATUS_DEVICE_NOT_FOUND &&
             enumeration_status != AMDCUID_STATUS_UNSUPPORTED) {
    return enumeration_status;
  }

  // Verified GPU paths can be described without reading a secret.
  amdcuid_status_t status;
  DevicePtr device;
  if (gpu_resolved) {
    amdcuid_gpu_info info{};
    status = CuidGpu::discover_single(&info, gpu_route.node);
    if (status != AMDCUID_STATUS_SUCCESS) return status;
    device = std::make_shared<CuidGpu>(info);
  } else {
    device = discover_device_by_path(real_dev_path.c_str(), device_type);
  }

  if (!device) {
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
  } else {
    return current_handle(device, handle);
  }
}

amdcuid_status_t amdcuid_get_handle_by_bdf(const char* bdf, amdcuid_device_type_t device_type,
                                           amdcuid_id_t* handle) {
  std::lock_guard<std::recursive_mutex> operation(cuid_operation_mutex());
  // The BDF becomes a sysfs path component below; anything but DDDD:BB:SS.F
  // could name somewhere else.
  if (!bdf || !handle || !CuidUtilities::is_valid_bdf(bdf)) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }

  if (device_type == AMDCUID_DEVICE_TYPE_CPU || device_type == AMDCUID_DEVICE_TYPE_PLATFORM) {
    return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
  }
  if (geteuid() == 0) global_hmac.reload_key();

  // check mgr first to see if device is already known
  for (const auto& device : mgr.devices()) {
    std::string device_bdf;
    amdcuid_status_t status = device->get_bdf(device_bdf);
    if (status != AMDCUID_STATUS_SUCCESS) {
      continue;
    }
    if (device_bdf == bdf && device->type() == device_type) {
      return current_handle(device, handle);
    }
  }

  const auto enumeration_status = enumerate_into_manager();
  if (enumeration_status == AMDCUID_STATUS_SUCCESS) {
    for (const auto& enumerated : mgr.devices()) {
      std::string device_bdf;
      if (enumerated->get_bdf(device_bdf) != AMDCUID_STATUS_SUCCESS) {
        continue;
      }
      if (device_bdf == bdf && enumerated->type() == device_type) {
        return current_handle(enumerated, handle);
      }
    }
  } else if (enumeration_status != AMDCUID_STATUS_DEVICE_NOT_FOUND &&
             enumeration_status != AMDCUID_STATUS_UNSUPPORTED) {
    return enumeration_status;
  }

  // Not already known: discover just this one device by BDF.
  DevicePtr device = nullptr;
  if (device_type == AMDCUID_DEVICE_TYPE_GPU) {
    CuidGpuRoute route;
    if (CuidGpu::resolve_path(std::string("/sys/bus/pci/devices/") + bdf, route) ==
            AMDCUID_STATUS_SUCCESS &&
        !route.partition)
      return amdcuid_get_handle_by_dev_path(route.node.c_str(), device_type, handle);
  }
  std::string device_path = CuidUtilities::bdf_to_device_path(bdf, device_type);
  if (device_path.empty()) {
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
  }
  // bdf_to_device_path() already returns a sysfs path, not a device node.
  device = discover_device_by_path(device_path.c_str(), device_type);

  if (!device) {
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
  } else {
    return current_handle(device, handle);
  }
}

amdcuid_status_t amdcuid_get_handle_by_fd(int fd, amdcuid_device_type_t device_type,
                                          amdcuid_id_t* handle) {
  if (fd < 0 || !handle) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }

  if (device_type == AMDCUID_DEVICE_TYPE_CPU || device_type == AMDCUID_DEVICE_TYPE_PLATFORM ||
      device_type == AMDCUID_DEVICE_TYPE_NIC) {
    return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
  }

  std::string device_path = CuidUtilities::real_dev_path_from_fd(fd);
  if (device_path.empty()) {
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
  }
  return amdcuid_get_handle_by_dev_path(device_path.c_str(), device_type, handle);
}

amdcuid_status_t amdcuid_refresh() {
  std::lock_guard<std::recursive_mutex> operation(cuid_operation_mutex());
  if (geteuid() == 0) global_hmac.reload_key();
  mgr.shutdown();
  return mgr.discover_devices();
}

amdcuid_status_t amdcuid_query_device_property(amdcuid_id_t handle, amdcuid_query_t query,
                                               void* data, uint32_t* length) {
  std::lock_guard<std::recursive_mutex> operation(cuid_operation_mutex());
  if (!length) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }
  auto device = mgr.lookup_by_handle(handle);
  if (!device) {
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
  }
  amdcuid_derived_id derived{};
  const auto identity_status = device->get_derived_cuid(derived, derivation_key());
  if (identity_status != AMDCUID_STATUS_SUCCESS) return identity_status;
  const amdcuid_id_t current = derived.UUIDv8_representation;
  if (std::memcmp(current.bytes, handle.bytes, sizeof(current.bytes)) != 0)
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
  // Must be initialized: most cases below only assign `status` inside
  // `if (data != nullptr)`, so the size-query form of this API
  // (data == nullptr, *length large enough) would otherwise return the value
  // of an uninitialized variable.
  amdcuid_status_t status = AMDCUID_STATUS_SUCCESS;
  switch (query) {
    case AMDCUID_QUERY_PRIMARY_CUID: {
      if (geteuid() != 0) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
      }
      if (*length < sizeof(amdcuid_id_t)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      amdcuid_primary_id id = {};
      status = device->get_primary_cuid(id);
      if (data != nullptr) {
        std::memcpy(data, &id.UUIDv8_representation, sizeof(amdcuid_id_t));
      }
      *length = sizeof(amdcuid_id_t);
    } break;
    case AMDCUID_QUERY_DERIVED_CUID: {
      if (*length < sizeof(amdcuid_id_t)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        std::memcpy(data, &current, sizeof(amdcuid_id_t));
      }
      *length = sizeof(amdcuid_id_t);
    } break;
    case AMDCUID_QUERY_HARDWARE_FINGERPRINT: {
      if (geteuid() != 0) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
      }
      if (*length < sizeof(uint64_t)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        std::memset(data, 0, sizeof(uint64_t));
        status = device->get_hardware_fingerprint(*(uint64_t*)data);
      }
      *length = sizeof(uint64_t);
    } break;
    case AMDCUID_QUERY_DEVICE_PATH: {
      std::string path;
      status = device->get_device_path(path);
      if (status != AMDCUID_STATUS_SUCCESS) {
        break;
      }
      uint32_t required_length = static_cast<uint32_t>(path.size() + 1);  // include null terminator
      if (*length < required_length) {
        *length = required_length;
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        std::memcpy(data, path.c_str(), required_length);
      }
      *length = required_length;
    } break;
    case AMDCUID_QUERY_SOURCE: {
      if (*length < sizeof(amdcuid_source_t)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        const amdcuid_source_t source = device->derived_source();
        std::memcpy(data, &source, sizeof(source));
      }
      *length = sizeof(amdcuid_source_t);
      break;
    }
    case AMDCUID_QUERY_DEVICE_TYPE: {
      if (*length < sizeof(amdcuid_device_type_t)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        amdcuid_device_type_t device_type = device->type();
        std::memcpy(data, &device_type, sizeof(amdcuid_device_type_t));
      }
      *length = sizeof(amdcuid_device_type_t);
      status = AMDCUID_STATUS_SUCCESS;
    } break;
    case AMDCUID_QUERY_VENDOR_ID: {
      if (*length < sizeof(uint16_t)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        uint16_t vendor_id = 0;
        status = device->get_vendor_id(vendor_id);
        std::memcpy(data, &vendor_id, sizeof(uint16_t));
      }
      *length = sizeof(uint16_t);
    } break;
    case AMDCUID_QUERY_DEVICE_ID: {
      if (*length < sizeof(uint16_t)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        uint16_t device_id = 0;
        status = device->get_device_id(device_id);
        std::memcpy(data, &device_id, sizeof(uint16_t));
      }
      *length = sizeof(uint16_t);
    } break;
    case AMDCUID_QUERY_REVISION_ID: {
      if (*length < sizeof(uint8_t)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        uint8_t revision_id = 0;
        status = device->get_revision_id(revision_id);
        std::memcpy(data, &revision_id, sizeof(uint8_t));
      }
      *length = sizeof(uint8_t);
    } break;
    case AMDCUID_QUERY_UNIT_ID: {
      if (*length < sizeof(uint16_t)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        uint16_t unit_id = 0;
        status = device->get_unit_id(unit_id);
        std::memcpy(data, &unit_id, sizeof(uint16_t));
      }
      *length = sizeof(uint16_t);
    } break;
    case AMDCUID_QUERY_FAMILY: {
      // only CPU devices will return a valid family
      if (*length < sizeof(uint16_t)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        uint16_t family = 0;
        status = device->get_family(family);
        std::memcpy(data, &family, sizeof(uint16_t));
      }
      *length = sizeof(uint16_t);
    } break;
    case AMDCUID_QUERY_MODEL: {
      // only CPU devices will return a valid model
      if (*length < sizeof(uint16_t)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        uint16_t model = 0;
        status = device->get_model(model);
        std::memcpy(data, &model, sizeof(uint16_t));
      }
      *length = sizeof(uint16_t);
    } break;
    case AMDCUID_QUERY_CORE_ID: {
      // only CPU devices will return a valid core ID
      if (*length < sizeof(uint16_t)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        uint16_t core = 0;
        status = device->get_core(core);
        std::memcpy(data, &core, sizeof(uint16_t));
      }
      *length = sizeof(uint16_t);
    } break;
    case AMDCUID_QUERY_PHYSICAL_ID: {
      // only CPU devices will return a valid physical package ID
      if (*length < sizeof(uint16_t)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        uint16_t physical_id = 0;
        status = device->get_physical_id(physical_id);
        std::memcpy(data, &physical_id, sizeof(uint16_t));
      }
      *length = sizeof(uint16_t);
    } break;
    case AMDCUID_QUERY_PCI_CLASS: {
      // only PCI devices (GPU, NIC) will return a valid PCI class
      if (*length < sizeof(uint16_t)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        uint16_t pci_class = 0;
        status = device->get_pci_class(pci_class);
        std::memcpy(data, &pci_class, sizeof(uint16_t));
      }
      *length = sizeof(uint16_t);
    } break;
    case AMDCUID_QUERY_BDF: {
      // only PCI devices (GPU, NIC) will return a valid BDF
      std::string bdf;
      status = device->get_bdf(bdf);
      if (status != AMDCUID_STATUS_SUCCESS) {
        break;
      }
      uint32_t required_length = static_cast<uint32_t>(bdf.size() + 1);  // include null terminator
      if (*length < required_length) {
        *length = required_length;
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        std::memcpy(data, bdf.c_str(), required_length);
      }
      *length = required_length;
    } break;
    case AMDCUID_QUERY_TEMPORARY_CUID: {
      if (*length < sizeof(bool)) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
      }
      if (data != nullptr) {
        bool is_temporary = false;
        status = device->is_temporary_cuid(&is_temporary, derivation_key());
        *(bool*)data = is_temporary;
      }
      *length = sizeof(bool);
    } break;
    default:
      status = AMDCUID_STATUS_INVALID_ARGUMENT;
      break;
  }

  return status;
}

amdcuid_status_t amdcuid_set_hash_key(const uint8_t key[32]) {
  std::lock_guard<std::recursive_mutex> operation(cuid_operation_mutex());
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }
  if (!key) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }

  if (CuidUtilities::is_rejected_key(key)) return AMDCUID_STATUS_INVALID_ARGUMENT;

  std::string bdf;
  const auto status =
      find_cuid_seed_device(bdf) ? write_driver_seed(bdf, key) : write_key_variable(key);
  if (status != AMDCUID_STATUS_SUCCESS) return status;
  restrict_key_variable();
  return amdcuid_refresh();
}

amdcuid_status_t amdcuid_get_key_info(amdcuid_key_info_t* info) {
  std::lock_guard<std::recursive_mutex> operation(cuid_operation_mutex());
  if (!info) return AMDCUID_STATUS_INVALID_ARGUMENT;

  std::memset(info, 0, sizeof(*info));
  if (geteuid() != 0) return AMDCUID_STATUS_PERMISSION_DENIED;

  const auto status = global_hmac.reload_key();
  return status == AMDCUID_STATUS_SUCCESS ? global_hmac.get_key_info(info) : status;
}

amdcuid_status_t amdcuid_generate_hash_key(uint8_t key[32]) {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }
  if (!key) return AMDCUID_STATUS_INVALID_ARGUMENT;

  return global_hmac.generate_key(key);
}
