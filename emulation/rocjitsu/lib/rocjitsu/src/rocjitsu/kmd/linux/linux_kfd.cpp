// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file linux_kfd.cpp
/// @brief Shared Linux KFD ioctl helpers.

#include "rocjitsu/kmd/linux/linux_kfd.h"

#include "rocjitsu/kmd/linux/kfd_ioctl_utils.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <string_view>

namespace rocjitsu {

namespace {

constexpr std::string_view kKfdTopologyPrefix = "/sys/devices/virtual/kfd/kfd/topology";
constexpr std::string_view kKfdTopologyAltPrefix = "/sys/class/kfd/kfd/topology";
constexpr std::string_view kDrmSysfsPrefix = "/sys/class/drm";
constexpr std::string_view kDrmRenderPrefix = "/sys/class/drm/renderD";
constexpr std::string_view kSysDevCharPrefix = "/sys/dev/char/";

} // namespace

std::string LinuxKfd::redirect_sysfs_root_path(const char *path, const std::string &topology_path,
                                               const std::string &drm_path) {
  if (!path)
    return {};

  std::string_view sv(path);
  if (!topology_path.empty()) {
    if (sv.starts_with(kKfdTopologyPrefix))
      return topology_path + std::string(sv.substr(kKfdTopologyPrefix.size()));
    if (sv.starts_with(kKfdTopologyAltPrefix))
      return topology_path + std::string(sv.substr(kKfdTopologyAltPrefix.size()));
  }

  if (!drm_path.empty() && sv.starts_with(kDrmSysfsPrefix))
    return drm_path + std::string(sv.substr(kDrmSysfsPrefix.size()));

  return {};
}

bool LinuxKfd::parse_drm_render_path(std::string_view path, uint32_t *minor,
                                     std::string_view *suffix) {
  if (!minor || !suffix || !path.starts_with(kDrmRenderPrefix))
    return false;

  auto rest = path.substr(kDrmRenderPrefix.size());
  auto slash = rest.find('/');
  auto number = slash == std::string_view::npos ? rest : rest.substr(0, slash);
  uint32_t parsed = 0;
  auto [ptr, err] = std::from_chars(number.data(), number.data() + number.size(), parsed);
  if (err != std::errc{} || ptr != number.data() + number.size())
    return false;

  *minor = parsed;
  *suffix = slash == std::string_view::npos ? std::string_view{} : rest.substr(slash);
  return true;
}

bool LinuxKfd::parse_sys_dev_drm_render_path(std::string_view path, uint32_t *minor,
                                             std::string_view *suffix) {
  if (!minor || !suffix || !path.starts_with(kSysDevCharPrefix))
    return false;

  auto rest = path.substr(kSysDevCharPrefix.size());
  auto colon = rest.find(':');
  if (colon == std::string_view::npos)
    return false;

  uint32_t major = 0;
  auto [major_ptr, major_err] = std::from_chars(rest.data(), rest.data() + colon, major);
  if (major_err != std::errc{} || major_ptr != rest.data() + colon || major != 226)
    return false;

  auto after_colon = rest.substr(colon + 1);
  auto slash = after_colon.find('/');
  auto number = slash == std::string_view::npos ? after_colon : after_colon.substr(0, slash);
  uint32_t parsed = 0;
  auto [ptr, err] = std::from_chars(number.data(), number.data() + number.size(), parsed);
  if (err != std::errc{} || ptr != number.data() + number.size())
    return false;

  *minor = parsed;
  *suffix = slash == std::string_view::npos ? std::string_view{} : after_colon.substr(slash);
  return true;
}

const char *LinuxKfd::ioctl_name(unsigned long request) {
  switch (canonical_ioctl_request(request)) {
  case AMDKFD_IOC_GET_VERSION:
    return "GET_VERSION";
  case AMDKFD_IOC_GET_CLOCK_COUNTERS:
    return "GET_CLOCK_COUNTERS";
  case AMDKFD_IOC_GET_PROCESS_APERTURES_NEW:
    return "GET_APERTURES";
  case AMDKFD_IOC_ACQUIRE_VM:
    return "ACQUIRE_VM";
  case AMDKFD_IOC_ALLOC_MEMORY_OF_GPU:
    return "ALLOC_MEMORY";
  case AMDKFD_IOC_FREE_MEMORY_OF_GPU:
    return "FREE_MEMORY";
  case AMDKFD_IOC_MAP_MEMORY_TO_GPU:
    return "MAP_MEMORY";
  case AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU:
    return "UNMAP_MEMORY";
  case AMDKFD_IOC_CREATE_QUEUE:
    return "CREATE_QUEUE";
  case AMDKFD_IOC_UPDATE_QUEUE:
    return "UPDATE_QUEUE";
  case AMDKFD_IOC_DESTROY_QUEUE:
    return "DESTROY_QUEUE";
  case AMDKFD_IOC_CREATE_EVENT:
    return "CREATE_EVENT";
  case AMDKFD_IOC_DESTROY_EVENT:
    return "DESTROY_EVENT";
  case AMDKFD_IOC_SET_EVENT:
    return "SET_EVENT";
  case AMDKFD_IOC_RESET_EVENT:
    return "RESET_EVENT";
  case AMDKFD_IOC_WAIT_EVENTS:
    return "WAIT_EVENTS";
  case AMDKFD_IOC_RUNTIME_ENABLE:
    return "RUNTIME_ENABLE";
  case AMDKFD_IOC_SET_SCRATCH_BACKING_VA:
    return "SET_SCRATCH_VA";
  case AMDKFD_IOC_SET_TRAP_HANDLER:
    return "SET_TRAP_HANDLER";
  case AMDKFD_IOC_SET_XNACK_MODE:
    return "SET_XNACK";
  case AMDKFD_IOC_SET_MEMORY_POLICY:
    return "SET_MEM_POLICY";
  case AMDKFD_IOC_AVAILABLE_MEMORY:
    return "AVAIL_MEMORY";
  case AMDKFD_IOC_GET_TILE_CONFIG:
    return "GET_TILE_CONFIG";
  case AMDKFD_IOC_SVM:
    return "SVM";
  default:
    return "UNKNOWN";
  }
}

int LinuxKfd::get_version_ioctl(void *arg) { return fill_get_version_ioctl(arg); }

int LinuxKfd::fill_get_version_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_version_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }
  args->major_version = KFD_IOCTL_MAJOR_VERSION;
  args->minor_version = KFD_IOCTL_MINOR_VERSION;
  return 0;
}

int LinuxKfd::get_clock_counters_ioctl(void *arg) { return fill_get_clock_counters_ioctl(arg); }

int LinuxKfd::fill_get_clock_counters_ioctl(void *arg) {
  auto *args = static_cast<kfd_ioctl_get_clock_counters_args *>(arg);
  if (!args) {
    errno = EINVAL;
    return -1;
  }

  auto now = std::chrono::steady_clock::now().time_since_epoch();
  uint64_t ns =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  args->system_clock_freq = 1000000000ULL;
  args->system_clock_counter = ns;
  args->cpu_clock_counter = ns;
  args->gpu_clock_counter = ns;
  return 0;
}

int LinuxKfd::get_process_apertures_ioctl(void *) {
  errno = ENOTSUP;
  return -1;
}

int LinuxKfd::acquire_vm_ioctl(void *) {
  errno = ENOTSUP;
  return -1;
}

int LinuxKfd::get_available_memory_ioctl(void *) {
  errno = ENOTSUP;
  return -1;
}

int LinuxKfd::set_memory_policy_ioctl(void *) {
  errno = ENOTSUP;
  return -1;
}

int LinuxKfd::alloc_memory_ioctl(void *) {
  errno = ENOTSUP;
  return -1;
}

int LinuxKfd::free_memory_ioctl(void *) {
  errno = ENOTSUP;
  return -1;
}

int LinuxKfd::map_memory_ioctl(void *) {
  errno = ENOTSUP;
  return -1;
}

int LinuxKfd::unmap_memory_ioctl(void *) {
  errno = ENOTSUP;
  return -1;
}

} // namespace rocjitsu
