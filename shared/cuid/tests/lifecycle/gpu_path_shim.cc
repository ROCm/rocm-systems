// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <linux/magic.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "src/cuid_device.h"

extern "C" int __real_stat(const char*, struct stat*);
extern "C" int __real_fstat(int, struct stat*);
extern "C" amdcuid_status_t
__real__ZN7CuidCpu8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE(
    std::vector<DevicePtr>&);
extern "C" amdcuid_status_t
__real__ZN7CuidNic8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE(
    std::vector<DevicePtr>&);
extern "C" amdcuid_status_t
__real__ZN12CuidPlatform8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE(
    std::vector<DevicePtr>&);

namespace {
__attribute__((constructor(101))) void require_private_namespace() {
  struct statfs fs{};
  struct stat node{};
  if (statfs("/sys", &fs) != 0 || fs.f_type == SYSFS_MAGIC ||
      access("/sys/.cuid-path-fixture", F_OK) != 0 ||
      __real_stat("/dev/dri/renderD128", &node) != 0 || !S_ISCHR(node.st_mode) ||
      node.st_rdev != makedev(1, 3))
    _exit(90);
}

// A fixture that provides SMBIOS and NIC data opts in to real CPU, NIC and
// Platform discovery; the rest see GPUs only.
bool components() { return access("/sys/.cuid-path-components", F_OK) == 0; }

void drm_number(const std::string& path, struct stat* st) {
  if (!S_ISCHR(st->st_mode) || st->st_rdev != makedev(1, 3)) return;
  char* real = realpath(path.c_str(), nullptr);
  if (!real) return;
  const std::string name(real);
  free(real);
  for (const auto& prefix : {std::string("/dev/dri/renderD"), std::string("/dev/dri/card")}) {
    if (name.compare(0, prefix.size(), prefix) != 0) continue;
    const auto digits = name.substr(prefix.size());
    if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) continue;
    st->st_rdev = makedev(226, std::stoul(digits));
  }
}
}  // namespace

extern "C" int __wrap_stat(const char* path, struct stat* st) {
  const int result = __real_stat(path, st);
  if (result == 0) drm_number(path, st);
  return result;
}

extern "C" int __wrap_fstat(int fd, struct stat* st) {
  const int result = __real_fstat(fd, st);
  if (result == 0) drm_number("/proc/self/fd/" + std::to_string(fd), st);
  return result;
}

extern "C" amdcuid_status_t
__wrap__ZN7CuidCpu8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE(
    std::vector<DevicePtr>& devices) {
  if (components())
    return __real__ZN7CuidCpu8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE(devices);
  return AMDCUID_STATUS_SUCCESS;  // The synthetic topology advertises no CPUs.
}
extern "C" amdcuid_status_t
__wrap__ZN7CuidNic8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE(
    std::vector<DevicePtr>& devices) {
  if (components())
    return __real__ZN7CuidNic8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE(devices);
  return AMDCUID_STATUS_UNSUPPORTED;
}
extern "C" amdcuid_status_t
__wrap__ZN7CuidNpu8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE(
    std::vector<DevicePtr>&) {
  return AMDCUID_STATUS_UNSUPPORTED;
}
extern "C" amdcuid_status_t
__wrap__ZN12CuidPlatform8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE(
    std::vector<DevicePtr>& devices) {
  if (components())
    return __real__ZN12CuidPlatform8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE(
        devices);
  return AMDCUID_STATUS_UNSUPPORTED;
}
