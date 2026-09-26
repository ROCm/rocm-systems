// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <linux/magic.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <string>

#include "src/cuid_device.h"
#include "src/cuid_util.h"

extern "C" int __real_stat(const char*, struct stat*);
extern "C" int __real_fstat(int, struct stat*);
extern "C" ssize_t __real_write(int, const void*, size_t);
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

std::string pci(const std::string& bdf) { return "/sys/bus/pci/devices/" + bdf; }

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

void publish(const std::string& dir, const uint8_t* key) {
  std::ifstream input(dir + "/cuid_primary");
  std::string text;
  if (!(input >> text)) return;
  amdcuid_primary_id primary{};
  if (CuidUtilities::uuid_string_to_uint8(text, primary.UUIDv8_representation.bytes) !=
      AMDCUID_STATUS_SUCCESS)
    _exit(91);
  CuidUtilities::remove_UUIDv8_bits(&primary.UUIDv8_representation, primary.raw_bits);
  cuid_hmac hmac(reinterpret_cast<const char*>(key), 32);
  amdcuid_derived_id derived{};
  if (CuidUtilities::generate_derived_cuid(&primary, &derived, &hmac) != AMDCUID_STATUS_SUCCESS)
    _exit(91);
  std::ofstream(dir + "/cuid_derived")
      << amdcuid_id_to_string(derived.UUIDv8_representation) << '\n';
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

extern "C" ssize_t __wrap_write(int fd, const void* data, size_t size) {
  const auto result = __real_write(fd, data, size);
  char name[4096];
  const auto n = readlink(("/proc/self/fd/" + std::to_string(fd)).c_str(), name, sizeof(name) - 1);
  if (n < 0) return result;
  name[n] = 0;
  const std::string path(name);
  if (result != 32 || size != 32 || path.compare(0, 5, "/sys/") != 0 || path.size() < 10 ||
      path.substr(path.size() - 10) != "/cuid_seed")
    return result;
  const auto* key = static_cast<const uint8_t*>(data);
  for (const auto& bdf : {"0000:03:00.0", "0000:63:00.0", "0000:05:00.0"}) {
    const auto dir = pci(bdf);
    if (access((dir + "/cuid_seed").c_str(), F_OK) == 0) {
      std::ofstream seed(dir + "/cuid_seed", std::ios::binary);
      seed.write(static_cast<const char*>(data), size);
      publish(dir, key);
    }
  }
  publish(pci("0000:03:00.0") + "/xcp", key);
  for (unsigned n = 1; n < 64; ++n)
    publish("/sys/devices/platform/amdgpu_xcp." + std::to_string(n) + "/xcp", key);
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
