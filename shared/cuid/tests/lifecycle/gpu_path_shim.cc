// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <linux/capability.h>
#include <linux/magic.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "src/cuid_device.h"
#include "src/cuid_util.h"

extern "C" int __real_stat(const char*, struct stat*);
extern "C" int __real_fstat(int, struct stat*);
extern "C" ssize_t __real_read(int, void*, size_t);
extern "C" ssize_t __real___read_chk(int, void*, size_t, size_t);
extern "C" ssize_t __real_write(int, const void*, size_t);
extern "C" int __real_madvise(void*, size_t, int);
extern "C" void __real__ZN4rocm4sha26sha2566updateEPKhm(void*, const uint8_t*, size_t);
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
    st->st_rdev = makedev(226, static_cast<unsigned>(std::stoul(digits)));
  }
}

std::string fd_path(int fd) {
  char name[4096];
  const auto n = readlink(("/proc/self/fd/" + std::to_string(fd)).c_str(), name, sizeof(name) - 1);
  if (n < 0) return {};
  name[n] = 0;
  return name;
}

bool ends_with(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool has_sys_admin() {
  __user_cap_header_struct header{_LINUX_CAPABILITY_VERSION_3, 0};
  __user_cap_data_struct data[_LINUX_CAPABILITY_U32S_3]{};
  if (syscall(SYS_capget, &header, data) != 0) _exit(91);
  return data[CAP_TO_INDEX(CAP_SYS_ADMIN)].effective & CAP_TO_MASK(CAP_SYS_ADMIN);
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

// amdgpu fails cuid_seed with EPERM without CAP_SYS_ADMIN, and cuid_seed and
// cuid_derived with ENODATA while it holds no key; the fixture stands for that
// with an empty file. /sys/.cuid-path-seed-eio fails every cuid_seed read with
// EIO.
int refused_read(int fd) {
  struct stat st{};
  if (__real_fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) return 0;
  const auto path = fd_path(fd);
  if (path.compare(0, 5, "/sys/") != 0) return 0;
  const bool seed = ends_with(path, "/cuid_seed");
  if (seed && !has_sys_admin()) return EPERM;
  if (seed && access("/sys/.cuid-path-seed-eio", F_OK) == 0) return EIO;
  if (st.st_size == 0 && (seed || ends_with(path, "/cuid_derived"))) return ENODATA;
  return 0;
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

extern "C" ssize_t __wrap_read(int fd, void* data, size_t size) {
  if (const int err = refused_read(fd)) {
    errno = err;
    return -1;
  }
  return __real_read(fd, data, size);
}

// _FORTIFY_SOURCE turns read() into a buffer of known size into __read_chk.
extern "C" ssize_t __wrap___read_chk(int fd, void* data, size_t size, size_t capacity) {
  if (const int err = refused_read(fd)) {
    errno = err;
    return -1;
  }
  return __real___read_chk(fd, data, size, capacity);
}

extern "C" ssize_t __wrap_write(int fd, const void* data, size_t size) {
  const auto result = __real_write(fd, data, size);
  const auto path = fd_path(fd);
  if (result != 32 || size != 32 || path.compare(0, 5, "/sys/") != 0 ||
      !ends_with(path, "/cuid_seed"))
    return result;
  const auto* key = static_cast<const uint8_t*>(data);
  for (const auto& bdf : {"0000:03:00.0", "0000:63:00.0", "0000:05:00.0"}) {
    const auto dir = pci(bdf);
    if (access((dir + "/cuid_seed").c_str(), F_OK) == 0) {
      std::ofstream seed(dir + "/cuid_seed", std::ios::binary);
      seed.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
      publish(dir, key);
    }
  }
  publish(pci("0000:03:00.0") + "/xcp", key);
  for (unsigned xcp = 1; xcp < 64; ++xcp)
    publish("/sys/devices/platform/amdgpu_xcp." + std::to_string(xcp) + "/xcp", key);
  return result;
}

// Defined by a probe that needs to act inside a library call, after the call
// has read the node key, or inside a SHA-256 update.
extern "C" __attribute__((weak)) void cuid_probe_during_discovery();
extern "C" __attribute__((weak)) void cuid_probe_during_hash_update(const uint8_t*, size_t);

#ifndef MADV_WIPEONFORK
#define MADV_WIPEONFORK 18
#endif

// /sys/.cuid-path-no-wipeonfork and /sys/.cuid-path-no-dontdump make that
// madvise() fail, as a kernel without it would.
extern "C" int __wrap_madvise(void* address, size_t length, int advice) {
  if ((advice == MADV_WIPEONFORK && access("/sys/.cuid-path-no-wipeonfork", F_OK) == 0) ||
      (advice == MADV_DONTDUMP && access("/sys/.cuid-path-no-dontdump", F_OK) == 0)) {
    errno = EINVAL;
    return -1;
  }
  return __real_madvise(address, length, advice);
}

// rocm::sha2::sha256::update(const uint8_t*, size_t), as libamdcuid calls it.
extern "C" void __wrap__ZN4rocm4sha26sha2566updateEPKhm(void* hash, const uint8_t* data,
                                                        size_t length) {
  if (cuid_probe_during_hash_update) cuid_probe_during_hash_update(data, length);
  __real__ZN4rocm4sha26sha2566updateEPKhm(hash, data, length);
}

extern "C" amdcuid_status_t
__wrap__ZN7CuidCpu8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE(
    std::vector<DevicePtr>& devices) {
  if (cuid_probe_during_discovery) cuid_probe_during_discovery();
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
