// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "src/cuid_cpu.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "src/cuid_util.h"
#include "src/hmac.h"
#include "src/smbios_util.h"

#ifdef __x86_64__
#include <cpuid.h>
#endif

CuidCpu::CuidCpu(const amdcuid_cpu_info& i) : m_info(i) {}

namespace {

/**
 * @brief Get CPU information from CPUID instruction (x86_64)
 */
bool get_cpuid_info(uint16_t& vendor_id, uint16_t& family, uint16_t& model, uint16_t& device_id,
                    uint8_t& stepping) {
#ifdef __x86_64__
  uint32_t eax, ebx, ecx, edx;

  // CPUID leaf 0: Get vendor string
  if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
    return false;
  }

  // AuthenticAMD or GenuineIntel; any other vendor has no CUID.
  if (ebx == 0x68747541 && edx == 0x69746E65 && ecx == 0x444D4163) {
    vendor_id = 0x1022;
  } else if (ebx == 0x756E6547 && edx == 0x49656E69 && ecx == 0x6C65746E) {
    vendor_id = 0x8086;
  } else {
    vendor_id = 0;
  }

  // CPUID leaf 1: Get family, model, stepping
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    return false;
  }

  // Extract CPU signature fields from EAX
  stepping = eax & 0x0F;                     // Bits 0-3: Stepping
  uint32_t base_model = (eax >> 4) & 0x0F;   // Bits 4-7: Base Model
  uint32_t base_family = (eax >> 8) & 0x0F;  // Bits 8-11: Base Family
  uint32_t ext_model = (eax >> 16) & 0x0F;   // Bits 16-19: Extended Model
  uint32_t ext_family = (eax >> 20) & 0xFF;  // Bits 20-27: Extended Family

  // Calculate DisplayFamily and DisplayModel per AMD/Intel spec
  if (base_family == 0x0F) {
    family = static_cast<uint16_t>(base_family + ext_family);
  } else {
    family = static_cast<uint16_t>(base_family);
  }

  if (base_family == 0x0F || (vendor_id == 0x8086 && base_family == 0x06)) {
    model = static_cast<uint16_t>((ext_model << 4) | base_model);
  } else {
    model = static_cast<uint16_t>(base_model);
  }

  // DeviceID = combination of Family and Model
  device_id = static_cast<uint16_t>((family << 8) | model);

  return true;
#else
  return false;  // Not x86_64, cannot use CPUID
#endif
}

// A non-negative decimal field, or -1 when malformed.
int parse_int(const std::string& text) {
  if (text.empty()) return -1;
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (errno == ERANGE || *end != '\0' || value < 0 || value > INT_MAX) return -1;
  return static_cast<int>(value);
}

}  // namespace

namespace {

struct CpuSignature {
  uint16_t family = 0;
  uint16_t model = 0;
  uint8_t stepping = 0;
};

// Family, model and stepping of package `package`, from the first
// /proc/cpuinfo block whose physical id names it; a malformed one is an error,
// not a reason to try the next. These are the values CPUID leaf 1 gives on a
// logical CPU of that package.
amdcuid_status_t package_signature(const std::string& cpuinfo_path, int package,
                                   CpuSignature& signature) {
  std::ifstream cpuinfo(cpuinfo_path);
  if (!cpuinfo) return AMDCUID_STATUS_CPUINFO_ERROR;
  int physical_id = -1, family = -1, model = -1, stepping = -1;
  std::string line;
  bool found = false;
  while (std::getline(cpuinfo, line)) {
    if (line.empty()) {
      if (physical_id == package) {
        found = true;
        break;
      }
      physical_id = family = model = stepping = -1;
      continue;
    }
    const size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string key = line.substr(0, colon);
    std::string value = line.substr(colon + 1);
    while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(0, 1);
    if (key == "physical id") {
      physical_id = parse_int(value);
    } else if (key == "cpu family") {
      family = parse_int(value);
    } else if (key == "model") {
      model = parse_int(value);
    } else if (key == "stepping") {
      stepping = parse_int(value);
    }
  }
  if (!found && physical_id != package) return AMDCUID_STATUS_CPUINFO_ERROR;
  if (family < 0 || family > 0xFFFF || model < 0 || model > 0xFF || stepping < 0 || stepping > 0xF)
    return AMDCUID_STATUS_INVALID_FORMAT;
  signature.family = static_cast<uint16_t>(family);
  signature.model = static_cast<uint16_t>(model);
  signature.stepping = static_cast<uint8_t>(stepping);
  return AMDCUID_STATUS_SUCCESS;
}

// Logical CPU number of a "cpuN" directory name, or -1.
int logical_cpu(const std::string& name) {
  if (name.size() < 4 || name.compare(0, 3, "cpu") != 0) return -1;
  return parse_int(name.substr(3));
}

amdcuid_status_t fill_package(amdcuid_cpu_info& info, uint16_t vendor_id, int package,
                              const std::string& cpuinfo_path) {
  CpuSignature signature;
  const amdcuid_status_t status = package_signature(cpuinfo_path, package, signature);
  if (status != AMDCUID_STATUS_SUCCESS) return status;
  std::memset(&info.header, 0, sizeof(info.header));
  info.header.device_type = AMDCUID_DEVICE_TYPE_CPU;
  info.header.fields.cpu.vendor_id = vendor_id;
  info.header.fields.cpu.family = signature.family;
  info.header.fields.cpu.model = signature.model;
  info.header.fields.cpu.device_id =
      static_cast<uint16_t>(((signature.family & 0xFF) << 8) | signature.model);
  info.header.fields.cpu.revision_id = signature.stepping;
  // UnitID names a sub-unit, and a package is published as a whole, so zero.
  // Not an APIC ID: that moves with firmware enumeration or a disabled core.
  // The physical package ID in S's routing ID separates the sockets.
  info.header.fields.cpu.unit_id = 0;
  info.header.fields.cpu.physical_id = static_cast<uint16_t>(package);
  return AMDCUID_STATUS_SUCCESS;
}

}  // namespace

amdcuid_status_t CuidCpu::discover(std::vector<DevicePtr>& cpus) {
  return discover("/sys/devices/system/cpu", "/proc/cpuinfo", cpus);
}

amdcuid_status_t CuidCpu::discover(const std::string& cpu_base_path,
                                   const std::string& cpuinfo_path, std::vector<DevicePtr>& cpus) {
  cpus.clear();
  uint16_t vendor_id = 0, family = 0, model = 0, device_id = 0;
  uint8_t stepping = 0;
  if (!get_cpuid_info(vendor_id, family, model, device_id, stepping) || vendor_id == 0) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }

  DIR* dir = opendir(cpu_base_path.c_str());
  if (!dir) return AMDCUID_STATUS_CPUINFO_ERROR;
  // Each package is represented by its lowest-numbered logical CPU.
  std::map<int, int> packages;
  struct dirent* entry;
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  while ((entry = readdir(dir)) != NULL) {
    const int cpu = logical_cpu(entry->d_name);
    if (cpu < 0) continue;
    const std::string id_path =
        cpu_base_path + "/" + entry->d_name + "/topology/physical_package_id";
    if (access(id_path.c_str(), F_OK) != 0) continue;
    const int package = parse_int(CuidUtilities::read_sysfs_file(id_path));
    if (package < 0 || package > UINT16_MAX) {
      LOG(DEBUG, "CUID discovery: skipping " << entry->d_name << " (bad physical_package_id)");
      continue;
    }
    auto found = packages.find(package);
    if (found == packages.end() || cpu < found->second) packages[package] = cpu;
  }
  closedir(dir);

  for (const auto& package : packages) {
    amdcuid_cpu_info info{};
    const amdcuid_status_t status = fill_package(info, vendor_id, package.first, cpuinfo_path);
    if (status != AMDCUID_STATUS_SUCCESS) {
      LOG(DEBUG, "CUID discovery: skipping CPU package "
                     << package.first << " (" << amdcuid_status_to_string(status) << ")");
      continue;
    }
    info.header.fields.cpu.core = static_cast<uint16_t>(package.second);
    info.device_node = "/sys/devices/system/cpu/cpu" + std::to_string(package.second);
    cpus.emplace_back(std::make_shared<CuidCpu>(info));
  }
  if (cpus.empty()) return AMDCUID_STATUS_CPUINFO_ERROR;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidCpu::discover_single(amdcuid_cpu_info* cpu_info,
                                          const std::string& device_path) {
  return discover_single(cpu_info, device_path, "/proc/cpuinfo");
}

amdcuid_status_t CuidCpu::discover_single(amdcuid_cpu_info* cpu_info,
                                          const std::string& device_path,
                                          const std::string& cpuinfo_path) {
  if (!cpu_info) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }
  if (device_path.find("cpu") == std::string::npos) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }

  const std::string physical_package_str =
      CuidUtilities::read_sysfs_file(device_path + "/topology/physical_package_id");
  if (physical_package_str.empty()) return AMDCUID_STATUS_DEVICE_NOT_FOUND;

  uint16_t vendor_id = 0, family = 0, model = 0, device_id = 0;
  uint8_t stepping = 0;
  if (!get_cpuid_info(vendor_id, family, model, device_id, stepping))
    return AMDCUID_STATUS_UNSUPPORTED;
  if (vendor_id == 0) return AMDCUID_STATUS_UNSUPPORTED;

  const int physical_id = parse_int(physical_package_str);
  if (physical_id < 0 || physical_id > UINT16_MAX) return AMDCUID_STATUS_INVALID_FORMAT;
  const size_t slash = device_path.find_last_of('/');
  const int cpu = logical_cpu(device_path.substr(slash == std::string::npos ? 0 : slash + 1));

  amdcuid_cpu_info info{};
  const amdcuid_status_t status = fill_package(info, vendor_id, physical_id, cpuinfo_path);
  if (status != AMDCUID_STATUS_SUCCESS) return status;
  info.header.fields.cpu.core = static_cast<uint16_t>(cpu > 0 && cpu <= UINT16_MAX ? cpu : 0);
  info.device_node = device_path;
  *cpu_info = info;
  return AMDCUID_STATUS_SUCCESS;
}

namespace {

// MSR_AMD_PPIN and MSR_PPIN in the kernel's msr-index.h.
constexpr uint32_t kAmdPpinCtlMsr = 0xC00102F0;
constexpr uint32_t kAmdPpinMsr = 0xC00102F1;
constexpr uint32_t kIntelPpinMsr = 0x4F;

amdcuid_status_t io_error_status(int error) {
  return (error == EACCES || error == EPERM) ? AMDCUID_STATUS_PERMISSION_DENIED
                                             : AMDCUID_STATUS_FILE_ERROR;
}

bool parse_hex(const char* text, uint64_t& value) {
  if (!std::isxdigit(static_cast<unsigned char>(*text))) return false;
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text, &end, 16);
  if (errno == ERANGE) return false;
  while (std::isspace(static_cast<unsigned char>(*end))) ++end;
  if (*end != '\0') return false;
  value = parsed;
  return true;
}

// The kernel's topology/ppin: 0400, "0x%llx\n", absent when the CPU has no
// usable PPIN. `exists` is false only when the file is not there.
amdcuid_status_t read_topology_ppin(const std::string& path, bool& exists, uint64_t& ppin) {
  exists = true;
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    if (errno == ENOENT) {
      exists = false;
      return AMDCUID_STATUS_SUCCESS;
    }
    return io_error_status(errno);
  }
  char text[64];
  const ssize_t length = read(fd, text, sizeof(text) - 1);
  const int error = errno;
  close(fd);
  if (length < 0) return io_error_status(error);
  text[length] = '\0';
  return parse_hex(text, ppin) ? AMDCUID_STATUS_SUCCESS : AMDCUID_STATUS_INVALID_FORMAT;
}

// Whether the kernel lists `flag` for logical CPU `cpu` in /proc/cpuinfo.
amdcuid_status_t cpuinfo_has_flag(const std::string& cpuinfo_path, int cpu, const char* flag,
                                  bool& has_flag) {
  has_flag = false;
  std::ifstream cpuinfo(cpuinfo_path);
  if (!cpuinfo) return AMDCUID_STATUS_CPUINFO_ERROR;
  std::string line;
  bool found = false;
  while (std::getline(cpuinfo, line)) {
    if (line.empty()) {
      if (found) break;
      continue;
    }
    const size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string key = line.substr(0, colon);
    while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
    if (key == "processor") {
      std::string value = line.substr(colon + 1);
      while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(0, 1);
      found = parse_int(value) == cpu;
    } else if (found && key == "flags") {
      std::istringstream flags(line.substr(colon + 1));
      std::string word;
      while (flags >> word) {
        if (word == flag) has_flag = true;
      }
    }
  }
  return found ? AMDCUID_STATUS_SUCCESS : AMDCUID_STATUS_CPUINFO_ERROR;
}

// The kernel sets amd_ppin only since 5.7; before that a missing flag says
// nothing. An unparsable release is treated as old.
bool kernel_at_least(const std::string& release, int major, int minor) {
  const size_t dot = release.find('.');
  if (dot == std::string::npos) return false;
  const size_t end = std::min(release.find_first_not_of("0123456789", dot + 1), release.size());
  const int found_major = parse_int(release.substr(0, dot));
  const int found_minor = end > dot + 1 ? parse_int(release.substr(dot + 1, end - dot - 1)) : -1;
  if (found_major < 0 || found_minor < 0) return false;
  return found_major > major || (found_major == major && found_minor >= minor);
}

amdcuid_status_t read_ppin_msr(const std::string& msr_base_path, int cpu, uint32_t msr,
                               uint64_t& ppin) {
  const std::string path = msr_base_path + "/" + std::to_string(cpu) + "/msr";
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return io_error_status(errno);
  const ssize_t length = pread(fd, &ppin, sizeof(ppin), static_cast<off_t>(msr));
  const int error = errno;
  close(fd);
  if (length < 0) return io_error_status(error);
  return length == sizeof(ppin) ? AMDCUID_STATUS_SUCCESS : AMDCUID_STATUS_FILE_ERROR;
}

}  // namespace

amdcuid_status_t read_cpu_ppin(const std::string& cpu_base_path, const std::string& cpuinfo_path,
                               const std::string& msr_base_path, const std::string& kernel_release,
                               bool amd_ppin_cpuid, uint16_t vendor_id, int cpu, uint64_t& ppin) {
  ppin = 0;
#ifndef __x86_64__
  return AMDCUID_STATUS_SUCCESS;
#else
  const std::string topology_path = cpu_base_path + "/cpu" + std::to_string(cpu) + "/topology/ppin";
  bool exists = false;
  amdcuid_status_t status = read_topology_ppin(topology_path, exists, ppin);
  if (status != AMDCUID_STATUS_SUCCESS) {
    ppin = 0;
    return status;
  }
  if (exists) return AMDCUID_STATUS_SUCCESS;

  // Before Linux 5.18 only the msr driver exposes PPIN. The kernel sets
  // amd_ppin (since 5.7) or intel_ppin only when PPIN is there to read, so on a
  // kernel that sets the flag its absence is the absence of PPIN, and its
  // presence makes a failed MSR read an error rather than a reason to hash the
  // SMBIOS UUID.
  const char* flag = nullptr;
  uint32_t msr = 0;
  if (vendor_id == 0x1022) {
    flag = "amd_ppin";
    msr = kAmdPpinMsr;
  } else if (vendor_id == 0x8086) {
    flag = "intel_ppin";
    msr = kIntelPpinMsr;
  } else {
    return AMDCUID_STATUS_SUCCESS;
  }
  bool has_flag = false;
  status = cpuinfo_has_flag(cpuinfo_path, cpu, flag, has_flag);
  if (status != AMDCUID_STATUS_SUCCESS) return status;
  if (!has_flag) {
    // A kernel older than 5.7 has no amd_ppin, so decide as its ppin_init()
    // does: the CPUID bit, then PPIN_CTL, which says whether firmware locked
    // PPIN out.
    if (vendor_id != 0x1022 || kernel_at_least(kernel_release, 5, 7) || !amd_ppin_cpuid)
      return AMDCUID_STATUS_SUCCESS;
    uint64_t control = 0;
    status = read_ppin_msr(msr_base_path, cpu, kAmdPpinCtlMsr, control);
    if (status != AMDCUID_STATUS_SUCCESS) return status;
    constexpr uint64_t kLockout = 1, kEnable = 2;
    if ((control & kLockout) && !(control & kEnable)) return AMDCUID_STATUS_SUCCESS;
  }
  status = read_ppin_msr(msr_base_path, cpu, msr, ppin);
  if (status != AMDCUID_STATUS_SUCCESS) ppin = 0;
  return status;
#endif
}

namespace {

// CPUID Fn8000_0008 EBX[23]: the CPU implements PPIN.
bool amd_ppin_cpuid_bit() {
#ifdef __x86_64__
  uint32_t eax, ebx, ecx, edx;
  return __get_cpuid(0x80000008, &eax, &ebx, &ecx, &edx) && (ebx & (1u << 23));
#else
  return false;
#endif
}

}  // namespace

/**
 * @brief Get hardware fingerprint for CPU at socket granularity.
 *
 * Priority:
 * 1. PPIN (per-socket eFuse serial), read from the kernel's topology/ppin or,
 *    on older kernels, the vendor's PPIN MSR; reading it needs root.
 * 2. Only when the CPU has no PPIN: SMBIOS system uuid + physical_id
 *    (platform-level, requires root)
 *
 * Both tiers produce a per-socket fingerprint consistent with the spec's
 * intent that CPU identity is tied to the physical package, not logical cores.
 * A PPIN the caller cannot read is an error, never a reason to take tier 2:
 * the same socket would otherwise get a different CUID depending on whether
 * the msr module is loaded.
 */
amdcuid_status_t CuidCpu::get_hardware_fingerprint(uint64_t& fingerprint) const {
  uint64_t ppin = 0;
  struct utsname name = {};
  uname(&name);
  const amdcuid_status_t ppin_status = read_cpu_ppin(
      "/sys/devices/system/cpu", "/proc/cpuinfo", "/dev/cpu", name.release, amd_ppin_cpuid_bit(),
      m_info.header.fields.cpu.vendor_id, m_info.header.fields.cpu.core, ppin);
  if (ppin_status != AMDCUID_STATUS_SUCCESS) return ppin_status;
  if (ppin != 0) {
    fingerprint = ppin;
    return AMDCUID_STATUS_SUCCESS;
  }

  // Fallback: SMBIOS system uuid hashed with the socket index (physical_id).
  // physical_id combined with the platform uuid uniquely identifies
  // "socket N on this board" for per-package scope.
  uint8_t smbios_uuid[16] = {};
  amdcuid_status_t status = SmbiosUtil::get_system_uuid(smbios_uuid);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }

  // Append physical_id as little-endian uint16_t so each socket on the same
  // board produces distinct fingerprint.
  uint16_t phys_id = m_info.header.fields.cpu.physical_id;
  uint8_t input[18];
  std::memcpy(input, smbios_uuid, 16);
  std::memcpy(input + 16, &phys_id, sizeof(phys_id));

  uint8_t digest[32];
  status = CuidUtilities::sha256_unkeyed(input, sizeof(input), digest);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }

  std::memcpy(&fingerprint, digest, sizeof(fingerprint));
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidCpu::get_primary_cuid(amdcuid_primary_id& id) const {
  // Hardware fingerprint: PPIN, or the SMBIOS serial plus the socket index. A
  // socket with no serial has no primary (HW_FINGERPRINT_NOT_FOUND) and takes
  // its temporary CUID; a serial this caller may not read is returned as that
  // error, since the identity a socket presents must not depend on the
  // privilege of the caller asking.
  uint64_t fingerprint = 0;
  amdcuid_status_t status = get_hardware_fingerprint(fingerprint);
  if (status != AMDCUID_STATUS_SUCCESS) return status;

  const auto& h = m_info.header;
  // An out-of-range UnitID is refused rather than masked onto UnitID 0.
  status = CuidUtilities::generate_primary_cuid(
      fingerprint, h.fields.cpu.unit_id, h.fields.cpu.revision_id, h.fields.cpu.device_id,
      h.fields.cpu.vendor_id, AMDCUID_DEVICE_TYPE_CPU, &id, false);
  if (status != AMDCUID_STATUS_SUCCESS) std::memset(&id, 0, sizeof(id));
  return status;
}

// The Routing ID carries the physical package ID, since UnitID is zero for
// every socket.
amdcuid_status_t CuidCpu::get_temporary_input(CuidUtilities::AuxiliaryInput& aux) const {
  aux = {};
  aux.format = CuidUtilities::kAuxFormatCpu;
  aux.routing_id = m_info.header.fields.cpu.physical_id;
  aux.revision_id = m_info.header.fields.cpu.revision_id;
  aux.device_id = m_info.header.fields.cpu.device_id;
  aux.vendor_id = m_info.header.fields.cpu.vendor_id;
  aux.component_type = static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_CPU);
  aux.unit_id = m_info.header.fields.cpu.unit_id;
  if (aux.unit_id > CuidUtilities::kMaxUnitId) return AMDCUID_STATUS_INVALID_ARGUMENT;
  return AMDCUID_STATUS_SUCCESS;
}

const amdcuid_cpu_info& CuidCpu::get_info() const { return m_info; }

amdcuid_status_t CuidCpu::get_vendor_id(uint16_t& vendor_id) const {
  vendor_id = m_info.header.fields.cpu.vendor_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidCpu::get_family(uint16_t& family) const {
  family = m_info.header.fields.cpu.family;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidCpu::get_model(uint16_t& model) const {
  model = m_info.header.fields.cpu.model;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidCpu::get_device_id(uint16_t& device_id) const {
  device_id = m_info.header.fields.cpu.device_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidCpu::get_revision_id(uint8_t& revision_id) const {
  revision_id = m_info.header.fields.cpu.revision_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidCpu::get_unit_id(uint16_t& unit_id) const {
  unit_id = m_info.header.fields.cpu.unit_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidCpu::get_core(uint16_t& core) const {
  core = m_info.header.fields.cpu.core;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidCpu::get_physical_id(uint16_t& physical_id) const {
  physical_id = m_info.header.fields.cpu.physical_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidCpu::get_device_path(std::string& path) const {
  // device_node is set from the processor number during discovery.
  if (!m_info.device_node.empty()) {
    path = m_info.device_node;
    return AMDCUID_STATUS_SUCCESS;
  }
  // No stored node. unit_id is zero for every socket, so it cannot name one;
  // report no path, as every other device type does.
  return AMDCUID_STATUS_UNSUPPORTED;
}
