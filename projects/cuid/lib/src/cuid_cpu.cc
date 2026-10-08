// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "src/cuid_cpu.h"

#include <unistd.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "src/cuid_file.h"
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

  // Check for AMD vendor (AuthenticAMD)
  if (ebx == 0x68747541 && edx == 0x69746E65 && ecx == 0x444D4163) {
    vendor_id = 0x1022;  // AMD vendor ID
  } else {
    vendor_id = 0x8086;  // Intel or other
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
    family = base_family + ext_family;
  } else {
    family = base_family;
  }

  if (base_family == 0x0F || (vendor_id == 0x8086 && base_family == 0x06)) {
    model = (ext_model << 4) | base_model;
  } else {
    model = base_model;
  }

  // DeviceID = combination of Family and Model
  device_id = (family << 8) | model;

  return true;
#else
  return false;  // Not x86_64, cannot use CPUID
#endif
}

}  // namespace

/**
 * @brief CPU information parsed from /proc/cpuinfo
 */
struct ProcCpuInfo {
  int processor = -1;   // Logical processor number
  int physical_id = 0;  // Socket/package ID
  int core_id = 0;      // Core ID within package
  int apic_id = 0;      // APIC ID
  int family = 0;
  int model = 0;
  int stepping = 0;
  std::string vendor_id;  // "AuthenticAMD" or "GenuineIntel"
  bool is_amd = false;
};

namespace {

/**
 * @brief Parse /proc/cpuinfo to discover CPUs without root privileges
 *
 * This is the primary method for CPU discovery that works without sudo.
 * Returns information about all logical processors on the system.
 */
amdcuid_status_t parse_proc_cpuinfo(std::vector<ProcCpuInfo>& cpus) {
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (!cpuinfo) {
    return AMDCUID_STATUS_CPUINFO_ERROR;
  }

  cpus.clear();
  ProcCpuInfo current;
  std::string line;

  while (std::getline(cpuinfo, line)) {
    // Empty line marks end of processor entry
    if (line.empty()) {
      if (current.processor >= 0) {
        cpus.push_back(current);
        current = ProcCpuInfo();
      }
      continue;
    }

    // Parse key : value
    size_t colon = line.find(':');
    if (colon == std::string::npos) continue;

    std::string key = line.substr(0, colon);
    std::string value = line.substr(colon + 1);

    // Trim whitespace
    while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(0, 1);

    if (key == "processor") {
      current.processor = std::stoi(value);
    } else if (key == "vendor_id") {
      current.vendor_id = value;
      current.is_amd = (value == "AuthenticAMD");
    } else if (key == "cpu family") {
      current.family = std::stoi(value);
    } else if (key == "model") {
      current.model = std::stoi(value);
    } else if (key == "stepping") {
      current.stepping = std::stoi(value);
    } else if (key == "physical id") {
      current.physical_id = std::stoi(value);
    } else if (key == "core id") {
      current.core_id = std::stoi(value);
    } else if (key == "apicid") {
      current.apic_id = std::stoi(value);
    }
  }

  // Don't forget the last entry
  if (current.processor >= 0) {
    cpus.push_back(current);
  }

  if (cpus.empty()) {
    return AMDCUID_STATUS_CPUINFO_ERROR;
  }

  return AMDCUID_STATUS_SUCCESS;
}

}  // namespace

/**
 * @brief Discover CPUs without root privileges using /proc/cpuinfo
 *
 * This implementation reads from /proc/cpuinfo which is readable by all users.
 * The ACPI MADT parsing (which requires root) is moved to
 * get_hardware_fingerprint() where privileged operations are expected.
 */
amdcuid_status_t CuidCpu::discover(std::vector<DevicePtr>& cpus) {
  cpus.clear();

  // Parse /proc/cpuinfo - works without root
  std::vector<ProcCpuInfo> proc_cpus;
  amdcuid_status_t status = parse_proc_cpuinfo(proc_cpus);

  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }

  // Get CPU information from CPUID (same for all cores)
  uint16_t vendor_id = 0, family = 0, model = 0, device_id = 0;
  uint8_t stepping = 0;
  bool cpuid_available = get_cpuid_info(vendor_id, family, model, device_id, stepping);

  // Group logical processors by physical package (socket). For each socket,
  // pick the representative entry with the lowest APIC ID — this is the
  // bootstrap processor convention used by firmware to represent the package.
  std::map<int, const ProcCpuInfo*> socket_rep;
  for (const auto& proc_info : proc_cpus) {
    auto it = socket_rep.find(proc_info.physical_id);
    if (it == socket_rep.end() || proc_info.apic_id < it->second->apic_id) {
      socket_rep[proc_info.physical_id] = &proc_info;
    }
  }

  // Create one CPU device per physical socket using the representative entry.
  for (const auto& kv : socket_rep) {
    const ProcCpuInfo* rep = kv.second;
    amdcuid_cpu_info info{};
    std::memset(&info.header, 0, sizeof(info.header));

    info.header.device_type = AMDCUID_DEVICE_TYPE_CPU;

    if (cpuid_available) {
      info.header.fields.cpu.vendor_id = vendor_id;
      info.header.fields.cpu.family = family;
      info.header.fields.cpu.model = model;
      info.header.fields.cpu.device_id = device_id;
      info.header.fields.cpu.revision_id = stepping;
    } else {
      info.header.fields.cpu.vendor_id = rep->is_amd ? 0x1022 : 0x8086;
      info.header.fields.cpu.family = static_cast<uint16_t>(rep->family);
      info.header.fields.cpu.model = static_cast<uint16_t>(rep->model);
      info.header.fields.cpu.device_id = static_cast<uint16_t>((rep->family << 8) | rep->model);
      info.header.fields.cpu.revision_id = static_cast<uint8_t>(rep->stepping);
    }

    // unit_id is the lowest APIC ID in this package — the bootstrap processor
    // representative used by firmware to identify the socket.
    info.header.fields.cpu.unit_id = static_cast<uint16_t>(rep->apic_id);
    info.header.fields.cpu.physical_id = static_cast<uint16_t>(kv.first);
    info.header.fields.cpu.core = static_cast<uint16_t>(rep->processor);  // logical CPU number

    // device_node points to the representative logical CPU's sysfs path.
    info.device_node = "/sys/devices/system/cpu/cpu" + std::to_string(rep->processor);

    cpus.emplace_back(std::make_shared<CuidCpu>(info));
  }

  return AMDCUID_STATUS_SUCCESS;
}

/**
 * @brief Discover a single CPU by its sysfs device path
 *
 * Device path should be like /sys/devices/system/cpu/cpu0
 * Reads topology info from sysfs and CPU info from CPUID.
 */
amdcuid_status_t CuidCpu::discover_single(amdcuid_cpu_info* cpu_info,
                                          const std::string& device_path) {
  if (!cpu_info) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }

  // Verify the path exists and extract CPU number
  size_t cpu_pos = device_path.find("cpu");
  if (cpu_pos == std::string::npos) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }

  // Read topology information from sysfs
  std::string topology_path = device_path + "/topology";

  std::string physical_package_str =
      CuidUtilities::read_sysfs_file(topology_path + "/physical_package_id");
  std::string core_id_str = CuidUtilities::read_sysfs_file(topology_path + "/core_id");

  if (physical_package_str.empty() || core_id_str.empty()) {
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
  }

  // Zero only the POD header; preserve std::string device_node
  std::memset(&cpu_info->header, 0, sizeof(cpu_info->header));
  cpu_info->device_node.clear();
  cpu_info->header.device_type = AMDCUID_DEVICE_TYPE_CPU;

  // Get CPU info from CPUID instruction
  uint16_t vendor_id = 0, family = 0, model = 0, device_id = 0;
  uint8_t stepping = 0;
  if (get_cpuid_info(vendor_id, family, model, device_id, stepping)) {
    cpu_info->header.fields.cpu.vendor_id = vendor_id;
    cpu_info->header.fields.cpu.family = family;
    cpu_info->header.fields.cpu.model = model;
    cpu_info->header.fields.cpu.device_id = device_id;
    cpu_info->header.fields.cpu.revision_id = stepping;
  } else {
    return AMDCUID_STATUS_CPUINFO_ERROR;
  }

  // Set topology fields from sysfs
  cpu_info->header.fields.cpu.physical_id = static_cast<uint16_t>(std::stoi(physical_package_str));
  cpu_info->header.fields.cpu.core = static_cast<uint16_t>(std::stoi(core_id_str));

  // Try to get APIC ID for unit_id (optional, may not be available via sysfs)
  // Extract CPU number from path for processor ID
  size_t num_start = device_path.rfind("cpu") + 3;
  if (num_start < device_path.length() && std::isdigit(device_path[num_start])) {
    int processor_id = std::stoi(device_path.substr(num_start));
    cpu_info->header.fields.cpu.unit_id = static_cast<uint16_t>(processor_id);
  }

  // Store the device path for later lookup
  cpu_info->device_node = device_path;

  return AMDCUID_STATUS_SUCCESS;
}

namespace {

/**
 * @brief Try to read PPIN (Protected Processor Inventory Number) from MSR
 *
 * PPIN is available on AMD CPUs (CPUID Fn8000_0008.EBX[23]) via MSR 0xC001_083B
 * Requires root privileges to read MSR.
 */
bool try_read_ppin(uint64_t& ppin, uint32_t core_id) {
#ifdef __x86_64__
  // Check if PPIN is supported via CPUID
  uint32_t eax, ebx, ecx, edx;
  if (__get_cpuid(0x80000008, &eax, &ebx, &ecx, &edx)) {
    if (!(ebx & (1 << 23))) {
      return false;  // PPIN not supported
    }
  } else {
    return false;
  }

  // Try to read PPIN from MSR 0xC001083B (AMD) or 0x4F (Intel)
  // Requires root privileges
  std::string msr_path = "/dev/cpu/" + std::to_string(core_id) + "/msr";
  std::ifstream msr(msr_path, std::ios::binary);
  if (!msr) {
    return false;  // No MSR access (need root or msr module)
  }

  // AMD PPIN MSR first, fallback to Intel if not found
  const uint64_t AMD_PPIN_MSR = 0xC001083B;
  msr.seekg(AMD_PPIN_MSR);
  if (!msr.read(reinterpret_cast<char*>(&ppin), sizeof(ppin))) {
    const uint64_t INTEL_PPIN_MSR = 0x4F;
    msr.seekg(INTEL_PPIN_MSR);
    if (!msr.read(reinterpret_cast<char*>(&ppin), sizeof(ppin))) {
      return false;
    }
  }

  return ppin != 0;
#else
  return false;
#endif
}

}  // namespace

/**
 * @brief Get hardware fingerprint for CPU at socket granularity.
 *
 * Priority:
 * 1. PPIN (per-socket eFuse serial, requires root + firmware enablement)
 * 2. SMBIOS system uuid + physical_id (platform-level, requires root)
 *
 * Both tiers produce a per-socket fingerprint consistent with the spec's
 * intent that CPU identity is tied to the physical package, not logical cores.
 */
amdcuid_status_t CuidCpu::get_hardware_fingerprint(uint64_t& fingerprint) const {
  // PPIN is an eFuse-burned per-socket serial number — the spec's primary
  // identifier. The core argument only selects which MSR fd to open.
  uint64_t ppin = 0;
  if (try_read_ppin(ppin, m_info.header.fields.cpu.core)) {
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
  status = sha256_unkeyed(input, sizeof(input), digest);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }

  std::memcpy(&fingerprint, digest, sizeof(fingerprint));
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidCpu::get_primary_cuid(amdcuid_primary_id& id) const {
  bool temp = false;
  uint64_t fingerprint = 0;
  amdcuid_status_t status = AMDCUID_STATUS_SUCCESS;

  if (geteuid() == 0) {
    // Attempt to read the CUID from the file first
    std::string cuid_file_path = CuidUtilities::priv_cuid_file();
    CuidFile primary_file(cuid_file_path, false);
    primary_file.load();
    std::vector<CuidFileEntry> entries = primary_file.get_entries();

    CuidFileEntry entry;

    // Primary lookup: device_node points to the representative logical CPU
    // sysfs path for this socket (set during discover()).
    if (!m_info.device_node.empty()) {
      status = primary_file.find_by_device_node(m_info.device_node, entry);
      if (status == AMDCUID_STATUS_SUCCESS && entry.is_temporary == false) {
        id.UUIDv8_representation = entry.primary_cuid;
        CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);
        return AMDCUID_STATUS_SUCCESS;
      }
    }

    status = primary_file.find_by_package_id(m_info.header.fields.cpu.physical_id, entry);
    if (status == AMDCUID_STATUS_SUCCESS && entry.is_temporary == false) {
      id.UUIDv8_representation = entry.primary_cuid;
      CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);
      return AMDCUID_STATUS_SUCCESS;
    }

    // Get hardware fingerprint (PPIN or SMBIOS serial + socket index)
    status = get_hardware_fingerprint(fingerprint);
  }
  if (geteuid() != 0 || status != AMDCUID_STATUS_SUCCESS) {
    // Non-root or fingerprint unavailable: use socket index + machine-id as
    // a stable but non-hardware fallback, flagged as temporary.
    std::string socket_id = "socket:" + std::to_string(m_info.header.fields.cpu.physical_id);
    amdcuid_status_t fb_status = CuidUtilities::make_fallback_fingerprint(socket_id, fingerprint);
    if (fb_status != AMDCUID_STATUS_SUCCESS) {
      return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
    }
    temp = true;
  }

  // Use CuidUtilities::generate_primary_cuid to generate CUID
  amdcuid_primary_id result = {};
  const auto& h = m_info.header;
  CuidUtilities::generate_primary_cuid(
      fingerprint, h.fields.cpu.unit_id, h.fields.cpu.revision_id, h.fields.cpu.device_id,
      h.fields.cpu.vendor_id, static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_CPU), &result, temp);

  id = result;
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
  // Use stored device_node (set from processor number during discovery or file
  // load)
  if (!m_info.device_node.empty()) {
    path = m_info.device_node;
    return AMDCUID_STATUS_SUCCESS;
  }
  // Fallback: construct from unit_id (APIC ID) for backward compatibility
  path = "/sys/devices/system/cpu/cpu" + std::to_string(m_info.header.fields.cpu.unit_id);
  return AMDCUID_STATUS_SUCCESS;
}
