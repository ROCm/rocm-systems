/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "src/cuid_cpu.h"
#include "src/cuid_util.h"
#include "src/acpi_parser.h"
#include "src/cuid_file.h"
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <map>

#ifdef __x86_64__
#include <cpuid.h>
#endif

CuidCpu::CuidCpu(const amdcuid_cpu_info& i)
    : m_info(i)
{}

/**
 * @brief Get CPU information from CPUID instruction (x86_64)
 */
static bool get_cpuid_info(uint16_t& vendor_id, uint16_t& family, uint16_t& model, 
                           uint16_t& device_id, uint8_t& stepping) {
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
    stepping = eax & 0x0F;                          // Bits 0-3: Stepping
    uint32_t base_model = (eax >> 4) & 0x0F;        // Bits 4-7: Base Model
    uint32_t base_family = (eax >> 8) & 0x0F;       // Bits 8-11: Base Family
    uint32_t ext_model = (eax >> 16) & 0x0F;        // Bits 16-19: Extended Model
    uint32_t ext_family = (eax >> 20) & 0xFF;       // Bits 20-27: Extended Family
    
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

/**
 * @brief CPU information parsed from /proc/cpuinfo
 */
struct ProcCpuInfo {
    int processor = -1;      // Logical processor number
    int physical_id = 0;     // Socket/package ID
    int core_id = 0;         // Core ID within package
    int apic_id = 0;         // APIC ID
    int family = 0;
    int model = 0;
    int stepping = 0;
    std::string vendor_id;   // "AuthenticAMD" or "GenuineIntel"
    bool is_amd = false;
};

/**
 * @brief Parse /proc/cpuinfo to discover CPUs without root privileges
 * 
 * This is the primary method for CPU discovery that works without sudo.
 * Returns information about all logical processors on the system.
 */
static amdcuid_status_t parse_proc_cpuinfo(std::vector<ProcCpuInfo>& cpus) {
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
        while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) 
            key.pop_back();
        while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) 
            value.erase(0, 1);
        
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

/**
 * @brief Discover CPUs without root privileges using /proc/cpuinfo
 * 
 * This implementation reads from /proc/cpuinfo which is readable by all users.
 * The ACPI MADT parsing (which requires root) is moved to get_hardware_fingerprint()
 * where privileged operations are expected.
 */
amdcuid_status_t CuidCpu::discover(std::vector<DevicePtr> &cpus) {
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
    
    // Create CPU device for each logical processor
    for (const auto& proc_info : proc_cpus) {
        amdcuid_cpu_info info;
        std::memset(&info, 0, sizeof(info));
        
        // Set device type
        info.header.device_type = AMDCUID_DEVICE_TYPE_CPU;
        
        // Fill CPU fields per CUID design spec
        if (cpuid_available) {
            info.header.fields.cpu.vendor_id = vendor_id;
            info.header.fields.cpu.family = family;
            info.header.fields.cpu.model = model;
            info.header.fields.cpu.device_id = device_id;
            info.header.fields.cpu.revision_id = stepping;
        } else {
            // Fallback to /proc/cpuinfo values
            info.header.fields.cpu.vendor_id = proc_info.is_amd ? 0x1022 : 0x8086;
            info.header.fields.cpu.family = static_cast<uint16_t>(proc_info.family);
            info.header.fields.cpu.model = static_cast<uint16_t>(proc_info.model);
            info.header.fields.cpu.device_id = static_cast<uint16_t>((proc_info.family << 8) | proc_info.model);
            info.header.fields.cpu.revision_id = static_cast<uint8_t>(proc_info.stepping);
        }
        
        // UnitID from APIC ID (from /proc/cpuinfo, similar to MADT)
        info.header.fields.cpu.unit_id = static_cast<uint16_t>(proc_info.apic_id);
        
        // Core and physical_id from /proc/cpuinfo
        info.header.fields.cpu.core = static_cast<uint16_t>(proc_info.core_id);
        info.header.fields.cpu.physical_id = static_cast<uint16_t>(proc_info.physical_id);
        
        auto cpu = std::make_shared<CuidCpu>(info);
        cpus.emplace_back(cpu);
    }
    
    return AMDCUID_STATUS_SUCCESS;
}

/**
 * @brief Discover a single CPU by its sysfs device path
 * 
 * Device path should be like /sys/devices/system/cpu/cpu0
 * Reads topology info from sysfs and CPU info from CPUID.
 */
amdcuid_status_t CuidCpu::discover_single(amdcuid_cpu_info* cpu_info, const std::string& device_path) {
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
    
    std::string physical_package_str = CuidUtilities::read_sysfs_file(topology_path + "/physical_package_id");
    std::string core_id_str = CuidUtilities::read_sysfs_file(topology_path + "/core_id");
    
    if (physical_package_str.empty() || core_id_str.empty()) {
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    }

    std::memset(cpu_info, 0, sizeof(amdcuid_cpu_info));
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

    return AMDCUID_STATUS_SUCCESS;
}

/**
 * @brief Try to read PPIN (Protected Processor Inventory Number) from MSR
 * 
 * PPIN is available on AMD CPUs (CPUID Fn8000_0008.EBX[23]) via MSR 0xC001_083B
 * Requires root privileges to read MSR.
 */
static bool try_read_ppin(uint64_t& ppin, uint32_t core_id) {
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

/**
 * @brief Get hardware fingerprint for CPU, optionally with ACPI MADT data
 * 
 * Requires root privileges for:
 * - ACPI MADT parsing (for processor UID)
 * - PPIN reading (from MSR)
 */
amdcuid_status_t CuidCpu::get_hardware_fingerprint(uint64_t& fingerprint) const {
    if (geteuid() != 0) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }
    
    // Try to get PPIN (Protected Processor Inventory Number) if available
    uint64_t ppin = 0;
    if (try_read_ppin(ppin, m_info.header.fields.cpu.core)) {
        fingerprint = ppin;  // Use PPIN as primary fingerprint if available
        return AMDCUID_STATUS_SUCCESS;
    }
    
    // Try to get processor UID from ACPI MADT table
    // This requires root to read /sys/firmware/acpi/tables/APIC
    std::vector<AcpiCpuInfo> acpi_cpus;
    amdcuid_status_t status = AcpiParser::parse_madt(acpi_cpus);
    
    if (status == AMDCUID_STATUS_SUCCESS) {
        // Find matching CPU by APIC ID
        uint32_t target_apic_id = m_info.header.fields.cpu.unit_id;
        for (const auto& acpi_info : acpi_cpus) {
            if (acpi_info.apic_id == target_apic_id) {
                // Use processor UID from ACPI as fingerprint
                fingerprint = static_cast<uint64_t>(acpi_info.processor_uid) |
                              (static_cast<uint64_t>(acpi_info.apic_id) << 32);
                return AMDCUID_STATUS_SUCCESS;
            }
        }
    }
    
    // Fallback: Use combination of APIC ID and core/physical_id
    fingerprint = static_cast<uint64_t>(m_info.header.fields.cpu.core) |
                  (static_cast<uint64_t>(m_info.header.fields.cpu.unit_id) << 16) |
                  (static_cast<uint64_t>(m_info.header.fields.cpu.physical_id) << 32);
    
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidCpu::get_primary_cuid(amdcuid_primary_id& id) const {
    if (geteuid() != 0) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    // Attempt to read the CUID from the file first
    std::string cuid_file_path = CuidUtilities::priv_cuid_file();
    CuidFile primary_file(cuid_file_path, false);
    primary_file.load();
    std::vector<CuidFileEntry> entries = primary_file.get_entries();

    CuidFileEntry entry;
    std::string package_core_id = std::to_string(m_info.header.fields.cpu.physical_id) + 
                                  ":" + std::to_string(m_info.header.fields.cpu.core);
    amdcuid_status_t status = primary_file.find_by_package_core_id(package_core_id, entry);
    if (status == AMDCUID_STATUS_SUCCESS) {
        id.UUIDv8_representation = entry.primary_cuid;
        CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);
        return AMDCUID_STATUS_SUCCESS;
    }

    // Get hardware fingerprint (PPIN, ACPI UID, or fallback)
    uint64_t fingerprint = 0;
    status = get_hardware_fingerprint(fingerprint);
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::memset(&id, 0, sizeof(id));
        return status;
    }

    // Use CuidUtilities::generate_primary_cuid to generate CUID
    amdcuid_primary_id result = {};
    const auto& h = m_info.header;
    CuidUtilities::generate_primary_cuid(
        fingerprint,
        h.fields.cpu.unit_id,
        h.fields.cpu.revision_id,
        h.fields.cpu.device_id,
        h.fields.cpu.vendor_id,
        static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_CPU),
        &result
    );

    id = result;
    return AMDCUID_STATUS_SUCCESS;
}

const amdcuid_cpu_info& CuidCpu::get_info() const {
    return m_info;
}

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
    // Construct device path based on unit_id (APIC ID)
    path = "/sys/devices/system/cpu/cpu" + std::to_string(m_info.header.fields.cpu.unit_id);
    return AMDCUID_STATUS_SUCCESS;
}
