
#include "cuid_cpu.h"
#include "cuid_util.h"
#include "acpi_parser.h"
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>

#ifdef __x86_64__
#include <cpuid.h>
#endif

AmdCuidCpu::AmdCuidCpu(const amdcuid_cpu_info& i)
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

amdcuid_status_t AmdCuidCpu::discover(std::vector<DevicePtr> &cpus) {
    cpus.clear();
    
    // Parse MADT table to get APIC IDs from ACPI
    std::vector<AcpiCpuInfo> acpi_cpus;
    amdcuid_status_t status = AcpiParser::parse_madt(acpi_cpus);
    
    if (status != AMDCUID_STATUS_SUCCESS) {
        return status;
    }
    
    // Get CPU information from CPUID (same for all cores)
    uint16_t vendor_id = 0, family = 0, model = 0, device_id = 0;
    uint8_t stepping = 0;
    bool cpuid_available = get_cpuid_info(vendor_id, family, model, device_id, stepping);
    
    // Create CPU device for each enabled CPU
    for (const auto& acpi_info : acpi_cpus) {
        if (!acpi_info.enabled) {
            continue;  // Skip disabled CPUs
        }
        
        amdcuid_cpu_info info;
        std::memset(&info, 0, sizeof(info));
        
        // Set device type
        info.header.device_type = AMDCUID_DEVICE_TYPE_CPU;
        
        // Fill CPU fields per CUID design spec:
        // - VendorID: CPU Vendor (from CPUID)
        // - DeviceID: Family & Model (from CPUID EAX=1)
        // - RevisionID: Stepping (from CPUID EAX=1)
        // - UnitID: from APIC ID (from ACPI MADT)
        if (cpuid_available) {
            info.header.fields.cpu.vendor_id = vendor_id;
            info.header.fields.cpu.family = family;
            info.header.fields.cpu.model = model;
            info.header.fields.cpu.device_id = device_id;
            info.header.fields.cpu.revision_id = stepping;
        }
        
        // UnitID from APIC ID (as per design spec)
        info.header.fields.cpu.unit_id = acpi_info.apic_id & 0xFFFF;
        
        // Core and physical_id (from topology if available, else use APIC ID)
        // For now, use APIC ID as approximation
        info.header.fields.cpu.core = acpi_info.processor_uid & 0xFFFF;
        info.header.fields.cpu.physical_id = 0;  // Would need NUMA/topology parsing
        
        auto cpu = std::make_shared<AmdCuidCpu>(info);
        cpus.push_back(cpu);
    }
    
    return AMDCUID_STATUS_SUCCESS;
}

/**
 * @brief Try to read PPIN (Protected Processor Inventory Number) from MSR
 * 
 * PPIN is available on AMD CPUs (CPUID Fn8000_0008.EBX[23]) via MSR 0xC001_083B
 * Requires root privileges to read MSR.
 */
static bool try_read_ppin(uint64_t& ppin) {
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
    std::ifstream msr("/dev/cpu/0/msr", std::ios::binary);
    if (!msr) {
        return false;  // No MSR access (need root or msr module)
    }
    
    // AMD PPIN MSR
    const uint64_t AMD_PPIN_MSR = 0xC001083B;
    msr.seekg(AMD_PPIN_MSR);
    if (!msr.read(reinterpret_cast<char*>(&ppin), sizeof(ppin))) {
        return false;
    }
    
    return ppin != 0;
#else
    return false;
#endif
}

amdcuid_status_t AmdCuidCpu::get_hardware_fingerprint(uint64_t& fingerprint) const {
    // Per manager feedback: Use processor _UID as fingerprint (often a UUID or serial)
    // This comes from ACPI and is stored in the core field
    fingerprint = static_cast<uint64_t>(m_info.header.fields.cpu.core) |
                  (static_cast<uint64_t>(m_info.header.fields.cpu.unit_id) << 16);
    
    // Try to get PPIN (Protected Processor Inventory Number) if available
    uint64_t ppin = 0;
    if (try_read_ppin(ppin)) {
        fingerprint = ppin;  // Use PPIN as primary fingerprint if available
    }
    
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t AmdCuidCpu::get_primary_cuid(amdcuid& id) const {
    // Get hardware fingerprint (PPIN or processor UID)
    uint64_t fingerprint = 0;
    amdcuid_status_t status = get_hardware_fingerprint(fingerprint);
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::memset(id.bytes, 0, sizeof(id.bytes));
        return status;
    }
    
    // Use AmdCuidUtilities::generate_primary_cuid to generate CUID
    // This follows the same pattern as GPU CUID generation
    amdcuid result = {};
    const auto& h = m_info.header;
    AmdCuidUtilities::generate_primary_cuid(
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

const amdcuid_cpu_info& AmdCuidCpu::get_info() const {
    return m_info;
}