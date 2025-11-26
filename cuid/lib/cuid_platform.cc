#include "cuid_platform.h"
#include "cuid_util.h"
#include "acpi_util.h"
#include <cstring>
#include <iostream>
#include <sstream>

AmdCuidPlatform::AmdCuidPlatform(const amdcuid_cuid_fields& i)
    : m_info({i})
{}

amdcuid_status_t AmdCuidPlatform::discover(std::vector<DevicePtr> &platforms) {
    // Platform is a singleton - only one platform per system
    amdcuid_cuid_fields header;
    std::memset(&header, 0, sizeof(header));
    
    // Set device type to platform
    header.device_type = AMDCUID_DEVICE_TYPE_PLATFORM;
    
    // Create platform device
    platforms.push_back(std::make_shared<AmdCuidPlatform>(header));
    
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t AmdCuidPlatform::get_hardware_fingerprint(uint64_t& fingerprint) const {
    // Generate fingerprint from system UUID
    std::string uuid;
    amdcuid_status_t status = AcpiUtil::get_system_uuid(uuid);
    
    if (status != AMDCUID_STATUS_SUCCESS) {
        // Fallback: try to use serial number
        std::string serial;
        status = AcpiUtil::get_system_serial(serial);
        if (status != AMDCUID_STATUS_SUCCESS) {
            fingerprint = 0;
            return AMDCUID_STATUS_UNSUPPORTED;
        }
        
        // Generate fingerprint from serial number
        fingerprint = 0;
        for (size_t i = 0; i < serial.length() && i < 8; ++i) {
            fingerprint |= (static_cast<uint64_t>(serial[i]) << (i * 8));
        }
        return AMDCUID_STATUS_SUCCESS;
    }
    
    // Generate fingerprint from UUID
    // Use first 8 bytes of UUID (excluding hyphens)
    fingerprint = 0;
    size_t byte_count = 0;
    
    for (char c : uuid) {
        if (c != '-' && byte_count < 8) {
            // Convert hex character to nibble
            uint8_t nibble = 0;
            if (c >= '0' && c <= '9') {
                nibble = c - '0';
            } else if (c >= 'a' && c <= 'f') {
                nibble = c - 'a' + 10;
            } else if (c >= 'A' && c <= 'F') {
                nibble = c - 'A' + 10;
            }
            
            fingerprint = (fingerprint << 4) | nibble;
            byte_count++;
        }
    }
    
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t AmdCuidPlatform::get_primary_cuid(amdcuid& id) const {
    std::memset(id.bytes, 0, sizeof(id.bytes));
    
    // Get system UUID from ACPI/SMBIOS
    std::string uuid;
    amdcuid_status_t status = AcpiUtil::get_system_uuid(uuid);
    
    if (status != AMDCUID_STATUS_SUCCESS) {
        // Fallback: generate from hardware fingerprint
        uint64_t fingerprint = 0;
        status = get_hardware_fingerprint(fingerprint);
        
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }
        
        // Use fingerprint as basis for CUID
        // Set device type in first byte
        id.bytes[0] = static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_PLATFORM);
        
        // Copy fingerprint to remaining bytes
        for (int i = 0; i < 8 && (i + 1) < 16; ++i) {
            id.bytes[i + 1] = static_cast<uint8_t>((fingerprint >> (i * 8)) & 0xFF);
        }
        
        return AMDCUID_STATUS_SUCCESS;
    }
    
    // Parse UUID string format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    // Convert to CUID bytes format
    std::string hex_only;
    for (char c : uuid) {
        if (c != '-') {
            hex_only += c;
        }
    }
    
    if (hex_only.length() != 32) {
        return AMDCUID_STATUS_INVALID_FORMAT;
    }
    
    // Convert hex string to bytes
    for (int i = 0; i < 16; ++i) {
        std::string byte_str = hex_only.substr(i * 2, 2);
        unsigned int byte_val;
        std::stringstream ss;
        ss << std::hex << byte_str;
        ss >> byte_val;
        id.bytes[i] = static_cast<uint8_t>(byte_val);
    }
    
    // Set device type in first nibble (upper 4 bits of first byte)
    id.bytes[0] = (id.bytes[0] & 0x0F) | 
                  (static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_PLATFORM) << 4);
    
    return AMDCUID_STATUS_SUCCESS;
}

const amdcuid_platform_info& AmdCuidPlatform::get_info() const {
    return m_info;
}
