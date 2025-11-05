#include "cuid_util.h"
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <mutex>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>


const char* Logger::LogLevelName(LogLevel level) const {
    switch (level) {
        case DEBUG: return "DEBUG";
        case INFO:  return "INFO";
        case WARN:  return "WARN";
        case ERROR: return "ERROR";
        default:    return "UNKNOWN";
    }
}

void Logger::log(LogLevel level, const std::string& msg) const {
    if (level < level_) return;
    std::cerr << "[" << LogLevelName(level) << "] " << msg << std::endl;
}


// Helper to read a sysfs file into a string
std::string AmdCuidUtilities::read_sysfs_file(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";
    std::stringstream ss;
    ss << file.rdbuf();
    std::string result = ss.str();
    // Remove trailing newline if present
    if (!result.empty() && result.back() == '\n') result.pop_back();
    return result;
}

// Helper to get BDF from symlink, filter out non-PCI BDFs (e.g., USB like 3-10.2:2.0)
std::string AmdCuidUtilities::readlink_bdf(const std::string &device_path) {
    char buf[256];
    ssize_t len = readlink(device_path.c_str(), buf, sizeof(buf)-1);
    if (len > 0) {
        buf[len] = '\0';
        // The symlink is typically ../../../0000:65:00.0 or ../../../3-10.2:2.0
        const char *bdf = strrchr(buf, '/');
        if (bdf && strlen(bdf+1) < 32) {
            std::string bdf_str = bdf + 1;
            // Only accept PCI BDFs of the form "dddd:bb:dd.f"
            // Example: 0000:65:00.0
            if (bdf_str.size() == 12 &&
                bdf_str[4] == ':' && bdf_str[7] == ':' && bdf_str[10] == '.') {
                return bdf_str;
            }
        }
    }
    return "";
}


amdcuid_status_t AmdCuidUtilities::generate_secondary_cuid(const amdcuid* primary_id, amdcuid* secondary_id, AMDCUID_HMAC* hmac) {
    if (!primary_id || !hmac) {
        // Return invalid on null input
        amdcuid empty = {};
        return AMDCUID_STATUS_INVALID_ARGUMENT;
    }

    uint8_t hash[EVP_MAX_MD_SIZE];
    size_t hash_len = 0;

    amdcuid_status_t status = static_cast<amdcuid_status_t>(hmac->generate_hmac_sha256(reinterpret_cast<const uint8_t*>(primary_id->bytes), sizeof(primary_id->bytes), hash, &hash_len));
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error generating HMAC" << std::endl;
        return status;
    }

    // Map the 256-bit hash to 122-bit CUID format
    uint8_t id_bits[16] = {0}; // 128 bits total (122 bits + 6 bits padding)

    // Copy first 15 bytes (120 bits) from hash
    memcpy(id_bits, hash, 15);

    // Copy 2 more bits from the 16th byte of hash to complete 122 bits
    id_bits[15] = hash[15] & 0x03; // Take only the lower 2 bits

    // Apply UUIDv8 format according to RFC 9562
    // Bits 0-47: ID value part 1 (LSB)
    secondary_id->bytes[0] = id_bits[0];
    secondary_id->bytes[1] = id_bits[1];
    secondary_id->bytes[2] = id_bits[2];
    secondary_id->bytes[3] = id_bits[3];
    secondary_id->bytes[4] = id_bits[4];
    secondary_id->bytes[5] = id_bits[5];

    // Bits 48-51: Version (8) + Bits 52-63: ID value part 2
    secondary_id->bytes[6] = (id_bits[6] & 0x0F) | 0x80; // Version 8 in upper 4 bits
    secondary_id->bytes[7] = id_bits[7];

    // Bits 64-65: Variant (10b) + Bits 66-127: ID value part 3 (MSB)
    secondary_id->bytes[8] = (id_bits[8] & 0x3F) | 0x80; // Variant 10b in upper 2 bits
    secondary_id->bytes[9] = id_bits[9];
    secondary_id->bytes[10] = id_bits[10];
    secondary_id->bytes[11] = id_bits[11];
    secondary_id->bytes[12] = id_bits[12];
    secondary_id->bytes[13] = id_bits[13];
    secondary_id->bytes[14] = id_bits[14];
    secondary_id->bytes[15] = id_bits[15];

    return AMDCUID_STATUS_SUCCESS;
}


amdcuid_status_t AmdCuidUtilities::generate_primary_cuid(uint64_t serial_number, uint16_t unit_id,
                                 uint8_t revision_id, uint16_t device_id, uint16_t vendor_id,
                                 uint8_t component_type, amdcuid* id) {
    // amdcuid id = {};

    // Build 122-bit value in little-endian order
    uint8_t id_bits[16] = {0}; // 128 bits total (122 bits + 6 bits padding)

    uint8_t unit_id_part1 = unit_id & 0xFF;
    uint8_t unit_id_part2 = (unit_id >> 8) & 0x3F;

    // Bits 0-63: Serial number (8 bytes)
    memcpy(id_bits, &serial_number, 8);
    
    // Bits 64-71: UnitID part 1 (1 byte)
    id_bits[8] = unit_id_part1;
    
    // Bits 72-79: RevisionID (1 byte) 
    id_bits[9] = revision_id;
    
    // Bits 80-95: DeviceID (2 bytes); These format changes are necessary to make the final ID little Endian, as specified in the design
    id_bits[10] = device_id & 0xFF;
    id_bits[11] = (device_id >> 8) & 0xFF;
    
    // Bits 96-111: VendorID (2 bytes)
    id_bits[12] = vendor_id & 0xFF;
    id_bits[13] = (vendor_id >> 8) & 0xFF;
    
    // Bits 112-117: UnitID part 2 (6 bits) + Bits 118-121: Component Type (4 bits)
    id_bits[14] = unit_id_part2 | (component_type << 6);
    
    // Apply UUIDv8 format according to RFC 9562
    // Bits 0-47: ID value part 1 (LSB)
    id->bytes[0] = id_bits[0];
    id->bytes[1] = id_bits[1]; 
    id->bytes[2] = id_bits[2];
    id->bytes[3] = id_bits[3];
    id->bytes[4] = id_bits[4];
    id->bytes[5] = id_bits[5];
    
    // Bits 48-51: Version (8) + Bits 52-63: ID value part 2
    id->bytes[6] = (id_bits[6] & 0x0F) | 0x80; // Version 8 in upper 4 bits
    id->bytes[7] = id_bits[7];

    // Bits 64-65: Variant (10b) + Bits 66-127: ID value part 3 (MSB)
    id->bytes[8] = (id_bits[8] & 0x3F) | 0x80; // Variant 10b in upper 2 bits
    id->bytes[9] = id_bits[9];
    id->bytes[10] = id_bits[10];
    id->bytes[11] = id_bits[11];
    id->bytes[12] = id_bits[12];
    id->bytes[13] = id_bits[13];
    id->bytes[14] = id_bits[14];
    id->bytes[15] = id_bits[15];

    return AMDCUID_STATUS_SUCCESS;
}

char* AmdCuidUtilities::get_cuid_as_string(const amdcuid *id) {
    // Format as UUIDv8 string: 8-4-4-4-12 hex digits from id->bytes[16]
    // UUID: xxxxxxxx-xxxx-8xxx-yxxx-xxxxxxxxxxxx
    static char uuid_str[37]; // 36 chars + null
    // Format the bytes into a UUID string
    snprintf(uuid_str, 37,
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             id->bytes[0], id->bytes[1], id->bytes[2], id->bytes[3],
             id->bytes[4], id->bytes[5],
             id->bytes[6], id->bytes[7],
             id->bytes[8], id->bytes[9],
             id->bytes[10], id->bytes[11], id->bytes[12], id->bytes[13], id->bytes[14], id->bytes[15]);

    return uuid_str;
}
