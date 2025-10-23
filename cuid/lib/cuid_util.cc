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
#include <openssl/sha.h>


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
std::string CuidUtilities::read_sysfs_file(const std::string &path) {
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
std::string CuidUtilities::readlink_bdf(const std::string &device_path) {
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


amdcuid CuidUtilities::get_secondary_cuid(amdcuid_salt_t salt, const amdcuid* primary_id) {
    if (!primary_id) {
        // Return empty CUID on null input
        amdcuid empty = {};
        return empty;
    }

    amdcuid secondary = {};

    // Prepare input: first 14 bytes (112 bits) of primary_id + salt
    uint8_t input[14 + sizeof(amdcuid_salt_t)];
    memcpy(input, primary_id->bytes, 14);
    memcpy(input + 14, &salt, sizeof(amdcuid_salt_t));

    // Hash with SHA-256
    unsigned char hash[SHA256_DIGEST_LENGTH]; // 32 bytes
    SHA256(input, sizeof(input), hash);

    // Map the 224-bit hash to 122-bit CUID format
    uint8_t id_bits[16] = {0}; // 128 bits total (122 bits + 6 bits padding)

    // Copy first 15 bytes (120 bits) from hash
    memcpy(id_bits, hash, 15);

    // Copy 2 more bits from the 16th byte of hash to complete 122 bits
    id_bits[15] = hash[15] & 0x03; // Take only the lower 2 bits

    // Apply UUIDv8 format according to RFC 9562
    // Bits 0-47: ID value part 1 (LSB)
    secondary.bytes[0] = id_bits[0];
    secondary.bytes[1] = id_bits[1];
    secondary.bytes[2] = id_bits[2];
    secondary.bytes[3] = id_bits[3];
    secondary.bytes[4] = id_bits[4];
    secondary.bytes[5] = id_bits[5];

    // Bits 48-51: Version (8) + Bits 52-63: ID value part 2
    secondary.bytes[6] = (id_bits[6] & 0x0F) | 0x80; // Version 8 in upper 4 bits
    secondary.bytes[7] = id_bits[7];

    // Bits 64-65: Variant (10b) + Bits 66-127: ID value part 3 (MSB)
    secondary.bytes[8] = (id_bits[8] & 0x3F) | 0x80; // Variant 10b in upper 2 bits
    secondary.bytes[9] = id_bits[9];
    secondary.bytes[10] = id_bits[10];
    secondary.bytes[11] = id_bits[11];
    secondary.bytes[12] = id_bits[12];
    secondary.bytes[13] = id_bits[13];
    secondary.bytes[14] = id_bits[14];
    secondary.bytes[15] = id_bits[15];

    return secondary;
}


amdcuid CuidUtilities::generate_primary_cuid(uint64_t serial_number, uint8_t unit_id_part1, uint8_t unit_id_part2,
                                 uint8_t revision_id, uint16_t device_id, uint16_t vendor_id,
                                 uint8_t component_type) {
    amdcuid id = {};
    
    // Build 122-bit value in little-endian order
    uint8_t id_bits[16] = {0}; // 128 bits total (122 bits + 6 bits padding)
    
    // Bits 0-63: Serial number (8 bytes)
    memcpy(id_bits, &serial_number, 8);
    
    // Bits 64-71: UnitID part 1 (1 byte)
    id_bits[8] = unit_id_part1;
    
    // Bits 72-79: RevisionID (1 byte) 
    id_bits[9] = revision_id;
    
    // Bits 80-95: DeviceID (2 bytes)
    id_bits[10] = device_id & 0xFF;
    id_bits[11] = (device_id >> 8) & 0xFF;
    
    // Bits 96-111: VendorID (2 bytes)
    id_bits[12] = vendor_id & 0xFF;
    id_bits[13] = (vendor_id >> 8) & 0xFF;
    
    // Bits 112-117: UnitID part 2 (6 bits) + Bits 118-121: Component Type (4 bits)
    id_bits[14] = unit_id_part2 | (component_type << 6);
    
    // Apply UUIDv8 format according to RFC 9562
    // Bits 0-47: ID value part 1 (LSB)
    id.bytes[0] = id_bits[0];
    id.bytes[1] = id_bits[1]; 
    id.bytes[2] = id_bits[2];
    id.bytes[3] = id_bits[3];
    id.bytes[4] = id_bits[4];
    id.bytes[5] = id_bits[5];
    
    // Bits 48-51: Version (8) + Bits 52-63: ID value part 2
    id.bytes[6] = (id_bits[6] & 0x0F) | 0x80; // Version 8 in upper 4 bits
    id.bytes[7] = id_bits[7];
    
    // Bits 64-65: Variant (10b) + Bits 66-127: ID value part 3 (MSB)
    id.bytes[8] = (id_bits[8] & 0x3F) | 0x80; // Variant 10b in upper 2 bits
    id.bytes[9] = id_bits[9];
    id.bytes[10] = id_bits[10]; 
    id.bytes[11] = id_bits[11];
    id.bytes[12] = id_bits[12];
    id.bytes[13] = id_bits[13];
    id.bytes[14] = id_bits[14];
    id.bytes[15] = id_bits[15];
    
    return id;
}

char* CuidUtilities::get_cuid_as_string(const amdcuid *id) {
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


const char *CuidUtilities::cuid_status_to_string(amdcuid_status_t status)
{
    switch (status)
    {
    case AMDCUID_STATUS_SUCCESS:
        return "SUCCESS";
    case AMDCUID_STATUS_FILE_NOT_FOUND:
        return "FILE_NOT_FOUND";
    case AMDCUID_STATUS_DEVICE_NOT_FOUND:
        return "DEVICE_NOT_FOUND";
    case AMDCUID_STATUS_INVALID_ARGUMENT:
        return "INVALID_ARGUMENT";
    case AMDCUID_STATUS_PERMISSION_DENIED:
        return "PERMISSION_DENIED";
    case AMDCUID_STATUS_UNSUPPORTED:
        return "UNSUPPORTED";
    case AMDCUID_STATUS_NOT_INIT:
        return "NOT_INIT";
    case AMDCUID_STATUS_WRONG_DEVICE_TYPE:
        return "WRONG_DEVICE_TYPE";
    case AMDCUID_STATUS_INSUFFICIENT_SIZE:
        return "INSUFFICIENT_SIZE";
    case AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND:
        return "AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND";
    case AMDCUID_STATUS_HW_FINGERPRINT_FORMAT_ERROR:
        return "AMDCUID_STATUS_HW_FINGERPRINT_FORMAT_ERROR";
    case AMDCUID_STATUS_HW_FINGERPRINT_PERMISSION_DENIED:
        return "AMDCUID_STATUS_HW_FINGERPRINT_PERMISSION_DENIED";
    default:
        return "UNKNOWN_ERROR";
    }
}





// Parse the CUID file into a map: section -> key -> value
using SectionMap = std::map<std::string, std::map<std::string, std::string>>;
SectionMap CuidUtilities::parse_cuid_file(const std::string &filename)
{
    SectionMap sections;
    std::ifstream infile(filename);
    std::string line, current_section;
    while (std::getline(infile, line))
    {
        if (line.empty())
            continue;
        if (line.front() == '[' && line.back() == ']')
        {
            current_section = line;
            sections[current_section] = {};
        }
        else
        {
            auto pos = line.find('=');
            if (pos != std::string::npos && !current_section.empty())
            {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                sections[current_section][key] = value;
            }
        }
    }
    return sections;
}

void CuidUtilities::write_cuid_file(const std::string &filename, const SectionMap &sections)
{
    std::ofstream outfile(filename);
    for (SectionMap::const_iterator section_it = sections.begin(); section_it != sections.end(); ++section_it)
    {
        outfile << section_it->first << "\n";
        const std::map<std::string, std::string> &kv = section_it->second;
        for (std::map<std::string, std::string>::const_iterator kv_it = kv.begin(); kv_it != kv.end(); ++kv_it)
        {
            outfile << kv_it->first << "=" << kv_it->second << "\n";
        }
    }
}

