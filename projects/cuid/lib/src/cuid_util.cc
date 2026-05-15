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

#include "cuid_util.h"
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
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

// Helper to get device path from BDF based on device type
// For GPUs: returns /sys/class/drm/renderDXXX/device path
// For NICs: returns /sys/class/net/<iface>/device path
std::string CuidUtilities::bdf_to_device_path(const std::string &bdf, amdcuid_device_type_t device_type) {
    if (bdf.empty()) return "";

    std::string pci_device_path = "/sys/bus/pci/devices/" + bdf;
    std::string subsystem_dir;

    if (device_type == AMDCUID_DEVICE_TYPE_GPU) {
        subsystem_dir = pci_device_path + "/drm";
        DIR* dir = opendir(subsystem_dir.c_str());
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                // Look for renderD* nodes
                if (strncmp(entry->d_name, "renderD", 7) == 0) {
                    std::string device_path = "/sys/class/drm/" + std::string(entry->d_name) + "/device";
                    closedir(dir);
                    return device_path;
                }
            }
            closedir(dir);
        }
    } else if (device_type == AMDCUID_DEVICE_TYPE_NIC) {
        subsystem_dir = pci_device_path + "/net";
        DIR* dir = opendir(subsystem_dir.c_str());
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                // Skip . and ..
                if (entry->d_name[0] == '.') continue;
                std::string device_path = "/sys/class/net/" + std::string(entry->d_name) + "/device";
                closedir(dir);
                return device_path;
            }
            closedir(dir);
        }
    }

    return "";
}

std::string CuidUtilities::real_dev_path_from_fd(int fd) {
    struct stat st;
    if (fstat(fd, &st) != 0) {
        return "";
    }
    dev_t dev = st.st_rdev;
    uint32_t major_num = major(dev);
    uint32_t minor_num = minor(dev);

    // Construct sysfs path from char device numbers first
    std::string sys_path = "/sys/dev/char/" + std::to_string(major_num) + ":" + std::to_string(minor_num);
    char buf[PATH_MAX];
    if (realpath(sys_path.c_str(), buf) != nullptr) {
        return std::string(buf) + "/device";
    }
    else {
        // attempt to find as a block device now
        sys_path = "/sys/dev/block/" + std::to_string(major_num) + ":" + std::to_string(minor_num);
        if (realpath(sys_path.c_str(), buf) != nullptr) {
            return std::string(buf) + "/device";
        }
    }
    // If all fails, return empty string
    return "";
}

std::string CuidUtilities::get_real_path(const std::string path) {
    char buf[PATH_MAX];
    if (realpath(path.c_str(), buf) != nullptr) {
        return std::string(buf) + "/device";
    }
    return "";
}

amdcuid_status_t CuidUtilities::generate_derived_cuid(const amdcuid_primary_id* primary_id, amdcuid_derived_id* derived_id, cuid_hmac* hmac) {
    if (!primary_id || !hmac) {
        // Return invalid on null input
        amdcuid_id_t empty = {};
        return AMDCUID_STATUS_INVALID_ARGUMENT;
    }

    uint8_t hash[EVP_MAX_MD_SIZE];
    size_t hash_len = 0;

    amdcuid_status_t status = static_cast<amdcuid_status_t>(hmac->generate_hmac_sha256(reinterpret_cast<const uint8_t*>(primary_id->raw_bits), sizeof(primary_id->raw_bits), hash, &hash_len));
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error generating HMAC" << std::endl;
        return status;
    }

    // copy 110 LSB bits of hash to derived_id->hash
    // Using only first 14 bytes (112 bits) of hash, then will mask last 2 bits
    memcpy(derived_id->hash, hash, 14);
    derived_id->hash[13] &= 0xFC; // 11111100

    // Get the unit id parts from the primary ID
    uint8_t reserved_1 = 0;
    uint8_t reserved_2 = 0;
    // Map the 256-bit hash to 122-bit CUID format
    uint8_t id_bits[16] = {0};

    // Copy first 8 bytes (64 bits) from hash
    memcpy(id_bits, derived_id->hash, 8);

    // insert unit id part 1 at bits 64-71
    id_bits[8] = reserved_1;

    // copy next 6 bytes (46 bits) from hash and mask off last 2 bits for bits 72-117 
    memcpy(id_bits + 9, derived_id->hash + 8, 6);
    id_bits[14] &= 0xFC;

    // bits 118-121: UnitID part 2 (4 bits)
    id_bits[14] |= (reserved_2 >> 2); // upper 2 bits of unit id part 2
    id_bits[15] |= (reserved_2 & 0x03) << 6; // lower 2 bits of unit id part 2
    // last 6 bits are padding (bits 122-127)
    id_bits[15] &= 0xC0;

    memcpy(derived_id->raw_bits, id_bits, 16);

    // Apply UUIDv8 format according to RFC 9562
    // Bits 0-47: ID value part 1 (LSB)
    derived_id->UUIDv8_representation.bytes[0] = id_bits[0];
    derived_id->UUIDv8_representation.bytes[1] = id_bits[1];
    derived_id->UUIDv8_representation.bytes[2] = id_bits[2];
    derived_id->UUIDv8_representation.bytes[3] = id_bits[3];
    derived_id->UUIDv8_representation.bytes[4] = id_bits[4];
    derived_id->UUIDv8_representation.bytes[5] = id_bits[5];

    // Bits 48-51: Version (8) + Bits 52-63: ID value part 2
    derived_id->UUIDv8_representation.bytes[6] = ((id_bits[6] & 0xF0) >> 4) | 0x80; // Version 8 in upper 4 bits
    derived_id->UUIDv8_representation.bytes[7] = ((id_bits[6] & 0x0F) << 4) | ((id_bits[7] & 0xF0) >> 4);

    // Bits 64-65: Variant (10b) + Bits 66-127: ID value part 3 (MSB)
    derived_id->UUIDv8_representation.bytes[8] = 0x80 | (id_bits[7] & 0x0F) << 2 | (id_bits[8] & 0xC0) >> 6;
    // everything past here is now shifted by 6 bits
    derived_id->UUIDv8_representation.bytes[9] = ((id_bits[8] & 0x3F) << 2) | ((id_bits[9] & 0xC0) >> 6);
    derived_id->UUIDv8_representation.bytes[10] = ((id_bits[9] & 0x3F) << 2) | ((id_bits[10] & 0xC0) >> 6);
    derived_id->UUIDv8_representation.bytes[11] = ((id_bits[10] & 0x3F) << 2) | ((id_bits[11] & 0xC0) >> 6);
    derived_id->UUIDv8_representation.bytes[12] = ((id_bits[11] & 0x3F) << 2) | ((id_bits[12] & 0xC0) >> 6);
    derived_id->UUIDv8_representation.bytes[13] = ((id_bits[12] & 0x3F) << 2) | ((id_bits[13] & 0xC0) >> 6);
    derived_id->UUIDv8_representation.bytes[14] = ((id_bits[13] & 0x3F) << 2) | ((id_bits[14] & 0xC0) >> 6);
    derived_id->UUIDv8_representation.bytes[15] = ((id_bits[14] & 0x3F) << 2) | ((id_bits[15] & 0xC0) >> 6);

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidUtilities::generate_primary_cuid(uint64_t serial_number, uint16_t unit_id,
                                 uint8_t revision_id, uint16_t device_id, uint16_t vendor_id,
                                 uint8_t device_type, amdcuid_primary_id* primary_id) {

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
    id_bits[14] = (unit_id_part2 << 2) | ((device_type & 0xC) >> 2);
    id_bits[15] = (device_type & 0x3) << 6; // Last 6 bits are padding

    memcpy(primary_id->raw_bits, id_bits, 16);

    // Apply UUIDv8 format according to RFC 9562
    // Bits 0-47: ID value part 1 (LSB)
    primary_id->UUIDv8_representation.bytes[0] = id_bits[0];
    primary_id->UUIDv8_representation.bytes[1] = id_bits[1]; 
    primary_id->UUIDv8_representation.bytes[2] = id_bits[2];
    primary_id->UUIDv8_representation.bytes[3] = id_bits[3];
    primary_id->UUIDv8_representation.bytes[4] = id_bits[4];
    primary_id->UUIDv8_representation.bytes[5] = id_bits[5];
    
    // Bits 48-51: Version (8) + Bits 52-63: ID value part 2
    primary_id->UUIDv8_representation.bytes[6] = ((id_bits[6] & 0xF0) >> 4) | 0x80; // Version 8 in upper 4 bits
    primary_id->UUIDv8_representation.bytes[7] = ((id_bits[6] & 0x0F) << 4) | ((id_bits[7] & 0xF0) >> 4);

    // Bits 64-65: Variant (10b) + Bits 66-127: ID value part 3 (MSB)
    primary_id->UUIDv8_representation.bytes[8] = 0x80 | (id_bits[7] & 0x0F) << 2 | (id_bits[8] & 0xC0) >> 6;
    // everything past here is now shifted by 6 bits
    primary_id->UUIDv8_representation.bytes[9] = ((id_bits[8] & 0x3F) << 2) | ((id_bits[9] & 0xC0) >> 6);
    primary_id->UUIDv8_representation.bytes[10] = ((id_bits[9] & 0x3F) << 2) | ((id_bits[10] & 0xC0) >> 6);
    primary_id->UUIDv8_representation.bytes[11] = ((id_bits[10] & 0x3F) << 2) | ((id_bits[11] & 0xC0) >> 6);
    primary_id->UUIDv8_representation.bytes[12] = ((id_bits[11] & 0x3F) << 2) | ((id_bits[12] & 0xC0) >> 6);
    primary_id->UUIDv8_representation.bytes[13] = ((id_bits[12] & 0x3F) << 2) | ((id_bits[13] & 0xC0) >> 6);
    primary_id->UUIDv8_representation.bytes[14] = ((id_bits[13] & 0x3F) << 2) | ((id_bits[14] & 0xC0) >> 6);
    primary_id->UUIDv8_representation.bytes[15] = ((id_bits[14] & 0x3F) << 2) | ((id_bits[15] & 0xC0) >> 6);

    return AMDCUID_STATUS_SUCCESS;
}

void CuidUtilities::remove_UUIDv8_bits(amdcuid_id_t* id, uint8_t out_raw_bits[16]) {
    if (!id || !out_raw_bits) {
        return;
    }

    // Reverse the UUIDv8 formatting to get back the original raw bits
    // Bits 0-47: ID value part 1 (LSB)
    out_raw_bits[0] = id->bytes[0];
    out_raw_bits[1] = id->bytes[1];
    out_raw_bits[2] = id->bytes[2];
    out_raw_bits[3] = id->bytes[3];
    out_raw_bits[4] = id->bytes[4];
    out_raw_bits[5] = id->bytes[5];

    // Bits 48-51: Version (8) + Bits 52-63: ID value part 2
    out_raw_bits[6] = ((id->bytes[6] & 0x0F) << 4) | ((id->bytes[7] & 0xFC) >> 2);
    out_raw_bits[7] = ((id->bytes[7] & 0x03) << 6) | ((id->bytes[8] & 0xFC) >> 2);

    // Bits 64-65: Variant (10b) + Bits 66-127: ID value part 3 (MSB)
    out_raw_bits[8] = ((id->bytes[8] & 0x03) << 6) | ((id->bytes[9] & 0xFC) >> 2);
    out_raw_bits[9] = ((id->bytes[9] & 0x03) << 6) | ((id->bytes[10] & 0xFC) >> 2);
    out_raw_bits[10] = ((id->bytes[10] & 0x03) << 6) | ((id->bytes[11] & 0xFC) >> 2);
    out_raw_bits[11] = ((id->bytes[11] & 0x03) << 6) | ((id->bytes[12] & 0xFC) >> 2);
    out_raw_bits[12] = ((id->bytes[12] & 0x03) << 6) | ((id->bytes[13] & 0xFC) >> 2);
    out_raw_bits[13] = ((id->bytes[13] & 0x03) << 6) | ((id->bytes[14] & 0xFC) >> 2);
    out_raw_bits[14] = ((id->bytes[14] & 0x03) << 6) | ((id->bytes[15] & 0xFC) >> 2);
    out_raw_bits[15] = (id->bytes[15] & 0x03) << 6; // last 6 bits are padding
}

std::string CuidUtilities::get_cuid_as_string(const amdcuid_id_t *id) {
    // Format as UUIDv8 string: 8-4-4-4-12 hex digits from id->bytes[16]
    // UUID: xxxxxxxx-xxxx-8xxx-yxxx-xxxxxxxxxxxx
    char uuid_str[37]; // 36 chars + null
    // Format the bytes into a UUID string
    snprintf(uuid_str, sizeof(uuid_str),
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             id->bytes[0], id->bytes[1], id->bytes[2], id->bytes[3],
             id->bytes[4], id->bytes[5],
             id->bytes[6], id->bytes[7],
             id->bytes[8], id->bytes[9],
             id->bytes[10], id->bytes[11], id->bytes[12], id->bytes[13], id->bytes[14], id->bytes[15]);

    return std::string(uuid_str);
}

amdcuid_status_t CuidUtilities::uuid_string_to_uint8(const std::string& uuid_str, uint8_t* uuid) {
    if (!uuid) {
        return AMDCUID_STATUS_INVALID_ARGUMENT;
    }

    // UUID format: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX (36 chars with hyphens)
    // Remove hyphens and validate length
    std::string hex_str;
    for (char c : uuid_str) {
        if (c != '-') {
            if (!isxdigit(c)) {
                std::cerr << "Invalid UUID format: non-hex character found" << std::endl;
                return AMDCUID_STATUS_INVALID_ARGUMENT;
            }
            hex_str += c;
        }
    }

    // UUID should be 128 bits = 32 hex characters
    if (hex_str.length() != 32) {
        std::cerr << "Invalid UUID length: expected 32 hex digits, got " << hex_str.length() << std::endl;
        return AMDCUID_STATUS_INVALID_ARGUMENT;
    }

    // convert hex_str to uint8_t array
    for (size_t i = 0; i < 16; ++i) {
        std::string byte_str = hex_str.substr(i * 2, 2);
        uuid[i] = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
    }

    return AMDCUID_STATUS_SUCCESS;
}

std::string CuidUtilities::device_type_to_string(amdcuid_device_type_t type) {
    switch (type) {
        case AMDCUID_DEVICE_TYPE_PLATFORM: return "PLATFORM";
        case AMDCUID_DEVICE_TYPE_CPU: return "CPU";
        case AMDCUID_DEVICE_TYPE_GPU: return "GPU";
        case AMDCUID_DEVICE_TYPE_NIC: return "NIC";
        default: return "UNKNOWN";
    }
}
