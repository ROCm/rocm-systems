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

#include "smbios_util.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <sys/stat.h>
#include "cuid_util.h"
#include <unistd.h>

amdcuid_status_t SmbiosUtil::get_system_uuid(uint8_t* uuid) {
    if (!uuid) {
        return AMDCUID_STATUS_INVALID_ARGUMENT;
    }

    if (geteuid() != 0){
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    // attempt to read UUID from sysfs first
    std::string uuid_str = CuidUtilities::read_sysfs_file(std::string(DMI_PATH) + "product_uuid");
    if (!uuid_str.empty()) {
        // Convert the UUID string to uint8_t array
        amdcuid_status_t status = CuidUtilities::uuid_string_to_uint8(uuid_str, uuid);
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }
    } else {
        // try and get UUID from SMBIOS table
        amdcuid_status_t status = SmbiosUtil::get_uuid_from_smbios_table(uuid);
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }
    }

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t SmbiosUtil::get_uuid_from_smbios_table(uint8_t* uuid) {

    const std::string smbios_path = std::string(DMI_TABLES_PATH) + "DMI";
    const std::string smbios_entry_path = std::string(DMI_TABLES_PATH) + "smbios_entry_point";

    if (geteuid() != 0){
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    // validate the SMBIOS entry point and version. If version is < 2.1, then there won't be a UUID and we should quit there
    std::ifstream validate_version(smbios_entry_path);
    if (validate_version.is_open()) {
        // check the anchor string first
        char anchor[4];
        validate_version.read(anchor, 4);
        if (std::string(anchor, 4) == "_SM_") {
            // version starts at offset 0x6
            validate_version.seekg(0x6);
            char version_major;
            char version_minor;
            validate_version.read(&version_major, 1);
            validate_version.read(&version_minor, 1);
            // Check if version is less than 2.1
            if (version_major < 2 || (version_major == 2 && version_minor < 1)) {
                return AMDCUID_STATUS_UNSUPPORTED;
            }
        }
        else if (std::string(anchor, 4) == "_SM3") {
            // version starts at offset 0x7
            validate_version.seekg(0x7);
            char version_major;
            char version_minor;
            validate_version.read(&version_major, 1);
            validate_version.read(&version_minor, 1);
            // Check if version is less than 3.0
            if (version_major < 3) {
                return AMDCUID_STATUS_UNSUPPORTED;
            }
        }
        else {
            // file is corrupted/malformed so
            return AMDCUID_STATUS_SMBIOS_ERROR;
        }
    }
    else {
        return AMDCUID_STATUS_SMBIOS_ERROR;
    }
    // that's all we needed with version so we can close the file
    validate_version.close();

    // now search through the DMI struct table to find the System Header and then the UUID
    std::ifstream struct_table(smbios_path);
    if (!struct_table)
    {
        // failed to open the SMBIOS struct table file
        return AMDCUID_STATUS_SMBIOS_ERROR;
    }

    uint8_t type = 0;
    uint32_t struct_offset = 0;
    const uint8_t TYPE_SYSTEM_INFO = 0x1;
    uint8_t smbios_header[4] = {0};
    struct_table.read(reinterpret_cast<char*>(smbios_header), 4);
    type = smbios_header[0];
    // iterate through the table to find the System Header and then the UUID
    while (type != TYPE_SYSTEM_INFO && !struct_table.eof())
    {
        // move to the next structure and get past the strings section
        struct_offset += smbios_header[1];
        struct_table.seekg(struct_offset, std::ios::beg);

        bool peek1, peek2;
        peek1 = peek2 = false;
        while (!peek1 || !peek2) {
            peek1 = (struct_table.peek() == 0);
            struct_table.ignore(1);
            struct_offset++;
            peek2 = (struct_table.peek() == 0);
        }
        struct_table.ignore(1); // Skip the double null terminator
        struct_offset++;
        struct_table.read(reinterpret_cast<char*>(smbios_header), 4);
        type = smbios_header[0];
    }

    if (type != TYPE_SYSTEM_INFO) {
        return AMDCUID_STATUS_SMBIOS_ERROR;
    }

    // now at the System Info structure, read the UUID at offset 0x8
    uint8_t raw_bytes[16] = {0};
    const uint8_t UUID_OFFSET = 0x8;
    struct_table.seekg(struct_offset + UUID_OFFSET, std::ios::beg);
    struct_table.read(reinterpret_cast<char*>(raw_bytes), 16);

    // we need to store the bytes in such a way that when we use get_cuid_as_string, it prints out the same way as regular UUIDs
    uuid[0] = raw_bytes[3];
    uuid[1] = raw_bytes[2];
    uuid[2] = raw_bytes[1];
    uuid[3] = raw_bytes[0];

    uuid[4] = raw_bytes[5];
    uuid[5] = raw_bytes[4];

    uuid[6] = raw_bytes[7];
    uuid[7] = raw_bytes[6];

    for (int i = 8; i < 16; ++i) {
        uuid[i] = raw_bytes[i];
    }

    struct_table.close();

    return AMDCUID_STATUS_SUCCESS;
}

// Get system serial number
amdcuid_status_t SmbiosUtil::get_system_serial(std::string &serial) {

    if (geteuid() != 0){
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    // Try multiple sources in order of preference
    const char* serial_files[] = {
        "product_serial",
        "board_serial",
        "chassis_serial"
    };
    
    for (const char* filename : serial_files) {
        std::string path = std::string(DMI_PATH) + filename;
        serial = CuidUtilities::read_sysfs_file(path);
        
        if (!serial.empty()) {
            return AMDCUID_STATUS_SUCCESS;
        }
    }
    
    return AMDCUID_STATUS_SMBIOS_ERROR;
}

// Get board information
amdcuid_status_t SmbiosUtil::get_board_info(std::string &vendor, 
                                          std::string &name, 
                                          std::string &version) {
    vendor = CuidUtilities::read_sysfs_file(std::string(DMI_PATH) + "board_vendor");
    name = CuidUtilities::read_sysfs_file(std::string(DMI_PATH) + "board_name");
    version = CuidUtilities::read_sysfs_file(std::string(DMI_PATH) + "board_version");
    
    // Consider success if at least one field is available
    if (!vendor.empty() || 
        !name.empty() ||
        !version.empty()) {
        return AMDCUID_STATUS_SUCCESS;
    }
    
    return AMDCUID_STATUS_SMBIOS_ERROR;
}

// Get BIOS information
amdcuid_status_t SmbiosUtil::get_bios_info(std::string &vendor,
                                         std::string &version,
                                         std::string &date) {
    vendor = CuidUtilities::read_sysfs_file(std::string(DMI_PATH) + "bios_vendor");
    version = CuidUtilities::read_sysfs_file(std::string(DMI_PATH) + "bios_version");
    date = CuidUtilities::read_sysfs_file(std::string(DMI_PATH) + "bios_date");
    
    // Consider success if at least one field is available
    if (!vendor.empty() || 
        !version.empty() ||
        !date.empty()) {
        return AMDCUID_STATUS_SUCCESS;
    }
    
    return AMDCUID_STATUS_SMBIOS_ERROR;
}

// Get product information
amdcuid_status_t SmbiosUtil::get_product_info(std::string &name,
                                            std::string &family) {
    name = CuidUtilities::read_sysfs_file(std::string(DMI_PATH) + "product_name");
    family = CuidUtilities::read_sysfs_file(std::string(DMI_PATH) + "product_family");
    
    // Consider success if at least one field is available
    if (!name.empty() || 
        !family.empty()) {
        return AMDCUID_STATUS_SUCCESS;
    }
    
    return AMDCUID_STATUS_SMBIOS_ERROR;
}