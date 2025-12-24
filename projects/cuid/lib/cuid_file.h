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

#ifndef CUID_FILE_H
#define CUID_FILE_H

#include "cuid.h"
#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <cstring>
#include <memory>

struct CuidFileEntry {
    amdcuid_device_type_t device_type;
    uint32_t device_index;  // e.g., 0 for GPU:0, 1 for GPU:1
    
    amdcuid primary_cuid;     // Only in privileged file
    amdcuid secondary_cuid;
    
    // Device-specific identifiers
    std::string device_node;       // For GPU: /sys/class/drm/renderD128, NIC: /sys/class/net/eth0
    std::string package_core_id;   // For CPU: "0:0" (package:core)
    std::string bdf;               // PCIe Bus:Device.Function
    std::string mac_address;       // For NIC
    
    time_t last_update;            // Unix timestamp
    
    CuidFileEntry() 
        : device_type(AMDCUID_DEVICE_TYPE_UNKNOWN)
        , device_index(0)
        , last_update(0)
    {
        memset(primary_cuid.bytes, 0, sizeof(primary_cuid.bytes));
        memset(secondary_cuid.bytes, 0, sizeof(secondary_cuid.bytes));
    }
};

/**
 * @brief CUID File handler for reading and writing device CUID information
 */
class CuidFile {
public:
    /**
     * @brief Constructor
     * @param file_path Path to the CUID file (e.g., /tmp/cuid or /tmp/priv_cuid)
     * @param is_privileged Whether this is a privileged file (includes primary CUIDs)
     */
    CuidFile(const std::string& file_path, bool is_privileged = false);
    
    amdcuid_status_t load();
    amdcuid_status_t save();
    amdcuid_status_t add_entry(const CuidFileEntry& entry);
    amdcuid_status_t remove_entry(const CuidFileEntry& entry);
    
    const std::vector<CuidFileEntry>& get_entries() const { return entries_; }

    amdcuid_status_t find_by_device_node(const std::string& device_node, CuidFileEntry& entry) const;
    amdcuid_status_t find_by_package_core_id(const std::string& package_core_id, CuidFileEntry& entry) const;
    amdcuid_status_t find_by_device_type(amdcuid_device_type_t device_type, CuidFileEntry& entry) const;
    amdcuid_status_t find_by_secondary_cuid(const amdcuid& secondary_cuid, CuidFileEntry& entry) const;
    
    void clear() { entries_.clear(); }
    bool exists() const;
    const std::string& get_file_path() const { return file_path_; }
    
    /**
     * @brief Check if this is a privileged file
     */
    bool is_privileged() const { return is_privileged_; }

    // static utility to group entries by device type
    void get_grouped_entries(std::map<amdcuid_device_type_t, std::vector<CuidFileEntry>>& grouped) const;

private:
    std::string file_path_;
    bool is_privileged_;
    std::vector<CuidFileEntry> entries_;
    
    // Helper functions
    amdcuid_device_type_t string_to_device_type(const std::string& str) const;
    amdcuid string_to_cuid(const std::string& str) const;
    std::string trim(const std::string& str) const;
    bool parse_section_header(const std::string& line, amdcuid_device_type_t& type, uint32_t& index) const;
};

/**
 * @brief Utility class for generating CUID files from discovered devices
 */
class CuidFileGenerator {
public:
    /**
     * @brief Generate CUID files from device manager
     * @param devices Vector of discovered devices
     * @param key_file_path Path to the HMAC key file for secondary CUID generation
     * @param unprivileged_file Path to write unprivileged CUID file (default: /tmp/cuid)
     * @param privileged_file Path to write privileged CUID file (default: /tmp/priv_cuid)
     * @return AMDCUID_STATUS_SUCCESS on success, error code otherwise
     */
    static amdcuid_status_t generate_from_devices(
        const std::vector<std::shared_ptr<class AmdCuidDevice>>& devices,
        const std::string& key_file_path,
        const std::string& unprivileged_file = "/tmp/cuid",
        const std::string& privileged_file = "/tmp/priv_cuid"
    );
};

#endif // CUID_FILE_H
