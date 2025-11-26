#include "acpi_util.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <sys/stat.h>
#include <errno.h>

// Helper function to trim whitespace
std::string AcpiUtil::trim(const std::string &str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

// Read a single sysfs file
amdcuid_status_t AcpiUtil::read_sysfs_file(const std::string &path, std::string &content) {
    std::ifstream file(path);
    
    if (!file.is_open()) {
        // Check if file exists
        struct stat buffer;
        if (stat(path.c_str(), &buffer) != 0) {
            if (errno == ENOENT) {
                return AMDCUID_STATUS_FILE_NOT_FOUND;
            } else if (errno == EACCES) {
                return AMDCUID_STATUS_PERMISSION_DENIED;
            }
            return AMDCUID_STATUS_FILE_NOT_FOUND;
        }
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    content = trim(buffer.str());
    
    file.close();
    
    if (content.empty()) {
        return AMDCUID_STATUS_FILE_NOT_FOUND;
    }
    
    return AMDCUID_STATUS_SUCCESS;
}

// Get system UUID from DMI/SMBIOS
amdcuid_status_t AcpiUtil::get_system_uuid(std::string &uuid) {
    std::string path = std::string(DMI_PATH) + "product_uuid";
    amdcuid_status_t status = read_sysfs_file(path, uuid);
    
    if (status != AMDCUID_STATUS_SUCCESS) {
        return status;
    }
    
    // Validate UUID format (should be xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
    if (uuid.length() != 36) {
        return AMDCUID_STATUS_INVALID_FORMAT;
    }
    
    // Convert to lowercase for consistency
    std::transform(uuid.begin(), uuid.end(), uuid.begin(), ::tolower);
    
    return AMDCUID_STATUS_SUCCESS;
}

// Get system serial number
amdcuid_status_t AcpiUtil::get_system_serial(std::string &serial) {
    // Try multiple sources in order of preference
    const char* serial_files[] = {
        "product_serial",
        "board_serial",
        "chassis_serial"
    };
    
    for (const char* filename : serial_files) {
        std::string path = std::string(DMI_PATH) + filename;
        amdcuid_status_t status = read_sysfs_file(path, serial);
        
        if (status == AMDCUID_STATUS_SUCCESS) {
            return AMDCUID_STATUS_SUCCESS;
        }
    }
    
    return AMDCUID_STATUS_FILE_NOT_FOUND;
}

// Get board information
amdcuid_status_t AcpiUtil::get_board_info(std::string &vendor, 
                                          std::string &name, 
                                          std::string &version) {
    amdcuid_status_t status_vendor = read_sysfs_file(
        std::string(DMI_PATH) + "board_vendor", vendor);
    amdcuid_status_t status_name = read_sysfs_file(
        std::string(DMI_PATH) + "board_name", name);
    amdcuid_status_t status_version = read_sysfs_file(
        std::string(DMI_PATH) + "board_version", version);
    
    // Consider success if at least one field is available
    if (status_vendor == AMDCUID_STATUS_SUCCESS || 
        status_name == AMDCUID_STATUS_SUCCESS ||
        status_version == AMDCUID_STATUS_SUCCESS) {
        return AMDCUID_STATUS_SUCCESS;
    }
    
    return AMDCUID_STATUS_FILE_NOT_FOUND;
}

// Get BIOS information
amdcuid_status_t AcpiUtil::get_bios_info(std::string &vendor,
                                         std::string &version,
                                         std::string &date) {
    amdcuid_status_t status_vendor = read_sysfs_file(
        std::string(DMI_PATH) + "bios_vendor", vendor);
    amdcuid_status_t status_version = read_sysfs_file(
        std::string(DMI_PATH) + "bios_version", version);
    amdcuid_status_t status_date = read_sysfs_file(
        std::string(DMI_PATH) + "bios_date", date);
    
    // Consider success if at least one field is available
    if (status_vendor == AMDCUID_STATUS_SUCCESS || 
        status_version == AMDCUID_STATUS_SUCCESS ||
        status_date == AMDCUID_STATUS_SUCCESS) {
        return AMDCUID_STATUS_SUCCESS;
    }
    
    return AMDCUID_STATUS_FILE_NOT_FOUND;
}

// Get product information
amdcuid_status_t AcpiUtil::get_product_info(std::string &name,
                                            std::string &family) {
    amdcuid_status_t status_name = read_sysfs_file(
        std::string(DMI_PATH) + "product_name", name);
    amdcuid_status_t status_family = read_sysfs_file(
        std::string(DMI_PATH) + "product_family", family);
    
    // Consider success if at least one field is available
    if (status_name == AMDCUID_STATUS_SUCCESS || 
        status_family == AMDCUID_STATUS_SUCCESS) {
        return AMDCUID_STATUS_SUCCESS;
    }
    
    return AMDCUID_STATUS_FILE_NOT_FOUND;
}

// Read raw ACPI table
amdcuid_status_t AcpiUtil::read_acpi_table(const std::string &table_name,
                                           uint8_t *buffer,
                                           size_t buffer_size,
                                           size_t *bytes_read) {
    if (table_name.empty() || !buffer || buffer_size == 0) {
        return AMDCUID_STATUS_INVALID_ARGUMENT;
    }
    
    std::string path = std::string(ACPI_TABLES_PATH) + table_name;
    
    // Check if file exists and get size
    struct stat file_stat;
    if (stat(path.c_str(), &file_stat) != 0) {
        if (errno == ENOENT) {
            return AMDCUID_STATUS_FILE_NOT_FOUND;
        } else if (errno == EACCES) {
            return AMDCUID_STATUS_PERMISSION_DENIED;
        }
        return AMDCUID_STATUS_FILE_NOT_FOUND;
    }
    
    // Check if buffer is large enough
    if (buffer_size < static_cast<size_t>(file_stat.st_size)) {
        if (bytes_read) {
            *bytes_read = file_stat.st_size;
        }
        return AMDCUID_STATUS_BUFFER_TOO_SMALL;
    }
    
    // Read the table
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    file.read(reinterpret_cast<char*>(buffer), file_size);
    
    if (!file) {
        file.close();
        return AMDCUID_STATUS_FILE_ERROR;
    }
    
    if (bytes_read) {
        *bytes_read = file_size;
    }
    
    file.close();
    return AMDCUID_STATUS_SUCCESS;
}
