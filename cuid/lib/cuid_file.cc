#include "cuid_file.h"
#include "cuid_util.h"
#include "cuid_device.h"
#include "cuid_gpu.h"
#include "cuid_cpu.h"
#include "cuid_nic.h"
#include "cuid_platform.h"
#include "hmac.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

// ============================================================================
// CuidFile Implementation
// ============================================================================

CuidFile::CuidFile(const std::string& file_path, bool is_privileged)
    : file_path_(file_path)
    , is_privileged_(is_privileged)
{
}

bool CuidFile::exists() const {
    struct stat buffer;
    return (stat(file_path_.c_str(), &buffer) == 0);
}

std::string CuidFile::trim(const std::string& str) const {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

std::string CuidFile::device_type_to_string(amdcuid_device_type_t type) const {
    switch (type) {
        case AMDCUID_DEVICE_TYPE_PLATFORM: return "PLATFORM";
        case AMDCUID_DEVICE_TYPE_CPU: return "CPU";
        case AMDCUID_DEVICE_TYPE_GPU: return "GPU";
        case AMDCUID_DEVICE_TYPE_NIC: return "NIC";
        case AMDCUID_DEVICE_TYPE_NPU: return "NPU";
        case AMDCUID_DEVICE_TYPE_STORAGE: return "STORAGE";
        case AMDCUID_DEVICE_TYPE_MEMORY: return "MEMORY";
        case AMDCUID_DEVICE_TYPE_OTHER: return "OTHER";
        default: return "UNKNOWN";
    }
}

amdcuid_device_type_t CuidFile::string_to_device_type(const std::string& str) const {
    if (str == "PLATFORM") return AMDCUID_DEVICE_TYPE_PLATFORM;
    if (str == "CPU") return AMDCUID_DEVICE_TYPE_CPU;
    if (str == "GPU") return AMDCUID_DEVICE_TYPE_GPU;
    if (str == "NIC") return AMDCUID_DEVICE_TYPE_NIC;
    if (str == "NPU") return AMDCUID_DEVICE_TYPE_NPU;
    if (str == "STORAGE") return AMDCUID_DEVICE_TYPE_STORAGE;
    if (str == "MEMORY") return AMDCUID_DEVICE_TYPE_MEMORY;
    if (str == "OTHER") return AMDCUID_DEVICE_TYPE_OTHER;
    return AMDCUID_DEVICE_TYPE_UNKNOWN;
}

std::string CuidFile::cuid_to_string(const amdcuid& id) const {
    char uuid_str[37];
    snprintf(uuid_str, sizeof(uuid_str),
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             id.bytes[0], id.bytes[1], id.bytes[2], id.bytes[3],
             id.bytes[4], id.bytes[5],
             id.bytes[6], id.bytes[7],
             id.bytes[8], id.bytes[9],
             id.bytes[10], id.bytes[11], id.bytes[12], id.bytes[13], id.bytes[14], id.bytes[15]);
    return std::string(uuid_str);
}

amdcuid CuidFile::string_to_cuid(const std::string& str) const {
    amdcuid id;
    memset(id.bytes, 0, sizeof(id.bytes));
    
    // Parse UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    // Remove dashes and parse hex bytes
    std::string hex_only;
    for (char c : str) {
        if (c != '-') hex_only += c;
    }
    
    if (hex_only.length() != 32) {
        return id; // Return zero-filled ID on parse error
    }
    
    for (int i = 0; i < 16; ++i) {
        std::string byte_str = hex_only.substr(i * 2, 2);
        id.bytes[i] = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
    }
    
    return id;
}

bool CuidFile::parse_section_header(const std::string& line, 
                                     amdcuid_device_type_t& type, 
                                     uint32_t& index) const {
    // Parse lines like [GPU:0] or [PLATFORM]
    if (line.empty() || line[0] != '[' || line.back() != ']') {
        return false;
    }
    
    std::string content = line.substr(1, line.length() - 2);
    size_t colon_pos = content.find(':');
    
    if (colon_pos != std::string::npos) {
        // Format: TYPE:INDEX
        std::string type_str = content.substr(0, colon_pos);
        std::string index_str = content.substr(colon_pos + 1);
        type = string_to_device_type(type_str);
        index = std::stoul(index_str);
    } else {
        // Format: PLATFORM (no index)
        type = string_to_device_type(content);
        index = 0;
    }
    
    return type != AMDCUID_DEVICE_TYPE_UNKNOWN;
}

amdcuid_status_t CuidFile::load() {
    entries_.clear();
    
    std::ifstream file(file_path_);
    if (!file.is_open()) {
        return AMDCUID_STATUS_FILE_NOT_FOUND;
    }
    
    std::string line;
    CuidFileEntry current_entry;
    bool in_section = false;
    
    while (std::getline(file, line)) {
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }
        
        // Check for section header
        if (line[0] == '[') {
            // Save previous entry if valid
            if (in_section) {
                entries_.push_back(current_entry);
            }
            
            // Start new section
            amdcuid_device_type_t type;
            uint32_t index;
            if (parse_section_header(line, type, index)) {
                current_entry = CuidFileEntry();
                current_entry.device_type = type;
                current_entry.device_index = index;
                in_section = true;
            } else {
                in_section = false;
            }
            continue;
        }
        
        // Parse key=value pairs
        if (in_section) {
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = trim(line.substr(0, eq_pos));
                std::string value = trim(line.substr(eq_pos + 1));
                
                if (key == "primary_cuid") {
                    current_entry.primary_cuid = string_to_cuid(value);
                } else if (key == "secondary_cuid") {
                    current_entry.secondary_cuid = string_to_cuid(value);
                } else if (key == "device_node") {
                    current_entry.device_node = value;
                } else if (key == "package_core_id") {
                    current_entry.package_core_id = value;
                } else if (key == "bdf") {
                    current_entry.bdf = value;
                } else if (key == "mac_address") {
                    current_entry.mac_address = value;
                } else if (key == "last_update") {
                    current_entry.last_update = std::stol(value);
                }
            }
        }
    }
    
    // Save last entry
    if (in_section) {
        entries_.push_back(current_entry);
    }
    
    file.close();
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidFile::save() {
    // Create temporary file first for atomic write
    std::string temp_path = file_path_ + ".tmp";
    std::ofstream file(temp_path);
    
    if (!file.is_open()) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }
    
    // Write header comment
    file << "# AMD CUID Device Information File\n";
    file << "# Auto-generated by AMD CUID library\n";
    file << "# DO NOT EDIT MANUALLY\n";
    file << "# File: " << file_path_ << "\n";
    if (is_privileged_) {
        file << "# Type: Privileged (contains primary CUIDs)\n";
        file << "# Permissions: Root access only\n";
    } else {
        file << "# Type: Unprivileged (secondary CUIDs only)\n";
        file << "# Permissions: Readable by all users\n";
    }
    file << "\n";
    
    // Group entries by type for better organization
    std::map<amdcuid_device_type_t, std::vector<CuidFileEntry>> grouped;
    for (const auto& entry : entries_) {
        grouped[entry.device_type].push_back(entry);
    }
    
    // Define output order
    std::vector<amdcuid_device_type_t> order = {
        AMDCUID_DEVICE_TYPE_GPU,
        AMDCUID_DEVICE_TYPE_CPU,
        AMDCUID_DEVICE_TYPE_NIC,
        AMDCUID_DEVICE_TYPE_NPU,
        AMDCUID_DEVICE_TYPE_STORAGE,
        AMDCUID_DEVICE_TYPE_MEMORY,
        AMDCUID_DEVICE_TYPE_PLATFORM,
        AMDCUID_DEVICE_TYPE_OTHER
    };
    
    for (auto type : order) {
        if (grouped.find(type) == grouped.end()) continue;
        
        for (const auto& entry : grouped[type]) {
            // Write section header
            if (entry.device_type == AMDCUID_DEVICE_TYPE_PLATFORM) {
                file << "[" << device_type_to_string(entry.device_type) << "]\n";
            } else {
                file << "[" << device_type_to_string(entry.device_type) 
                     << ":" << entry.device_index << "]\n";
            }
            
            // Write primary CUID (privileged file only)
            if (is_privileged_) {
                file << "primary_cuid=" << cuid_to_string(entry.primary_cuid) << "\n";
            }
            
            // Write secondary CUID
            file << "secondary_cuid=" << cuid_to_string(entry.secondary_cuid) << "\n";
            
            // Write device-specific fields
            if (!entry.device_node.empty()) {
                file << "device_node=" << entry.device_node << "\n";
            }
            if (!entry.package_core_id.empty()) {
                file << "package_core_id=" << entry.package_core_id << "\n";
            }
            if (!entry.bdf.empty()) {
                file << "bdf=" << entry.bdf << "\n";
            }
            if (!entry.mac_address.empty()) {
                file << "mac_address=" << entry.mac_address << "\n";
            }
            
            // Write timestamp
            file << "last_update=" << entry.last_update << "\n";
            file << "\n";
        }
    }
    
    file.close();
    
    // Atomically move temp file to actual file
    if (rename(temp_path.c_str(), file_path_.c_str()) != 0) {
        unlink(temp_path.c_str());
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }
    
    // Set permissions
    if (!is_privileged_) {
        // Unprivileged file: readable by all (644)
        chmod(file_path_.c_str(), S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    } else {
        // Privileged file: readable by root only (600)
        chmod(file_path_.c_str(), S_IRUSR | S_IWUSR);
    }
    
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidFile::add_entry(const CuidFileEntry& entry) {
    // Check if entry with same type and index exists
    for (auto& existing : entries_) {
        if (existing.device_type == entry.device_type && 
            existing.device_index == entry.device_index) {
            // Update existing entry
            existing = entry;
            return AMDCUID_STATUS_SUCCESS;
        }
    }
    
    // Add new entry
    entries_.push_back(entry);
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidFile::find_by_device_node(const std::string& device_node, 
                                                 CuidFileEntry& entry) const {
    for (const auto& e : entries_) {
        if (e.device_node == device_node) {
            entry = e;
            return AMDCUID_STATUS_SUCCESS;
        }
    }
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t CuidFile::find_by_package_core_id(const std::string& package_core_id, 
                                                     CuidFileEntry& entry) const {
    for (const auto& e : entries_) {
        if (e.package_core_id == package_core_id) {
            entry = e;
            return AMDCUID_STATUS_SUCCESS;
        }
    }
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t CuidFile::find_by_device_type(amdcuid_device_type_t device_type, 
                                                 CuidFileEntry& entry) const {
    for (const auto& e : entries_) {
        if (e.device_type == device_type) {
            entry = e;
            return AMDCUID_STATUS_SUCCESS;
        }
    }
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t CuidFile::find_by_secondary_cuid(const amdcuid& secondary_cuid, 
                                                    CuidFileEntry& entry) const {
    for (const auto& e : entries_) {
        if (memcmp(e.secondary_cuid.bytes, secondary_cuid.bytes, sizeof(amdcuid::bytes)) == 0) {
            entry = e;
            return AMDCUID_STATUS_SUCCESS;
        }
    }
    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

// ============================================================================
// CuidFileGenerator Implementation
// ============================================================================

amdcuid_status_t CuidFileGenerator::generate_from_devices(
    const std::vector<std::shared_ptr<AmdCuidDevice>>& devices,
    const std::string& key_file_path,
    const std::string& unprivileged_file,
    const std::string& privileged_file)
{
    // Check if we have root privileges
    bool is_root = (geteuid() == 0);
    
    // Initialize HMAC for secondary CUID generation
    AMDCUID_HMAC hmac(key_file_path);
    if (!hmac.is_valid()) {
        std::cerr << "Error: Failed to initialize HMAC with key file" << std::endl;
        return AMDCUID_STATUS_KEY_ERROR;
    }
    
    // Create file handlers
    CuidFile unpriv_cuid_file(unprivileged_file, false);
    CuidFile priv_cuid_file(privileged_file, true);
    
    // Clear existing entries
    unpriv_cuid_file.clear();
    priv_cuid_file.clear();
    
    // Get current timestamp
    time_t now = time(nullptr);
    
    // Track device indices per type
    std::map<amdcuid_device_type_t, uint32_t> device_counters;
    
    // Process each device
    for (const auto& device : devices) {
        if (!device) continue;
        
        CuidFileEntry entry;
        entry.device_type = device->type();
        entry.device_index = device_counters[entry.device_type]++;
        entry.last_update = now;
        
        // Get primary CUID
        amdcuid primary_id = {};
        amdcuid_status_t status = device->get_primary_cuid(primary_id);
        if (status != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Warning: Failed to get primary CUID for device type " 
                      << entry.device_type << std::endl;
            continue;
        }
        entry.primary_cuid = primary_id;
        
        // Generate secondary CUID using HMAC
        amdcuid secondary_id = {};
        status = AmdCuidUtilities::generate_secondary_cuid(&primary_id, &secondary_id, &hmac);
        if (status != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Warning: Failed to generate secondary CUID for device type " 
                      << entry.device_type << std::endl;
            continue;
        }
        entry.secondary_cuid = secondary_id;
        
        // Fill in device-specific information
        switch (entry.device_type) {
            case AMDCUID_DEVICE_TYPE_GPU: {
                auto gpu = std::dynamic_pointer_cast<AmdCuidGpu>(device);
                if (gpu) {
                    const auto& info = gpu->get_info();
                    entry.device_node = info.render_node;
                    entry.bdf = info.bdf;
                }
                break;
            }
            case AMDCUID_DEVICE_TYPE_CPU: {
                auto cpu = std::dynamic_pointer_cast<AmdCuidCpu>(device);
                if (cpu) {
                    const auto& info = cpu->get_info();
                    // Format: package:core
                    entry.package_core_id = std::to_string(info.header.fields.cpu.physical_id) + 
                                          ":" + std::to_string(info.header.fields.cpu.core);
                }
                break;
            }
            case AMDCUID_DEVICE_TYPE_NIC: {
                auto nic = std::dynamic_pointer_cast<AmdCuidNic>(device);
                if (nic) {
                    const auto& info = nic->get_info();
                    entry.device_node = info.network_interface;
                    // MAC address would need to be read from sysfs, for now skip
                    entry.bdf = info.bdf;
                }
                break;
            }
            case AMDCUID_DEVICE_TYPE_PLATFORM: {
                // Platform has no additional fields
                break;
            }
            default:
                break;
        }
        
        // Add to both files
        unpriv_cuid_file.add_entry(entry);
        if (is_root) {
            priv_cuid_file.add_entry(entry);
        }
    }
    
    // Save unprivileged file (always)
    amdcuid_status_t status = unpriv_cuid_file.save();
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to save unprivileged CUID file: " 
                  << unprivileged_file << std::endl;
        return status;
    }
    
    std::cout << "Successfully generated: " << unprivileged_file << std::endl;
    
    // Save privileged file (only if root)
    if (is_root) {
        status = priv_cuid_file.save();
        if (status != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Error: Failed to save privileged CUID file: " 
                      << privileged_file << std::endl;
            return status;
        }
        std::cout << "Successfully generated: " << privileged_file << std::endl;
    } else {
        std::cout << "Note: Skipping privileged file (requires root access)" << std::endl;
    }
    
    return AMDCUID_STATUS_SUCCESS;
}
