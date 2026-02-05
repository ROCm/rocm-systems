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

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cstring>
#include <getopt.h>
#include <unistd.h>
#include <errno.h>
#include "include/amd_cuid.h"
#include "src/cuid_file.h"
#include "src/cuid_device_manager.h"
#include "src/cuid_device.h"
#include "src/cuid_gpu.h"
#include "src/cuid_cpu.h"
#include "src/cuid_nic.h"
#include "src/cuid_util.h"

/**
 * @file amdcuid_tool.cc
 * @brief AMD CUID command-line tool for generating and querying CUIDs
 * 
 * This tool provides functionality to:
 * - Generate CUID files from discovered hardware
 * - List and query device CUIDs from the CUID file
 * - Always reads from /tmp/cuid (or /tmp/priv_cuid with sudo for primary CUIDs)
 */


// Default CUID file paths - use functions to get paths at runtime
static const char* get_default_cuid_file() { return CuidUtilities::cuid_file().c_str(); }
static const char* get_default_priv_cuid_file() { return CuidUtilities::priv_cuid_file().c_str(); }

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "AMD Component Unified Identifier (CUID) Tool\n\n";
    std::cout << "Options:\n";
    std::cout << "  --generate-cuid <key_file>   Generate CUID files from discovered devices\n";
    std::cout << "                               Requires HMAC key file for derived CUID generation\n";
    std::cout << "                               Creates /tmp/cuid and /tmp/priv_cuid (if root)\n";
    std::cout << "  --list                       List all devices and their CUIDs from CUID file\n";
    std::cout << "                               Reads from /tmp/cuid (or /tmp/priv_cuid with --show-primary)\n";
    std::cout << "  --type <type>                Filter by device type (gpu, cpu, nic, platform)\n";
    std::cout << "                               Use with --list or --query-device\n";
    std::cout << "  --show-primary               Show primary CUIDs (requires root privileges)\n";
    std::cout << "                               Use with --list or --query-device\n";
    std::cout << "  --query-device <identifier>  Query specific device by node path or package:core ID\n";
    std::cout << "  --help, -h                   Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  # Generate CUID files (requires root for priv_cuid)\n";
    std::cout << "  sudo " << program_name << " --generate-cuid /path/to/hmac_key.bin\n\n";
    std::cout << "  # List all devices with their CUIDs\n";
    std::cout << "  " << program_name << " --list\n\n";
    std::cout << "  # List all GPUs with their CUIDs\n";
    std::cout << "  " << program_name << " --list --type gpu\n\n";
    std::cout << "  # List all devices with primary CUIDs (requires root)\n";
    std::cout << "  sudo " << program_name << " --list --show-primary\n\n";
    std::cout << "  # Query specific device\n";
    std::cout << "  " << program_name << " --query-device /sys/class/drm/renderD128\n\n";
    std::cout << "  # Query device with primary CUID (requires root)\n";
    std::cout << "  sudo " << program_name << " --query-device 0:0 --show-primary\n\n";
}

const char* device_type_to_string(amdcuid_device_type_t type) {
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

amdcuid_device_type_t string_to_device_type(const std::string& type_str) {
    std::string upper = type_str;
    for (auto& c : upper) c = toupper(c);
    
    if (upper == "PLATFORM") return AMDCUID_DEVICE_TYPE_PLATFORM;
    if (upper == "CPU") return AMDCUID_DEVICE_TYPE_CPU;
    if (upper == "GPU") return AMDCUID_DEVICE_TYPE_GPU;
    if (upper == "NIC") return AMDCUID_DEVICE_TYPE_NIC;
    if (upper == "NPU") return AMDCUID_DEVICE_TYPE_NPU;
    if (upper == "STORAGE") return AMDCUID_DEVICE_TYPE_STORAGE;
    if (upper == "MEMORY") return AMDCUID_DEVICE_TYPE_MEMORY;
    return AMDSMI_DEVICE_TYPE_NONE;
}

std::string cuid_to_string(const amdcuid_id_t& id) {
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

std::string format_timestamp(time_t timestamp) {
    char buffer[64];
    struct tm* tm_info = localtime(&timestamp);
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
    return std::string(buffer);
}

/**
 * @brief Check if running with root privileges
 */
bool is_root() {
    return geteuid() == 0;
}

/**
 * @brief Get appropriate error message for file access failures
 */
std::string get_file_error_message(const std::string& file_path, bool show_primary) {
    if (access(file_path.c_str(), F_OK) != 0) {
        return "CUID file not found: " + file_path + "\n"
               "Please run 'sudo amdcuid_tool --generate-cuid <key_file>' first to generate CUID files.";
    }
    if (access(file_path.c_str(), R_OK) != 0) {
        if (show_primary) {
            return "Permission denied: Cannot read " + file_path + "\n"
                   "Reading primary CUIDs requires root privileges. Try running with sudo.";
        }
        return "Permission denied: Cannot read " + file_path;
    }
    return "Failed to load CUID file: " + file_path;
}

int generate_cuid_files(const std::string& key_file) {
    std::cout << "Generating CUID files...\n" << std::endl;
    
    // Initialize device manager and discover devices
    auto& mgr = CuidDeviceManager::instance();
    amdcuid_status_t status = mgr.init();
    
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to initialize device manager (status: " << status << ")" << std::endl;
        if (status == AMDCUID_STATUS_PERMISSION_DENIED) {
            std::cerr << "Some devices may require root privileges to discover." << std::endl;
        }
        return 1;
    }
    
    std::cout << "Discovered " << mgr.devices().size() << " device(s)" << std::endl;
    
    // Generate CUID files
    status = CuidFileGenerator::generate_from_devices(
        mgr.devices(),
        key_file,
        get_default_cuid_file(),
        get_default_priv_cuid_file()
    );
    
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to generate CUID files (status: " << status << ")" << std::endl;
        return 1;
    }
    
    std::cout << "\nCUID files generated successfully!" << std::endl;
    std::cout << "  Public CUID file:     " << get_default_cuid_file() << std::endl;
    if (is_root()) {
        std::cout << "  Privileged CUID file: " << get_default_priv_cuid_file() << std::endl;
    }
    return 0;
}

int list_devices(bool show_primary, const std::string* filter_type) {
    // Determine which file to read based on show_primary flag
    std::string file_path = show_primary ? get_default_priv_cuid_file() : get_default_cuid_file();
    
    // Check for root privileges if requesting primary CUIDs
    if (show_primary && !is_root()) {
        std::cerr << "Error: Permission denied\n";
        std::cerr << "Reading primary CUIDs requires root privileges. Try running with sudo." << std::endl;
        return 1;
    }
    
    CuidFile cuid_file(file_path, show_primary);
    
    if (!cuid_file.exists()) {
        std::cerr << "Error: " << get_file_error_message(file_path, show_primary) << std::endl;
        return 1;
    }
    
    amdcuid_status_t status = cuid_file.load();
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: " << get_file_error_message(file_path, show_primary) << std::endl;
        return 1;
    }
    
    // const auto& entries = cuid_file.get_entries();
    

    
    // Group by type
    std::map<amdcuid_device_type_t, std::vector<CuidFileEntry>> grouped;
    cuid_file.get_grouped_entries(grouped);

    if (grouped.empty()) {
        std::cout << "No entries found in CUID file." << std::endl;
        return 0;
    }
    
    // Parse filter type if provided
    amdcuid_device_type_t filter_device_type = AMDSMI_DEVICE_TYPE_NONE;
    if (filter_type) {
        filter_device_type = string_to_device_type(*filter_type);
        if (filter_device_type == AMDSMI_DEVICE_TYPE_NONE) {
            std::cerr << "Error: Unknown device type '" << *filter_type << "'" << std::endl;
            std::cerr << "Valid types: platform, cpu, gpu, nic, npu, storage, memory" << std::endl;
            return 1;
        }
    }
    
    if (grouped.empty()) {
        if (filter_type) {
            std::cout << "No " << *filter_type << " devices found." << std::endl;
        } else {
            std::cout << "No devices found." << std::endl;
        }
        return 0;
    }
    
    // Count total entries after filtering
    size_t total = 0;
    for (const auto& kv : grouped) {
        total += kv.second.size();
    }
    
    std::cout << "Found " << total << " device(s)";
    if (filter_type) {
        std::cout << " of type '" << *filter_type << "'";
    }
    std::cout << ":\n" << std::endl;
    
    for (const auto& kv : grouped) {
        amdcuid_device_type_t type = kv.first;
        const std::vector<CuidFileEntry>& entry_list = kv.second;
        std::string type_str = device_type_to_string(type);
        std::cout << "---- " << type_str << " Devices ----" << std::endl;
        
        for (const auto& entry : entry_list) {
            if (type == AMDCUID_DEVICE_TYPE_PLATFORM) {
                std::cout << type_str;
            } else {
                std::cout << type_str << " #" << entry.device_index;
            }
            
            if (show_primary && cuid_file.is_privileged()) {
                std::cout << "\n  Primary CUID:   " << CuidUtilities::get_cuid_as_string(&entry.primary_cuid);
            }
            std::cout << "\n  CUID:           " << CuidUtilities::get_cuid_as_string(&entry.derived_cuid);
            
            if (!entry.device_node.empty()) {
                std::cout << "\n  Device Node:    " << entry.device_node;
            }
            if (!entry.package_core_id.empty()) {
                std::cout << "\n  Package:Core:   " << entry.package_core_id;
            }
            if (!entry.bdf.empty()) {
                std::cout << "\n  BDF:            " << entry.bdf;
            }
            if (!entry.mac_address.empty()) {
                std::cout << "\n  MAC Address:    " << entry.mac_address;
            }
            if (entry.last_update > 0) {
                std::cout << "\n  Last Update:    " << format_timestamp(entry.last_update);
            }
            
            std::cout << "\n" << std::endl;
        }
    }
    
    return 0;
}

int query_device(const std::string& identifier, bool show_primary) {
    // Determine which file to read based on show_primary flag
    std::string file_path = show_primary ? get_default_priv_cuid_file() : get_default_cuid_file();
    
    // Check for root privileges if requesting primary CUIDs
    if (show_primary && !is_root()) {
        std::cerr << "Error: Permission denied\n";
        std::cerr << "Reading primary CUIDs requires root privileges. Try running with sudo." << std::endl;
        return 1;
    }
    
    CuidFile cuid_file(file_path, show_primary);
    
    if (!cuid_file.exists()) {
        std::cerr << "Error: " << get_file_error_message(file_path, show_primary) << std::endl;
        return 1;
    }
    
    amdcuid_status_t status = cuid_file.load();
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: " << get_file_error_message(file_path, show_primary) << std::endl;
        return 1;
    }
    
    // Try different search methods
    CuidFileEntry entry;
    
    // Try as device node
    status = cuid_file.find_by_device_node(identifier, entry);
    if (status == AMDCUID_STATUS_SUCCESS) {
        std::cout << "Device Found:" << std::endl;
        std::cout << "  Type:           " << device_type_to_string(entry.device_type) << std::endl;
        if (show_primary && cuid_file.is_privileged()) {
            std::cout << "  Primary CUID:   " << cuid_to_string(entry.primary_cuid) << std::endl;
        }
        std::cout << "  CUID:           " << cuid_to_string(entry.derived_cuid) << std::endl;
        std::cout << "  Device Node:    " << entry.device_node << std::endl;
        if (!entry.bdf.empty()) {
            std::cout << "  BDF:            " << entry.bdf << std::endl;
        }
        std::cout << "  Last Update:    " << format_timestamp(entry.last_update) << std::endl;
        return 0;
    }
    
    // Try as package:core ID
    status = cuid_file.find_by_package_core_id(identifier, entry);
    if (status == AMDCUID_STATUS_SUCCESS) {
        std::cout << "Device Found:" << std::endl;
        std::cout << "  Type:           " << device_type_to_string(entry.device_type) << std::endl;
        if (show_primary && cuid_file.is_privileged()) {
            std::cout << "  Primary CUID:   " << cuid_to_string(entry.primary_cuid) << std::endl;
        }
        std::cout << "  CUID:           " << cuid_to_string(entry.derived_cuid) << std::endl;
        std::cout << "  Package:Core:   " << entry.package_core_id << std::endl;
        std::cout << "  Last Update:    " << format_timestamp(entry.last_update) << std::endl;
        return 0;
    }
    
    std::cerr << "Error: Device not found: " << identifier << std::endl;
    return 1;
}

int main(int argc, char* argv[]) {
    static struct option long_options[] = {
        {"generate-cuid",      required_argument, 0, 'g'},
        {"list",               no_argument,       0, 'l'},
        {"type",               required_argument, 0, 't'},
        {"show-primary",       no_argument,       0, 'p'},
        {"query-device",       required_argument, 0, 'q'},
        {"help",               no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    std::string key_file;
    std::string filter_type;
    std::string query_identifier;
    bool do_generate = false;
    bool do_list = false;
    bool show_primary = false;
    bool do_query = false;
    
    int opt;
    int option_index = 0;
    
    while ((opt = getopt_long(argc, argv, "g:lt:pq:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'g':
                do_generate = true;
                key_file = optarg;
                break;
            case 'l':
                do_list = true;
                break;
            case 't':
                filter_type = optarg;
                break;
            case 'p':
                show_primary = true;
                break;
            case 'q':
                do_query = true;
                query_identifier = optarg;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // Execute requested operation
    if (do_generate) {
        if (key_file.empty()) {
            std::cerr << "Error: HMAC key file required for --generate-cuid" << std::endl;
            return 1;
        }
        return generate_cuid_files(key_file);
    } else if (do_list) {
        return list_devices(show_primary, filter_type.empty() ? nullptr : &filter_type);
    } else if (do_query) {
        return query_device(query_identifier, show_primary);
    } else {
        print_usage(argv[0]);
        return 1;
    }
    
    return 0;
}
