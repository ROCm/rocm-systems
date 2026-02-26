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
#include <fstream>
#include "include/amd_cuid.h"

/**
 * @file amdcuid_tool.cc
 * @brief AMD CUID command-line tool for generating and querying CUIDs
 * 
 * This tool provides functionality to:
 * - Generate CUID files from discovered hardware using amdcuid_refresh()
 * - List and query device CUIDs using the public amd_cuid.h API
 * - Always reads from the library's internal registry
 */

/**
 * @brief Check if running with root privileges
 */
bool is_root() {
    return geteuid() == 0;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "AMD Component Unified Identifier (CUID) Tool\n\n";
    std::cout << "Options:\n";
    std::cout << "  --generate-cuid              Generate/refresh CUID registry from discovered devices\n";
    std::cout << "                               Requires root privileges\n";
    std::cout << "                               Uses existing key or specify --generate-key/--set-key\n";
    std::cout << "  --generate-key               Generate a new random HMAC key (use with --generate-cuid)\n";
    std::cout << "  --set-key <key_file>         Set HMAC key from file (32 bytes, use with --generate-cuid)\n";
    std::cout << "  --notify-daemon              Notify daemon to refresh device registry (for udev integration)\n";
    std::cout << "  --list                       List all devices and their CUIDs\n";
    std::cout << "  --type <type>                Filter by device type (gpu, cpu, nic, platform)\n";
    std::cout << "                               Use with --list or --query-device\n";
    std::cout << "  --show-primary               Show primary CUIDs (requires root privileges)\n";
    std::cout << "                               Use with --list or --query-device\n";
    std::cout << "  --query-device <identifier>  Query specific device by device path or BDF\n";
    std::cout << "  --version                    Show library version\n";
    std::cout << "  --help, -h                   Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  # Generate CUID registry with a new random key (requires root)\n";
    std::cout << "  sudo " << program_name << " --generate-cuid --generate-key\n\n";
    std::cout << "  # Generate CUID registry with existing key file\n";
    std::cout << "  sudo " << program_name << " --generate-cuid --set-key /path/to/hmac_key.bin\n\n";
    std::cout << "  # Generate CUID registry using previously set key\n";
    std::cout << "  sudo " << program_name << " --generate-cuid\n\n";
    std::cout << "  # Notify daemon of device changes (called by udev)\n";
    std::cout << "  " << program_name << " --notify-daemon\n\n";
    std::cout << "  # List all devices with their CUIDs\n";
    std::cout << "  " << program_name << " --list\n\n";
    std::cout << "  # List all GPUs with their CUIDs\n";
    std::cout << "  " << program_name << " --list --type gpu\n\n";
    std::cout << "  # List all devices with primary CUIDs (requires root)\n";
    std::cout << "  sudo " << program_name << " --list --show-primary\n\n";
    std::cout << "  # Query specific device by path\n";
    std::cout << "  " << program_name << " --query-device /sys/class/drm/renderD128\n\n";
    std::cout << "  # Query device by BDF\n";
    std::cout << "  " << program_name << " --query-device 0000:03:00.0 --type gpu\n\n";
}

const char* device_type_to_string(amdcuid_device_type_t type) {
    switch (type) {
        case AMDCUID_DEVICE_TYPE_PLATFORM: return "PLATFORM";
        case AMDCUID_DEVICE_TYPE_CPU: return "CPU";
        case AMDCUID_DEVICE_TYPE_GPU: return "GPU";
        case AMDCUID_DEVICE_TYPE_NIC: return "NIC";
        case AMDCUID_DEVICE_TYPE_NONE: return "NONE";
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
    return AMDCUID_DEVICE_TYPE_NONE;
}

/**
 * @brief Load HMAC key from a file
 * @param key_file Path to the key file (must be 32 bytes)
 * @param key Output buffer for the key (32 bytes)
 * @return true on success, false on failure
 */
bool load_key_from_file(const std::string& key_file, uint8_t key[32]) {
    std::ifstream file(key_file, std::ios::binary);
    if (!file) {
        return false;
    }
    file.read(reinterpret_cast<char*>(key), 32);
    return file.gcount() == 32;
}

/**
 * @brief Notify the daemon to refresh device registry
 * 
 * This function is called by udev when device changes are detected.
 * It triggers the daemon to re-scan devices via IPC.
 * 
 * @return 0 on success, 1 on failure
 */
int notify_daemon() {
    amdcuid_status_t status = amdcuid_refresh();
    
    if (status != AMDCUID_STATUS_SUCCESS) {
        // Silently fail for udev context - don't spam logs
        // The daemon may not be running, which is acceptable
        return (status == AMDCUID_STATUS_IPC_ERROR) ? 0 : 1;
    }
    
    return 0;
}

/**
 * @brief Query a string property from a device handle
 * @param handle The device handle
 * @param query The query type
 * @param result Output string
 * @return Status code
 */
amdcuid_status_t query_string_property(amdcuid_id_t handle, amdcuid_query_t query, std::string& result) {
    // First query to get required size
    uint32_t length = 0;
    amdcuid_status_t status = amdcuid_query_device_property(handle, query, nullptr, &length);
    
    if (status == AMDCUID_STATUS_INSUFFICIENT_SIZE && length > 0) {
        // Allocate buffer and query again
        std::vector<char> buffer(length);
        status = amdcuid_query_device_property(handle, query, buffer.data(), &length);
        if (status == AMDCUID_STATUS_SUCCESS) {
            // Remove null terminator if present
            result = std::string(buffer.data(), length > 0 && buffer[length-1] == '\0' ? length - 1 : length);
        }
    } else if (status == AMDCUID_STATUS_SUCCESS) {
        result.clear();
    }
    
    return status;
}

int generate_cuid_files(const std::string& key_file, bool generate_key) {
    std::cout << "Generating/refreshing CUID registry...\n" << std::endl;
    
    // Check for root privileges
    if (!is_root()) {
        std::cerr << "Error: Generating CUIDs requires root privileges.\n";
        std::cerr << "Please run with sudo." << std::endl;
        return 1;
    }
    
    // Validate key options - cannot use both together
    if (!key_file.empty() && generate_key) {
        std::cerr << "Error: Cannot use both --generate-key and --set-key together.\n";
        std::cerr << "Use --generate-key to create a new random key, or\n";
        std::cerr << "Use --set-key <file> to load an existing key from a file." << std::endl;
        return 1;
    }
    
    // Handle key setup
    if (!key_file.empty()) {
        // Load key from file
        uint8_t key[32];
        if (!load_key_from_file(key_file, key)) {
            std::cerr << "Error: Failed to read HMAC key from file: " << key_file << std::endl;
            std::cerr << "Key file must be exactly 32 bytes." << std::endl;
            return 1;
        }
        
        amdcuid_status_t status = amdcuid_set_hash_key(key);
        if (status != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Error: Failed to set HMAC key: " << amdcuid_status_to_string(status) << std::endl;
            return 1;
        }
        std::cout << "HMAC key loaded from: " << key_file << std::endl;
    } else if (generate_key) {
        // Generate a new random key
        uint8_t key[32];
        amdcuid_status_t status = amdcuid_generate_hash_key(key);
        if (status != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Error: Failed to generate HMAC key: " << amdcuid_status_to_string(status) << std::endl;
            return 1;
        }
        
        status = amdcuid_set_hash_key(key);
        if (status != AMDCUID_STATUS_SUCCESS) {
            std::cerr << "Error: Failed to set generated HMAC key: " << amdcuid_status_to_string(status) << std::endl;
            return 1;
        }
        std::cout << "Generated new HMAC key." << std::endl;
    }
    // else: No key option specified - attempt to use existing key (will be validated by amdcuid_refresh)
    // else: No key option specified - attempt to use existing key (will be validated by amdcuid_refresh)
    
    // Use amdcuid_refresh() to discover devices and generate CUID files
    amdcuid_status_t status = amdcuid_refresh();
    
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to refresh CUID registry: " << amdcuid_status_to_string(status) << std::endl;
        if (status == AMDCUID_STATUS_KEY_ERROR) {
            std::cerr << "No HMAC key found. Please use --generate-key or --set-key <file> to create a key first." << std::endl;
        } else if (status == AMDCUID_STATUS_PERMISSION_DENIED) {
            std::cerr << "Some devices may require root privileges to discover." << std::endl;
        }
        return 1;
    }
    
    // Count discovered devices
    uint32_t count = 0;
    status = amdcuid_get_all_handles(nullptr, &count);
    if (status == AMDCUID_STATUS_INSUFFICIENT_SIZE || status == AMDCUID_STATUS_SUCCESS) {
        std::cout << "Discovered " << count << " device(s)" << std::endl;
    }
    
    std::cout << "\nCUID registry refreshed successfully!" << std::endl;
    return 0;
}

int list_devices(bool show_primary, const std::string* filter_type) {
    // Check for root privileges if requesting primary CUIDs
    if (show_primary && !is_root()) {
        std::cerr << "Error: Permission denied\n";
        std::cerr << "Reading primary CUIDs requires root privileges. Try running with sudo." << std::endl;
        return 1;
    }
    
    // Parse filter type if provided
    amdcuid_device_type_t filter_device_type = AMDCUID_DEVICE_TYPE_NONE;
    if (filter_type) {
        filter_device_type = string_to_device_type(*filter_type);
        if (filter_device_type == AMDCUID_DEVICE_TYPE_NONE) {
            std::cerr << "Error: Unknown device type '" << *filter_type << "'" << std::endl;
            std::cerr << "Valid types: platform, cpu, gpu, nic" << std::endl;
            return 1;
        }
    }
    
    // Get all device handles using public API
    uint32_t count = 0;
    amdcuid_status_t status = amdcuid_get_all_handles(nullptr, &count);
    
    if (status != AMDCUID_STATUS_INSUFFICIENT_SIZE && status != AMDCUID_STATUS_SUCCESS) {
        if (status == AMDCUID_STATUS_UNSUPPORTED) {
            std::cout << "No devices found.\n";
            std::cout << "Please run 'sudo amdcuid_tool --generate-cuid' first to generate CUID registry." << std::endl;
            return 0;
        }
        std::cerr << "Error: Failed to get device count: " << amdcuid_status_to_string(status) << std::endl;
        return 1;
    }
    
    if (count == 0) {
        std::cout << "No devices found." << std::endl;
        return 0;
    }
    
    std::vector<amdcuid_id_t> handles(count);
    status = amdcuid_get_all_handles(handles.data(), &count);
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to get device handles: " << amdcuid_status_to_string(status) << std::endl;
        return 1;
    }
    
    // Group handles by device type
    std::map<amdcuid_device_type_t, std::vector<amdcuid_id_t>> grouped;
    
    for (const auto& handle : handles) {
        amdcuid_device_type_t device_type;
        uint32_t len = sizeof(device_type);
        status = amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_TYPE, &device_type, &len);
        if (status != AMDCUID_STATUS_SUCCESS) {
            continue;
        }
        
        // Apply filter if specified
        if (filter_type && device_type != filter_device_type) {
            continue;
        }
        
        grouped[device_type].push_back(handle);
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
        const std::vector<amdcuid_id_t>& handle_list = kv.second;
        std::string type_str = device_type_to_string(type);
        std::cout << "---- " << type_str << " Devices ----" << std::endl;
        
        int device_index = 0;
        for (const auto& handle : handle_list) {
            if (type == AMDCUID_DEVICE_TYPE_PLATFORM) {
                std::cout << type_str;
            } else {
                std::cout << type_str << " #" << device_index;
            }
            
            // Query primary CUID if requested and has permission
            if (show_primary) {
                amdcuid_id_t primary_cuid;
                uint32_t len = sizeof(primary_cuid);
                status = amdcuid_query_device_property(handle, AMDCUID_QUERY_PRIMARY_CUID, &primary_cuid, &len);
                if (status == AMDCUID_STATUS_SUCCESS) {
                    std::cout << "\n  Primary CUID:   " << amdcuid_id_to_string(primary_cuid);
                }
            }
            
            // Query derived CUID (the handle itself is the derived CUID)
            std::cout << "\n  CUID:           " << amdcuid_id_to_string(handle);
            
            // Query device path
            std::string device_path;
            if (query_string_property(handle, AMDCUID_QUERY_DEVICE_PATH, device_path) == AMDCUID_STATUS_SUCCESS) {
                std::cout << "\n  Device Path:    " << device_path;
            }
            
            std::cout << "\n" << std::endl;
            device_index++;
        }
    }
    
    return 0;
}

int query_device(const std::string& identifier, bool show_primary, const std::string* device_type_str) {
    // Check for root privileges if requesting primary CUIDs
    if (show_primary && !is_root()) {
        std::cerr << "Error: Permission denied\n";
        std::cerr << "Reading primary CUIDs requires root privileges. Try running with sudo." << std::endl;
        return 1;
    }
    
    amdcuid_id_t handle;
    amdcuid_status_t status;
    
    // Determine device type for lookup
    amdcuid_device_type_t device_type = AMDCUID_DEVICE_TYPE_GPU;  // Default to GPU
    if (device_type_str) {
        device_type = string_to_device_type(*device_type_str);
        if (device_type == AMDCUID_DEVICE_TYPE_NONE) {
            std::cerr << "Error: Unknown device type '" << *device_type_str << "'" << std::endl;
            std::cerr << "Valid types: platform, cpu, gpu, nic" << std::endl;
            return 1;
        }
    }
    
    // Check if identifier looks like a BDF (contains ':' and '.')
    bool is_bdf = (identifier.find(':') != std::string::npos && identifier.find('.') != std::string::npos);
    
    if (is_bdf) {
        // Try as BDF
        status = amdcuid_get_handle_by_bdf(identifier.c_str(), device_type, &handle);
    } else {
        // Try as device path
        status = amdcuid_get_handle_by_dev_path(identifier.c_str(), device_type, &handle);
    }
    
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: Device not found: " << identifier << std::endl;
        std::cerr << "Status: " << amdcuid_status_to_string(status) << std::endl;
        return 1;
    }
    
    // Device found, display information
    std::cout << "Device Found:" << std::endl;
    
    // Query device type
    amdcuid_device_type_t queried_type;
    uint32_t len = sizeof(queried_type);
    status = amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_TYPE, &queried_type, &len);
    if (status == AMDCUID_STATUS_SUCCESS) {
        std::cout << "  Type:           " << device_type_to_string(queried_type) << std::endl;
    }
    
    // Query primary CUID if requested
    if (show_primary) {
        amdcuid_id_t primary_cuid;
        len = sizeof(primary_cuid);
        status = amdcuid_query_device_property(handle, AMDCUID_QUERY_PRIMARY_CUID, &primary_cuid, &len);
        if (status == AMDCUID_STATUS_SUCCESS) {
            std::cout << "  Primary CUID:   " << amdcuid_id_to_string(primary_cuid) << std::endl;
        } else if (status == AMDCUID_STATUS_PERMISSION_DENIED) {
            std::cout << "  Primary CUID:   (requires root)" << std::endl;
        }
    }
    
    // Display derived CUID (the handle)
    std::cout << "  CUID:           " << amdcuid_id_to_string(handle) << std::endl;
    
    // Query device path
    std::string device_path;
    if (query_string_property(handle, AMDCUID_QUERY_DEVICE_PATH, device_path) == AMDCUID_STATUS_SUCCESS) {
        std::cout << "  Device Path:    " << device_path << std::endl;
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    static struct option long_options[] = {
        {"generate-cuid",      no_argument,       0, 'g'},
        {"generate-key",       no_argument,       0, 'k'},
        {"set-key",            required_argument, 0, 's'},
        {"notify-daemon",      no_argument,       0, 'n'},
        {"list",               no_argument,       0, 'l'},
        {"type",               required_argument, 0, 't'},
        {"show-primary",       no_argument,       0, 'p'},
        {"query-device",       required_argument, 0, 'q'},
        {"version",            no_argument,       0, 'v'},
        {"help",               no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    std::string key_file;
    std::string filter_type;
    std::string query_identifier;
    bool do_generate = false;
    bool generate_key = false;
    bool do_notify = false;
    bool do_list = false;
    bool show_primary = false;
    bool do_query = false;
    
    int opt;
    int option_index = 0;
    
    while ((opt = getopt_long(argc, argv, "gks:nlt:pq:vh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'g':
                do_generate = true;
                break;
            case 'k':
                generate_key = true;
                break;
            case 's':
                key_file = optarg;
                break;
            case 'n':
                do_notify = true;
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
            case 'v':
                std::cout << "AMD CUID Library Version: " << amdcuid_library_version_to_string() << std::endl;
                return 0;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // Validate key options usage
    if ((generate_key || !key_file.empty()) && !do_generate) {
        std::cerr << "Error: --generate-key and --set-key can only be used with --generate-cuid" << std::endl;
        return 1;
    }
    
    // Execute requested operation
    if (do_generate) {
        return generate_cuid_files(key_file, generate_key);
    } else if (do_notify) {
        return notify_daemon();
    } else if (do_list) {
        return list_devices(show_primary, filter_type.empty() ? nullptr : &filter_type);
    } else if (do_query) {
        return query_device(query_identifier, show_primary, filter_type.empty() ? nullptr : &filter_type);
    } else {
        print_usage(argv[0]);
        return 1;
    }
    
    return 0;
}
