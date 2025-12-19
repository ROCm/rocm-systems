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
#include "cuid.h"
#include "cuid_file.h"
#include "cuid_device_manager.h"
#include "cuid_device.h"
#include "cuid_gpu.h"
#include "cuid_cpu.h"
#include "cuid_nic.h"
#include "cuid_util.h"

/**
 * @file amdcuid_tool.cc
 * @brief AMD CUID command-line tool for generating and querying CUIDs
 * 
 * This tool provides functionality to:
 * - Generate CUID files from discovered hardware
 * - Query and display device CUIDs
 * - Update CUID files
 * - Monitor device changes (future daemon mode)
 */

inline const char* cuid_status_to_string(amdcuid_status_t status) {
    switch (status) {
        case AMDCUID_STATUS_SUCCESS: return "SUCCESS";
        case AMDCUID_STATUS_FILE_NOT_FOUND: return "FILE_NOT_FOUND";
        case AMDCUID_STATUS_DEVICE_NOT_FOUND: return "DEVICE_NOT_FOUND";
        case AMDCUID_STATUS_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case AMDCUID_STATUS_PERMISSION_DENIED: return "PERMISSION_DENIED";
        case AMDCUID_STATUS_UNSUPPORTED: return "UNSUPPORTED";
        case AMDCUID_STATUS_WRONG_DEVICE_TYPE: return "WRONG_DEVICE_TYPE";
        case AMDCUID_STATUS_INSUFFICIENT_SIZE: return "INSUFFICIENT_SIZE";
        case AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND: return "AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND";
        case AMDCUID_STATUS_HW_FINGERPRINT_FORMAT_ERROR: return "AMDCUID_STATUS_HW_FINGERPRINT_FORMAT_ERROR";
        case AMDCUID_STATUS_HW_FINGERPRINT_PERMISSION_DENIED: return "AMDCUID_STATUS_HW_FINGERPRINT_PERMISSION_DENIED";
        default: return "UNKNOWN_ERROR";
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "AMD Component Unified Identifier (CUID) Tool\n\n";
    std::cout << "Options:\n";
    std::cout << "  --generate-cuid <key_file>  Generate CUID files from discovered devices\n";
    std::cout << "                               Requires HMAC key file for secondary CUID generation\n";
    std::cout << "                               Creates /tmp/cuid and /tmp/priv_cuid (if root)\n";
    std::cout << "  --list                       List all devices and their secondary CUIDs\n";
    std::cout << "  --list-file <file>           List devices from existing CUID file\n";
    std::cout << "  --type <type>                Filter by device type (gpu, cpu, nic, platform)\n";
    std::cout << "  --show-primary               Show primary CUIDs (requires root and priv_cuid file)\n";
    std::cout << "  --query-device <identifier>  Query specific device by node/core_id\n";
    std::cout << "  --output-file <path>         Specify output file path (default: /tmp/cuid)\n";
    std::cout << "  --priv-output-file <path>    Specify privileged output file path (default: /tmp/priv_cuid)\n";
    std::cout << "  --help, -h                   Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  # Generate CUID files (requires root for priv_cuid)\n";
    std::cout << "  sudo " << program_name << " --generate-cuid /path/to/hmac_key.bin\n\n";
    std::cout << "  # List all GPUs with their CUIDs\n";
    std::cout << "  " << program_name << " --list --type gpu\n\n";
    std::cout << "  # List devices from existing CUID file\n";
    std::cout << "  " << program_name << " --list-file /tmp/cuid\n\n";
    std::cout << "  # Query specific device\n";
    std::cout << "  " << program_name << " --query-device /sys/class/drm/renderD128\n\n";
}

std::string format_timestamp(time_t timestamp) {
    char buffer[64];
    struct tm* tm_info = localtime(&timestamp);
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
    return std::string(buffer);
}

int generate_cuid_files(const std::string& key_file, 
                        const std::string& output_file,
                        const std::string& priv_output_file) {
    std::cout << "Generating CUID files...\n" << std::endl;
    
    // Initialize device manager and discover devices
    auto& mgr = AmdCuidDeviceManager::instance();
    amdcuid_status_t status = mgr.init(AMDCUID_DEVICE_TYPE_SET_ALL);
    
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to initialize device manager: " << cuid_status_to_string(status) << std::endl;
        return 1;
    }
    
    std::cout << "Discovered " << mgr.devices().size() << " device(s)" << std::endl;
    
    // Generate CUID files
    status = CuidFileGenerator::generate_from_devices(
        mgr.devices(),
        key_file,
        output_file,
        priv_output_file
    );
    
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to generate CUID files; error status: " << cuid_status_to_string(status) << std::endl;
        return 1;
    }
    
    std::cout << "\nCUID files generated successfully!" << std::endl;
    return 0;
}

int list_devices(bool show_primary, const std::string* filter_type = nullptr) {
    // Initialize device manager
    auto& mgr = AmdCuidDeviceManager::instance();
    amdcuid_status_t status = mgr.init(AMDCUID_DEVICE_TYPE_SET_ALL);
    
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to initialize device manager; error status: " << cuid_status_to_string(status) << std::endl;
        return 1;
    }
    
    // const auto& devices = mgr.devices();
    
    
    
    // Group devices by type
    std::map<amdcuid_device_type_t, std::vector<std::shared_ptr<AmdCuidDevice>>> grouped;
    mgr.get_grouped_devices(grouped);

    if (grouped.empty()) {
        std::cout << "No devices found." << std::endl;
        return 0;
    }
    
    std::cout << "Discovered " << grouped.size() << " device(s):\n" << std::endl;
    
    // Display devices
    for (const auto& kv : grouped) {
        amdcuid_device_type_t type = kv.first;
        const std::vector<std::shared_ptr<AmdCuidDevice>>& device_list = kv.second;
        std::string type_str = AmdCuidUtilities::device_type_to_string(type);
        
        // Case-insensitive type filter comparison
        if (filter_type) {
            std::string filter_upper = *filter_type;
            for (auto& c : filter_upper) c = toupper(c);
            if (filter_upper != type_str) {
                continue;
            }
        }
        
        std::cout << "---- " << type_str << " Devices ----" << std::endl;
        
        for (size_t i = 0; i < device_list.size(); ++i) {
            const auto& device = device_list[i];
            
            std::cout << type_str << " #" << i;
            
            // Get primary CUID
            amdcuid primary_id = {};
            device->get_primary_cuid(primary_id);
            
            if (show_primary) {
                std::cout << "\n  Primary CUID:   " << AmdCuidUtilities::get_cuid_as_string(&primary_id);
            }
            
            // Get secondary CUID (would need HMAC key in real implementation)
            // For now, just show primary
            std::cout << "\n  CUID:           " << AmdCuidUtilities::get_cuid_as_string(&primary_id);
            
            // Show device-specific info
            if (type == AMDCUID_DEVICE_TYPE_GPU) {
                auto gpu = std::dynamic_pointer_cast<AmdCuidGpu>(device);
                if (gpu) {
                    const auto& info = gpu->get_info();
                    std::cout << "\n  Vendor:         0x" << std::hex << info.header.fields.gpu.vendor_id << std::dec;
                    std::cout << "\n  Device:         0x" << std::hex << info.header.fields.gpu.device_id << std::dec;
                    std::cout << "\n  PCI Class:      0x" << std::hex << info.header.fields.gpu.pci_class << std::dec;
                    std::cout << "\n  Revision:       0x" << std::hex << (int)info.header.fields.gpu.revision_id << std::dec;
                    if (!info.bdf.empty()) {
                        std::cout << "\n  BDF:            " << info.bdf;
                    }
                    if (!info.render_node.empty()) {
                        std::cout << "\n  Render Node:    " << info.render_node;
                    }
                }
            }
            
            std::cout << "\n" << std::endl;
        }
    }
    
    return 0;
}

int list_from_file(const std::string& file_path, bool show_primary) {
    CuidFile cuid_file(file_path, show_primary);
    
    if (!cuid_file.exists()) {
        std::cerr << "Error: CUID file not found: " << file_path << std::endl;
        return 1;
    }
    
    amdcuid_status_t status = cuid_file.load();
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to load CUID file; error status: " << cuid_status_to_string(status) << std::endl;
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
    
    std::cout << "CUID File: " << file_path << std::endl;
    std::cout << "Type: " << (cuid_file.is_privileged() ? "Privileged" : "Unprivileged") << "\n" << std::endl;
    
    for (const auto& kv : grouped) {
        amdcuid_device_type_t type = kv.first;
        const std::vector<CuidFileEntry>& entry_list = kv.second;
        std::string type_str = AmdCuidUtilities::device_type_to_string(type);
        std::cout << "---- " << type_str << " CUIDs ----" << std::endl;
        
        for (const auto& entry : entry_list) {
            if (type == AMDCUID_DEVICE_TYPE_PLATFORM) {
                std::cout << type_str;
            } else {
                std::cout << type_str << " #" << entry.device_index;
            }
            
            if (show_primary && cuid_file.is_privileged()) {
                std::cout << "\n  Primary CUID:   " << AmdCuidUtilities::get_cuid_as_string(&entry.primary_cuid);
            }
            std::cout << "\n  CUID:           " << AmdCuidUtilities::get_cuid_as_string(&entry.secondary_cuid);
            
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

int query_device(const std::string& identifier) {
    // Try to load from /tmp/cuid first
    CuidFile cuid_file("/tmp/cuid", false);
    
    if (!cuid_file.exists()) {
        std::cerr << "Error: CUID file not found at /tmp/cuid" << std::endl;
        std::cerr << "Please run with --generate-cuid first" << std::endl;
        return 1;
    }
    
    amdcuid_status_t status = cuid_file.load();
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "Error: Failed to load CUID file; error status: " << cuid_status_to_string(status) << std::endl;
        return 1;
    }
    
    // Try different search methods
    CuidFileEntry entry;
    
    // Try as device node
    status = cuid_file.find_by_device_node(identifier, entry);
    if (status == AMDCUID_STATUS_SUCCESS) {
        std::cout << "Device Found:" << std::endl;
        std::cout << "  Type:           " << AmdCuidUtilities::device_type_to_string(entry.device_type) << std::endl;
        std::cout << "  CUID:           " << AmdCuidUtilities::get_cuid_as_string(&entry.secondary_cuid) << std::endl;
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
        std::cout << "  Type:           " << AmdCuidUtilities::device_type_to_string(entry.device_type) << std::endl;
        std::cout << "  CUID:           " << AmdCuidUtilities::get_cuid_as_string(&entry.secondary_cuid) << std::endl;
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
        {"list-file",          required_argument, 0, 'f'},
        {"type",               required_argument, 0, 't'},
        {"show-primary",       no_argument,       0, 'p'},
        {"query-device",       required_argument, 0, 'q'},
        {"output-file",        required_argument, 0, 'o'},
        {"priv-output-file",   required_argument, 0, 'P'},
        {"help",               no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    std::string key_file;
    std::string list_file_path;
    std::string filter_type;
    std::string query_identifier;
    std::string output_file = "/tmp/cuid";
    std::string priv_output_file = "/tmp/priv_cuid";
    bool do_generate = false;
    bool do_list = false;
    bool do_list_file = false;
    bool show_primary = false;
    bool do_query = false;
    
    int opt;
    int option_index = 0;
    
    while ((opt = getopt_long(argc, argv, "g:lf:t:pq:o:P:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'g':
                do_generate = true;
                key_file = optarg;
                break;
            case 'l':
                do_list = true;
                break;
            case 'f':
                do_list_file = true;
                list_file_path = optarg;
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
            case 'o':
                output_file = optarg;
                break;
            case 'P':
                priv_output_file = optarg;
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
        return generate_cuid_files(key_file, output_file, priv_output_file);
    } else if (do_list) {
        return list_devices(show_primary, filter_type.empty() ? nullptr : &filter_type);
    } else if (do_list_file) {
        return list_from_file(list_file_path, show_primary);
    } else if (do_query) {
        return query_device(query_identifier);
    } else {
        print_usage(argv[0]);
        return 1;
    }
    
    return 0;
}
