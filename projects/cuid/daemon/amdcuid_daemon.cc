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

#include "cuid.h"
#include "cuid_file.h"
#include "cuid_device_manager.h"
#include "cuid_device.h"
#include "cuid_gpu.h"
#include "cuid_cpu.h"
#include "cuid_nic.h"
#include "cuid_util.h"
#include "hmac.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <map>
#include <unistd.h>
#include <memory>

// Global log file stream
static std::unique_ptr<std::ofstream> g_log_file;
static bool g_logging_to_file = false;

static std::ostream& log_out() {
    if (g_logging_to_file && g_log_file && g_log_file->is_open()) {
        return *g_log_file;
    }
    return std::cout;
}

static std::ostream& log_err() {
    if (g_logging_to_file && g_log_file && g_log_file->is_open()) {
        return *g_log_file;
    }
    return std::cerr;
}

static void init_logging(bool enabled) {
    if (enabled) {
        g_log_file = std::make_unique<std::ofstream>("/var/log/amdcuid.log", std::ios::app);
        if (g_log_file->is_open()) {
            g_logging_to_file = true;
            // Add timestamp to log entry
            time_t now = time(nullptr);
            *g_log_file << "\n=== Log started at " << ctime(&now);
        }
    }
}

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

amdcuid_status_t remove_device(std::string output_file,
                                std::string priv_output_file,
                                CuidFileEntry *device) {

    // Load existing CUID files
    CuidFile unpriv_file(output_file, false);
    CuidFile priv_file(priv_output_file, true);
    unpriv_file.load();
    priv_file.load();

    log_out() << "Attempting removal of device with secondary CUID: " << AmdCuidUtilities::get_cuid_as_string(&device->secondary_cuid) << std::endl;

    amdcuid_status_t status;
    // Remove entry from both files
    status = unpriv_file.remove_entry(*device);
    if (status != AMDCUID_STATUS_SUCCESS) {
        log_err() << "Error removing device from unprivileged file: " << cuid_status_to_string(status) << std::endl;
        return status;
    }
    status = priv_file.remove_entry(*device);
    if (status != AMDCUID_STATUS_SUCCESS) {
        log_err() << "Error removing device from privileged file: " << cuid_status_to_string(status) << std::endl;
        return status;
    }

    // Save updated CUID files
    unpriv_file.save();
    priv_file.save();

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t update_device(std::string output_file,
                                std::string priv_output_file,
                                CuidFileEntry *device) {
    // Load existing CUID files
    CuidFile unpriv_file(output_file, false);
    CuidFile priv_file(priv_output_file, true);
    unpriv_file.load();
    priv_file.load();

    log_out() << "Attempting update of device with secondary CUID: " << AmdCuidUtilities::get_cuid_as_string(&device->secondary_cuid) << std::endl;

    amdcuid_status_t status;
    // Remove entry from both files
    status = unpriv_file.add_entry(*device);
    if (status != AMDCUID_STATUS_SUCCESS) {
        log_err() << "Error updating device in unprivileged file: " << cuid_status_to_string(status) << std::endl;
        return status;
    }
    status = priv_file.add_entry(*device);
    if (status != AMDCUID_STATUS_SUCCESS) {
        log_err() << "Error updating device in privileged file: " << cuid_status_to_string(status) << std::endl;
        return status;
    }


    // Save updated CUID files
    unpriv_file.save();
    priv_file.save();

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t get_device_from_udev(std::string *action_output, CuidFileEntry *device_entry, AMDCUID_HMAC* hmac) {
    // udev passes device information as environment variables when triggering rules
    const char* action = std::getenv("ACTION");
    const char* devpath = std::getenv("DEVPATH");
    const char* subsystem = std::getenv("SUBSYSTEM");
    const char* devname = std::getenv("DEVNAME");
    const char* pci_slot = std::getenv("PCI_SLOT_NAME");

    // Validate required environment variables
    if (!devpath || !subsystem) {
        log_err() << "Error: Missing required udev environment variables (DEVPATH, SUBSYSTEM)" << std::endl;
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    }

    // Log udev event info
    log_out() << "udev event received:" << std::endl;
    log_out() << "  ACTION: " << (action ? action : "(null)") << std::endl;
    log_out() << "  DEVPATH: " << devpath << std::endl;
    log_out() << "  SUBSYSTEM: " << subsystem << std::endl;
    log_out() << "  DEVNAME: " << (devname ? devname : "(null)") << std::endl;
    log_out() << "  PCI_SLOT_NAME: " << (pci_slot ? pci_slot : "(null)") << std::endl;

    // Build sysfs path from DEVPATH
    std::string syspath = "/sys" + std::string(devpath);

    // Parse uevent file for additional properties
    std::map<std::string, std::string> uevent_props;
    std::ifstream uevent_file(syspath + "/uevent");
    if (uevent_file.is_open()) {
        std::string line;
        while (std::getline(uevent_file, line)) {
            size_t eq = line.find('=');
            if (eq != std::string::npos) {
                uevent_props[line.substr(0, eq)] = line.substr(eq + 1);
            }
        }
        uevent_file.close();
    }
    else {
        log_out() << "Failed to open uevent file. Exiting." << std::endl;
        return AMDCUID_STATUS_FILE_NOT_FOUND;
    }

    // Determine device type from subsystem and create appropriate device entry
    std::string subsys_str(subsystem);
    CuidFileEntry entry = CuidFileEntry();

    if (subsys_str == "drm") {
        // GPU device
        amdcuid_gpu_info info = {};
        amdcuid_status_t status = AmdCuidGpu::discover_single(&info, syspath + "/device");
        auto gpu_device = std::make_shared<AmdCuidGpu>(info);

        entry.device_type = AMDCUID_DEVICE_TYPE_GPU;
        amdcuid primary_id;
        status = gpu_device->get_primary_cuid(primary_id);
        if (status != AMDCUID_STATUS_SUCCESS) {
            log_err() << "Error: Failed to get primary CUID for GPU device" << std::endl;
            return status;
        }
        entry.primary_cuid = primary_id;
        amdcuid secondary_id;
        status = gpu_device->get_secondary_cuid(secondary_id, hmac);
        if (status != AMDCUID_STATUS_SUCCESS) {
            log_err() << "Error: Failed to generate secondary CUID for GPU device" << std::endl;
            return status;
        }
        log_out() << "Generated secondary CUID for GPU device: " << AmdCuidUtilities::get_cuid_as_string(&secondary_id) << std::endl;
        entry.secondary_cuid = secondary_id;
        entry.device_node = info.render_node;
        entry.bdf = info.bdf;
        entry.device_index = 0; // could be set based on existing entries
        entry.last_update = time(nullptr);
    } else if (subsys_str == "net") {
        // NIC device
        amdcuid_nic_info info = {};
        amdcuid_status_t status = AmdCuidNic::discover_single(&info, syspath + "/device");
        auto nic_device = std::make_shared<AmdCuidNic>(info);

        entry.device_type = AMDCUID_DEVICE_TYPE_NIC;
        amdcuid primary_id;
        status = nic_device->get_primary_cuid(primary_id);
        if (status != AMDCUID_STATUS_SUCCESS) {
            log_err() << "Error: Failed to get primary CUID for NIC device" << std::endl;
            return status;
        }
        entry.primary_cuid = primary_id;
        amdcuid secondary_id;
        status = nic_device->get_secondary_cuid(secondary_id, hmac);
        if (status != AMDCUID_STATUS_SUCCESS) {
            log_err() << "Error: Failed to generate secondary CUID for NIC device" << std::endl;
            return status;
        }
        entry.secondary_cuid = secondary_id;
        entry.device_node = info.network_interface;
        entry.bdf = info.bdf;
        entry.device_index = 0; // could be set based on existing entries
        entry.last_update = time(nullptr);
    } else {
        // additional subsystems can be added later as support expands
        log_err() << "Error: Unsupported subsystem: " << subsystem << std::endl;
        return AMDCUID_STATUS_UNSUPPORTED;
    }

    if (!device_entry) {
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    }

    // Set action output
    if (action_output) {
        *action_output = std::string(action);
    } else {
        *action_output = "unknown";
    }
    // Store the device entry
    *device_entry = entry;

    // For a daemon that processes one event and exits, this is sufficient.
    // For a long-running daemon, consider using AmdCuidDeviceManager.

    return AMDCUID_STATUS_SUCCESS;
}

int main() {
    // Note: We can't log to file yet until we read the config
    std::cout << "AMD CUID Daemon Starting..." << std::endl;

    if (geteuid() != 0) {
        std::cerr << "Root privileges required to detect relevant devices and generate CUID. Exiting" << std::endl;
        return 1;
    }

    // read config file first to set key file path, logging options, and whether to run as a daemon or only on boot
    std::ifstream config_file("/opt/cuid/amdcuid_daemon.conf");
    std::vector<std::string> config_lines;

    if (config_file.is_open()) {
        std::string line;
        while (std::getline(config_file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }
            size_t eq = line.find('=');
            if (eq != std::string::npos) {
                config_lines.push_back(line.substr(eq + 1));
            }
        }
        config_file.close();
    }
    else {
        std::cerr << "Failed to open config file. Exiting." << std::endl;
        return 1;
    }

    if (config_lines.size() < 3) {
        std::cerr << "Insufficient config parameters. Exiting." << std::endl;
        return 1;
    }

    std::string key_file = config_lines[1];
    bool logging_enabled = (config_lines[2] == "true");
    std::string output_file = "/tmp/cuid";
    std::string priv_output_file = "/tmp/priv_cuid";

    // Initialize file logging if enabled
    init_logging(logging_enabled);
    log_out() << "AMD CUID Daemon initialized with logging " << (logging_enabled ? "enabled" : "disabled") << std::endl;

    if (config_lines[0] == "true") {
        // in daemon mode, we expect to be triggered by udev on device add/remove/change

        // create HMAC instance with key file
        AMDCUID_HMAC hmac(key_file);
        if (!hmac.is_valid()) {
            log_err() << "Error: Failed to initialize HMAC with key file" << std::endl;
            return 1;
        }
        // get udev input in argv, fill out device handle, and then update CUIDs
        CuidFileEntry device_handle;
        std::string action;
        amdcuid_status_t status = get_device_from_udev(&action, &device_handle, &hmac);
        if (status != AMDCUID_STATUS_SUCCESS) {
            log_err() << "Error: Failed to get device from udev. status: " << cuid_status_to_string(status) << std::endl;
            return 1;
        }
        // actions we have to worry about: add, remove, change, move
        if (action == "remove") {
            status = remove_device(output_file, priv_output_file, &device_handle);
            if (status != AMDCUID_STATUS_SUCCESS) {
                log_err() << "Error: Failed to remove device CUID. status: " << cuid_status_to_string(status) << std::endl;
                return 1;
            }
        }
        else if (action == "add" || action == "change" || action == "move") {
            status = update_device(output_file, priv_output_file, &device_handle);
            if (status != AMDCUID_STATUS_SUCCESS) {
                log_err() << "Error: Failed to update device CUID. status: " << cuid_status_to_string(status) << std::endl;
                return 1;
            }
        }
        else {
            log_err() << "Error: Unsupported udev action: " << action << std::endl;
            return 1;
        }
    }
    else {
        // non-daemon mode discovers devices on bootup and updates their CUIDs once
        // Initialize device manager and discover devices
        auto& mgr = AmdCuidDeviceManager::instance();
        amdcuid_status_t status = mgr.init(AMDCUID_DEVICE_TYPE_SET_ALL);

        if (status != AMDCUID_STATUS_SUCCESS) {
            log_err() << "Error: Failed to initialize device manager (status: " << status << ")" << std::endl;
            if (status == AMDCUID_STATUS_PERMISSION_DENIED) {
                log_err() << "Some devices may require root privileges to discover." << std::endl;
            }
            return 1;
        }

        log_out() << "Discovered " << mgr.devices().size() << " device(s)" << std::endl;

        // Generate CUID files
        status = CuidFileGenerator::generate_from_devices(
            mgr.devices(),
            key_file,
            output_file,
            priv_output_file
        );

        if (status != AMDCUID_STATUS_SUCCESS) {
            log_err() << "Error: Failed to generate CUID files (status: " << status << ")" << std::endl;
            return 1;
        }
    }

    log_out() << "AMD CUID Daemon Exiting..." << std::endl;
    return 0;
}