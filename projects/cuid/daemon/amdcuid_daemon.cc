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

#include "include/amd_cuid.h"
#include "src/cuid_file.h"
#include "src/cuid_device_manager.h"
#include "src/cuid_device.h"
#include "src/cuid_gpu.h"
#include "src/cuid_cpu.h"
#include "src/cuid_nic.h"
#include "src/cuid_platform.h"
#include "src/cuid_util.h"
#include "src/hmac.h"
#include "src/ipc_protocol.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <map>
#include <unistd.h>
#include <memory>

// static hmac instance for daemon
static cuid_hmac daemon_hmac = cuid_hmac();

// Daemon Server
class CuidDaemonServer
{
public:
    CuidDaemonServer() : is_running_(false), server_fd_(-1) {}
    ~CuidDaemonServer() {
        stop();
    }

    amdcuid_status_t start() {
        if (is_running_) {
            return AMDCUID_STATUS_SUCCESS; // Already running
        }

        server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (server_fd_ < 0) {
            return AMDCUID_STATUS_IPC_ERROR;
        }

        unlink(AMDCUID_SOCKET_PATH); // Remove existing socket file

        // bind to socket path
        struct sockaddr_un server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sun_family = AF_UNIX;
        strncpy(server_addr.sun_path, AMDCUID_SOCKET_PATH, sizeof(server_addr.sun_path) - 1);

        if (bind(server_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            close(server_fd_);
            return AMDCUID_STATUS_IPC_ERROR;
        }

        // Set permissions; only root permissions can access
        chmod(AMDCUID_SOCKET_PATH, 0600);

        if (listen(server_fd_, 5) < 0) {
            close(server_fd_);
            return AMDCUID_STATUS_IPC_ERROR;
        }

        is_running_ = true;
        server_thread_ = std::thread(&CuidDaemonServer::accept_loop, this);

        return AMDCUID_STATUS_SUCCESS;
    }

    void stop() {
        if (!is_running_) {
            return;
        }
        is_running_ = false;
        if (server_fd_ >= 0) {
            shutdown(server_fd_, SHUT_RDWR);
            close(server_fd_);
            server_fd_ = -1;
        }
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        unlink(AMDCUID_SOCKET_PATH);
    }

private:
    std::atomic<bool> is_running_;
    int server_fd_;
    std::thread server_thread_;

    void accept_loop() {
        while (is_running_) {
            int client_fd = accept(server_fd_, nullptr, nullptr);
            if (client_fd < 0) {
                continue;
            }

            handle_client(client_fd);
            close(client_fd);
        }
    }

    void handle_client(int client_fd) {
        IpcRequest request;
        if (recv(client_fd, &request, sizeof(request), 0) != sizeof(request)) {
            return;
        }

        IpcResponse response;
        memset(&response, 0, sizeof(response));

        // Handle different request types
        switch (request.type) {
            case IpcMessageType::ADD_DEVICE:
                response.status = handle_add_device(request, &response.device);
                break;
            case IpcMessageType::REMOVE_DEVICE:
                response.status = handle_remove_device(request.handle);
                break;
            default:
                response.status = AMDCUID_STATUS_INVALID_ARGUMENT;
                break;
        }

        send(client_fd, &response, sizeof(response), 0);
    }

    amdcuid_status_t handle_add_device(
        const IpcRequest& request,
        DevicePtr* out_device
    ) {
        // Create device based on type and path
        CuidFileEntry entry = {};
        entry.device_type = request.device_type;

        amdcuid_status_t status = AMDCUID_STATUS_SUCCESS;
        amdcuid_primary_id primary_id;
        amdcuid_secondary_id secondary_id;

        switch (request.device_type) {
            case AMDCUID_DEVICE_TYPE_GPU: {
                amdcuid_gpu_info gpu_info = {};
                status = CuidGpu::discover_single(&gpu_info, request.device_path);
                if (status != AMDCUID_STATUS_SUCCESS) {
                    return status;
                }
                auto gpu_device = std::make_shared<CuidGpu>(gpu_info);
                *out_device = gpu_device;
                // convert gpu_device to CuidFileEntry and add to files using update_device
                status = gpu_device->get_primary_cuid(primary_id);
                entry.primary_cuid = primary_id.UUIDv8_representation;
                status = gpu_device->get_secondary_cuid(secondary_id, &daemon_hmac);
                entry.secondary_cuid = secondary_id.UUIDv8_representation;
                entry.vendor_id = gpu_info.header.fields.gpu.vendor_id;
                entry.device_id = gpu_info.header.fields.gpu.device_id;
                entry.pci_class = gpu_info.header.fields.gpu.pci_class;
                entry.revision_id = gpu_info.header.fields.gpu.revision_id;
                entry.unit_id = gpu_info.header.fields.gpu.unit_id;
                entry.device_node = gpu_info.render_node;
                entry.bdf = gpu_info.bdf;
                break;
            }
            case AMDCUID_DEVICE_TYPE_NIC: {
                amdcuid_nic_info nic_info = {};
                status = CuidNic::discover_single(&nic_info, request.device_path);
                if (status != AMDCUID_STATUS_SUCCESS) {
                    return status;
                }
                auto nic_device = std::make_shared<CuidNic>(nic_info);
                *out_device = nic_device;
                // convert nic_device to CuidFileEntry and add to files using update_device
                status = nic_device->get_primary_cuid(primary_id);
                entry.primary_cuid = primary_id.UUIDv8_representation;
                status = nic_device->get_secondary_cuid(secondary_id, &daemon_hmac);
                entry.secondary_cuid = secondary_id.UUIDv8_representation;
                entry.vendor_id = nic_info.header.fields.nic.vendor_id;
                entry.device_id = nic_info.header.fields.nic.device_id;
                entry.pci_class = nic_info.header.fields.nic.pci_class;
                entry.revision_id = nic_info.header.fields.nic.revision_id;
                entry.bdf = nic_info.bdf;
                entry.device_node = nic_info.network_interface;
                break;
            }
            case AMDCUID_DEVICE_TYPE_CPU: {
                amdcuid_cpu_info cpu_info = {};
                status = CuidCpu::discover_single(&cpu_info, request.device_path);
                if (status != AMDCUID_STATUS_SUCCESS) {
                    return status;
                }
                auto cpu_device = std::make_shared<CuidCpu>(cpu_info);
                *out_device = cpu_device;
                // convert cpu_device to CuidFileEntry and add to files using update_device
                status = cpu_device->get_primary_cuid(primary_id);
                status = cpu_device->get_secondary_cuid(secondary_id, &daemon_hmac);
                entry.primary_cuid = primary_id.UUIDv8_representation;
                entry.secondary_cuid = secondary_id.UUIDv8_representation;
                entry.vendor_id = cpu_info.header.fields.cpu.vendor_id;
                entry.family = cpu_info.header.fields.cpu.family;
                entry.model = cpu_info.header.fields.cpu.model;
                entry.unit_id = cpu_info.header.fields.cpu.unit_id;
                entry.package_core_id = std::to_string(cpu_info.header.fields.cpu.physical_id) + ":" + std::to_string(cpu_info.header.fields.cpu.core);
                break;
            }
            default:
                return AMDCUID_STATUS_UNSUPPORTED;
        }
        entry.last_update = time(nullptr);
        update_device(CuidUtilities::cuid_file, CuidUtilities::priv_cuid_file, &entry);

        return status;
    }

    amdcuid_status_t handle_remove_device(
        const amdcuid_id_t& handle
    ) {
        return remove_device(
            CuidUtilities::cuid_file,
            CuidUtilities::priv_cuid_file,
            &handle
        );
    }
};


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


amdcuid_status_t remove_device(std::string output_file,
                                std::string priv_output_file,
                                const amdcuid_id_t *device) {

    // Load existing CUID files
    CuidFile unpriv_file(output_file, false);
    CuidFile priv_file(priv_output_file, true);
    unpriv_file.load();
    priv_file.load();

    log_out() << "Attempting removal of device with secondary CUID: " << CuidUtilities::get_cuid_as_string(device) << std::endl;

    amdcuid_status_t status;
    // Remove entry from both files
    status = unpriv_file.remove_entry(*device);
    if (status != AMDCUID_STATUS_SUCCESS) {
        log_err() << "Error removing device from unprivileged file: " << amdcuid_status_to_string(status) << std::endl;
        return status;
    }
    status = priv_file.remove_entry(*device);
    if (status != AMDCUID_STATUS_SUCCESS) {
        log_err() << "Error removing device from privileged file: " << amdcuid_status_to_string(status) << std::endl;
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

    log_out() << "Attempting update of device with secondary CUID: " << CuidUtilities::get_cuid_as_string(&device->secondary_cuid) << std::endl;

    amdcuid_status_t status;
    // Remove entry from both files
    status = unpriv_file.add_entry(*device);
    if (status != AMDCUID_STATUS_SUCCESS) {
        log_err() << "Error updating device in unprivileged file: " << amdcuid_status_to_string(status) << std::endl;
        return status;
    }
    status = priv_file.add_entry(*device);
    if (status != AMDCUID_STATUS_SUCCESS) {
        log_err() << "Error updating device in privileged file: " << amdcuid_status_to_string(status) << std::endl;
        return status;
    }


    // Save updated CUID files
    unpriv_file.save();
    priv_file.save();

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t get_device_from_udev(std::string *action_output, CuidFileEntry *device_entry, cuid_hmac* hmac) {
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
        amdcuid_status_t status = CuidGpu::discover_single(&info, syspath + "/device");
        auto gpu_device = std::make_shared<CuidGpu>(info);

        entry.device_type = AMDCUID_DEVICE_TYPE_GPU;
        amdcuid_primary_id primary_id;
        status = gpu_device->get_primary_cuid(primary_id);
        if (status != AMDCUID_STATUS_SUCCESS) {
            log_err() << "Error: Failed to get primary CUID for GPU device" << std::endl;
            return status;
        }
        entry.primary_cuid = primary_id.UUIDv8_representation;
        amdcuid_secondary_id secondary_id;
        status = gpu_device->get_secondary_cuid(secondary_id, hmac);
        if (status != AMDCUID_STATUS_SUCCESS) {
            log_err() << "Error: Failed to generate secondary CUID for GPU device" << std::endl;
            return status;
        }
        log_out() << "Generated secondary CUID for GPU device: " << CuidUtilities::get_cuid_as_string(&secondary_id.UUIDv8_representation) << std::endl;
        entry.secondary_cuid = secondary_id.UUIDv8_representation;
        entry.device_node = info.render_node;
        entry.bdf = info.bdf;
        entry.device_index = 0; // could be set based on existing entries
        entry.last_update = time(nullptr);
    } else if (subsys_str == "net") {
        // NIC device
        amdcuid_nic_info info = {};
        amdcuid_status_t status = CuidNic::discover_single(&info, syspath + "/device");
        auto nic_device = std::make_shared<CuidNic>(info);

        entry.device_type = AMDCUID_DEVICE_TYPE_NIC;
        amdcuid_primary_id primary_id;
        status = nic_device->get_primary_cuid(primary_id);
        if (status != AMDCUID_STATUS_SUCCESS) {
            log_err() << "Error: Failed to get primary CUID for NIC device" << std::endl;
            return status;
        }
        entry.primary_cuid = primary_id.UUIDv8_representation;
        amdcuid_secondary_id secondary_id;
        status = nic_device->get_secondary_cuid(secondary_id, hmac);
        if (status != AMDCUID_STATUS_SUCCESS) {
            log_err() << "Error: Failed to generate secondary CUID for NIC device" << std::endl;
            return status;
        }
        entry.secondary_cuid = secondary_id.UUIDv8_representation;
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

    bool logging_enabled = (config_lines[1] == "true");
    std::string output_file = CuidUtilities::cuid_file;
    std::string priv_output_file = CuidUtilities::priv_cuid_file;

    // Initialize file logging if enabled
    init_logging(logging_enabled);
    log_out() << "AMD CUID Daemon initialized with logging " << (logging_enabled ? "enabled" : "disabled") << std::endl;

    if (config_lines[0] == "true") {
        // in daemon mode, we expect to be triggered by udev on device add/remove/change

        CuidDaemonServer server;
        amdcuid_status_t status = server.start();
        if (status != AMDCUID_STATUS_SUCCESS) {
            log_err() << "Error: Failed to start daemon server (status: " << amdcuid_status_to_string(status) << ")" << std::endl;
            return 1;
        }
        log_out() << "Daemon server started, listening for device events..." << std::endl;

        // Keep the main thread alive while the server is running
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }

        // On shutdown (not reachable in current code)
        server.stop();

        // if (!daemon_hmac.is_valid()) {
        //     log_err() << "Error: Failed to initialize HMAC" << std::endl;
        //     return 1;
        // }
        // // get udev input in argv, fill out device handle, and then update CUIDs
        // CuidFileEntry device_info;
        // std::string action;
        // amdcuid_status_t status = get_device_from_udev(&action, &device_info, &daemon_hmac);
        // if (status != AMDCUID_STATUS_SUCCESS) {
        //     log_err() << "Error: Failed to get device from udev. status: " << amdcuid_status_to_string(status) << std::endl;
        //     return 1;
        // }
        // // actions we have to worry about: add, remove, change, move
        // if (action == "remove") {
        //     status = remove_device(output_file, priv_output_file, &device_info.secondary_cuid);
        //     if (status != AMDCUID_STATUS_SUCCESS) {
        //         log_err() << "Error: Failed to remove device CUID. status: " << amdcuid_status_to_string(status) << std::endl;
        //         return 1;
        //     }
        // }
        // else if (action == "add" || action == "change" || action == "move") {
        //     status = update_device(output_file, priv_output_file, &device_info);
        //     if (status != AMDCUID_STATUS_SUCCESS) {
        //         log_err() << "Error: Failed to update device CUID. status: " << amdcuid_status_to_string(status) << std::endl;
        //         return 1;
        //     }
        // }
        // else {
        //     log_err() << "Error: Unsupported udev action: " << action << std::endl;
        //     return 1;
        // }
    }
    else {
        // non-daemon mode discovers devices on bootup and updates their CUIDs once
        // discover devices
        std::vector<std::shared_ptr<CuidDevice>> devices;

        // Platform discovery
        std::vector<DevicePtr> platforms;
        amdcuid_status_t status = CuidPlatform::discover(platforms);
        if (status == AMDCUID_STATUS_SUCCESS) {
            devices.insert(devices.end(), platforms.begin(), platforms.end());
        }

        // GPU discovery
        std::vector<DevicePtr> gpus;
        status = CuidGpu::discover(gpus);
        if (status == AMDCUID_STATUS_SUCCESS) {
            devices.insert(devices.end(), gpus.begin(), gpus.end());
        }

         // CPU discovery
        std::vector<DevicePtr> cpus;
        status = CuidCpu::discover(cpus);
        if (status == AMDCUID_STATUS_SUCCESS) {
            devices.insert(devices.end(), cpus.begin(), cpus.end());
        }

        // NIC discovery
        std::vector<DevicePtr> nics;
        status = CuidNic::discover(nics);
        if (status == AMDCUID_STATUS_SUCCESS) {
            devices.insert(devices.end(), nics.begin(), nics.end());
        }

        if (status != AMDCUID_STATUS_SUCCESS) {
            log_err() << "Error: Failed to initialize device manager (status: " << amdcuid_status_to_string(status) << ")" << std::endl;
            if (status == AMDCUID_STATUS_PERMISSION_DENIED) {
                log_err() << "Some devices may require root privileges to discover." << std::endl;
            }
            return 1;
        }

        log_out() << "Discovered " << devices.size() << " device(s)" << std::endl;

        // Generate CUID files
        status = CuidFileGenerator::generate_from_devices(
            devices,
            daemon_hmac.key_file_path,
            output_file,
            priv_output_file
        );

        if (status != AMDCUID_STATUS_SUCCESS) {
            log_err() << "Error: Failed to generate CUID files (status: " << amdcuid_status_to_string(status) << ")" << std::endl;
            return 1;
        }
    }

    log_out() << "AMD CUID Daemon Exiting..." << std::endl;
    return 0;
}