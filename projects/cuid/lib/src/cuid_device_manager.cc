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
#include "src/cuid_device_manager.h"
#include "src/cuid_gpu.h"
#include "src/cuid_cpu.h"
#include "src/cuid_nic.h"
#include "src/cuid_platform.h"
#include "src/cuid_util.h"
#include "src/ipc_protocol.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <iostream>
#include <algorithm>
#include <unistd.h>

class CuidDaemonIpcClientUtils {
public:
    static amdcuid_status_t request_add_device(
        const char* dev_path,
        amdcuid_device_type_t device_type,
        amdcuid_id_t* device_handle
    ) {
        int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (sock_fd < 0) {
            return AMDCUID_STATUS_IPC_ERROR;
        }

        struct sockaddr_un server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sun_family = AF_UNIX;
        strncpy(server_addr.sun_path, AMDCUID_SOCKET_PATH, sizeof(server_addr.sun_path));

        if (connect(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            close(sock_fd);
            return AMDCUID_STATUS_IPC_ERROR;
        }

        IpcRequest request;
        request.type = IpcMessageType::ADD_DEVICE;
        strncpy(request.device_path, dev_path, sizeof(request.device_path));
        request.device_type = device_type;

        if (send(sock_fd, &request, sizeof(request), 0) != sizeof(request)) {
            close(sock_fd);
            return AMDCUID_STATUS_IPC_ERROR;
        }

        IpcResponse response;
        if (recv(sock_fd, &response, sizeof(response), 0) != sizeof(response)) {
            close(sock_fd);
            return AMDCUID_STATUS_IPC_ERROR;
        }

        close(sock_fd);
        if (response.status == AMDCUID_STATUS_SUCCESS && device_handle) {
            *device_handle = response.device_handle;
        }
        return response.status;
    }

    static amdcuid_status_t request_refresh_devices() {
        int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (sock_fd < 0) {
            return AMDCUID_STATUS_IPC_ERROR;
        }

        struct sockaddr_un server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sun_family = AF_UNIX;
        strncpy(server_addr.sun_path, AMDCUID_SOCKET_PATH, sizeof(server_addr.sun_path));

        if (connect(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            close(sock_fd);
            return AMDCUID_STATUS_IPC_ERROR;
        }

        IpcRequest request;
        request.type = IpcMessageType::REFRESH_DEVICES;
        memset(request.device_path, 0, sizeof(request.device_path)); // Not used for REFRESH_DEVICES
        request.device_type = AMDCUID_DEVICE_TYPE_NONE; // Not used for REFRESH_DEVICES

        if (send(sock_fd, &request, sizeof(request), 0) != sizeof(request)) {
            close(sock_fd);
            return AMDCUID_STATUS_IPC_ERROR;
        }

        IpcResponse response;
        if (recv(sock_fd, &response, sizeof(response), 0) != sizeof(response)) {
            close(sock_fd);
            return AMDCUID_STATUS_IPC_ERROR;
        }

        close(sock_fd);
        return response.status;
    }

    static bool is_daemon_running() {
        int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (sock_fd < 0) {
            return false;
        }

        struct sockaddr_un server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sun_family = AF_UNIX;
        strncpy(server_addr.sun_path, AMDCUID_SOCKET_PATH, sizeof(server_addr.sun_path));

        if (connect(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            close(sock_fd);
            return false;
        }

        close(sock_fd);
        return true;
    }
};

amdcuid_status_t CuidDeviceManager::get_devices_on_system() {

    amdcuid_status_t status;
    std::vector<DevicePtr> discovered_devices;

    // discover devices on the system
    // Discover platform device
    std::vector<DevicePtr> platform_devices;
    status = CuidPlatform::discover(platform_devices);
    if (status == AMDCUID_STATUS_SUCCESS) {
        for (const auto& platform_device : platform_devices) {
            discovered_devices.push_back(platform_device);
        }
    }
    // Discover CPU devices
    std::vector<DevicePtr> cpu_devices;
    status = CuidCpu::discover(cpu_devices);
    if (status == AMDCUID_STATUS_SUCCESS) {
        for (const auto& cpu : cpu_devices) {
            discovered_devices.push_back(cpu);
        }
    }
    // Discover GPU devices
    std::vector<DevicePtr> gpu_devices;
    status = CuidGpu::discover(gpu_devices);
    if (status == AMDCUID_STATUS_SUCCESS) {
        for (const auto& gpu : gpu_devices) {
            discovered_devices.push_back(gpu);
        }
    }
    // Discover NIC devices
    std::vector<DevicePtr> nic_devices;
    status = CuidNic::discover(nic_devices);
    if (status == AMDCUID_STATUS_SUCCESS) {
        for (const auto& nic : nic_devices) {
            discovered_devices.push_back(nic);
        }
    }

    if (status == AMDCUID_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(manager_mutex_);
        devices_.clear();
        devices_ = discovered_devices;
    }

    return status;
}

// helper function to convert CuidFileEntry to appropriate CuidDevice
void _convert_entry_to_device(CuidFileEntry& entry, DevicePtr& device){

    switch (entry.device_type) {
        case AMDCUID_DEVICE_TYPE_PLATFORM: {
            amdcuid_platform_info platform_info = {};
            platform_info.header.fields.platform.vendor_id = entry.vendor_id;
            device = std::make_shared<CuidPlatform>(platform_info);
            break;
        }
        case AMDCUID_DEVICE_TYPE_GPU: {
            amdcuid_gpu_info gpu_info = {};
            gpu_info.header.fields.gpu.vendor_id = entry.vendor_id;
            gpu_info.header.fields.gpu.device_id = entry.device_id;
            gpu_info.header.fields.gpu.pci_class = entry.pci_class;
            gpu_info.header.fields.gpu.revision_id = entry.revision_id;
            gpu_info.header.fields.gpu.unit_id = entry.unit_id;
            gpu_info.render_node = entry.device_node;
            gpu_info.bdf = entry.bdf;
            device = std::make_shared<CuidGpu>(gpu_info);
            break;
        }
        case AMDCUID_DEVICE_TYPE_CPU: {
            amdcuid_cpu_info cpu_info = {};
            cpu_info.header.fields.cpu.vendor_id = entry.vendor_id;
            cpu_info.header.fields.cpu.family = entry.family;
            cpu_info.header.fields.cpu.model = entry.model;
            cpu_info.header.fields.cpu.device_id = entry.device_id;
            cpu_info.header.fields.cpu.revision_id = entry.revision_id;
            cpu_info.header.fields.cpu.unit_id = entry.unit_id;

            // Split package_core_id by colon into package and core
            uint16_t package = 0;
            uint16_t core = 0;
            size_t colon_pos = entry.package_core_id.find(':');
            if (colon_pos != std::string::npos) {
                package = static_cast<uint16_t>(std::stoul(entry.package_core_id.substr(0, colon_pos)));
                core = static_cast<uint16_t>(std::stoul(entry.package_core_id.substr(colon_pos + 1)));
            }

            cpu_info.header.fields.cpu.physical_id = package;
            cpu_info.header.fields.cpu.core = core;
            device = std::make_shared<CuidCpu>(cpu_info);
            break;
        }
        case AMDCUID_DEVICE_TYPE_NIC: {
            amdcuid_nic_info nic_info = {};
            nic_info.header.fields.nic.vendor_id = entry.vendor_id;
            nic_info.header.fields.nic.device_id = entry.device_id;
            nic_info.header.fields.nic.pci_class = entry.pci_class;
            nic_info.header.fields.nic.revision_id = entry.revision_id;
            nic_info.network_interface = entry.device_node;
            nic_info.bdf = entry.bdf;
            device = std::make_shared<CuidNic>(nic_info);
            break;
        }
        // Add cases for other device types as needed
        default: {
            device = nullptr;
            break;
        }
    }
}

amdcuid_status_t CuidDeviceManager::get_devices_from_file_entries(CuidFile& cuid_file) {
    std::lock_guard<std::mutex> lock(manager_mutex_);

    amdcuid_status_t status = cuid_file.load();
    if (status != AMDCUID_STATUS_SUCCESS) {
        return status;
    }

    devices_.clear();
    for (const auto& entry : cuid_file.get_entries()) {
        DevicePtr device = nullptr;
        _convert_entry_to_device(const_cast<CuidFileEntry&>(entry), device);
        if (device) {
            devices_.push_back(device);
        }
    }

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidDeviceManager::add_device(DevicePtr device) {
    std::lock_guard<std::mutex> lock(manager_mutex_);

    if (device) {
        // Update the CUID index
        amdcuid_derived_id derived;
        if (device->get_derived_cuid(derived) == AMDCUID_STATUS_SUCCESS) {
            cuid_index_[derived.UUIDv8_representation] = device;
        }

        devices_.push_back(device);
    }
    else {
        return AMDCUID_STATUS_INVALID_ARGUMENT;
    }

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidDeviceManager::get_device_from_file_by_id(amdcuid_id_t& derived_cuid, DevicePtr& device) {
    amdcuid_status_t status = AMDCUID_STATUS_DEVICE_NOT_FOUND;

    // Search in privileged CUID file first
    CuidFileEntry entry;
    if (geteuid() == 0) {
        status = priv_cuid_file_.find_by_derived_cuid(derived_cuid, entry);
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }
    } else {
        status = unpriv_cuid_file_.find_by_derived_cuid(derived_cuid, entry);
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }
    }

    // Create device based on the found entry
    _convert_entry_to_device(entry, device);
    if (device) {
        add_device(device);
    }
    return status;
}

amdcuid_status_t CuidDeviceManager::get_device_from_file_by_dev_path(const std::string& device_path, DevicePtr& device) {
    amdcuid_status_t status = AMDCUID_STATUS_DEVICE_NOT_FOUND;

    // Search in privileged CUID file first
    CuidFileEntry entry;
    if (geteuid() == 0) {
        status = priv_cuid_file_.find_by_device_node(device_path, entry);
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status; // Not found in either file
        }
    } else {
        status = unpriv_cuid_file_.find_by_device_node(device_path, entry);
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status; // Not found in unprivileged file
        }
    }

    // Create device based on the found entry
    _convert_entry_to_device(entry, device);
    if (device) {
        add_device(device);
    }
    return status;
}

amdcuid_status_t CuidDeviceManager::get_device_from_file_by_bdf(const std::string& bdf, DevicePtr& device) {
    amdcuid_status_t status = AMDCUID_STATUS_DEVICE_NOT_FOUND;

    // Search in privileged CUID file first
    CuidFileEntry entry;
    if (geteuid() == 0) {
        status = priv_cuid_file_.find_by_bdf(bdf, entry);
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status; // Not found in privileged file
        }
    } else {
        status = unpriv_cuid_file_.find_by_bdf(bdf, entry);
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status; // Not found in unprivileged file
        }
    }

    // Create device based on the found entry
    _convert_entry_to_device(entry, device);
    if (device) {
        add_device(device);
    }
    return status;
}

amdcuid_status_t CuidDeviceManager::request_device(const std::string& device_path, amdcuid_device_type_t device_type, DevicePtr& device) {
    amdcuid_status_t status;
    if (CuidDaemonIpcClientUtils::is_daemon_running()) {
        amdcuid_id_t device_handle;
        status = CuidDaemonIpcClientUtils::request_add_device(device_path.c_str(), device_type, &device_handle);
        if (status == AMDCUID_STATUS_SUCCESS && device_handle.bytes != nullptr) {
            // Lookup device by handle and return it
            status = get_device_from_file_by_id(device_handle, device);
            if (status != AMDCUID_STATUS_SUCCESS) {
                return status;
            }
            add_device(device);
        }
        return status;
    }
    else {
        status = AMDCUID_STATUS_IPC_ERROR;
    }

    return status;
}

amdcuid_status_t CuidDeviceManager::request_refresh() {
    amdcuid_status_t status;
    if (CuidDaemonIpcClientUtils::is_daemon_running()) {
        status = CuidDaemonIpcClientUtils::request_refresh_devices();
        return status;
    }
    else {
        status = AMDCUID_STATUS_IPC_ERROR;
    }

    return status;
}

amdcuid_status_t CuidDeviceManager::discover_devices() {
    amdcuid_status_t status;
    // search for devices in the appropriate CUID file first
    if (geteuid() == 0) {
        status = get_devices_from_file_entries(priv_cuid_file_);
        // if there are no devices found, search the system for devices
        if (devices_.empty() || status != AMDCUID_STATUS_SUCCESS) {
            status = get_devices_on_system();
            if (status != AMDCUID_STATUS_SUCCESS) {
                return status;
            }
        }
    }
    else {
        status = get_devices_from_file_entries(unpriv_cuid_file_);
        if (status != AMDCUID_STATUS_SUCCESS || devices_.empty()) {
            // refresh to get devices from system since none found
            status = request_refresh();
            return status;
        }
    }

    // if devices still empty, return error
    if (devices_.empty()) {
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    }

    // Build the CUID index for lookups
    build_cuid_index();

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidDeviceManager::shutdown() {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    devices_.clear();
    cuid_index_.clear();
    return AMDCUID_STATUS_SUCCESS;
}

CuidDeviceManager& CuidDeviceManager::instance() {
    static CuidDeviceManager instance;
    return instance;
}

void CuidDeviceManager::get_grouped_devices(std::map<amdcuid_device_type_t, std::vector<DevicePtr>>& grouped) {
    grouped.clear();
    for (const auto& entry : devices_) {
        grouped[entry->type()].push_back(entry);
    }
}

void CuidDeviceManager::build_cuid_index() {
    std::lock_guard<std::mutex> lock(manager_mutex_);

    cuid_index_.clear();
    for (const auto& device : devices_) {
        amdcuid_derived_id derived;
        if (device->get_derived_cuid(derived, &manager_hmac) == AMDCUID_STATUS_SUCCESS) {
            cuid_index_[derived.UUIDv8_representation] = device;
        }
    }
}

DevicePtr CuidDeviceManager::lookup_by_handle(const amdcuid_id_t& handle) const {
    // Since amdcuid_id_t is our handle, we can use it directly as key
    auto it = cuid_index_.find(handle);
    return (it != cuid_index_.end()) ? it->second : nullptr;
}

std::vector<amdcuid_id_t> CuidDeviceManager::get_all_handles() const {
    std::vector<amdcuid_id_t> handles;
    handles.reserve(cuid_index_.size());

    for (const auto& pair : cuid_index_) {
        // amdcuid_id_t is the handle, so just copy directly
        handles.push_back(pair.first);
    }
    return handles;
}

amdcuid_status_t CuidDeviceManager::save_registry_to_files() {
    if (geteuid() != 0) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    std::lock_guard<std::mutex> lock(manager_mutex_);

    // Generate new CUID files from current device list
    amdcuid_status_t status = CuidFileGenerator::generate_from_devices(
        devices_,
        manager_hmac.key_file_path,
        CuidUtilities::cuid_file(),
        CuidUtilities::priv_cuid_file()
    );
    if (status != AMDCUID_STATUS_SUCCESS) {

        return status;
    }

    // ensure new files generated are reloaded into the file objects
    unpriv_cuid_file_.load();
    priv_cuid_file_.load();

    return status;
}