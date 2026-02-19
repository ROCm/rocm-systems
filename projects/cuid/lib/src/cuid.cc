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
#include "src/cuid_internal.h"
#include "src/cuid_device.h"
#include "src/cuid_util.h"
#include "src/cuid_device_manager.h"
#include "src/cuid_cpu.h"
#include "src/cuid_gpu.h"
#include "src/cuid_nic.h"
#include "src/cuid_platform.h"
#include "src/hmac.h"
#include <cstring>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <mutex>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <fcntl.h>

// Static instance for API
static CuidDeviceManager& mgr = CuidDeviceManager::instance();
static cuid_hmac global_hmac = cuid_hmac();

void amdcuid_get_library_version(uint32_t* major, uint32_t* minor, uint32_t* patch) {
    if (major) *major = AMDCUID_LIB_VERSION_MAJOR;
    if (minor) *minor = AMDCUID_LIB_VERSION_MINOR;
    if (patch) *patch = AMDCUID_LIB_VERSION_PATCH;
}

const char* amdcuid_library_version_to_string() {
    static std::string version_str = std::to_string(AMDCUID_LIB_VERSION_MAJOR) + "." +
                                      std::to_string(AMDCUID_LIB_VERSION_MINOR) + "." +
                                      std::to_string(AMDCUID_LIB_VERSION_PATCH);
    return version_str.c_str();
}

const char* amdcuid_status_to_string(amdcuid_status_t status) {
    switch (status) {
        case AMDCUID_STATUS_SUCCESS: return "SUCCESS";
        case AMDCUID_STATUS_FILE_NOT_FOUND: return "FILE_NOT_FOUND";
        case AMDCUID_STATUS_DEVICE_NOT_FOUND: return "DEVICE_NOT_FOUND";
        case AMDCUID_STATUS_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case AMDCUID_STATUS_PERMISSION_DENIED: return "PERMISSION_DENIED";
        case AMDCUID_STATUS_UNSUPPORTED: return "UNSUPPORTED";
        case AMDCUID_STATUS_WRONG_DEVICE_TYPE: return "WRONG_DEVICE_TYPE";
        case AMDCUID_STATUS_INSUFFICIENT_SIZE: return "INSUFFICIENT_SIZE";
        case AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND: return "HARDWARE_FINGERPRINT_NOT_FOUND";
        case AMDCUID_STATUS_KEY_ERROR: return "KEY_ERROR";
        case AMDCUID_STATUS_HMAC_ERROR: return "HMAC_ERROR";
        case AMDCUID_STATUS_FILE_ERROR: return "FILE_ERROR";
        case AMDCUID_STATUS_INVALID_FORMAT: return "INVALID_FORMAT";
        case AMDCUID_STATUS_PCI_ERROR: return "PCI_ERROR";
        case AMDCUID_STATUS_SMBIOS_ERROR: return "SMBIOS_ERROR";
        case AMDCUID_STATUS_ACPI_ERROR: return "ACPI_ERROR";
        case AMDCUID_STATUS_CPUINFO_ERROR: return "CPUINFO_ERROR";
        case AMDCUID_STATUS_IPC_ERROR: return "IPC_ERROR";
        default: return "UNKNOWN_ERROR";
    }
}

const char* amdcuid_id_to_string(amdcuid_id_t cuid_value) {
    // Use thread_local static buffer to avoid returning dangling pointer from temporary string
    thread_local static char uuid_str[37]; // 36 chars + null terminator
    std::string result = CuidUtilities::get_cuid_as_string(&cuid_value);
    std::strncpy(uuid_str, result.c_str(), sizeof(uuid_str) - 1);
    uuid_str[sizeof(uuid_str) - 1] = '\0';
    return uuid_str;
}

amdcuid_status_t amdcuid_get_all_handles(amdcuid_id_t *handles, uint32_t *count) {
    amdcuid_status_t status;
    // get all the devices on the system first
    if (mgr.devices().empty()) {
        status = mgr.discover_devices();
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }
    }

    auto handle_list = mgr.get_all_handles();
    auto handle_count = static_cast<uint32_t>(handle_list.size());
    if (handle_count == 0) {
        handles = nullptr;
        return AMDCUID_STATUS_UNSUPPORTED;
    }
    if (*count < handle_count) {
        *count = handle_count;
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
    }
    for (uint32_t i = 0; i < handle_count; ++i) {
        std::memcpy(handles[i].bytes, handle_list[i].bytes, 16);
    }
    *count = handle_count;
    return AMDCUID_STATUS_SUCCESS;
}

// helper function to discover device given dev_path
DevicePtr discover_device_by_path(const char* dev_path, amdcuid_device_type_t device_type) {
    DevicePtr device = nullptr;
    amdcuid_status_t status;
    int fd = open(dev_path, O_RDONLY);
    if (fd < 0) {
        // unable to open device path
        return nullptr;
    }

    // find real device path in case of symlink
    std::string real_dev_path = CuidUtilities::real_dev_path_from_fd(fd);
    close(fd);
    if (real_dev_path.empty()) {
        return nullptr;
    }
    switch (device_type) {
        case AMDCUID_DEVICE_TYPE_CPU: {
            amdcuid_cpu_info cpu_info = {};
            status = CuidCpu::discover_single(&cpu_info, real_dev_path);
            if (status != AMDCUID_STATUS_SUCCESS) {
                return nullptr;
            }
            device = std::make_shared<CuidCpu>(cpu_info);
            break;
        }
        case AMDCUID_DEVICE_TYPE_GPU: {
            amdcuid_gpu_info gpu_info = {};
            status = CuidGpu::discover_single(&gpu_info, real_dev_path);
            if (status != AMDCUID_STATUS_SUCCESS) {
                return nullptr;
            }
            device = std::make_shared<CuidGpu>(gpu_info);
            break;
        }
        case AMDCUID_DEVICE_TYPE_NIC: {
            amdcuid_nic_info nic_info = {};
            status = CuidNic::discover_single(&nic_info, real_dev_path);
            if (status != AMDCUID_STATUS_SUCCESS) {
                return nullptr;
            }
            device = std::make_shared<CuidNic>(nic_info);
            break;
        }
        default:
            return nullptr;
    }
    return device;
}

amdcuid_status_t amdcuid_get_handle_by_dev_path(const char* dev_path, amdcuid_device_type_t device_type, amdcuid_id_t* handle) {
    if (!dev_path || !handle) {
        return AMDCUID_STATUS_INVALID_ARGUMENT;
    }

    std::string real_dev_path = CuidUtilities::get_real_path(dev_path);
    if (real_dev_path.empty()) {
         return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    }

    amdcuid_status_t status;
    // check mgr first to see if device is already known
    for (const auto& device : mgr.devices()) {
        std::string device_path;
        status = device->get_device_path(device_path);
        if (status != AMDCUID_STATUS_SUCCESS) {
            continue;
        }
        if (device_path == real_dev_path && device->type() == device_type) {
            amdcuid_derived_id derived;
            status = device->get_derived_cuid(derived);
            if (status != AMDCUID_STATUS_SUCCESS) {
                return status;
            }
            std::memcpy(handle->bytes, &derived.UUIDv8_representation, 16);
            return AMDCUID_STATUS_SUCCESS;
        }
    }

    // next check cuid files for device
    DevicePtr device = nullptr;
    status = mgr.get_device_from_file_by_dev_path(real_dev_path, device);
    if (status == AMDCUID_STATUS_SUCCESS) {
        amdcuid_derived_id derived;
        status = device->get_derived_cuid(derived);
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }
        std::memcpy(handle->bytes, &derived.UUIDv8_representation, 16);
        return AMDCUID_STATUS_SUCCESS;
    }

    // finally, attempt to discover device, this would require elevated permissions since it will require reading of protected hardware info
    if (geteuid() == 0) {
       device = discover_device_by_path(real_dev_path.c_str(), device_type);
    }
    else {
        status = mgr.request_device(real_dev_path.c_str(), device_type, device);
    }

    if (!device) {
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    } else {
        // add device to mgr
        amdcuid_derived_id derived;
        status = device->get_derived_cuid(derived, &global_hmac);
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }
        std::memcpy(handle->bytes, &derived.UUIDv8_representation, 16);
        mgr.add_device(device);
        return AMDCUID_STATUS_SUCCESS;
    }
}

amdcuid_status_t amdcuid_get_handle_by_bdf(const char* bdf, amdcuid_device_type_t device_type, amdcuid_id_t* handle) {
    if (!bdf || !handle) {
        return AMDCUID_STATUS_INVALID_ARGUMENT;
    }

    if (device_type == AMDCUID_DEVICE_TYPE_CPU || device_type == AMDCUID_DEVICE_TYPE_PLATFORM) {
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }

    // check mgr first to see if device is already known
    for (const auto& device : mgr.devices()) {
        std::string device_bdf;
        amdcuid_status_t status = device->get_bdf(device_bdf);
        if (status != AMDCUID_STATUS_SUCCESS) {
            continue;
        }
        if (device_bdf == bdf && device->type() == device_type) {
            amdcuid_derived_id derived;
            status = device->get_derived_cuid(derived);
            if (status != AMDCUID_STATUS_SUCCESS) {
                return status;
            }
            std::memcpy(handle->bytes, &derived.UUIDv8_representation, 16);
            return AMDCUID_STATUS_SUCCESS;
        }
    }

    // next check cuid files for device
    DevicePtr device = nullptr;
    amdcuid_status_t status = mgr.get_device_from_file_by_bdf(bdf, device);
    if (status == AMDCUID_STATUS_SUCCESS) {
        amdcuid_derived_id derived;
        status = device->get_derived_cuid(derived);
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }
        std::memcpy(handle->bytes, &derived.UUIDv8_representation, 16);
        return AMDCUID_STATUS_SUCCESS;
    }

    // finally, attempt to discover device, this would require elevated permissions since it will require reading of protected hardware info
    std::string device_path = CuidUtilities::bdf_to_device_path(bdf, device_type);
    if (device_path.empty()) {
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    }
    std::string real_dev_path = device_path;
    // if device is not a nic, attempt to resolve real device path in case of symlink for more reliable matching
    if (device_type != AMDCUID_DEVICE_TYPE_NIC
        || device_path.find("net") == std::string::npos) {
        int fd = open(device_path.c_str(), O_RDONLY);
        if (fd < 0) {
            return AMDCUID_STATUS_DEVICE_NOT_FOUND;
        }
        real_dev_path = CuidUtilities::real_dev_path_from_fd(fd);
        close(fd);
    }

    if (real_dev_path.empty()) {
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    }

    if (geteuid() == 0) {
        device = discover_device_by_path(real_dev_path.c_str(), device_type);
    }
    else {
        status = mgr.request_device(real_dev_path.c_str(), device_type, device);
    }

    if (!device) {
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    } else {
        // add device to mgr
        amdcuid_derived_id derived;
        status = device->get_derived_cuid(derived, &global_hmac);
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }
        std::memcpy(handle->bytes, &derived.UUIDv8_representation, 16);
        mgr.add_device(device);
        return AMDCUID_STATUS_SUCCESS;
    }
}

amdcuid_status_t amdcuid_get_handle_by_fd(int fd, amdcuid_device_type_t device_type, amdcuid_id_t* handle) {
    if (fd < 0 || !handle) {
        return AMDCUID_STATUS_INVALID_ARGUMENT;
    }

    if (device_type == AMDCUID_DEVICE_TYPE_CPU || device_type == AMDCUID_DEVICE_TYPE_PLATFORM || device_type == AMDCUID_DEVICE_TYPE_NIC) {
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }

    std::string device_path = CuidUtilities::real_dev_path_from_fd(fd);
    if (device_path.empty()) {
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    }
    return amdcuid_get_handle_by_dev_path(device_path.c_str(), device_type, handle);
}

amdcuid_status_t amdcuid_refresh() {
    // discover existing devices on the system and update mgr
    amdcuid_status_t status;
    if (geteuid() == 0)
    {
        status = mgr.get_devices_on_system();
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }
        mgr.build_cuid_index();

        // save updated device list to files
        status = mgr.save_registry_to_files();
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }

        return status;
    }
    else
    {
        status = mgr.request_refresh();
        return status;
    }

}

amdcuid_status_t amdcuid_query_device_property(amdcuid_id_t handle, amdcuid_query_t query, void *data, uint32_t *length) {

    auto device = mgr.lookup_by_handle(handle);
    if (!device) {
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    }
    amdcuid_status_t status;
    switch (query)
    {
        case AMDCUID_QUERY_PRIMARY_CUID: {
                if (geteuid() != 0) {
                    return AMDCUID_STATUS_PERMISSION_DENIED;
                }
                if (*length < sizeof(amdcuid_id_t)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                amdcuid_primary_id id = {};
                status = device->get_primary_cuid(id);
                *(amdcuid_id_t *)data = id.UUIDv8_representation;
                *length = sizeof(amdcuid_id_t);
            }
            break;
        case AMDCUID_QUERY_DERIVED_CUID: {
                if (*length < sizeof(amdcuid_id_t)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                amdcuid_derived_id sec_id = {};
                if (geteuid() == 0) {
                    // if elevated, provide hmac in case it needs to get generated
                    status = device->get_derived_cuid(sec_id, &global_hmac);
                } else {
                    // if not elevated, only get existing derived cuid
                    status = device->get_derived_cuid(sec_id);
                }
                *(amdcuid_id_t *)data = sec_id.UUIDv8_representation;
                *length = sizeof(amdcuid_id_t);
            }
            break;
        case AMDCUID_QUERY_HARDWARE_FINGERPRINT: {
                if (geteuid() != 0) {
                    return AMDCUID_STATUS_PERMISSION_DENIED;
                }
                if (*length < sizeof(uint64_t)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                status = device->get_hardware_fingerprint(*(uint64_t*)data);
                *length = sizeof(uint64_t);
            }
            break;
        case AMDCUID_QUERY_DEVICE_PATH: {
                std::string path;
                status = device->get_device_path(path);
                if (status != AMDCUID_STATUS_SUCCESS) {
                    break;
                }
                uint32_t required_length = static_cast<uint32_t>(path.size() + 1); // include null terminator
                if (*length < required_length) {
                    *length = required_length;
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                std::memcpy(data, path.c_str(), required_length);
                *length = required_length;
            }
            break;
        case AMDCUID_QUERY_DEVICE_TYPE: {
                if (*length < sizeof(amdcuid_device_type_t)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                *(amdcuid_device_type_t*)data = device->type();
                *length = sizeof(amdcuid_device_type_t);
                status = AMDCUID_STATUS_SUCCESS;
            }
            break;
        case AMDCUID_QUERY_VENDOR_ID: {
                if (*length < sizeof(uint16_t)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                status = device->get_vendor_id(*(uint16_t*)data);
                *length = sizeof(uint16_t);
            }
            break;
        case AMDCUID_QUERY_DEVICE_ID: {
                if (*length < sizeof(uint16_t)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                status = device->get_device_id(*(uint16_t*)data);
                *length = sizeof(uint16_t);
            }
            
            break;
        case AMDCUID_QUERY_REVISION_ID: {
                if (*length < sizeof(uint8_t)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                status = device->get_revision_id(*(uint8_t*)data);
                *length = sizeof(uint8_t);
            }
            break;
        case AMDCUID_QUERY_UNIT_ID: {
                if (*length < sizeof(uint16_t)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                status = device->get_unit_id(*(uint16_t*)data);
                *length = sizeof(uint16_t);
            }
            break;
        case AMDCUID_QUERY_FAMILY: {
                // only CPU devices will return a valid family
                if (*length < sizeof(uint16_t)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                status = device->get_family(*(uint16_t*)data);
                *length = sizeof(uint16_t);
            }
            break;
        case AMDCUID_QUERY_MODEL: {
                // only CPU devices will return a valid model
                if (*length < sizeof(uint16_t)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                status = device->get_model(*(uint16_t*)data);
                *length = sizeof(uint16_t);
            }
            break;
        case AMDCUID_QUERY_CORE_ID: {
                // only CPU devices will return a valid core ID
                if (*length < sizeof(uint16_t)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                status = device->get_core(*(uint16_t*)data);
                *length = sizeof(uint16_t);
            }
            break;
        case AMDCUID_QUERY_PHYSICAL_ID: {
                // only CPU devices will return a valid physical package ID
                if (*length < sizeof(uint16_t)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                status = device->get_physical_id(*(uint16_t*)data);
                *length = sizeof(uint16_t);
            }
            break;
        case AMDCUID_QUERY_PCI_CLASS: {
                // only PCI devices (GPU, NIC) will return a valid PCI class
                if (*length < sizeof(uint16_t)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                status = device->get_pci_class(*(uint16_t*)data);
                *length = sizeof(uint16_t);
            }
            break;
        default:
            status = AMDCUID_STATUS_INVALID_ARGUMENT;
            break;
    }
    if (status != AMDCUID_STATUS_SUCCESS) {
        return status;
    }

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t amdcuid_set_hash_key(const uint8_t key[32]) {
    if (geteuid() != 0) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    global_hmac.set_hmac_key(key);

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t amdcuid_generate_hash_key(uint8_t key[32]) {
    if (geteuid() != 0) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }
    if (!key)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    global_hmac.generate_key(key);
    return AMDCUID_STATUS_SUCCESS;
}