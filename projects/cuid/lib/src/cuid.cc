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

amdcuid_status_t amdcuid_add_device(const char* dev_path, amdcuid_device_type_t device_type, amdcuid_id_t* handle) {
    if (!dev_path || !handle)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    if (geteuid() != 0) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    return mgr.add_device(dev_path, device_type, handle);
}

amdcuid_status_t amdcuid_remove_device(amdcuid_id_t handle) {
    if (geteuid() != 0) {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    if (mgr.is_valid_handle(handle) != AMDCUID_STATUS_SUCCESS) {
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    }

    return mgr.remove_device(handle);
}

amdcuid_status_t amdcuid_get_num_handles(uint32_t* count) {
    if (!count)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    amdcuid_status_t status;
    // get all the devices on the system first
    if (mgr.devices().empty()) {
        status = mgr.discover_devices();
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }
    }

    *count = static_cast<uint32_t>(mgr.get_handle_count());
    if (*count == 0) {
        return AMDCUID_STATUS_UNSUPPORTED;
    }
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t amdcuid_get_all_handles(amdcuid_id_t *handles, uint32_t count) {
    auto handle_list = mgr.get_all_handles();
    auto handle_count = static_cast<uint32_t>(handle_list.size());
    if (handle_count == 0) {
        handles = nullptr;
        return AMDCUID_STATUS_UNSUPPORTED;
    }
    if (count < handle_count) {
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
    }
    for (uint32_t i = 0; i < handle_count; ++i) {
        std::memcpy(handles[i].bytes, handle_list[i].bytes, 16);
    }
    return AMDCUID_STATUS_SUCCESS;
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
                if (*length < sizeof(amdcuid_primary_id)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                amdcuid_primary_id id = {};
                status = device->get_primary_cuid(id);
                *(amdcuid_id_t *)data = id.UUIDv8_representation;
                *length = sizeof(amdcuid_id_t);
            }
            break;
        case AMDCUID_QUERY_SECONDARY_CUID: {
                if (*length < sizeof(amdcuid_secondary_id)) {
                    return AMDCUID_STATUS_INSUFFICIENT_SIZE;
                }
                amdcuid_secondary_id sec_id = {};
                status = device->get_secondary_cuid(sec_id);
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