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
#include "cuid_device.h"
#include "cuid_util.h"
#include "cuid_device_manager.h"
#include "cuid_cpu.h"
#include "cuid_gpu.h"
#include "cuid_nic.h"
#include "cuid_platform.h"
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <mutex>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>

// Static instance for C API
static AmdCuidDeviceManager& mgr = AmdCuidDeviceManager::instance();

amdcuid_status_t amdcuid_get_handles(
    amdcuid_device_type_set_t component_types,
    uint32_t *handle_count,
    amdcuid_handle *handles,
    uint32_t *total_available_handles)
{
    if (!total_available_handles)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    if (!mgr.is_initialized() || mgr.device_types() != component_types)
        mgr.init(component_types);
    const auto &devices = mgr.devices();

    std::vector<amdcuid_handle> temp_handles;
    for (const auto &dev : devices) {
        // TODO: implement device type mapping if needed
        temp_handles.push_back(amdcuid_handle{const_cast<void*>(reinterpret_cast<const void*>(dev.get()))});
    }

    *total_available_handles = temp_handles.size();
    if (!handles || *handle_count == 0) {
        return AMDCUID_STATUS_SUCCESS;
    }

    uint32_t to_copy = std::min(*handle_count, static_cast<uint32_t>(temp_handles.size()));
    for (uint32_t i = 0; i < to_copy; ++i) {
        handles[i] = temp_handles[i];
    }

    return (to_copy < temp_handles.size()) ? AMDCUID_STATUS_INSUFFICIENT_SIZE : AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t amdcuid_get_primary_cuid(amdcuid_handle handle, amdcuid *primary_cuid) {
    if (geteuid() != 0)
        return AMDCUID_STATUS_PERMISSION_DENIED;

    if (!handle.impl || !primary_cuid)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    AmdCuidDevice* dev = mgr.get_device_by_handle<AmdCuidDevice>(handle.impl);
    if (!dev)
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;

    amdcuid id = {};
    amdcuid_status_t status = (amdcuid_status_t)dev->get_primary_cuid(reinterpret_cast<amdcuid&>(id));
    if (status != AMDCUID_STATUS_SUCCESS)
        return AMDCUID_STATUS_UNSUPPORTED;

    std::memcpy(primary_cuid->bytes, id.bytes, 16);

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t amdcuid_get_secondary_cuid(amdcuid_handle handle, amdcuid *secondary_cuid) {
    if (!handle.impl || !secondary_cuid)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    AmdCuidDevice* dev = mgr.get_device_by_handle<AmdCuidDevice>(handle.impl);
    if (!dev)
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;

    amdcuid id = {};
    amdcuid_status_t status = (amdcuid_status_t)dev->get_secondary_cuid(reinterpret_cast<amdcuid&>(id));
    if (status != AMDCUID_STATUS_SUCCESS)
        return AMDCUID_STATUS_UNSUPPORTED;

    std::memcpy(secondary_cuid->bytes, id.bytes, 16);

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t amdcuid_get_device_type(amdcuid_handle handle, amdcuid_device_type_t *dev_type) {
    if (!handle.impl || !dev_type)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    AmdCuidDevice* dev = mgr.get_device_by_handle<AmdCuidDevice>(handle.impl);
    if (!dev)
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;

    *dev_type = dev->type();

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t amdcuid_get_vendor_id(amdcuid_handle handle, uint16_t *vendor_id) {
    if (!handle.impl || !vendor_id)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    AmdCuidDevice* dev = mgr.get_device_by_handle<AmdCuidDevice>(handle.impl);
    if (!dev)
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;

    amdcuid_status_t status = dev->get_vendor_id(*vendor_id);
    if (status != AMDCUID_STATUS_SUCCESS)
        *vendor_id = 0;
    return status;
}

amdcuid_status_t amdcuid_get_revision_id(amdcuid_handle handle, uint16_t *revision_id) {
    if (!handle.impl || !revision_id)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    AmdCuidDevice* dev = mgr.get_device_by_handle<AmdCuidDevice>(handle.impl);
    if (!dev)
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;

    uint8_t rev_id = 0;
    amdcuid_status_t status = dev->get_revision_id(rev_id);
    *revision_id = rev_id;
    return status;
}

amdcuid_status_t amdcuid_get_partition_info(amdcuid_handle handle, uint32_t *partition_info) {
    if (!handle.impl || !partition_info)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    AmdCuidDevice* dev = mgr.get_device_by_handle<AmdCuidDevice>(handle.impl);
    if (!dev)
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;

    amdcuid_device_type_t type = dev->type();
    switch (type) {
        case AMDCUID_DEVICE_TYPE_GPU:
            *partition_info = static_cast<AmdCuidGpu*>(dev)->get_info().header.fields.gpu.unit_id;
            return AMDCUID_STATUS_SUCCESS;
        default:
            *partition_info = 0;
            return AMDCUID_STATUS_UNSUPPORTED;
    }
}

amdcuid_status_t amdcuid_get_bdf(amdcuid_handle handle, char *bdf, uint32_t *length) {
    if (!handle.impl || !bdf || !length)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    AmdCuidDevice* dev = mgr.get_device_by_handle<AmdCuidDevice>(handle.impl);
    if (!dev)
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;

    std::string bdf_str;
    amdcuid_status_t status = dev->get_bdf(bdf_str);
    if (status != AMDCUID_STATUS_SUCCESS) {
        *length = 0;
        bdf[0] = '\0';
        return status;
    }

    uint32_t bdf_len = bdf_str.length();
    if (*length < bdf_len + 1) {
        *length = bdf_len + 1;
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
    }
    std::strcpy(bdf, bdf_str.c_str());
    *length = bdf_len;
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t amdcuid_get_render_node(amdcuid_handle handle, char *render_node, uint32_t *length) {
    if (!handle.impl || !render_node || !length)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    AmdCuidDevice* dev = mgr.get_device_by_handle<AmdCuidDevice>(handle.impl);
    if (!dev)
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;

    amdcuid_device_type_t type = dev->type();
    switch (type) {
        case AMDCUID_DEVICE_TYPE_GPU: {
            const amdcuid_gpu_info& info = static_cast<AmdCuidGpu*>(dev)->get_info();
            if (info.render_node.empty()) {
                *length = 0;
                render_node[0] = '\0';
                return AMDCUID_STATUS_UNSUPPORTED;
            }
            uint32_t rn_len = info.render_node.length();
            if (*length < rn_len + 1) {
                *length = rn_len + 1;
                return AMDCUID_STATUS_INSUFFICIENT_SIZE;
            }
            std::strcpy(render_node, info.render_node.c_str());
            *length = rn_len;
            return AMDCUID_STATUS_SUCCESS;
        }
        default:
            *length = 0;
            render_node[0] = '\0';
            return AMDCUID_STATUS_UNSUPPORTED;
    }
}

amdcuid_status_t amdcuid_get_cpu_core(amdcuid_handle handle, uint16_t *core) {
    if (!handle.impl || !core)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    AmdCuidDevice* dev = mgr.get_device_by_handle<AmdCuidDevice>(handle.impl);
    if (!dev)
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;

    amdcuid_device_type_t type = dev->type();
    switch (type) {
        case AMDCUID_DEVICE_TYPE_CPU:
            *core = static_cast<AmdCuidCpu*>(dev)->get_info().header.fields.cpu.core;
            return AMDCUID_STATUS_SUCCESS;
        default:
            *core = 0;
            return AMDCUID_STATUS_UNSUPPORTED;
    }
}

amdcuid_status_t amdcuid_get_network_interface(amdcuid_handle handle, char *network_interface, uint32_t *length) {
    if (!handle.impl || !network_interface || !length)
        return AMDCUID_STATUS_INVALID_ARGUMENT;

    AmdCuidDevice* dev = mgr.get_device_by_handle<AmdCuidDevice>(handle.impl);
    if (!dev)
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;

    amdcuid_device_type_t type = dev->type();
    switch (type) {
        case AMDCUID_DEVICE_TYPE_NIC: {
            const amdcuid_nic_info& info = static_cast<AmdCuidNic*>(dev)->get_info();
            if (info.network_interface.empty()) {
                *length = 0;
                network_interface[0] = '\0';
                return AMDCUID_STATUS_UNSUPPORTED;
            }
            uint32_t rn_len = info.network_interface.length();
            if (*length < rn_len + 1) {
                *length = rn_len + 1;
                return AMDCUID_STATUS_INSUFFICIENT_SIZE;
            }
            std::strcpy(network_interface, info.network_interface.c_str());
            *length = rn_len;
            return AMDCUID_STATUS_SUCCESS;
        }
        default:
            *length = 0;
            network_interface[0] = '\0';
            return AMDCUID_STATUS_UNSUPPORTED;
    }
}

