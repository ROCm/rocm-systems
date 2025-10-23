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
#include <openssl/sha.h>

// Static instance for C API
static CuidDeviceManager& mgr = CuidDeviceManager::instance();

amdcuid_status_t amdcuid_init(amdcuid_device_type_set_t device_types) {
    return mgr.init(device_types);
}

amdcuid_status_t amdcuid_shutdown() {
    return mgr.shutdown();
}

amdcuid_status_t amdcuid_get_handles(
    amdcuid_device_type_set_t component_types,
    uint32_t handle_count,
    amdcuid_handle *handles,
    uint32_t *total_available_handles)
{
    if (!total_available_handles) return AMDCUID_STATUS_INVALID_ARGUMENT;
    if (!mgr.is_initialized()) mgr.init(component_types);
    const auto &devices = mgr.devices();
    std::vector<amdcuid_handle> temp_handles;
    for (const auto &dev : devices) {
        // TODO: implement device type mapping if needed
        temp_handles.push_back(amdcuid_handle{const_cast<void*>(reinterpret_cast<const void*>(dev.get()))});
    }
    *total_available_handles = temp_handles.size();
    if (!handles || handle_count == 0) {
        return AMDCUID_STATUS_SUCCESS;
    }
    uint32_t to_copy = std::min(handle_count, static_cast<uint32_t>(temp_handles.size()));
    for (uint32_t i = 0; i < to_copy; ++i) {
        handles[i] = temp_handles[i];
    }
    return (to_copy < temp_handles.size()) ? AMDCUID_STATUS_INSUFFICIENT_SIZE : AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t amdcuid_get_primary_cuid(amdcuid_handle handle, amdcuid *primary_cuid) {
    if (!handle.impl || !primary_cuid) return AMDCUID_STATUS_INVALID_ARGUMENT;
    CuidDevice* dev = mgr.get_device_by_handle<CuidDevice>(handle.impl);
    if (!dev) return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    amdcuid id = {};
    amdcuid_status_t status = (amdcuid_status_t)dev->get_primary_cuid(reinterpret_cast<amdcuid&>(id));
    if (status != AMDCUID_STATUS_SUCCESS) return AMDCUID_STATUS_UNSUPPORTED;
    std::memcpy(primary_cuid->bytes, id.bytes, 16);
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t amdcuid_get_secondary_cuid(amdcuid_handle handle, amdcuid *secondary_cuid) {
    if (!handle.impl || !secondary_cuid) return AMDCUID_STATUS_INVALID_ARGUMENT;
    CuidDevice* dev = mgr.get_device_by_handle<CuidDevice>(handle.impl);
    if (!dev) return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    amdcuid id = {};
    amdcuid_salt_t salt = {};
    amdcuid_status_t status = (amdcuid_status_t)dev->get_secondary_cuid(reinterpret_cast<amdcuid_salt_t&>(salt), reinterpret_cast<amdcuid&>(id));
    if (status != AMDCUID_STATUS_SUCCESS) return AMDCUID_STATUS_UNSUPPORTED;
    std::memcpy(secondary_cuid->bytes, id.bytes, 16);
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t amdcuid_get_device_type(amdcuid_handle handle, amdcuid_device_type_t *dev_type) {
    if (!handle.impl || !dev_type) return AMDCUID_STATUS_INVALID_ARGUMENT;
    CuidDevice* dev = static_cast<CuidDevice*>(handle.impl);
    if (!dev) return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    *dev_type = dev->type();
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t amdcuid_get_vendor_id(amdcuid_handle handle, uint16_t *vendor_id) {
    if (!handle.impl || !vendor_id) return AMDCUID_STATUS_INVALID_ARGUMENT;
    CuidDevice* dev = static_cast<CuidDevice*>(handle.impl);
    if (!dev) return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    amdcuid_device_type_t type = dev->type();
    switch (type) {
        case AMDCUID_DEVICE_TYPE_CPU:
            *vendor_id = static_cast<CuidCpu*>(dev)->get_info().header.vid;
            return AMDCUID_STATUS_SUCCESS;
        case AMDCUID_DEVICE_TYPE_GPU:
            *vendor_id = static_cast<CuidGpu*>(dev)->get_info().header.vid;
            return AMDCUID_STATUS_SUCCESS;
        default:
            *vendor_id = 0;
            return AMDCUID_STATUS_UNSUPPORTED;
    }
}

amdcuid_status_t amdcuid_get_revision_id(amdcuid_handle handle, uint16_t *revision_id) {
    if (!handle.impl || !revision_id) return AMDCUID_STATUS_INVALID_ARGUMENT;
    CuidDevice* dev = static_cast<CuidDevice*>(handle.impl);
    if (!dev) return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    amdcuid_device_type_t type = dev->type();
    switch (type) {
        case AMDCUID_DEVICE_TYPE_CPU:
            *revision_id = static_cast<CuidCpu*>(dev)->get_info().header.revision;
            return AMDCUID_STATUS_SUCCESS;
        case AMDCUID_DEVICE_TYPE_GPU:
            *revision_id = static_cast<CuidGpu*>(dev)->get_info().header.revision_id;
            return AMDCUID_STATUS_SUCCESS;
        default:
            *revision_id = 0;
            return AMDCUID_STATUS_UNSUPPORTED;
    }
}

amdcuid_status_t amdcuid_get_partition_info(amdcuid_handle handle, uint32_t *partition_info) {
    if (!handle.impl || !partition_info) return AMDCUID_STATUS_INVALID_ARGUMENT;
    CuidDevice* dev = static_cast<CuidDevice*>(handle.impl);
    if (!dev) return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    amdcuid_device_type_t type = dev->type();
    switch (type) {
        case AMDCUID_DEVICE_TYPE_GPU:
            *partition_info = static_cast<CuidGpu*>(dev)->get_info().header.partition_info;
            return AMDCUID_STATUS_SUCCESS;
        default:
            *partition_info = 0;
            return AMDCUID_STATUS_UNSUPPORTED;
    }
}

amdcuid_status_t amdcuid_get_bdf(amdcuid_handle handle, char *bdf, uint32_t *length) {
    if (!handle.impl || !bdf || !length) return AMDCUID_STATUS_INVALID_ARGUMENT;
    CuidDevice* dev = static_cast<CuidDevice*>(handle.impl);
    if (!dev) return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    amdcuid_device_type_t type = dev->type();
    switch (type) {
        case AMDCUID_DEVICE_TYPE_GPU: {
            const amdcuid_gpu_info& info = static_cast<CuidGpu*>(dev)->get_info();
            if (info.bdf.empty()) {
                *length = 0;
                bdf[0] = '\0';
                return AMDCUID_STATUS_UNSUPPORTED;
            }
            uint32_t bdf_len = info.bdf.length();
            if (*length < bdf_len + 1) {
                *length = bdf_len + 1;
                return AMDCUID_STATUS_INSUFFICIENT_SIZE;
            }
            std::strcpy(bdf, info.bdf.c_str());
            *length = bdf_len;
            return AMDCUID_STATUS_SUCCESS;
        }
        default:
            *length = 0;
            bdf[0] = '\0';
            return AMDCUID_STATUS_UNSUPPORTED;
    }
}

amdcuid_status_t amdcuid_get_render_node(amdcuid_handle handle, char *render_node, uint32_t *length) {
    if (!handle.impl || !render_node || !length) return AMDCUID_STATUS_INVALID_ARGUMENT;
    CuidDevice* dev = static_cast<CuidDevice*>(handle.impl);
    if (!dev) return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    amdcuid_device_type_t type = dev->type();
    switch (type) {
        case AMDCUID_DEVICE_TYPE_GPU: {
            const amdcuid_gpu_info& info = static_cast<CuidGpu*>(dev)->get_info();
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
    if (!handle.impl || !core) return AMDCUID_STATUS_INVALID_ARGUMENT;
    CuidDevice* dev = static_cast<CuidDevice*>(handle.impl);
    if (!dev) return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    amdcuid_device_type_t type = dev->type();
    switch (type) {
        case AMDCUID_DEVICE_TYPE_CPU:
            // amdcuid_cpu_info does not have 'core' member; return unsupported
            *core = 0;
            return AMDCUID_STATUS_UNSUPPORTED;
        default:
            *core = 0;
            return AMDCUID_STATUS_UNSUPPORTED;
    }
}

amdcuid_status_t amdcuid_get_network_interface(amdcuid_handle handle, char *network_interface, uint32_t *length) {
    if (!handle.impl || !network_interface || !length) return AMDCUID_STATUS_INVALID_ARGUMENT;
    CuidDevice* dev = static_cast<CuidDevice*>(handle.impl);
    if (!dev) return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    amdcuid_device_type_t type = dev->type();
    switch (type) {
        case AMDCUID_DEVICE_TYPE_NIC: {
            // TODO: add
            return AMDCUID_STATUS_UNSUPPORTED;
        }
        default:
            *length = 0;
            network_interface[0] = '\0';
            return AMDCUID_STATUS_UNSUPPORTED;
    }
}


