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

#include "cuid_gpu.h"
#include "cuid_util.h"
#include "pci_util.h"
#include "cuid_file.h"
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>

CuidGpu::CuidGpu(const amdcuid_gpu_info& i)
    : m_info(i)
{}

amdcuid_status_t CuidGpu::discover(std::vector<DevicePtr> &gpus) {
    const char *drm_path = "/sys/class/drm";
    DIR *dir = opendir(drm_path);
    if (!dir) return AMDCUID_STATUS_UNSUPPORTED;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "renderD", 7) == 0 && isdigit(entry->d_name[7])) {
            std::string render_name(entry->d_name);
            std::string device_path = std::string(drm_path) + "/" + render_name + "/device";
            amdcuid_gpu_info info = {};
            amdcuid_status_t status = discover_single(&info, device_path);

            gpus.emplace_back(std::make_shared<CuidGpu>(info));
        }
    }
    closedir(dir);
    return AMDCUID_STATUS_SUCCESS;
}

std::string get_parent_device_path(const std::string& device_path, uint16_t unit_id) {
    if (unit_id == 0) return device_path;

    size_t render_pos = device_path.find("renderD");
    if (render_pos != std::string::npos) {
        size_t num_start = render_pos + 7; // length of "renderD"
        size_t num_end = device_path.find('/', num_start);
        if (num_end != std::string::npos) {
            std::string render_num_str = device_path.substr(num_start, num_end - num_start);
            int render_num = std::stoi(render_num_str);
            int parent_render_num = render_num - unit_id;
            std::string parent_device_path = device_path.substr(0, render_pos) + "renderD" +
                                             std::to_string(parent_render_num) +
                                             device_path.substr(num_end);
            return parent_device_path;
        }
    }
    return device_path;
}

amdcuid_status_t CuidGpu::discover_single(amdcuid_gpu_info *gpu_info, const std::string& device_path) {

    std::string device_path_in_use = device_path;
    amdcuid_gpu_info info = {};
    std::string bdf = CuidUtilities::readlink_bdf(device_path);
    info.header.fields.gpu.unit_id = 0;

    // If BDF is empty, this might be a partition device - try to get BDF from parent
    // and determine unit_id from the renderD number difference
    if (bdf.empty()) {
        // Extract renderD number from device_path
        size_t render_pos = device_path.find("renderD");
        if (render_pos != std::string::npos) {
            size_t num_start = render_pos + 7;
            size_t num_end = device_path.find('/', num_start);
            if (num_end != std::string::npos) {
                int render_num = std::stoi(device_path.substr(num_start, num_end - num_start));
                // Try parent renderD nodes (up to 7 partitions possible based on function numbers 1-7)
                for (int offset = 1; offset <= 7; ++offset) {
                    std::string parent_path = device_path.substr(0, render_pos) + "renderD" +
                                              std::to_string(render_num - offset) +
                                              device_path.substr(num_end);
                    std::string parent_bdf = CuidUtilities::readlink_bdf(parent_path);
                    if (!parent_bdf.empty()) {
                        // Found parent - unit_id is the offset, construct partition BDF
                        info.header.fields.gpu.unit_id = static_cast<uint16_t>(offset);
                        // Replace the function number in BDF with the unit_id
                        size_t dot_pos = parent_bdf.rfind('.');
                        if (dot_pos != std::string::npos) {
                            bdf = parent_bdf.substr(0, dot_pos + 1) + std::to_string(offset);
                        }
                        device_path_in_use = parent_path;
                        break;
                    }
                }
            }
        }
    }
    // if bdf ends in non zero digit, then we have a partitioned device and we will use that as unit id
    else {
        size_t last_dot = bdf.rfind('.');
        if (last_dot != std::string::npos && last_dot + 1 < bdf.size()) {
            char last_char = bdf[last_dot + 1];
            if (isdigit(last_char) && last_char != '0') {
                uint16_t unit_id = static_cast<uint16_t>(last_char - '0');
                info.header.fields.gpu.unit_id = unit_id;

                // partition device path will not have any of the other info, so we need to get parent device path
                // Calculate parent renderD number by subtracting unit_id from current renderD number
                // e.g., /sys/class/drm/renderD129/device with unit_id=1 -> /sys/class/drm/renderD128/device
                device_path_in_use = get_parent_device_path(device_path, unit_id);
            }
        }
    }

    std::string vendor = CuidUtilities::read_sysfs_file(device_path_in_use + "/vendor");
    // if this is a partitioned device, we won't be able to get information from pci config space so skip in the check
    if (vendor.empty() && !bdf.empty() && info.header.fields.gpu.unit_id == 0){
        // if file read fails, attempt to get from pci config
        uint8_t vendor_id_bytes[2] = {0};
        const uint16_t offset = 0x0;
        amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, vendor_id_bytes, 2, offset);
        uint16_t vendor_id_int = PciUtil::le16_to_be16(*reinterpret_cast<uint16_t*>(vendor_id_bytes));
        info.header.fields.gpu.vendor_id = (status == AMDCUID_STATUS_SUCCESS) ? vendor_id_int : 0;
    }
    else
    {
        info.header.fields.gpu.vendor_id = (uint16_t)strtol(vendor.c_str(), nullptr, 0);
    }

    std::string device = CuidUtilities::read_sysfs_file(device_path_in_use + "/device");
    if (device.empty() && !bdf.empty() && info.header.fields.gpu.unit_id == 0){
        // if file read fails, attempt to get from pci config
        uint8_t device_id_bytes[2] = {0};
        const uint16_t offset = 0x2;
        amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, device_id_bytes, 2, offset);
        uint16_t device_id_int = PciUtil::le16_to_be16(*reinterpret_cast<uint16_t*>(device_id_bytes));
        info.header.fields.gpu.device_id = (status ==  AMDCUID_STATUS_SUCCESS) ? device_id_int : 0;
    }
    else
    {
        info.header.fields.gpu.device_id = (uint16_t)strtol(device.c_str(), nullptr, 0);
    }

    std::string pci_class = CuidUtilities::read_sysfs_file(device_path_in_use + "/class");
    uint32_t pci_class_integer = 0;
    if (pci_class.empty() && !bdf.empty() && info.header.fields.gpu.unit_id == 0){
        // if file read fails, attempt to get from pci config
        uint8_t class_id_bytes[2] = {0};
        const uint16_t offset = 0xa;
        amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, class_id_bytes, 2, offset);
        uint16_t class_id_int = PciUtil::le16_to_be16(*reinterpret_cast<uint16_t*>(class_id_bytes));
        pci_class_integer = (status ==  AMDCUID_STATUS_SUCCESS) ? class_id_int : 0;
    }
    else
    {
        pci_class_integer = (uint16_t)strtol(pci_class.c_str(), nullptr, 0);
    }
    info.header.fields.gpu.pci_class = (pci_class_integer >> 8) & 0xFFFF;

    std::string revision_id = CuidUtilities::read_sysfs_file(device_path_in_use + "/revision");
    if (revision_id.empty() && !bdf.empty() && info.header.fields.gpu.unit_id == 0){
        // if file read fails, attempt to get from pci config
        uint8_t revision_id_bytes[2] = {0};
        const uint16_t offset = 0x8;
        amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, revision_id_bytes, 2, offset);
        uint16_t revision_id_int = PciUtil::le16_to_be16(*reinterpret_cast<uint16_t*>(revision_id_bytes));
        info.header.fields.gpu.revision_id = (status ==  AMDCUID_STATUS_SUCCESS) ? revision_id_int : 0;
    }
    else
    {
        info.header.fields.gpu.revision_id = (uint16_t)strtol(revision_id.c_str(), nullptr, 0);
    }

    // we use the original device path to get render node
    std::string full_device_node;
    size_t last_slash = device_path.rfind('/');
    if (last_slash != std::string::npos && last_slash > 0) {
        // Trim to just /sys/class/drm/renderDXXX;
        full_device_node = device_path.substr(0, last_slash);
    }
    else {
        full_device_node = device_path;
    }

    info.header.device_type = AMDCUID_DEVICE_TYPE_GPU;
    info.bdf = bdf;
    info.render_node = full_device_node;

    *gpu_info = info;

    return AMDCUID_STATUS_DEVICE_NOT_FOUND;
}

amdcuid_status_t CuidGpu::get_hardware_fingerprint(uint64_t& fingerprint) const {
    if (geteuid() != 0)
    {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    std::string unique_id_path = m_info.render_node + "/device/unique_id";
    // check if device is partition first by checking unit id
    if (m_info.header.fields.gpu.unit_id != 0) {
        unique_id_path = get_parent_device_path(unique_id_path, m_info.header.fields.gpu.unit_id);
    }
    // Try to read the unique_id from the device sysfs file
    std::ifstream fin(unique_id_path);
    if (fin.is_open()) {
        std::string hex_str;
        std::getline(fin, hex_str);
        fin.close();
        if (hex_str.empty()) {
            fingerprint = 0;
            return AMDCUID_STATUS_UNSUPPORTED;
        }
        // Parse as 64-bit hex value (if possible)
        try {
            fingerprint = std::stoull(hex_str, nullptr, 16);
        } catch (...) {
            fingerprint = 0;
            return AMDCUID_STATUS_UNSUPPORTED;
        }
    }
    else if (m_info.header.fields.gpu.unit_id == 0) {
        // attempt to get fingerprint through PCI Config Space if not a partition
        uint16_t offset = 0;
        amdcuid_status_t status = PciUtil::get_pci_cap_offset(m_info.bdf, 0x03, offset);
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }

        uint8_t fingerprint_size = 8;
        uint8_t* fingerprint_buffer = new uint8_t[fingerprint_size];
        status = PciUtil::read_pci_config_space(m_info.bdf, fingerprint_buffer, fingerprint_size, offset);
        if (status != AMDCUID_STATUS_SUCCESS) {
            fingerprint = 0;
            delete[] fingerprint_buffer;
            return status;
        }
        // pcie config file is little endian, so need to convert to big endian
        fingerprint = PciUtil::le64_to_be64(*reinterpret_cast<uint64_t*>(fingerprint_buffer));
        delete[] fingerprint_buffer;
    }
    else {
        // partitioned device without unique_id file cannot get fingerprint
        fingerprint = 0;
        return AMDCUID_STATUS_UNSUPPORTED;
    }
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_primary_cuid(amdcuid_primary_id& id) const {
    if (geteuid() != 0)
    {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    // attempt to read the CUID from the file first
    std::string cuid_file_path = CuidUtilities::priv_cuid_file();
    CuidFile primary_file(cuid_file_path, false);
    primary_file.load();
    std::vector<CuidFileEntry> entries = primary_file.get_entries();

    CuidFileEntry entry;
    amdcuid_status_t status = primary_file.find_by_device_node(m_info.render_node, entry);
    if (status == AMDCUID_STATUS_SUCCESS) {
        id.UUIDv8_representation = entry.primary_cuid;
        CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);
        return AMDCUID_STATUS_SUCCESS;
    }

    // primary CUID not found in file so generate it
    uint64_t fingerprint = 0;
    status = get_hardware_fingerprint(fingerprint);
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::memset(&id, 0, sizeof(id));
        return status;
    }
    // Use header fields for the rest
    amdcuid_primary_id result = {};
    const auto& h = m_info.header;
    CuidUtilities::generate_primary_cuid(
        fingerprint,
        h.fields.gpu.unit_id,
        h.fields.gpu.revision_id,
        h.fields.gpu.device_id,
        h.fields.gpu.vendor_id,
        static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_GPU),
        &result
    );

    id = result;
    return AMDCUID_STATUS_SUCCESS;
}

const amdcuid_gpu_info& CuidGpu::get_info() const {
    return m_info;
}

amdcuid_status_t CuidGpu::get_vendor_id(uint16_t& vendor_id) const {
    vendor_id = m_info.header.fields.gpu.vendor_id;
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_device_id(uint16_t& device_id) const {
    device_id = m_info.header.fields.gpu.device_id;
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_pci_class(uint16_t& pci_class) const {
    pci_class = m_info.header.fields.gpu.pci_class;
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_revision_id(uint8_t& revision_id) const {
    revision_id = m_info.header.fields.gpu.revision_id;
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_unit_id(uint16_t& unit_id) const {
    unit_id = m_info.header.fields.gpu.unit_id;
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_bdf(std::string& bdf) const {
    if (m_info.bdf.empty()) {
        return AMDCUID_STATUS_UNSUPPORTED;
    }
    bdf = m_info.bdf;
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_device_path(std::string& path) const {
    if (m_info.render_node.empty()) {
        return AMDCUID_STATUS_UNSUPPORTED;
    }
    path = m_info.render_node;
    return AMDCUID_STATUS_SUCCESS;
}
