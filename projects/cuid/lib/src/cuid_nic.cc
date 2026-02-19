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

#include "cuid_nic.h"
#include "cuid_util.h"
#include "pci_util.h"
#include "cuid_file.h"
#include <cstring>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>

CuidNic::CuidNic(const amdcuid_nic_info& i)
    : m_info(i)
{}

amdcuid_status_t CuidNic::discover(std::vector<DevicePtr> &nics) {
    std::string nic_base_path = "/sys/class/net";
    DIR *dir = opendir(nic_base_path.c_str());
    if (!dir) return AMDCUID_STATUS_DEVICE_NOT_FOUND;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        // grab everything except the loopback device and hidden entries
        if (strncmp(entry->d_name, "lo", 2) != 0 && entry->d_name[0] != '.') {
            amdcuid_nic_info info = {};
            std::string device_path = std::string(nic_base_path) + "/" + entry->d_name + "/device";
            discover_single(&info, device_path);

            nics.emplace_back(std::make_shared<CuidNic>(info));
        }
    }
    if (nics.size() == 0)
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;

    closedir(dir);
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::discover_single(amdcuid_nic_info* nic_info, const std::string& device_path) {
    amdcuid_nic_info info = {};
    info.header.device_type = AMDCUID_DEVICE_TYPE_NIC;

    std::string bdf = CuidUtilities::readlink_bdf(device_path);

    std::string vendor = CuidUtilities::read_sysfs_file(device_path + "/vendor");
    if (vendor.empty() && !bdf.empty()){
        // if file read fails, attempt to get from pci config
        uint8_t vendor_id_bytes[2] = {0};
        const uint16_t offset = 0x0;
        amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, vendor_id_bytes, 2, offset);
        uint16_t vendor_id_int = PciUtil::le16_to_be16(*reinterpret_cast<uint16_t*>(vendor_id_bytes));
        info.header.fields.nic.vendor_id = (status == AMDCUID_STATUS_SUCCESS) ? vendor_id_int : 0;
    }
    else
    {
        info.header.fields.nic.vendor_id = (uint16_t)strtol(vendor.c_str(), nullptr, 0);
    }

    std::string device = CuidUtilities::read_sysfs_file(device_path + "/device");
    if (device.empty() && !bdf.empty()){
        // if file read fails, attempt to get from pci config
        uint8_t device_id_bytes[2] = {0};
        const uint16_t offset = 0x2;
        amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, device_id_bytes, 2, offset);
        uint16_t device_id_int = PciUtil::le16_to_be16(*reinterpret_cast<uint16_t*>(device_id_bytes));
        info.header.fields.nic.device_id = (status ==  AMDCUID_STATUS_SUCCESS) ? device_id_int : 0;
    }
    else
    {
        info.header.fields.nic.device_id = (uint16_t)strtol(device.c_str(), nullptr, 0);
    }

    std::string pci_class = CuidUtilities::read_sysfs_file(device_path + "/class");
    uint32_t pci_class_integer = 0;
    if (pci_class.empty() && !bdf.empty()){
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
    info.header.fields.nic.pci_class = (pci_class_integer >> 8) & 0xFFFF;

    std::string revision_id = CuidUtilities::read_sysfs_file(device_path + "/revision");
    if (revision_id.empty() && !bdf.empty()){
        // if file read fails, attempt to get from pci config
        uint8_t revision_id_bytes[2] = {0};
        const uint16_t offset = 0x8;
        amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, revision_id_bytes, 2, offset);
        uint16_t revision_id_int = PciUtil::le16_to_be16(*reinterpret_cast<uint16_t*>(revision_id_bytes));
        info.header.fields.nic.revision_id = (status ==  AMDCUID_STATUS_SUCCESS) ? revision_id_int : 0;
    }
    else
    {
        info.header.fields.nic.revision_id = (uint16_t)strtol(revision_id.c_str(), nullptr, 0);
    }
    info.bdf = bdf;
    std::string full_device_node;
    size_t last_slash = device_path.rfind('/');
    if (last_slash != std::string::npos && last_slash > 0) {
        full_device_node = device_path.substr(0, last_slash);
    }
    else {
        full_device_node = device_path;
    }
    info.network_interface = full_device_node;
    *nic_info = info;

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_hardware_fingerprint(uint64_t& fingerprint) const {
    if (geteuid() != 0)
    {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    uint32_t cap_id = 0x3;
    uint16_t offset = 0;
    amdcuid_status_t status = PciUtil::get_pci_cap_offset(m_info.bdf, cap_id, offset);
    if (status != AMDCUID_STATUS_UNSUPPORTED)
    {
        const uint8_t fingerprint_size = 8;
        uint8_t fingerprint_bytes[fingerprint_size] = {0};
        status = PciUtil::read_pci_config_space(m_info.bdf, fingerprint_bytes, fingerprint_size, offset);
        if (status == AMDCUID_STATUS_SUCCESS)
        {
            fingerprint = PciUtil::le64_to_be64(*reinterpret_cast<uint64_t*>(fingerprint_bytes));
            return AMDCUID_STATUS_SUCCESS;
        }
    }
    // TODO: serial ID coudl be found in FRU_ID so should attempt to look there
    // pci config space file does not exist or read failed, so create fingerprint from MAC address
    std::string mac_address;
    status = get_mac_address(mac_address);
    if (status != AMDCUID_STATUS_SUCCESS)
    {
        return status;
    }
    if (!mac_address.empty())
    {
        // convert MAC address string to bytes
        uint8_t mac_bytes[6] = {0};
        sscanf(mac_address.c_str(), "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx",
               &mac_bytes[0], &mac_bytes[1], &mac_bytes[2],
               &mac_bytes[3], &mac_bytes[4], &mac_bytes[5]);
        fingerprint = *reinterpret_cast<uint64_t*>(mac_bytes);
        return AMDCUID_STATUS_SUCCESS;
    }

    return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
}

amdcuid_status_t CuidNic::get_primary_cuid(amdcuid_primary_id& id) const {
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
    amdcuid_status_t status =primary_file.find_by_device_node(m_info.network_interface, entry);
    if (status == AMDCUID_STATUS_SUCCESS) {
        id.UUIDv8_representation = entry.primary_cuid;
        CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);
        return AMDCUID_STATUS_SUCCESS;
    }

    uint64_t fingerprint = 0;
    status = get_hardware_fingerprint(fingerprint);
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::memset(&id, 0, sizeof(id));
        return status;
    }

    status = CuidUtilities::generate_primary_cuid(
        fingerprint, 
        0, 
        m_info.header.fields.nic.revision_id,
        m_info.header.fields.nic.device_id,
        m_info.header.fields.nic.vendor_id,
        static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_NIC), 
        &id);
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::memset(&id, 0, sizeof(id));
        return status;
    }

    return status;
}

const amdcuid_nic_info& CuidNic::get_info() const {
    return m_info;
}

amdcuid_status_t CuidNic::get_vendor_id(uint16_t& vendor_id) const {
    vendor_id = m_info.header.fields.nic.vendor_id;
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_device_id(uint16_t& device_id) const {
    device_id = m_info.header.fields.nic.device_id;
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_pci_class(uint16_t& pci_class) const {
    pci_class = m_info.header.fields.nic.pci_class;
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_revision_id(uint8_t& revision_id) const {
    revision_id = m_info.header.fields.nic.revision_id;
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_bdf(std::string& bdf) const {
    if (m_info.bdf.empty()) {
        return AMDCUID_STATUS_UNSUPPORTED;
    }
    bdf = m_info.bdf;
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_device_path(std::string& path) const {
    if (m_info.network_interface.empty()) {
        return AMDCUID_STATUS_UNSUPPORTED;
    }
    path = m_info.network_interface;
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_mac_address(std::string& mac_address) const {
    std::string mac_path = m_info.network_interface + "/address";
    mac_address = CuidUtilities::read_sysfs_file(mac_path);

    if (mac_address.empty()) {
        return AMDCUID_STATUS_UNSUPPORTED;
    }
    return AMDCUID_STATUS_SUCCESS;
}