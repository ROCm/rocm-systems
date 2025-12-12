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

AmdCuidNic::AmdCuidNic(const amdcuid_nic_info& i)
    : m_info(i)
{}

amdcuid_status_t AmdCuidNic::discover(std::vector<DevicePtr> &nics) {
    std::string nic_base_path = "/sys/class/net";
    DIR *dir = opendir(nic_base_path.c_str());
    if (!dir) return AMDCUID_STATUS_FILE_NOT_FOUND;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        // grab everything except the loopback device and hidden entries
        if (strncmp(entry->d_name, "lo", 2) != 0 && entry->d_name[0] != '.') {
            amdcuid_nic_info info = {};
            info.header.device_type = AMDCUID_DEVICE_TYPE_NIC;
            info.network_interface = entry->d_name;
            std::string device_path = std::string(nic_base_path) + "/" + entry->d_name + "/device";

            std::string bdf = AmdCuidUtilities::readlink_bdf(device_path);

            std::string vendor = AmdCuidUtilities::read_sysfs_file(device_path + "/vendor");
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

            std::string device = AmdCuidUtilities::read_sysfs_file(device_path + "/device");
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

            std::string pci_class = AmdCuidUtilities::read_sysfs_file(device_path + "/class");
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

            std::string revision_id = AmdCuidUtilities::read_sysfs_file(device_path + "/revision");
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

            nics.emplace_back(std::make_shared<AmdCuidNic>(info));
        }
    }
    if (nics.size() == 0)
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;

    closedir(dir);
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t AmdCuidNic::get_hardware_fingerprint(uint64_t& fingerprint) const {
    if (geteuid() != 0)
    {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    uint32_t cap_id = 0x3;
    uint16_t offset = 0;
    amdcuid_status_t status = PciUtil::get_pci_cap_offset(m_info.bdf, cap_id, offset);
    if (status != AMDCUID_STATUS_FILE_NOT_FOUND)
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
    // pci config space file does not exist or read failed, so create fingerprint from MAC address
    std::string mac_path = "/sys/class/net/" + m_info.network_interface + "/address";
    std::string mac_address = AmdCuidUtilities::read_sysfs_file(mac_path);
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

amdcuid_status_t AmdCuidNic::get_primary_cuid(amdcuid& id) const {
    if (geteuid() != 0)
    {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    // attempt to read the CUID from the file first
    std::string cuid_file_path = "/tmp/priv_cuid";
    CuidFile primary_file(cuid_file_path, false);
    primary_file.load();
    std::vector<CuidFileEntry> entries = primary_file.get_entries();

    CuidFileEntry entry;
    amdcuid_status_t status =primary_file.find_by_device_node(m_info.network_interface, entry);
    if (status == AMDCUID_STATUS_SUCCESS) {
        id = entry.primary_cuid;
        return AMDCUID_STATUS_SUCCESS;
    }

    uint64_t fingerprint = 0;
    status = get_hardware_fingerprint(fingerprint);
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::memset(id.bytes, 0, sizeof(id.bytes));
        return status;
    }

    status = AmdCuidUtilities::generate_primary_cuid(
        fingerprint, 
        0, 
        m_info.header.fields.nic.revision_id,
        m_info.header.fields.nic.device_id,
        m_info.header.fields.nic.vendor_id,
        static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_NIC), 
        &id);
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::memset(id.bytes, 0, sizeof(id.bytes));
        return status;
    }

    return status;
}

const amdcuid_nic_info& AmdCuidNic::get_info() const {
    return m_info;
}
