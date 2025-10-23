
#include "cuid_gpu.h"
#include "cuid_util.h"
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
    if (!dir) return AMDCUID_STATUS_FILE_NOT_FOUND;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "renderD", 7) == 0 && isdigit(entry->d_name[7])) {
            std::string render_name(entry->d_name);
            std::string device_path = std::string(drm_path) + "/" + render_name + "/device";

            std::string vendor = CuidUtilities::read_sysfs_file(device_path + "/vendor");
            std::string device = CuidUtilities::read_sysfs_file(device_path + "/device");
            std::string pci_class = CuidUtilities::read_sysfs_file(device_path + "/class");
            std::string revision_id = CuidUtilities::read_sysfs_file(device_path + "/revision");
            std::string partition_info = CuidUtilities::read_sysfs_file(device_path + "/partition_info");
            std::string bdf = CuidUtilities::readlink_bdf(device_path);
            std::string full_device_node = std::string(drm_path) + "/" + render_name;

            amdcuid_gpu_info info = {};
            info.header.hdr.device_type = AMDCUID_DEVICE_TYPE_GPU;
            info.header.vid = vendor.empty() ? 0 : (uint16_t)strtol(vendor.c_str(), nullptr, 0);
            info.header.did = device.empty() ? 0 : (uint16_t)strtol(device.c_str(), nullptr, 0);
            uint32_t pci_class_integer = (uint32_t)strtoul(pci_class.c_str(), nullptr, 16);
            info.header.pci_class = (pci_class_integer >> 8) & 0xFFFF;
            info.header.revision_id = revision_id.empty() ? 0 : (uint16_t)strtol(revision_id.c_str(), nullptr, 0);
            info.header.partition_info = partition_info.empty() ? 0 : (uint32_t)strtoul(partition_info.c_str(), nullptr, 0);
            info.bdf = bdf;
            info.render_node = full_device_node;

            gpus.emplace_back(std::make_shared<CuidGpu>(info));
        }
    }
    closedir(dir);
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_hardware_fingerprint(uint64_t& fingerprint) const {
    // Try to read the unique_id from the device sysfs
    std::string unique_id_path = m_info.render_node + "/device/unique_id";
    std::ifstream fin(unique_id_path);
    if (!fin.is_open()) {
        fingerprint = 0;
        return AMDCUID_STATUS_FILE_NOT_FOUND;
    }
    std::string hex_str;
    std::getline(fin, hex_str);
    fin.close();
    if (hex_str.empty()) {
        fingerprint = 0;
        return AMDCUID_STATUS_FILE_NOT_FOUND;
    }
    // Parse as 64-bit hex value (if possible)
    try {
        fingerprint = std::stoull(hex_str, nullptr, 16);
    } catch (...) {
        fingerprint = 0;
        return AMDCUID_STATUS_UNSUPPORTED;
    }
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_primary_cuid(amdcuid& id) const {
    uint64_t fingerprint = 0;
    amdcuid_status_t status = get_hardware_fingerprint(fingerprint);
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::memset(id.bytes, 0, sizeof(id.bytes));
        return status;
    }
    // Use header fields for the rest
    const auto& h = m_info.header;
    id = CuidUtilities::generate_primary_cuid(
        fingerprint,
        0, // unit_id_part1 TODO:
        0, // unit_id_part2 TODO:
        h.revision_id,
        h.did,
        h.vid,
        static_cast<uint8_t>(h.hdr.device_type)
    );
    return AMDCUID_STATUS_SUCCESS;
}

const amdcuid_gpu_info& CuidGpu::get_info() const {
    return m_info;
}