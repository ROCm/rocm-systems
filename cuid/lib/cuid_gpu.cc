
#include "cuid_gpu.h"
#include "cuid_util.h"
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>

AmdCuidGpu::AmdCuidGpu(const amdcuid_gpu_info& i)
    : m_info(i)
{}

amdcuid_status_t AmdCuidGpu::discover(std::vector<DevicePtr> &gpus) {
    const char *drm_path = "/sys/class/drm";
    DIR *dir = opendir(drm_path);
    if (!dir) return AMDCUID_STATUS_FILE_NOT_FOUND;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "renderD", 7) == 0 && isdigit(entry->d_name[7])) {
            std::string render_name(entry->d_name);
            std::string device_path = std::string(drm_path) + "/" + render_name + "/device";

            std::string vendor = AmdCuidUtilities::read_sysfs_file(device_path + "/vendor");
            std::string device = AmdCuidUtilities::read_sysfs_file(device_path + "/device");
            std::string pci_class = AmdCuidUtilities::read_sysfs_file(device_path + "/class");
            std::string revision_id = AmdCuidUtilities::read_sysfs_file(device_path + "/revision");
            std::string partition_info = AmdCuidUtilities::read_sysfs_file(device_path + "/partition_info");
            std::string bdf = AmdCuidUtilities::readlink_bdf(device_path);
            std::string full_device_node = std::string(drm_path) + "/" + render_name;

            amdcuid_gpu_info info = {};
            // amdcuid_cuid_fields_gpu gpu_fields = {};
            info.header.device_type = AMDCUID_DEVICE_TYPE_GPU;
            info.header.fields.gpu.vendor_id = vendor.empty() ? 0 : (uint16_t)strtol(vendor.c_str(), nullptr, 0);
            info.header.fields.gpu.device_id= device.empty() ? 0 : (uint16_t)strtol(device.c_str(), nullptr, 0);
            uint32_t pci_class_integer = (uint32_t)strtoul(pci_class.c_str(), nullptr, 16);
            info.header.fields.gpu.pci_class = (pci_class_integer >> 8) & 0xFFFF;
            info.header.fields.gpu.revision_id = revision_id.empty() ? 0 : (uint16_t)strtol(revision_id.c_str(), nullptr, 0);
            info.header.fields.gpu.unit_id = partition_info.empty() ? 0 : (uint32_t)strtoul(partition_info.c_str(), nullptr, 0);
            info.bdf = bdf;
            info.render_node = full_device_node;
            // info.header.fields.gpu = gpu_fields;

            gpus.emplace_back(std::make_shared<AmdCuidGpu>(info));
        }
    }
    closedir(dir);
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t AmdCuidGpu::get_hardware_fingerprint(uint64_t& fingerprint) const {
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

amdcuid_status_t AmdCuidGpu::get_primary_cuid(amdcuid& id) const {
    uint64_t fingerprint = 0;
    amdcuid_status_t status = get_hardware_fingerprint(fingerprint);
    if (status != AMDCUID_STATUS_SUCCESS) {
        std::memset(id.bytes, 0, sizeof(id.bytes));
        return status;
    }
    // Use header fields for the rest
    amdcuid result = {};
    const auto& h = m_info.header;
    AmdCuidUtilities::generate_primary_cuid(
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

const amdcuid_gpu_info& AmdCuidGpu::get_info() const {
    return m_info;
}