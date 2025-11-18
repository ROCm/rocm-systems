#include "pci_util.h"
#include "cuid.h"
#include "cuid_util.h"
#include "cuid_device.h"
#include "cuid_gpu.h"
#include "cuid_nic.h"
#include "cuid_lib_loader.h"
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>

// Endianness conversion functions
uint16_t PciUtil::le16_to_be16(uint16_t value) {
    return ((value & 0x00FF) << 8) | ((value & 0xFF00) >> 8);
}

uint32_t PciUtil::le32_to_be32(uint32_t value) {
    return ((value & 0x000000FF) << 24) |
           ((value & 0x0000FF00) << 8)  |
           ((value & 0x00FF0000) >> 8)  |
           ((value & 0xFF000000) >> 24);
}

uint64_t PciUtil::le64_to_be64(uint64_t value) {
    return ((value & 0x00000000000000FFULL) << 56) |
           ((value & 0x000000000000FF00ULL) << 40) |
           ((value & 0x0000000000FF0000ULL) << 24) |
           ((value & 0x00000000FF000000ULL) << 8)  |
           ((value & 0x000000FF00000000ULL) >> 8)  |
           ((value & 0x0000FF0000000000ULL) >> 24) |
           ((value & 0x00FF000000000000ULL) >> 40) |
           ((value & 0xFF00000000000000ULL) >> 56);
}

// function should only work with PCI devices, which so far includes GPUs and NICs. May later incude NPU and storage.
amdcuid_status_t PciUtil::read_pci_config_space(std::string bdf, uint8_t *buffer, size_t buffer_size, uint16_t offset) {
    if (bdf.empty() || !buffer || buffer_size == 0) return AMDCUID_STATUS_INVALID_ARGUMENT;

    std::string pci_config_path = "/sys/bus/pci/devices/" + bdf + "/config";

    // Read the PCI config space
    std::ifstream config_file(pci_config_path, std::ios::binary);
    if (!config_file){
        std::cout << "Error opening PCI config space from " << pci_config_path << std::endl;
        config_file.close();
        return AMDCUID_STATUS_FILE_NOT_FOUND;
    }

    config_file.seekg(0, std::ios::end);
    int length = config_file.tellg();
    config_file.seekg(0, std::ios::beg);
    std::cout << "length of config space is: " << length << std::endl;

    // TODO: need to determine where precisely serial ID is located in PCI config space to get
    config_file.seekg(offset, std::ios::beg);
    config_file.read(reinterpret_cast<char*>(buffer), buffer_size);
    int err = errno;
    if (config_file.bad())
    {
        std::cout << "Error reading PCI config space from " << pci_config_path << " errno: " << err << std::endl;
        std::cout << "bytes read: " << config_file.gcount() << std::endl;
        std::cout << "file size: " << length << std::endl;
        config_file.close();
        return AMDCUID_STATUS_INSUFFICIENT_SIZE; // give this error for now and figure out real error later
    }
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t PciUtil::get_pci_bdf_from_handle(amdcuid_handle handle, std::string &bdf) {
    if (!handle.impl) return AMDCUID_STATUS_INVALID_ARGUMENT;
    AmdCuidDevice* dev = static_cast<AmdCuidDevice*>(handle.impl);
    if (!dev) return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    amdcuid_device_type_t type = dev->type();
    switch (type) {
        case AMDCUID_DEVICE_TYPE_GPU: {
            const amdcuid_gpu_info& info = static_cast<AmdCuidGpu*>(dev)->get_info();
            if (info.bdf.empty()) {
                bdf.clear();
                return AMDCUID_STATUS_UNSUPPORTED;
            }
            bdf = info.bdf;
            return AMDCUID_STATUS_SUCCESS;
        }
        // will add this back once NIC is defined
        // case AMDCUID_DEVICE_TYPE_NIC: {
        //     const amdcuid_nic_info& info = static_cast<AmdCuidNic*>(dev)->get_info();
        //     if (info.bdf.empty()) {
        //         bdf.clear();
        //         return AMDCUID_STATUS_UNSUPPORTED;
        //     }
        //     bdf = info.bdf;
        //     return AMDCUID_STATUS_SUCCESS;
        // }
        default:
            bdf.clear();
            return AMDCUID_STATUS_UNSUPPORTED;
    }
}

// iterate capabilities list to find the relevant capability
amdcuid_status_t PciUtil::get_pci_cap_offset(std::string bdf, uint32_t cap_id, uint16_t &offset)
{
    if (bdf.empty()) return AMDCUID_STATUS_INVALID_ARGUMENT;

    // Get the whole PCI config space header
    uint8_t config_space[4096] = {0};
    amdcuid_status_t status = read_pci_config_space(bdf, config_space, 4096, 0);
    if (status != AMDCUID_STATUS_SUCCESS)
    {
        std::cout << "read_pci_config_space failed on " << bdf << std::endl;
        return status;
    }

    std::cout << "read_pci_config_space succeeded on " << bdf << std::endl;
    std::cout << "bytes read: " << sizeof(config_space) << std::endl;

    // check the 4th bit in the status register first to determine if the capabilities list exists
    uint8_t status_reg = config_space[0x06];
    if ((status_reg & 0b10000) == 0)
    {
        std::cout << "No capabilities list found in PCI config space." << std::endl;
        return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
    }

    std::cout << "Capabilities list indicator found, continuing." << std::endl;

    // Get the capabilities pointer from the PCI config space
    // Device Serial Number is a PCIE Extended Capability, so we need to look in the extended capabilities list
    uint16_t cap_ptr = 0x100;

    uint16_t cap_id_local = config_space[cap_ptr];
    // Iterate through the capabilities list
    while (cap_ptr != 0) {
        // Check if this is the capability we're looking for
        if (cap_id_local == cap_id) {
            // Found the capability, set the offset to the capability's data
            offset = cap_ptr + 4;
            std::cout << "Found capability " << cap_id << " at offset " << offset << std::endl;
            return AMDCUID_STATUS_SUCCESS;
        }

        // Move to the next capability
        cap_ptr = config_space[cap_ptr + 2];
        cap_id_local = config_space[cap_ptr];
    }

    std::cout << "Capability not found." << std::endl;
    return AMDCUID_STATUS_UNSUPPORTED;
}
