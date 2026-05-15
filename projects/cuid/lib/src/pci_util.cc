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

#include "pci_util.h"
#include "include/amd_cuid.h"
#include "cuid_util.h"
#include "cuid_device.h"
#include "cuid_device_manager.h"
#include "cuid_gpu.h"
#include "cuid_nic.h"
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unistd.h>

// Endianness conversion functions
uint16_t PciUtil::le16_to_be16(uint16_t value) {
    return ((value & 0x00FF) << 8) | ((value & 0xFF00) >> 8);
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

// function should only work with PCI devices, which so far includes GPUs and NICs. May later include NPU and storage.
amdcuid_status_t PciUtil::read_pci_config_space(std::string bdf, uint8_t *buffer, size_t buffer_size, uint16_t offset) {
    if (geteuid() != 0){
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }
    if (bdf.empty() || !buffer || buffer_size == 0) return AMDCUID_STATUS_INVALID_ARGUMENT;

    std::string pci_config_path = "/sys/bus/pci/devices/" + bdf + "/config";

    // Read the PCI config space
    std::ifstream config_file(pci_config_path, std::ios::binary);
    if (!config_file){
        config_file.close();
        return AMDCUID_STATUS_PCI_ERROR;
    }

    config_file.seekg(0, std::ios::end);
    int length = config_file.tellg();
    config_file.seekg(0, std::ios::beg);

    config_file.seekg(offset, std::ios::beg);
    config_file.read(reinterpret_cast<char*>(buffer), buffer_size);
    int err = errno;
    if (config_file.bad())
    {
        config_file.close();
        return AMDCUID_STATUS_PCI_ERROR;
    }
    return AMDCUID_STATUS_SUCCESS;
}

// Helper to check if a handle has non-zero bytes
static bool is_pci_handle_nonzero(const amdcuid_id_t& handle) {
    for (int i = 0; i < 16; ++i) {
        if (handle.bytes[i] != 0) return true;
    }
    return false;
}

// iterate capabilities list to find the relevant capability
amdcuid_status_t PciUtil::get_pci_cap_offset(std::string bdf, uint32_t cap_id, uint16_t &offset)
{
    if (geteuid() != 0){
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }
    if (bdf.empty()) return AMDCUID_STATUS_INVALID_ARGUMENT;

    // Get the whole PCI config space header
    uint8_t config_space[4096] = {0};
    amdcuid_status_t status = read_pci_config_space(bdf, config_space, 4096, 0);
    if (status != AMDCUID_STATUS_SUCCESS)
    {
        return status;
    }

    // check the 4th bit in the status register first to determine if the capabilities list exists
    uint8_t status_reg = config_space[0x06];
    if ((status_reg & 0b10000) == 0)
    {
        return AMDCUID_STATUS_UNSUPPORTED;
    }

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
            return AMDCUID_STATUS_SUCCESS;
        }

        // Move to the next capability
        cap_ptr = config_space[cap_ptr + 2];
        cap_id_local = config_space[cap_ptr];
    }

    return AMDCUID_STATUS_UNSUPPORTED;
}
