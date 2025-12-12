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

#ifndef CUID_NIC_H
#define CUID_NIC_H


#include "src/cuid_device.h"
#include "include/amd_cuid.h"
#include "src/cuid_internal.h"
#include <vector>
#include <memory>
#include <string>

struct amdcuid_nic_info {
    amdcuid_cuid_public_fields header;
    std::string bdf;
    std::string network_interface;
};

class CuidNic : public CuidDevice {
public:
    CuidNic(const amdcuid_nic_info& i);
    amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_NIC; }
    amdcuid_status_t get_primary_cuid(amdcuid_primary_id& id) const override;
    amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const override;
    static amdcuid_status_t discover(std::vector<DevicePtr> &nics);
    static amdcuid_status_t discover_single(amdcuid_nic_info* nic_info, const std::string& device_path);

    // Virtual accessor overrides
    amdcuid_status_t get_vendor_id(uint16_t& vendor_id) const override;
    amdcuid_status_t get_device_id(uint16_t& device_id) const override;
    amdcuid_status_t get_pci_class(uint16_t& pci_class) const override;
    amdcuid_status_t get_revision_id(uint8_t& revision_id) const override;
    amdcuid_status_t get_bdf(std::string& bdf) const override;
    amdcuid_status_t get_device_path(std::string& path) const override;

    // MAC address accessor
    amdcuid_status_t get_mac_address(std::string& mac_address) const;

    const amdcuid_nic_info& get_info() const;
private:
    amdcuid_nic_info m_info;
};

#endif // CUID_NIC_H