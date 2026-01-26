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

#include "cuid_platform.h"
#include "cuid_util.h"
#include "smbios_util.h"
#include "cuid_file.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <unistd.h>

CuidPlatform::CuidPlatform(const amdcuid_platform_info& i)
    : m_info({i})
{}

amdcuid_status_t CuidPlatform::discover(std::vector<DevicePtr> &platforms) {
    // Platform is a singleton - only one platform per system
    amdcuid_platform_info info = {};
    info.header.device_type = AMDCUID_DEVICE_TYPE_PLATFORM;

    std::string vendor, name, version;
    amdcuid_status_t status = SmbiosUtil::get_board_info(vendor, name, version);
    if (status != AMDCUID_STATUS_SUCCESS) {
        // FILE NOT FOUND status given here, which means smbios info not available
        return AMDCUID_STATUS_UNSUPPORTED;
    }
    uint16_t vendor_id = (uint16_t)strtol(vendor.c_str(), nullptr, 0);
    info.header.fields.platform.vendor_id = vendor_id;

    // Create platform device
    platforms.emplace_back(std::make_shared<CuidPlatform>(info));

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidPlatform::get_hardware_fingerprint(uint64_t& fingerprint) const {
    if (geteuid() != 0)
    {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    std::string serial;
    amdcuid_status_t status = SmbiosUtil::get_system_serial(serial);
    if (status != AMDCUID_STATUS_SUCCESS) {
        fingerprint = 0;
        return AMDCUID_STATUS_UNSUPPORTED;
    }

    // Generate fingerprint from serial number
    fingerprint = 0;
    for (size_t i = 0; i < serial.length() && i < 8; ++i) {
        fingerprint |= (static_cast<uint64_t>(serial[i]) << (i * 8));
    }
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidPlatform::get_primary_cuid(amdcuid_primary_id& id) const {
    if (geteuid() != 0)
    {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }

    //attempt to find the primary CUID in file first
    std::string cuid_file_path = CuidUtilities::priv_cuid_file();
    CuidFile primary_file(cuid_file_path, false);
    primary_file.load();
    std::vector<CuidFileEntry> entries = primary_file.get_entries();

    // for platform, just return the first entry found
    CuidFileEntry entry;
    amdcuid_status_t status = primary_file.find_by_device_type(AMDCUID_DEVICE_TYPE_PLATFORM, entry);
    if (status == AMDCUID_STATUS_SUCCESS) {
        id.UUIDv8_representation = entry.primary_cuid;
        CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);
        return AMDCUID_STATUS_SUCCESS;
    }

    // Get system UUID from SMBIOS
    uint8_t uuid[16] = {0};
    status = SmbiosUtil::get_system_uuid(uuid);

    if (status != AMDCUID_STATUS_SUCCESS) {
        // Fallback: generate from hardware fingerprint
        uint64_t fingerprint = 0;
        status = get_hardware_fingerprint(fingerprint);

        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }

        // Use fingerprint as basis for CUID
        status = CuidUtilities::generate_primary_cuid(
            fingerprint,
            0,
            0,
            0,
            m_info.header.fields.platform.vendor_id,
            AMDCUID_DEVICE_TYPE_PLATFORM,
            &id
        );

        return AMDCUID_STATUS_SUCCESS;
    }

    // use UUID directly as CUID, if found
    std::memcpy(&id.UUIDv8_representation, uuid, sizeof(uuid));
    CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);

    return AMDCUID_STATUS_SUCCESS;
}

const amdcuid_platform_info& CuidPlatform::get_info() const {
    return m_info;
}

amdcuid_status_t CuidPlatform::get_vendor_id(uint16_t& vendor_id) const {
    vendor_id = m_info.header.fields.platform.vendor_id;
    return AMDCUID_STATUS_SUCCESS;
}
