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

#ifndef CUID_PLATFORM_H
#define CUID_PLATFORM_H

#include "cuid_device.h"
#include "include/amd_cuid.h"
#include "cuid_internal.h"
#include <vector>
#include <memory>

struct amdcuid_platform_info {
    amdcuid_cuid_public_fields header;
    // Add more fields as needed
};

class CuidPlatform : public CuidDevice {
public:
    CuidPlatform(const amdcuid_platform_info& i);
    amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_PLATFORM; }
    amdcuid_status_t get_primary_cuid(amdcuid_primary_id& id) const override;
    amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const override;
    static amdcuid_status_t discover(std::vector<DevicePtr> &platforms);

    // Virtual accessor overrides
    amdcuid_status_t get_vendor_id(uint16_t& vendor_id) const override;

    const amdcuid_platform_info& get_info() const;
private:
    amdcuid_platform_info m_info;
};

#endif // CUID_PLATFORM_H