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

#ifndef CUID_DEVICE_H
#define CUID_DEVICE_H

#include "include/amd_cuid.h"
#include "src/cuid_internal.h"
#include "src/hmac.h"
#include <memory>
#include <cstdint>
#include <string>

class CuidDevice {
public:
    virtual ~CuidDevice() = default;
    virtual amdcuid_device_type_t type() const = 0;
    virtual amdcuid_status_t get_primary_cuid(amdcuid_primary_id& id) const = 0;
    virtual amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const = 0;
    amdcuid_status_t get_derived_cuid(amdcuid_derived_id& id, cuid_hmac * hmac = nullptr) const;

    // Virtual accessors for common device properties with default wrong device type implementations
    virtual amdcuid_status_t get_vendor_id(uint16_t& vendor_id) const {
        (void)vendor_id;
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }
    virtual amdcuid_status_t get_family(uint16_t& family) const {
        (void)family;
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }
    virtual amdcuid_status_t get_model(uint16_t& model) const {
        (void)model;
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }
    virtual amdcuid_status_t get_device_id(uint16_t& device_id) const {
        (void)device_id;
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }
    virtual amdcuid_status_t get_revision_id(uint8_t& revision_id) const {
        (void)revision_id;
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }
    virtual amdcuid_status_t get_unit_id(uint16_t& unit_id) const {
        (void)unit_id;
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }
    virtual amdcuid_status_t get_pci_class(uint16_t& pci_class) const {
        (void)pci_class;
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }
    virtual amdcuid_status_t get_core(uint16_t& core) const {
        (void)core;
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }
    virtual amdcuid_status_t get_physical_id(uint16_t& physical_id) const {
        (void)physical_id;
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }
    virtual amdcuid_status_t get_bdf(std::string& bdf) const {
        (void)bdf;
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }
    virtual amdcuid_status_t get_device_path(std::string& path) const {
        (void)path;
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }
};

typedef std::shared_ptr<CuidDevice> DevicePtr;

#endif // CUID_DEVICE_H