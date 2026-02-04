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

#include "cuid_device.h"
#include "cuid_util.h"
#include "cuid_cpu.h"
#include "cuid_gpu.h"
#include "cuid_nic.h"
#include "cuid_platform.h"
#include "cuid_file.h"
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <mutex>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <openssl/sha.h>

// helper function to get a hash from the raw bytes of a derived ID
void get_hash_from_raw(uint8_t raw_bytes[16], uint8_t out_hash[14]) {
    // just remove the reserved bits from the raw bytes to get the hash
    memcpy(out_hash, raw_bytes, 8);

    // byte 8 of raw bits is reserved which we can skip
    memcpy(&out_hash[8], &raw_bytes[9], 5);
    // byte 14 of raw bits has 2 reserved bits in the MSBs, so mask those off
    out_hash[13] = raw_bytes[14] & 0x3F;
}

void build_derived_id_from_file_entry(const CuidFileEntry& entry, amdcuid_derived_id& id) {
    id.UUIDv8_representation = entry.derived_cuid;
    CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);
    get_hash_from_raw(id.raw_bits, id.hash);
}

amdcuid_status_t CuidDevice::get_derived_cuid(amdcuid_derived_id& id, cuid_hmac * hmac) const {
    // attempt to find the derived CUID in file first
    CuidFile derived_file(CuidUtilities::cuid_file(), false);
    amdcuid_status_t status = derived_file.load();

    if (status == AMDCUID_STATUS_SUCCESS) {
        amdcuid_device_type_t type = this->type();
        // there's only 1 platform entry, so handle that case first
        switch (type){
            case AMDCUID_DEVICE_TYPE_PLATFORM:
            {
                // for platform, just return the first entry found
                CuidFileEntry entry;
                status = derived_file.find_by_device_type(AMDCUID_DEVICE_TYPE_PLATFORM, entry);
                if (status == AMDCUID_STATUS_SUCCESS) {
                    build_derived_id_from_file_entry(entry, id);
                    return AMDCUID_STATUS_SUCCESS;
                }
            }break;
            case AMDCUID_DEVICE_TYPE_GPU:
            // search by render node
            {
                auto gpu = reinterpret_cast<CuidGpu*>(const_cast<CuidDevice*>(this));
                if (gpu) {
                    auto info = gpu->get_info();
                    CuidFileEntry entry;
                    status = derived_file.find_by_device_node(info.render_node, entry);
                    if (status == AMDCUID_STATUS_SUCCESS) {
                        build_derived_id_from_file_entry(entry, id);
                        return AMDCUID_STATUS_SUCCESS;
                    }
                }
            }
            break;
            case AMDCUID_DEVICE_TYPE_CPU:
                // search by package_core_id
                {
                    auto cpu = reinterpret_cast<CuidCpu*>(const_cast<CuidDevice*>(this));
                    if (cpu) {
                        const auto& info = cpu->get_info();
                        std::string core_id = std::to_string(info.header.fields.cpu.physical_id) + 
                                        ":" + std::to_string(info.header.fields.cpu.core);
                        CuidFileEntry entry;
                        status = derived_file.find_by_package_core_id(core_id, entry);
                        if (status == AMDCUID_STATUS_SUCCESS) {
                            build_derived_id_from_file_entry(entry, id);
                            return AMDCUID_STATUS_SUCCESS;
                        }
                    }
                }
                break;
            case AMDCUID_DEVICE_TYPE_NIC:
                // search by device node
                {
                    auto nic = reinterpret_cast<CuidNic*>(const_cast<CuidDevice*>(this));
                    if (nic) {
                        const auto& info = nic->get_info();
                        CuidFileEntry entry;
                        amdcuid_status_t status = derived_file.find_by_device_node(info.network_interface, entry);
                        if (status == AMDCUID_STATUS_SUCCESS) {
                            build_derived_id_from_file_entry(entry, id);
                            return AMDCUID_STATUS_SUCCESS;
                        }
                    }
                }
                break;
            default:
                break;
                    // Will expand with different devices as we implement them
        }
    }

    // if not found, generate derived CUID
    if (geteuid() != 0)
    {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }
    if (!hmac) {
        // if we must generate the derived CUID, then require HMAC
        return AMDCUID_STATUS_INVALID_ARGUMENT;
    }
    amdcuid_primary_id primary;
    status = get_primary_cuid(primary);
    if (status != AMDCUID_STATUS_SUCCESS) {
        return status;
    }

    status = CuidUtilities::generate_derived_cuid(&primary, &id, hmac);
    return status;
}
