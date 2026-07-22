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
#include "cuid_cpu.h"
#include "cuid_file.h"
#include "cuid_gpu.h"
#include "cuid_nic.h"
#include "cuid_npu.h"
#include "cuid_platform.h"
#include "cuid_util.h"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <sstream>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

// helper function to get a hash from the raw bytes of a derived ID
void get_hash_from_raw(uint8_t raw_bytes[16], uint8_t out_hash[14]) {
  // just remove the reserved bits from the raw bytes to get the hash
  memcpy(out_hash, raw_bytes, 8);

  // byte 8 of raw bits is reserved which we can skip
  memcpy(&out_hash[8], &raw_bytes[9], 5);
  // byte 14 of raw bits has 2 reserved bits in the MSBs, so mask those off
  out_hash[13] = raw_bytes[14] & 0x3F;
}

void build_derived_id_from_file_entry(const CuidFileEntry &entry,
                                      amdcuid_derived_id &id) {
  id.UUIDv8_representation = entry.derived_cuid;
  CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);
  get_hash_from_raw(id.raw_bits, id.hash);
}

amdcuid_status_t CuidDevice::get_derived_cuid(amdcuid_derived_id &id,
                                              cuid_hmac *hmac) const {
  // attempt to find the derived CUID in file first
  CuidFile derived_file(CuidUtilities::cuid_file(), false);
  amdcuid_status_t status = derived_file.load();

  if (status == AMDCUID_STATUS_SUCCESS) {
    amdcuid_device_type_t type = this->type();
    // there's only 1 platform entry, so handle that case first
    switch (type) {
    case AMDCUID_DEVICE_TYPE_PLATFORM: {
      // for platform, just return the first entry found
      CuidFileEntry entry;
      status =
          derived_file.find_by_device_type(AMDCUID_DEVICE_TYPE_PLATFORM, entry);
      if (status == AMDCUID_STATUS_SUCCESS) {
        build_derived_id_from_file_entry(entry, id);
        return AMDCUID_STATUS_SUCCESS;
      }
    } break;
    case AMDCUID_DEVICE_TYPE_GPU:
      // search by render node
      {
        auto gpu = reinterpret_cast<CuidGpu *>(const_cast<CuidDevice *>(this));
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
      // search by device_node first (unique per logical CPU),
      // then fall back to package_core_id for backward compatibility
      {
        auto cpu = reinterpret_cast<CuidCpu *>(const_cast<CuidDevice *>(this));
        if (cpu) {
          // Try device_node first - unique per logical CPU on SMT systems
          std::string device_path;
          if (cpu->get_device_path(device_path) == AMDCUID_STATUS_SUCCESS &&
              !device_path.empty()) {
            CuidFileEntry entry;
            status = derived_file.find_by_device_node(device_path, entry);
            if (status == AMDCUID_STATUS_SUCCESS) {
              build_derived_id_from_file_entry(entry, id);
              return AMDCUID_STATUS_SUCCESS;
            }
          }
          // Fallback: package_core_id (not unique on SMT, but needed
          // for backward compatibility with old CUID files)
          const auto &info = cpu->get_info();
          std::string core_id =
              std::to_string(info.header.fields.cpu.physical_id) + ":" +
              std::to_string(info.header.fields.cpu.core);
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
        auto nic = reinterpret_cast<CuidNic *>(const_cast<CuidDevice *>(this));
        if (nic) {
          const auto &info = nic->get_info();
          CuidFileEntry entry;
          amdcuid_status_t status =
              derived_file.find_by_device_node(info.network_interface, entry);
          if (status == AMDCUID_STATUS_SUCCESS) {
            build_derived_id_from_file_entry(entry, id);
            return AMDCUID_STATUS_SUCCESS;
          }
        }
      }
      break;
    case AMDCUID_DEVICE_TYPE_NPU:
      // search by accel node
      {
        auto npu = reinterpret_cast<CuidNpu *>(const_cast<CuidDevice *>(this));
        if (npu) {
          const auto &info = npu->get_info();
          CuidFileEntry entry;
          amdcuid_status_t status =
              derived_file.find_by_device_node(info.accel_node, entry);
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
  amdcuid_primary_id primary;
  status = get_primary_cuid(primary);
  // check the temporary bit in the primary CUID to determine whether to use the
  // real HMAC key or the temp key for derived CUID generation
  bool temp = primary.raw_bits[14] &
              0x04; // check the temp indicator bit in the reserved bits
  if (temp) {
    // machine id is what needs to be protected and under HMAC system, message
    // is not guaranteed protection when output is well known so use machine id
    // (recorded in the primary CUID) as the key for generating the derived
    // CUID, and use a fixed application ID as the message to generate a
    // consistent derived CUID for non-privileged users without access to the
    // real HMAC key or primary CUID
    amdcuid_primary_id fixed_app_id = {};
    // Application UUID: UUID_v5(NAMESPACE_DNS, "com.amd.cuid.v1")
    static const uint8_t CUID_APP_UUID[16] = {
        0xac, 0x05, 0xca, 0x9f, 0x1a, 0xc4, 0x58, 0xb9,
        0x92, 0x7e, 0x2e, 0x17, 0x51, 0x47, 0x9c, 0x01};
    memcpy(fixed_app_id.raw_bits, CUID_APP_UUID, 16);
    CuidUtilities::add_UUIDv8_bits(fixed_app_id.raw_bits,
                                   &fixed_app_id.UUIDv8_representation);

    // Use the machine ID from the primary CUID as the key for HMAC, so that the
    // derived CUID is consistent for the same machine even for non-privileged
    // users
    uint8_t padded_key[key_length] = {};
    memcpy(padded_key, primary.raw_bits, sizeof(primary.raw_bits));
    // not using set_key here because we don't want to write to overwrite user
    // key
    cuid_hmac temp_hmac = cuid_hmac(padded_key);
    temp_hmac.set_hmac_algorithm("SHA256");
    status =
        CuidUtilities::generate_derived_cuid(&fixed_app_id, &id, &temp_hmac);
  } else {
    status = CuidUtilities::generate_derived_cuid(&primary, &id, hmac);
  }

  return status;
}

amdcuid_status_t CuidDevice::is_temporary_cuid(bool *is_temp) const {
  if (!is_temp) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }
  *is_temp = false;
  // Check the temporary bit in the primary CUID to determine if the CUID is
  // temporary
  amdcuid_primary_id primary;
  amdcuid_status_t status = get_primary_cuid(primary);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }
  *is_temp = primary.raw_bits[14] &
             0x20; // check the temp indicator bit in the reserved bits

  return AMDCUID_STATUS_SUCCESS;
}
