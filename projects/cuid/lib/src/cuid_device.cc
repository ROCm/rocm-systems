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
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <openssl/sha.h>
#include <sstream>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#define TEMP_DERIVED_CUID_PUBLIC_KEY                                           \
  "329jf8d+024j1l8i89f4hako1gm2n3-2" // 32-byte key for generating synthetic
                                     // derived CUIDs when primary CUID is not
                                     // available

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
  //
  amdcuid_status_t primary_status = AMDCUID_STATUS_DEVICE_NOT_FOUND;
  if (geteuid() == 0) {
    if (!hmac) {
      // since primary id is available here, should use real key for derived
      // CUID generation for fully function CUID
      return AMDCUID_STATUS_INVALID_ARGUMENT;
    }
    status = get_primary_cuid(primary);

    if (status == AMDCUID_STATUS_SUCCESS) {
      status = CuidUtilities::generate_derived_cuid(&primary, &id, hmac, false);
    }
  }
  if (geteuid() != 0 || status != AMDCUID_STATUS_SUCCESS) {
    // If we can't get the primary ID (e.g. non-privileged user) or the
    // generation failed for some reason, fallback to using the temp generation
    // method
    uint64_t fingerprint = 0;

    // if the device is a pci, use bdf for fingerprint
    if (this->type() == AMDCUID_DEVICE_TYPE_GPU ||
        this->type() == AMDCUID_DEVICE_TYPE_NIC ||
        this->type() == AMDCUID_DEVICE_TYPE_NPU) {
      std::string bdf;
      status = this->get_bdf(bdf);
      if (status != AMDCUID_STATUS_SUCCESS) {
        return status;
      }
      std::string bdf_hex;
      std::copy_if(bdf.begin(), bdf.end(), std::back_inserter(bdf_hex),
                   [](unsigned char c) { return std::isxdigit(c); });
      if (bdf_hex.empty()) {
        return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
      }
      fingerprint = std::stoull(bdf_hex, nullptr, 16);

    } else {
      // for non-pci devices, use machine id
      std::ifstream machine_id_file("/etc/machine-id");
      if (!machine_id_file.is_open()) {
        return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
      }
      std::string machine_id;
      std::getline(machine_id_file, machine_id);
      machine_id_file.close();
      std::string machine_id_hex = machine_id.substr(0, 16);
      if (machine_id_hex.empty()) {
        return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
      }
      try {
        fingerprint = std::stoull(machine_id_hex, nullptr, 16);
      } catch (...) {
        return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
      }
    }

    // create primary ID with fingerprint
    uint16_t unit_id = 0;
    this->get_unit_id(unit_id);
    uint8_t revision_id = 0;
    this->get_revision_id(revision_id);
    uint16_t device_id = 0;
    this->get_device_id(device_id);
    uint16_t vendor_id = 0;
    this->get_vendor_id(vendor_id);
    status = CuidUtilities::generate_primary_cuid(
        fingerprint, unit_id, revision_id, device_id, vendor_id, this->type(),
        &primary);

    // use a fixed hmac context with a hardcoded public key since we can't rely
    // on the caller to provide a valid HMAC context without access to the key,
    // and we just need something consistent for the derived CUID generation
    // since we don't have access to the real primary CUID
    uint8_t temp_key[key_length];
    static_assert(sizeof(TEMP_DERIVED_CUID_PUBLIC_KEY) - 1 == key_length,
                  "TEMP_DERIVED_CUID_PUBLIC_KEY length must match key_length");
    std::memcpy(temp_key, TEMP_DERIVED_CUID_PUBLIC_KEY, key_length);

    cuid_hmac temp_hmac = cuid_hmac(temp_key);
    temp_hmac.set_hmac_algorithm(EVP_sha256());

    // Use hardware fingerprint and device type to generate a synthetic primary
    // ID for derived CUID generation
    status =
        CuidUtilities::generate_derived_cuid(&primary, &id, &temp_hmac, true);
  }

  return status;
}
