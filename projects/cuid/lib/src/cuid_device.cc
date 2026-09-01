// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cuid_device.h"

#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <vector>

#include "cuid_cpu.h"
#include "cuid_file.h"
#include "cuid_gpu.h"
#include "cuid_nic.h"
#include "cuid_npu.h"
#include "cuid_platform.h"
#include "cuid_util.h"

namespace cuid {

// helper function to get a hash from the raw bytes of a derived ID
void get_hash_from_raw(uint8_t raw_bytes[16], uint8_t out_hash[14]) {
  // just remove the reserved bits from the raw bytes to get the hash
  memcpy(out_hash, raw_bytes, 8);

  // byte 8 of raw bits is reserved which we can skip
  memcpy(&out_hash[8], &raw_bytes[9], 5);
  // The derived slot is 45 bits, so byte 14 carries only 5 hash bits: bit 5 is
  // the Auxiliary Value Identifier (payload bit 117) and bits 6:7 are reserved.
  // Masking 6 bits instead of 5 would fold the auxiliary marker into the hash.
  out_hash[13] = raw_bytes[14] & 0x1F;
}

}  // namespace cuid

namespace {

void build_derived_id_from_file_entry(const CuidFileEntry& entry, amdcuid_derived_id& id) {
  id.UUIDv8_representation = entry.derived_cuid;
  CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);
  cuid::get_hash_from_raw(id.raw_bits, id.hash);
}

// Is this primary an auxiliary (temporary) identifier?
//
// Only a constructed CUID has a payload to read the marker out of. In a
// Platform CUID adopted verbatim from firmware, bit 117 is whatever the
// firmware wrote, so reading it reports roughly half of all machines as
// synthesised. An adopted identifier is a genuine firmware identity, so it is
// never auxiliary.
//
// One definition, shared by get_derived_cuid(), which picks the derivation key
// from it, and is_temporary_cuid(), which reports it.
bool primary_is_auxiliary(const amdcuid_primary_id& primary) {
  if (!CuidUtilities::is_constructed(&primary.UUIDv8_representation)) {
    return false;
  }
  // The Auxiliary Value Identifier, payload bit 117.
  return (primary.raw_bits[14] & 0x20) != 0;
}

}  // namespace

amdcuid_status_t CuidDevice::read_driver_published(const std::string& attribute, amdcuid_id_t& out,
                                                   uint8_t raw_bits[16]) const {
  std::string bdf;
  if (this->get_bdf(bdf) != AMDCUID_STATUS_SUCCESS || bdf.empty()) {
    // No BDF: a CPU, the platform, or a GIM-only device that sysfs does not
    // enumerate. There is nothing to look up under /sys/bus/pci/devices.
    return AMDCUID_STATUS_UNSUPPORTED;
  }

  amdcuid_id_t published = {};
  const amdcuid_status_t status = CuidUtilities::read_driver_cuid(bdf, attribute, &published);
  switch (status) {
    case AMDCUID_STATUS_SUCCESS:
      out = published;
      CuidUtilities::remove_UUIDv8_bits(&out, raw_bits);
      return AMDCUID_STATUS_SUCCESS;
    case AMDCUID_STATUS_FILE_NOT_FOUND:
      // The driver does not implement the CUID interface, or found no serial
      // and so created none of the attributes. No kernel value to defer to.
      return AMDCUID_STATUS_UNSUPPORTED;
    default:
      return status;
  }
}

amdcuid_status_t CuidDevice::driver_primary_cuid(amdcuid_primary_id& id) const {
  amdcuid_primary_id published = {};
  const amdcuid_status_t drv = read_driver_published(
      CuidUtilities::kDriverPrimaryAttribute, published.UUIDv8_representation, published.raw_bits);
  if (drv == AMDCUID_STATUS_SUCCESS) {
    id = published;
  }
  return drv;
}

amdcuid_status_t CuidDevice::get_derived_cuid(amdcuid_derived_id& id, cuid_hmac* hmac) const {
  // cuid_secondary is 0444, so this stage answers for an unprivileged caller
  // even where cuid_primary does not: an ordinary user must get the kernel's
  // value rather than falling through and deriving a competing one.
  {
    amdcuid_derived_id published = {};
    const amdcuid_status_t drv =
        read_driver_published(CuidUtilities::kDriverSecondaryAttribute,
                              published.UUIDv8_representation, published.raw_bits);
    if (drv == AMDCUID_STATUS_SUCCESS) {
      cuid::get_hash_from_raw(published.raw_bits, published.hash);
      id = published;
      return AMDCUID_STATUS_SUCCESS;
    }
    if (drv != AMDCUID_STATUS_UNSUPPORTED) {
      return drv;
    }
  }

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
        status = derived_file.find_by_device_type(AMDCUID_DEVICE_TYPE_PLATFORM, entry);
        if (status == AMDCUID_STATUS_SUCCESS) {
          build_derived_id_from_file_entry(entry, id);
          return AMDCUID_STATUS_SUCCESS;
        }
      } break;
      case AMDCUID_DEVICE_TYPE_GPU: {
        // search by render node
        const auto& info = static_cast<const CuidGpu*>(this)->get_info();
        CuidFileEntry entry;
        status = derived_file.find_by_device_node(info.render_node, entry);
        if (status == AMDCUID_STATUS_SUCCESS) {
          build_derived_id_from_file_entry(entry, id);
          return AMDCUID_STATUS_SUCCESS;
        }
      } break;
      case AMDCUID_DEVICE_TYPE_CPU: {
        const auto* cpu = static_cast<const CuidCpu*>(this);
        // Try device_node first - unique per logical CPU on SMT systems
        std::string device_path;
        if (cpu->get_device_path(device_path) == AMDCUID_STATUS_SUCCESS && !device_path.empty()) {
          CuidFileEntry entry;
          status = derived_file.find_by_device_node(device_path, entry);
          if (status == AMDCUID_STATUS_SUCCESS) {
            build_derived_id_from_file_entry(entry, id);
            return AMDCUID_STATUS_SUCCESS;
          }
        }
        const auto& info = cpu->get_info();
        CuidFileEntry entry;
        status = derived_file.find_by_package_id(info.header.fields.cpu.physical_id, entry);
        if (status == AMDCUID_STATUS_SUCCESS) {
          build_derived_id_from_file_entry(entry, id);
          return AMDCUID_STATUS_SUCCESS;
        }
      } break;
      case AMDCUID_DEVICE_TYPE_NIC: {
        // search by device node
        const auto& info = static_cast<const CuidNic*>(this)->get_info();
        CuidFileEntry entry;
        status = derived_file.find_by_device_node(info.network_interface, entry);
        if (status == AMDCUID_STATUS_SUCCESS) {
          build_derived_id_from_file_entry(entry, id);
          return AMDCUID_STATUS_SUCCESS;
        }
      } break;
      case AMDCUID_DEVICE_TYPE_NPU: {
        // search by accel node
        const auto& info = static_cast<const CuidNpu*>(this)->get_info();
        CuidFileEntry entry;
        status = derived_file.find_by_device_node(info.accel_node, entry);
        if (status == AMDCUID_STATUS_SUCCESS) {
          build_derived_id_from_file_entry(entry, id);
          return AMDCUID_STATUS_SUCCESS;
        }
      } break;
      // No other Component Type has a device class here yet, so there is no
      // record entry to look up; the derivation below answers for them.
      default:
        break;
    }
  }

  // Nothing published, nothing recorded: derive. That needs a primary, and
  // without one there is no derived CUID to be had. Do not substitute a zeroed
  // payload with the auxiliary bit set: it holds no per-device input, so every
  // component whose primary lookup failed would HMAC the same zero octets with
  // the fixed public temporary key and collide on one identifier. The kernel
  // takes the same position (amdgpu_cuid.c): with no serial it publishes
  // nothing. A device class that can build an auxiliary identifier does so
  // inside its own get_primary_cuid().
  amdcuid_primary_id primary = {};
  status = get_primary_cuid(primary);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }

  // An auxiliary primary is derived with the fixed public temporary key rather
  // than the node key; primary_is_auxiliary() reads the marker only where it
  // means something.
  if (primary_is_auxiliary(primary)) {
    // Same operand order as every other derivation: the key is the constant,
    // the message is the 16 auxiliary primary octets. Do not swap them to
    // protect the machine ID in the primary. HMAC with a public key is a keyed
    // hash whose preimage resistance covers the message either way, and
    // generate_derived_cuid() reads bit 117 out of whatever it is handed as the
    // primary, so a fixed constant there leaves the derived value unmarked.
    cuid_hmac temp_hmac(kTemporaryKey, kTemporaryKeyLen);
    status = temp_hmac.set_hmac_algorithm("SHA256");
    if (status != AMDCUID_STATUS_SUCCESS) return status;
    status = CuidUtilities::generate_derived_cuid(&primary, &id, &temp_hmac);
  } else {
    status = CuidUtilities::generate_derived_cuid(&primary, &id, hmac);
  }

  return status;
}

amdcuid_status_t CuidDevice::is_temporary_cuid(bool* is_temp) const {
  if (!is_temp) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }
  amdcuid_primary_id primary = {};
  amdcuid_status_t status = get_primary_cuid(primary);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }

  *is_temp = primary_is_auxiliary(primary);

  return AMDCUID_STATUS_SUCCESS;
}
