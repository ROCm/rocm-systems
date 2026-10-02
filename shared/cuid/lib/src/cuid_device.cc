// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cuid_device.h"

#include <unistd.h>

#include <cstring>

#include "cuid_util.h"
#include "rocm/sha2/sha256.h"

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

cuid::DeviceRoute cuid::device_route(const CuidDeviceEntry& entry) {
  return {entry.device_type, entry.bdf, entry.device_node};
}

namespace {

// Is this id an auxiliary (temporary) identifier?
//
// Only a constructed CUID has a payload to read the marker out of. In a
// Platform CUID adopted verbatim from firmware, bit 117 is whatever the
// firmware wrote, so reading it reports roughly half of all machines as
// synthesised. An adopted identifier is a genuine firmware identity, so it is
// never auxiliary.
//
// Templated so the same check works on both amdcuid_primary_id and
// amdcuid_derived_id: both carry raw_bits and UUIDv8_representation at the
// same layout, and get_derived_cuid() checks a primary while
// is_temporary_cuid() checks whichever derived ID it obtained.
template <typename IdT>
bool id_is_auxiliary(const IdT& id) {
  if (!CuidUtilities::is_constructed(&id.UUIDv8_representation)) {
    return false;
  }
  // The Auxiliary Value Identifier, payload bit 117.
  return (id.raw_bits[14] & 0x20) != 0;
}

}  // namespace

amdcuid_status_t CuidDevice::driver_attribute_path(const std::string& attribute,
                                                   std::string& path) const {
  std::string bdf;
  if (this->get_bdf(bdf) != AMDCUID_STATUS_SUCCESS || bdf.empty()) {
    // No BDF: a CPU, the platform, or a GIM-only device that sysfs does not
    // enumerate. There is nothing to look up under /sys/bus/pci/devices.
    return AMDCUID_STATUS_UNSUPPORTED;
  }

  path = "/sys/bus/pci/devices/" + bdf + "/" + attribute;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidDevice::read_driver_published(const std::string& attribute, amdcuid_id_t& out,
                                                   uint8_t raw_bits[16]) const {
  std::string path;
  if (this->driver_attribute_path(attribute, path) != AMDCUID_STATUS_SUCCESS) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }

  amdcuid_id_t published = {};
  const amdcuid_status_t status = CuidUtilities::read_driver_cuid_from_path(path, &published);
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
  // cuid_unit_id is what says the driver publishes a CUID here. A
  // cuid_primary without it comes from a driver with another interface, whose
  // identity this library cannot complete, so the device is treated as
  // unpublished.
  std::string unit_id_path;
  if (driver_attribute_path(CuidUtilities::kDriverUnitIdAttribute, unit_id_path) !=
          AMDCUID_STATUS_SUCCESS ||
      access(unit_id_path.c_str(), F_OK) != 0) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  amdcuid_primary_id published = {};
  const amdcuid_status_t drv = read_driver_published(
      CuidUtilities::kDriverPrimaryAttribute, published.UUIDv8_representation, published.raw_bits);
  if (drv == AMDCUID_STATUS_SUCCESS) {
    id = published;
  }
  return drv;
}

amdcuid_status_t CuidDevice::get_derived_cuid(amdcuid_derived_id& id, cuid_hmac* hmac) const {
  // A key-gated component takes its auxiliary identity outright when no key
  // is available.
  const bool key_available = hmac && hmac->is_valid();
  const bool force_auxiliary = key_gated_identity() && !key_available;

  // Derive. That needs a primary, and without one there is no derived CUID to
  // be had. Do not substitute a zeroed payload with the auxiliary bit set: it
  // holds no per-device input, so every component on this host whose primary
  // lookup failed would collide on one identifier. The kernel takes the same
  // position (amdgpu_cuid.c): with no serial it publishes nothing. A device
  // class that can build an auxiliary identifier does so inside its own
  // get_primary_cuid()/get_auxiliary_primary_cuid().
  amdcuid_primary_id primary = {};
  amdcuid_status_t status =
      force_auxiliary ? get_auxiliary_primary_cuid(primary) : get_primary_cuid(primary);
  if (status == AMDCUID_STATUS_SUCCESS) {
    // An auxiliary primary is derived with the machine-id application key
    // rather than the node key; id_is_auxiliary() reads the marker only where
    // it means something.
    if (id_is_auxiliary(primary)) {
      uint8_t k_app[key_length];
      status = CuidUtilities::temporary_key(k_app);
      if (status == AMDCUID_STATUS_SUCCESS) {
        cuid_hmac temp_hmac(k_app);
        status = CuidUtilities::generate_derived_cuid(&primary, &id, &temp_hmac);
      }
      rocm::sha2::secure_zero(k_app, sizeof(k_app));
    } else {
      status = CuidUtilities::generate_derived_cuid(&primary, &id, hmac);
    }
  }
  last_source_ = status == AMDCUID_STATUS_SUCCESS ? AMDCUID_SOURCE_LIBRARY : AMDCUID_SOURCE_UNKNOWN;
  return status;
}

amdcuid_status_t CuidDevice::is_temporary_cuid(bool* is_temp, cuid_hmac* hmac) const {
  if (!is_temp) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }
  const bool force_auxiliary = key_gated_identity() && !(hmac && hmac->is_valid());
  amdcuid_primary_id primary = {};
  const amdcuid_status_t status =
      force_auxiliary ? get_auxiliary_primary_cuid(primary) : get_primary_cuid(primary);
  if (status != AMDCUID_STATUS_SUCCESS) return status;
  *is_temp = id_is_auxiliary(primary);
  return AMDCUID_STATUS_SUCCESS;
}
