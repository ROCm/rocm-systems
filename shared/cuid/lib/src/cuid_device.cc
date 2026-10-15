// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cuid_device.h"

#include <unistd.h>

#include <cstring>

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

cuid::DeviceRoute cuid::device_route(const CuidDeviceEntry& entry) {
  return {entry.device_type, entry.bdf, entry.device_node};
}

namespace {

// get_primary_cuid() reports a component that has no permanent primary, and so
// takes its temporary CUID, as HW_FINGERPRINT_NOT_FOUND.
bool no_permanent_primary(amdcuid_status_t status) {
  return status == AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
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

amdcuid_status_t CuidDevice::get_temporary_cuid(amdcuid_derived_id& id) const {
  CuidUtilities::AuxiliaryInput input;
  const amdcuid_status_t status = get_temporary_input(input);
  if (status != AMDCUID_STATUS_SUCCESS) return status;
  return CuidUtilities::temporary_cuid(input, &id);
}

amdcuid_status_t CuidDevice::get_derived_cuid(amdcuid_derived_id& id, cuid_hmac* hmac) const {
  // A key-gated component takes its temporary identity outright when no key
  // is available, and any component does when it has no permanent primary.
  // Without a primary for another reason there is no derived CUID to be had.
  amdcuid_status_t status;
  const bool key_available = hmac && hmac->is_valid();
  if (key_gated_identity() && !key_available) {
    status = get_temporary_cuid(id);
  } else {
    amdcuid_primary_id primary = {};
    status = get_primary_cuid(primary);
    if (status == AMDCUID_STATUS_SUCCESS)
      status = CuidUtilities::generate_derived_cuid(&primary, &id, hmac);
    else if (no_permanent_primary(status))
      status = get_temporary_cuid(id);
  }
  last_source_ = status == AMDCUID_STATUS_SUCCESS ? AMDCUID_SOURCE_LIBRARY : AMDCUID_SOURCE_UNKNOWN;
  return status;
}

amdcuid_status_t CuidDevice::is_temporary_cuid(bool* is_temp, cuid_hmac* hmac) const {
  if (!is_temp) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }
  amdcuid_derived_id derived = {};
  amdcuid_status_t status;
  if (!(key_gated_identity() && !(hmac && hmac->is_valid()))) {
    amdcuid_primary_id primary = {};
    status = get_primary_cuid(primary);
    if (status == AMDCUID_STATUS_SUCCESS) {
      *is_temp = false;
      return AMDCUID_STATUS_SUCCESS;
    }
    if (!no_permanent_primary(status)) return status;
  }
  status = get_temporary_cuid(derived);
  if (status != AMDCUID_STATUS_SUCCESS) return status;
  *is_temp = true;
  return AMDCUID_STATUS_SUCCESS;
}
