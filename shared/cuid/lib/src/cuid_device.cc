// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cuid_device.h"

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
