// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cuid_platform.h"

#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>

#include "cuid_file.h"
#include "cuid_util.h"
#include "smbios_util.h"

CuidPlatform::CuidPlatform(const amdcuid_platform_info& i) : m_info({i}) {}

amdcuid_status_t CuidPlatform::discover(std::vector<DevicePtr>& platforms) {
  // Platform is a singleton - only one platform per system
  amdcuid_platform_info info = {};
  info.header.device_type = AMDCUID_DEVICE_TYPE_PLATFORM;

  std::string vendor, name, version;
  amdcuid_status_t status = SmbiosUtil::get_board_info(vendor, name, version);
  if (status != AMDCUID_STATUS_SUCCESS) {
    // FILE NOT FOUND status given here, which means smbios info not available
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  uint16_t vendor_id = (uint16_t)strtol(vendor.c_str(), nullptr, 16);
  info.header.fields.platform.vendor_id = vendor_id;

  // Create platform device
  platforms.emplace_back(std::make_shared<CuidPlatform>(info));

  return AMDCUID_STATUS_SUCCESS;
}

bool CuidPlatform::get_system_uuid(uint8_t uuid[16]) {
  if (SmbiosUtil::get_system_uuid(uuid) != AMDCUID_STATUS_SUCCESS) return false;
  // Reject the "not set" sentinels firmware writes when it has no UUID.
  for (int i = 0; i < 16; ++i) {
    if (uuid[i] != 0x00 && uuid[i] != 0xFF) return true;
  }
  return false;
}

amdcuid_status_t CuidPlatform::get_hardware_fingerprint(uint64_t& fingerprint) const {
  // Only reached when the platform has no system UUID: where one exists the
  // Platform CUID is that UUID verbatim, with no 64-bit fingerprint at all (see
  // get_primary_cuid()). This is the specification's else-branch, the system
  // serial through the normal 122-bit layout. Hashed rather than packed as
  // ASCII, so a serial longer than 8 characters is not truncated to its first
  // 8, which on a fleet with a common prefix gives every machine one value.
  std::string serial;
  if (SmbiosUtil::get_system_serial(serial) != AMDCUID_STATUS_SUCCESS || serial.empty()) {
    fingerprint = 0;
    return AMDCUID_STATUS_UNSUPPORTED;
  }

  uint8_t digest[32];
  const amdcuid_status_t status = CuidUtilities::sha256_unkeyed(
      reinterpret_cast<const uint8_t*>(serial.data()), serial.size(), digest);
  if (status != AMDCUID_STATUS_SUCCESS) {
    fingerprint = 0;
    return status;
  }

  fingerprint = 0;
  for (size_t i = 0; i < sizeof(fingerprint); ++i) {
    fingerprint |= static_cast<uint64_t>(digest[i]) << (8 * i);
  }
  return AMDCUID_STATUS_SUCCESS;
}

namespace {

// Populate raw_bits for a Platform primary, which is the HMAC message, so the
// two constructions need different treatment.
//
// An adopted firmware UUID is not a framed 122-bit payload: remove_UUIDv8_bits()
// on it drops six bits and shifts the rest, so two platforms whose UUIDs differ
// only in the version and variant bits de-frame alike and derive the same
// secondary CUID. It has to be the sixteen octets as firmware wrote them. A
// serial-derived primary is framed as a UUIDv8, so its raw_bits are the
// de-framed payload; copying the framed octets there gives a platform read back
// from the record a different HMAC message from the same platform computed
// fresh. is_constructed() discriminates on the version nibble.
void set_platform_raw_bits(amdcuid_primary_id& id) {
  if (CuidUtilities::is_constructed(&id.UUIDv8_representation)) {
    CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);
    return;
  }
  std::memcpy(id.raw_bits, id.UUIDv8_representation.bytes, sizeof(id.raw_bits));
}

}  // namespace

amdcuid_status_t CuidPlatform::get_primary_cuid(amdcuid_primary_id& id) const {
  // attempt to find the primary CUID in file first
  std::string cuid_file_path = CuidUtilities::priv_cuid_file();
  CuidFile primary_file(cuid_file_path, false);
  primary_file.load();
  std::vector<CuidFileEntry> entries = primary_file.get_entries();

  // for platform, just return the first entry found
  CuidFileEntry entry;
  amdcuid_status_t status = primary_file.find_by_device_type(AMDCUID_DEVICE_TYPE_PLATFORM, entry);
  if (status == AMDCUID_STATUS_SUCCESS) {
    id.UUIDv8_representation = entry.primary_cuid;
    set_platform_raw_bits(id);
    return AMDCUID_STATUS_SUCCESS;
  }

  // Where the firmware supplies a system UUID, the Platform CUID is those 16
  // octets used directly: no reframing, no component type, no vendor field, no
  // fold. Collapsing the UUID to its first 8 octets and packing it through the
  // normal layout discards half of an identifier firmware had already made
  // unique, and makes the result depend on which producer folded it.
  uint8_t system_uuid[16];
  if (get_system_uuid(system_uuid)) {
    std::memcpy(id.UUIDv8_representation.bytes, system_uuid, sizeof(system_uuid));
    set_platform_raw_bits(id);
    return AMDCUID_STATUS_SUCCESS;
  }

  // No system UUID: the system serial through the normal layout, or no CUID.
  uint64_t fingerprint = 0;
  status = get_hardware_fingerprint(fingerprint);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }

  // Component Type 0x0, and UnitID, Revision, Device and Vendor all zero: the
  // platform is not a PCI function, so it has no vendor or device ID. The
  // SMBIOS vendor string names whoever built the machine rather than the
  // machine, and packing it makes two producers disagree on the same platform.
  return CuidUtilities::generate_primary_cuid(fingerprint, 0, 0, 0, 0, AMDCUID_DEVICE_TYPE_PLATFORM,
                                              &id, false);
}

const amdcuid_platform_info& CuidPlatform::get_info() const { return m_info; }

amdcuid_status_t CuidPlatform::get_vendor_id(uint16_t& vendor_id) const {
  vendor_id = m_info.header.fields.platform.vendor_id;
  return AMDCUID_STATUS_SUCCESS;
}
