// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_PLATFORM_H
#define CUID_PLATFORM_H

#include <memory>
#include <vector>

#include "cuid_device.h"
#include "cuid_internal.h"
#include "include/amd_cuid.h"

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

  // The SMBIOS system UUID, if the firmware supplies a real one. Returns false
  // when it is absent or an all-0x00 / all-0xFF sentinel. When present, it is
  // the Platform CUID, used verbatim rather than folded.
  static bool get_system_uuid(uint8_t uuid[16]);
  static amdcuid_status_t discover(std::vector<DevicePtr>& platforms);

  // Virtual accessor overrides
  amdcuid_status_t get_vendor_id(uint16_t& vendor_id) const override;

  const amdcuid_platform_info& get_info() const;

 private:
  amdcuid_platform_info m_info;
};

#endif  // CUID_PLATFORM_H
