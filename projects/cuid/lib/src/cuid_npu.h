// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_NPU_H
#define CUID_NPU_H

#include <memory>
#include <string>
#include <vector>

#include "include/amd_cuid.h"
#include "src/cuid_device.h"
#include "src/cuid_internal.h"

struct amdcuid_npu_info {
  amdcuid_cuid_public_fields header;
  // Accel device node: /sys/class/accel/accelN
  std::string accel_node;
  std::string bdf;
};

class CuidNpu : public CuidDevice {
 public:
  CuidNpu(const amdcuid_npu_info& i);
  amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_NPU; }
  amdcuid_status_t get_primary_cuid(amdcuid_primary_id& id) const override;
  amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const override;
  static amdcuid_status_t discover(std::vector<DevicePtr>& npus);
  static amdcuid_status_t discover_single(amdcuid_npu_info* npu_info,
                                          const std::string& device_path);

  // Virtual accessor overrides
  amdcuid_status_t get_vendor_id(uint16_t& vendor_id) const override;
  amdcuid_status_t get_device_id(uint16_t& device_id) const override;
  amdcuid_status_t get_pci_class(uint16_t& pci_class) const override;
  amdcuid_status_t get_revision_id(uint8_t& revision_id) const override;
  amdcuid_status_t get_bdf(std::string& bdf) const override;
  amdcuid_status_t get_device_path(std::string& path) const override;

  const amdcuid_npu_info& get_info() const;

 private:
  amdcuid_npu_info m_info;
};

#endif  // CUID_NPU_H
