// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_CPU_H
#define CUID_CPU_H


#include "src/cuid_device.h"
#include "include/amd_cuid.h"
#include "src/cuid_internal.h"
#include <vector>
#include <memory>
#include <string>

struct amdcuid_cpu_info {
    amdcuid_cuid_public_fields header;
    std::string device_node;  // sysfs path e.g. /sys/devices/system/cpu/cpu0
};

class CuidCpu : public CuidDevice {
public:
    CuidCpu(const amdcuid_cpu_info& i);
    amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_CPU; }
    amdcuid_status_t get_primary_cuid(amdcuid_primary_id& id) const override;
    amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const override;
    static amdcuid_status_t discover(std::vector<DevicePtr> &cpus);
    static amdcuid_status_t discover_single(amdcuid_cpu_info* cpu_info, const std::string& device_path);

    // Virtual accessor overrides
    amdcuid_status_t get_vendor_id(uint16_t& vendor_id) const override;
    amdcuid_status_t get_family(uint16_t& family) const override;
    amdcuid_status_t get_model(uint16_t& model) const override;
    amdcuid_status_t get_device_id(uint16_t& device_id) const override;
    amdcuid_status_t get_revision_id(uint8_t& revision_id) const override;
    amdcuid_status_t get_unit_id(uint16_t& unit_id) const override;
    amdcuid_status_t get_core(uint16_t& core) const override;
    amdcuid_status_t get_physical_id(uint16_t& physical_id) const override;
    amdcuid_status_t get_device_path(std::string& path) const override;

    const amdcuid_cpu_info& get_info() const;
private:
    amdcuid_cpu_info m_info;
};

#endif // CUID_CPU_H
