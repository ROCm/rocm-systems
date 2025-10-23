#ifndef CUID_CPU_H
#define CUID_CPU_H


#include "cuid_device.h"
#include "cuid.h"
#include "cuid_spec.h"
#include <vector>
#include <memory>
#include <string>

struct amdcuid_cpu_info {
    amdcuid_header_cpu header;
    // Add more fields as needed (e.g., model name, apic id, etc.)
};

class CuidCpu : public CuidDevice {
public:
    CuidCpu(const amdcuid_cpu_info& i);
    amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_CPU; }
    amdcuid_status_t get_primary_cuid(amdcuid& id) const override;
    amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const override;
    static amdcuid_status_t discover(std::vector<DevicePtr> &cpus);

    const amdcuid_cpu_info& get_info() const;
private:
    amdcuid_cpu_info m_info;
};

#endif // CUID_CPU_H
