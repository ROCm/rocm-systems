#ifndef CUID_PLATFORM_H
#define CUID_PLATFORM_H

#include "cuid_device.h"
#include "cuid.h"
#include <vector>
#include <memory>

class CuidPlatform : public CuidDevice {
public:
    CuidPlatform(const amdcuid_header_platform& i);
    amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_PLATFORM; }
    amdcuid_status_t get_primary_cuid(amdcuid& id) const override;
    amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const override;
    static amdcuid_status_t discover(std::vector<DevicePtr> &platforms);

    const amdcuid_header_platform& get_info() const;
private:
    amdcuid_header_platform m_info;
};

#endif // CUID_PLATFORM_H