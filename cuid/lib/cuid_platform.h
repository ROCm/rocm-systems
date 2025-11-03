#ifndef CUID_PLATFORM_H
#define CUID_PLATFORM_H

#include "cuid_device.h"
#include "cuid.h"
#include <vector>
#include <memory>

struct amdcuid_platform_info {
    amdcuid_cuid_fields header;
    // Add more fields as needed
};

class AmdCuidPlatform : public AmdCuidDevice {
public:
    AmdCuidPlatform(const amdcuid_cuid_fields& i);
    amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_PLATFORM; }
    amdcuid_status_t get_primary_cuid(amdcuid& id) const override;
    amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const override;
    static amdcuid_status_t discover(std::vector<DevicePtr> &platforms);

    const amdcuid_platform_info& get_info() const;
private:
    amdcuid_platform_info m_info;
};

#endif // CUID_PLATFORM_H