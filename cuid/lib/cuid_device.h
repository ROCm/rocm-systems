#ifndef CUID_DEVICE_H
#define CUID_DEVICE_H

#include "cuid.h"
#include <memory>
#include <cstdint>

class CuidDevice {
public:
    virtual ~CuidDevice() = default;
    virtual amdcuid_device_type_t type() const = 0;
    virtual amdcuid_status_t get_primary_cuid(amdcuid& id) const = 0;
    virtual amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const = 0;
    amdcuid_status_t get_secondary_cuid(amdcuid_salt_t salt, amdcuid& id) const;
};

typedef std::shared_ptr<CuidDevice> DevicePtr;

#endif // CUID_DEVICE_H