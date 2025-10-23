#ifndef CUID_NIC_H
#define CUID_NIC_H


#include "cuid_device.h"
#include "cuid.h"
#include "cuid_spec.h"
#include <vector>
#include <memory>
#include <string>

struct amdcuid_nic_info {
    amdcuid_header_nic header;
    std::string bdf;
    std::string network_interface;
};

class CuidNic : public CuidDevice {
public:
    CuidNic(const amdcuid_nic_info& i);
    amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_NIC; }
    amdcuid_status_t get_primary_cuid(amdcuid& id) const override;
    amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const override;
    static amdcuid_status_t discover(std::vector<DevicePtr> &nics);

    const amdcuid_nic_info& get_info() const;
private:
    amdcuid_nic_info m_info;
};

#endif // CUID_NIC_H