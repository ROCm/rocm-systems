#ifndef CUID_GPU_H
#define CUID_GPU_H


#include "cuid_device.h"
#include "cuid.h"
#include "cuid_spec.h"
#include <vector>
#include <memory>
#include <string>

struct amdcuid_gpu_info {
    amdcuid_header_gpu header;
    std::string render_node;
    std::string bdf;
};

class CuidGpu : public CuidDevice {
public:
    CuidGpu(const amdcuid_gpu_info& i);
    amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_GPU; }
    amdcuid_status_t get_primary_cuid(amdcuid& id) const override;
    amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const override;
    static amdcuid_status_t discover(std::vector<DevicePtr> &gpus);

    const amdcuid_gpu_info& get_info() const;
private:
    amdcuid_gpu_info m_info;
};

#endif // CUID_GPU_H