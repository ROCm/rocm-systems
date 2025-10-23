
#include "cuid_cpu.h"
#include "cuid_util.h"
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>

CuidCpu::CuidCpu(const amdcuid_cpu_info& i)
    : m_info(i)
{}

amdcuid_status_t CuidCpu::discover(std::vector<DevicePtr> &cpus) {
    // This is a stub. Implement as needed for your platform.
    // You may want to parse /proc/cpuinfo and fill amdcuid_cpu_info.
    return AMDCUID_STATUS_UNSUPPORTED;
}

amdcuid_status_t CuidCpu::get_hardware_fingerprint(uint64_t& fingerprint) const {
    // This is a stub. Implement as needed for your platform.
    fingerprint = 0;
    return AMDCUID_STATUS_UNSUPPORTED;
}

amdcuid_status_t CuidCpu::get_primary_cuid(amdcuid& id) const {
    // This is a stub. Implement as needed for your platform.
    std::memset(id.bytes, 0, sizeof(id.bytes));
    return AMDCUID_STATUS_UNSUPPORTED;
}

const amdcuid_cpu_info& CuidCpu::get_info() const {
    return m_info;
}