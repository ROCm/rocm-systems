#include "cuid_device_manager.h"
#include "cuid_gpu.h"
#include "cuid_cpu.h"
#include "cuid_nic.h"
#include "cuid_platform.h"
#include "cuid.h"
#include <iostream>


amdcuid_status_t AmdCuidDeviceManager::init(amdcuid_device_type_set_t device_types) {
    devices_.clear();
    initialized_ = false;
    if (device_types & AMDCUID_DEVICE_TYPE_SET_GPU) {
        std::vector<DevicePtr> gpus;
        amdcuid_status_t status = AmdCuidGpu::discover(gpus);
        if (status != AMDCUID_STATUS_SUCCESS) {
            return status;
        }
        devices_.insert(devices_.end(), gpus.begin(), gpus.end());
    }
    if (device_types & AMDCUID_DEVICE_TYPE_SET_CPU) {
        std::vector<DevicePtr> cpus;
        amdcuid_status_t status = AmdCuidCpu::discover(cpus);
        if (status != AMDCUID_STATUS_SUCCESS && status != AMDCUID_STATUS_UNSUPPORTED) {
            return status;
        }
        if (status == AMDCUID_STATUS_SUCCESS) {
            devices_.insert(devices_.end(), cpus.begin(), cpus.end());
        }
    }
    initialized_ = true;
    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t AmdCuidDeviceManager::shutdown() {
    devices_.clear();
    initialized_ = false;
    return AMDCUID_STATUS_SUCCESS;
}

AmdCuidDeviceManager& AmdCuidDeviceManager::instance() {
    static AmdCuidDeviceManager instance;
    return instance;
}