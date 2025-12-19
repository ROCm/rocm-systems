/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

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
    if (device_types & AMDCUID_DEVICE_TYPE_SET_PLATFORM) {
        std::vector<DevicePtr> platforms;
        amdcuid_status_t status = AmdCuidPlatform::discover(platforms);
        if (status != AMDCUID_STATUS_SUCCESS && status != AMDCUID_STATUS_UNSUPPORTED) {
            return status;
        }
        if (status == AMDCUID_STATUS_SUCCESS) {
            devices_.insert(devices_.end(), platforms.begin(), platforms.end());
        }
    }
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
    if (device_types & AMDCUID_DEVICE_TYPE_SET_NIC) {
        std::vector<DevicePtr> nics;
        amdcuid_status_t status = AmdCuidNic::discover(nics);
        if (status != AMDCUID_STATUS_SUCCESS && status != AMDCUID_STATUS_UNSUPPORTED) {
            return status;
        }
        if (status == AMDCUID_STATUS_SUCCESS) {
            devices_.insert(devices_.end(), nics.begin(), nics.end());
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

void AmdCuidDeviceManager::get_grouped_devices(std::map<amdcuid_device_type_t, std::vector<DevicePtr>>& grouped) {
    grouped.clear();
    for (const auto& entry : devices_) {
        grouped[entry->type()].push_back(entry);
    }
}