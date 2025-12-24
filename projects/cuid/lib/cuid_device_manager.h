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

#ifndef CUID_DEVICE_MANAGER_H
#define CUID_DEVICE_MANAGER_H

#include "cuid_device.h"
#include "cuid.h"
#include <vector>
#include <memory>
#include <map>

class AmdCuidDeviceManager {
public:
    static AmdCuidDeviceManager& instance();
    amdcuid_status_t init(amdcuid_device_type_set_t device_types);
    amdcuid_status_t shutdown();

    const std::vector<DevicePtr>& devices() const { return devices_; }
    void get_grouped_devices(std::map<amdcuid_device_type_t, std::vector<DevicePtr>>& grouped);
    const amdcuid_device_type_set_t& device_types() const { return device_types_; }

    template <class T>
    T* get_device_by_handle(void* handle) const {
        for (const auto& device : devices_) {
            if (reinterpret_cast<void*>(device.get()) == handle) {
                return dynamic_cast<T*>(device.get());
            }
        }
        return nullptr;
    }

    bool is_initialized() const { return initialized_; }

private:
    AmdCuidDeviceManager() : initialized_(false) {}
    ~AmdCuidDeviceManager() = default;
    AmdCuidDeviceManager(const AmdCuidDeviceManager&) = delete;
    AmdCuidDeviceManager& operator=(const AmdCuidDeviceManager&) = delete;

    std::vector<DevicePtr> devices_;
    amdcuid_device_type_set_t device_types_;
    bool initialized_;
};

#endif // CUID_DEVICE_MANAGER_H