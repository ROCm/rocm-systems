#ifndef CUID_DEVICE_MANAGER_H
#define CUID_DEVICE_MANAGER_H

#include "cuid_device.h"
#include "cuid.h"
#include <vector>
#include <memory>

class CuidDeviceManager {
public:
    static CuidDeviceManager& instance();
    amdcuid_status_t init(amdcuid_device_type_set_t device_types);
    amdcuid_status_t shutdown();

    const std::vector<DevicePtr>& devices() const { return devices_; }

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
    CuidDeviceManager() : initialized_(false) {}
    ~CuidDeviceManager() = default;
    CuidDeviceManager(const CuidDeviceManager&) = delete;
    CuidDeviceManager& operator=(const CuidDeviceManager&) = delete;

    std::vector<DevicePtr> devices_;
    bool initialized_;
};

#endif // CUID_DEVICE_MANAGER_H