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

#include "src/cuid_device.h"
#include "src/cuid_file.h"
#include "include/amd_cuid.h"
#include <vector>
#include <memory>
#include <map>
#include <cstring>

/**
 * @brief Bitmask set of device types for querying multiple device types.
 *
 * These values are used as bitmasks to specify which device types to include in queries.
 * Each value corresponds to a specific device type, and AMDCUID_DEVICE_TYPE_SET_ALL selects all types.
 */
typedef enum {
    AMDCUID_DEVICE_TYPE_SET_NONE      = 0U,                                    ///< No device types
    AMDCUID_DEVICE_TYPE_SET_PLATFORM  = 1U << AMDCUID_DEVICE_TYPE_PLATFORM, ///< Platform devices (chassis, motherboard)
    AMDCUID_DEVICE_TYPE_SET_CPU       = 1U << AMDCUID_DEVICE_TYPE_CPU,      ///< CPU devices
    AMDCUID_DEVICE_TYPE_SET_GPU       = 1U << AMDCUID_DEVICE_TYPE_GPU,      ///< GPU devices
    AMDCUID_DEVICE_TYPE_SET_NIC       = 1U << AMDCUID_DEVICE_TYPE_NIC,      ///< NIC devices
    AMDCUID_DEVICE_TYPE_SET_ALL       = -1U                                    ///< All device types
} amdcuid_device_type_set_t;

/**
 * @brief Comparator for amdcuid_id_t (16-byte array comparison) for use as map key.
 */
struct CuidComparator {
    bool operator()(const amdcuid_id_t& a, const amdcuid_id_t& b) const {
        return std::memcmp(a.bytes, b.bytes, 16) < 0;
    }
};

class CuidDeviceManager {
public:
    static CuidDeviceManager& instance();
    amdcuid_status_t discover_devices(); // should rename to discover_devices() or similar
    amdcuid_status_t shutdown(); // no need for this function as actual shutdown function, may be useful for unti testing though?

    const std::vector<DevicePtr>& devices() const { return devices_; }
    void get_grouped_devices(std::map<amdcuid_device_type_t, std::vector<DevicePtr>>& grouped);
    const amdcuid_device_type_set_t& device_types() const { return device_types_; }

    amdcuid_status_t add_device(const char* dev_path, amdcuid_device_type_t device_type, amdcuid_id_t* handle);
    amdcuid_status_t remove_device(const amdcuid_id_t& handle);

    /**
     * @brief Look up a device by its handle (secondary CUID).
     * @param handle The handle containing the secondary CUID.
     * @return Pointer to the device, or nullptr if not found.
     */
    DevicePtr lookup_by_handle(const amdcuid_id_t& handle) const;

    int get_handle_count() const { return static_cast<int>(cuid_index_.size()); }

    /**
     * @brief Get all device handles (secondary CUIDs).
     * @return Vector of all valid handles.
     */
    std::vector<amdcuid_id_t> get_all_handles() const;

    /**
     * @brief Get handles filtered by device type bitmask.
     * @param types Bitmask of device types to include.
     * @return Vector of handles matching the specified types.
     */
    std::vector<amdcuid_id_t> get_handles_by_type(amdcuid_device_type_set_t types) const;

    /**
     * @brief Check if a handle is valid (device exists in registry).
     * @param handle The handle to validate.
     * @return true if valid, false otherwise.
     */
    bool is_valid_handle(const amdcuid_id_t& handle) const;

private:
    CuidDeviceManager() = default;
    ~CuidDeviceManager() = default;
    CuidDeviceManager(const CuidDeviceManager&) = delete;
    CuidDeviceManager& operator=(const CuidDeviceManager&) = delete;

    /**
     * @brief Build the CUID index after device discovery.
     * Called internally after init() discovers devices.
     */
    void build_cuid_index();

    /**
     * @brief Create devices from CUID file entries.
     * @param[in] cuid_file The CUID file containing device entries.
     * 
     * @return AMDCUID_STATUS_SUCCESS on success, error code otherwise
     */
    amdcuid_status_t get_devices_from_file_entries(CuidFile& cuid_file);

    /**
     * @brief Get device from CUID file by secondary CUID and add to device list.
     * @param[in] secondary_cuid The secondary CUID to look for.
     * 
     * @return AMDCUID_STATUS_SUCCESS on success, error code otherwise
     */
    amdcuid_status_t get_device_from_file_by_id(amdcuid_id_t& secondary_cuid);

    std::vector<DevicePtr> devices_;
    amdcuid_device_type_set_t device_types_;

    /// Index lookup by secondary CUID
    std::map<amdcuid_id_t, DevicePtr, CuidComparator> cuid_index_;

    // Cuid Files
    CuidFile unpriv_cuid_file_{CuidUtilities::cuid_file, true};
    CuidFile priv_cuid_file_{CuidUtilities::priv_cuid_file, false};
};

#endif // CUID_DEVICE_MANAGER_H