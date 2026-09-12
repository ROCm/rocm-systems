// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_DEVICE_MANAGER_H
#define CUID_DEVICE_MANAGER_H

#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "include/amd_cuid.h"
#include "src/cuid_device.h"
#include "src/cuid_file.h"
#include "src/hmac.h"

/**
 * @brief Bitmask set of device types for querying multiple device types.
 *
 * These values are used as bitmasks to specify which device types to include in queries.
 * Each value corresponds to a specific device type, and AMDCUID_DEVICE_TYPE_SET_ALL selects all
 * types.
 */
typedef enum {
  AMDCUID_DEVICE_TYPE_SET_NONE = 0U,  ///< No device types
  AMDCUID_DEVICE_TYPE_SET_PLATFORM =
      1U << AMDCUID_DEVICE_TYPE_PLATFORM,  ///< Platform devices (chassis, motherboard)
  AMDCUID_DEVICE_TYPE_SET_CPU = 1U << AMDCUID_DEVICE_TYPE_CPU,  ///< CPU devices
  AMDCUID_DEVICE_TYPE_SET_GPU = 1U << AMDCUID_DEVICE_TYPE_GPU,  ///< GPU devices
  AMDCUID_DEVICE_TYPE_SET_NIC = 1U << AMDCUID_DEVICE_TYPE_NIC,  ///< NIC devices
  AMDCUID_DEVICE_TYPE_SET_NPU = 1U << AMDCUID_DEVICE_TYPE_NPU,  ///< NPU devices
  AMDCUID_DEVICE_TYPE_SET_ALL = -1U                             ///< All device types
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
  // Mutex for thread-safe access
  mutable std::mutex manager_mutex_;

  static CuidDeviceManager& instance();
  amdcuid_status_t discover_devices();  // should rename to discover_devices() or similar
  amdcuid_status_t shutdown();  // no need for this function as actual shutdown function, may be
                                // useful for unit testing

  // Returns a snapshot by value, deliberately not a reference. discover_devices()
  // replaces devices_ wholesale under manager_mutex_, so a caller iterating a
  // reference to it can have the vector reallocated underneath its iterators by
  // another thread. Copying a vector of shared_ptr is cheap and keeps every
  // device alive for as long as the caller holds the snapshot.
  std::vector<DevicePtr> devices() const {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    return devices_;
  }
  void get_grouped_devices(std::map<amdcuid_device_type_t, std::vector<DevicePtr>>& grouped);
  const amdcuid_device_type_set_t& device_types() const { return device_types_; }

  amdcuid_status_t add_device(DevicePtr device);

  // Share an externally owned cuid_hmac so build_cuid_index() derives CUIDs
  // with the same in-memory key amdcuid_set_hash_key() updates.
  // Not owned; caller must outlive this manager. Pass nullptr to fall back to
  // deriving without an HMAC.
  void set_hmac(cuid_hmac* hmac) { hmac_ = hmac; }

  /**
   * @brief Discover all devices currently present on the system.
   *
   * @return AMDCUID_STATUS_SUCCESS on success, error code otherwise
   */
  amdcuid_status_t get_devices_on_system();

  /**
   * @brief Create devices from CUID file entries.
   * @param[in] cuid_file The CUID file containing device entries.
   *
   * @return AMDCUID_STATUS_SUCCESS on success, error code otherwise
   */
  amdcuid_status_t get_devices_from_file_entries(CuidFile& cuid_file);

  /**
   * @brief Get device from CUID file by derived CUID and add to device list.
   * @param[in] derived_cuid The derived CUID to look for.
   * @param[out] device The device pointer to populate.
   *
   * @return AMDCUID_STATUS_SUCCESS on success, error code otherwise
   */
  amdcuid_status_t get_device_from_file_by_id(amdcuid_id_t& derived_cuid, DevicePtr& device);

  /**
   * @brief Get device from CUID file by device path and add to device list.
   * @param[in] device_path The device path to look for.
   * @param[out] device The device pointer to populate.
   *
   * @return AMDCUID_STATUS_SUCCESS on success, error code otherwise
   */
  amdcuid_status_t get_device_from_file_by_dev_path(const std::string& device_path,
                                                    DevicePtr& device);

  /**
   * @brief Get device from CUID file by BDF and add to device list.
   * @param[in] bdf The BDF to look for.
   * @param[out] device The device pointer to populate.
   *
   * @return AMDCUID_STATUS_SUCCESS on success, error code otherwise
   */
  amdcuid_status_t get_device_from_file_by_bdf(const std::string& bdf, DevicePtr& device);

  /**
   * @brief Request addition of a device by its path and type.
   * @param[in] device_path The device path of the target device.
   * @param[in] device_type The type of the device (see amdcuid_device_type_t).
   * @param[out] device Pointer to the device pointer that will be filled with the requested device.
   *
   * @return AMDCUID_STATUS_SUCCESS on success, error code otherwise
   */
  amdcuid_status_t request_device(const std::string& device_path, amdcuid_device_type_t device_type,
                                  DevicePtr& device);

  /**
   * @brief Request a refresh of the device list from the system.
   *
   * @return AMDCUID_STATUS_SUCCESS on success, error code otherwise
   */
  amdcuid_status_t request_refresh();

  /**
   * @brief Build the CUID index after device discovery.
   */
  void build_cuid_index();

  /**
   * @brief Look up a device by its handle (derived CUID).
   * @param handle The handle containing the derived CUID.
   * @return Pointer to the device, or nullptr if not found.
   */
  DevicePtr lookup_by_handle(const amdcuid_id_t& handle) const;

  /**
   * @brief Get all device handles (derived CUIDs).
   * @return Vector of all handles of devices present on the system.
   */
  std::vector<amdcuid_id_t> get_all_handles() const;

  /**
   * @brief Save the current device registry to the CUID files.
   *
   * @return AMDCUID_STATUS_SUCCESS on success, error code otherwise
   */
  amdcuid_status_t save_registry_to_files();

  /**
   * @brief Discard every derived CUID recorded under the previous seed and
   *        re-record the surviving devices under @p new_key.
   *
   * CuidDevice::get_derived_cuid() consults the record file before deriving, so
   * the records are removed before anything is recomputed; otherwise the
   * regenerated file copies the stale values back and the node serves
   * pre-re-key values indefinitely.
   *
   * @param[in] new_key The 32-octet seed now in effect, or nullptr to drop the
   *                    records without adopting a new one.
   * @return AMDCUID_STATUS_SUCCESS on success, error code otherwise
   */
  amdcuid_status_t invalidate_derived_cuids(const uint8_t new_key[key_length]);

 private:
  CuidDeviceManager() = default;
  ~CuidDeviceManager() = default;
  CuidDeviceManager(const CuidDeviceManager&) = delete;
  CuidDeviceManager& operator=(const CuidDeviceManager&) = delete;

  std::vector<DevicePtr> devices_;
  amdcuid_device_type_set_t device_types_;

  /// Index lookup by derived CUID
  std::map<amdcuid_id_t, DevicePtr, CuidComparator> cuid_index_;

  // Cuid Files
  // The second argument is is_privileged. CuidFile::save() uses it to decide
  // whether to write primary_cuid/hardware_fingerprint and whether to chmod
  // 0600 or 0644, so having these swapped would make the privileged file
  // world-readable and strip the primary CUIDs from it.
  CuidFile unpriv_cuid_file_{CuidUtilities::cuid_file(), false};
  CuidFile priv_cuid_file_{CuidUtilities::priv_cuid_file(), true};

  // Externally owned hmac shared via set_hmac(); see that method's comment.
  cuid_hmac* hmac_ = nullptr;
};

#endif  // CUID_DEVICE_MANAGER_H
