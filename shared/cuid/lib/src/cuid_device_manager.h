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
#include "src/hmac.h"

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
  amdcuid_status_t discover_devices();
  amdcuid_status_t shutdown();

  // Returns a snapshot by value, deliberately not a reference. discover_devices()
  // replaces devices_ wholesale under manager_mutex_, so a caller iterating a
  // reference to it can have the vector reallocated underneath its iterators by
  // another thread. Copying a vector of shared_ptr is cheap and keeps every
  // device alive for as long as the caller holds the snapshot.
  std::vector<DevicePtr> devices() const {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    return devices_;
  }
  // Share an externally owned cuid_hmac so build_cuid_index() derives CUIDs
  // with the node key the public API reloads. Not owned; the caller must
  // outlive this manager.
  void set_hmac(cuid_hmac* hmac) { hmac_ = hmac; }

  /**
   * @brief Discover all devices currently present on the system.
   *
   * @return AMDCUID_STATUS_SUCCESS on success, error code otherwise
   */
  amdcuid_status_t get_devices_on_system();

  /**
   * @brief Build the CUID index after device discovery.
   */
  amdcuid_status_t build_cuid_index();
  amdcuid_status_t index_handle(const DevicePtr& device, const amdcuid_id_t& handle);

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

 private:
  CuidDeviceManager() = default;
  ~CuidDeviceManager() = default;
  CuidDeviceManager(const CuidDeviceManager&) = delete;
  CuidDeviceManager& operator=(const CuidDeviceManager&) = delete;

  std::vector<DevicePtr> devices_;

  /// Index lookup by derived CUID
  std::map<amdcuid_id_t, DevicePtr, CuidComparator> cuid_index_;
  std::map<cuid::DeviceRoute, CuidDeviceEntry> observations_;

  // Externally owned hmac shared via set_hmac(); see that method's comment.
  cuid_hmac* hmac_ = nullptr;
};

#endif  // CUID_DEVICE_MANAGER_H
