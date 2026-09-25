// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "src/cuid_device_manager.h"

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <iostream>

#include "include/amd_cuid.h"
#include "src/cuid_cpu.h"
#include "src/cuid_gpu.h"
#include "src/cuid_nic.h"
#include "src/cuid_npu.h"
#include "src/cuid_platform.h"
#include "src/cuid_util.h"

amdcuid_status_t CuidDeviceManager::get_devices_on_system() {
  std::lock_guard<std::recursive_mutex> operation(cuid_operation_mutex());
  std::vector<DevicePtr> discovered_devices;
  const struct {
    const char* name;
    amdcuid_status_t (*discover)(std::vector<DevicePtr>&);
  } subsystems[] = {{"platform", CuidPlatform::discover},
                    {"CPU", CuidCpu::discover},
                    {"GPU", CuidGpu::discover},
                    {"NIC", CuidNic::discover},
                    {"NPU", CuidNpu::discover}};
  for (const auto& subsystem : subsystems) {
    std::vector<DevicePtr> found;
    const auto status = subsystem.discover(found);
    // Absence of an entire subsystem is expected. A returned component whose
    // identity cannot be constructed is handled separately by the strict index.
    if (found.empty() &&
        (status == AMDCUID_STATUS_SUCCESS || status == AMDCUID_STATUS_UNSUPPORTED ||
         status == AMDCUID_STATUS_DEVICE_NOT_FOUND)) {
      LOG(DEBUG, "CUID discovery: no " << subsystem.name << " components ("
                                       << amdcuid_status_to_string(status) << ")");
      continue;
    }
    if (status != AMDCUID_STATUS_SUCCESS) {
      shutdown();
      return status;
    }
    discovered_devices.insert(discovered_devices.end(), found.begin(), found.end());
  }
  std::lock_guard<std::mutex> lock(manager_mutex_);
  devices_ = std::move(discovered_devices);
  cuid_index_.clear();
  observations_.clear();
  return devices_.empty() ? AMDCUID_STATUS_DEVICE_NOT_FOUND : AMDCUID_STATUS_SUCCESS;
}

namespace {

CuidDeviceEntry device_entry(const DevicePtr& device) {
  CuidDeviceEntry entry;
  entry.device_type = device->type();
  device->get_bdf(entry.bdf);
  device->get_device_path(entry.device_node);
  device->get_vendor_id(entry.vendor_id);
  device->get_device_id(entry.device_id);
  device->get_revision_id(entry.revision_id);
  uint16_t unit = 0;
  if (device->type() == AMDCUID_DEVICE_TYPE_NIC)
    entry.unit_id = 0;
  else if (device->get_unit_id(unit) == AMDCUID_STATUS_SUCCESS)
    entry.unit_id = unit;
  return entry;
}

bool same_gpu_object(const DevicePtr& a, const DevicePtr& b) {
  const auto left = std::dynamic_pointer_cast<CuidGpu>(a);
  const auto right = std::dynamic_pointer_cast<CuidGpu>(b);
  return left && right && left->same_device(*right);
}

}  // namespace

amdcuid_status_t CuidDeviceManager::index_handle(const DevicePtr& device,
                                                 const amdcuid_id_t& handle) {
  std::lock_guard<std::recursive_mutex> operation(cuid_operation_mutex());
  auto route = cuid::device_route(device_entry(device));
  std::unique_lock<std::mutex> lock(manager_mutex_);
  auto old_devices = devices_;
  auto old_index = cuid_index_;
  auto old_observations = observations_;
  {
    auto found = std::find_if(devices_.begin(), devices_.end(), [&](const auto& existing) {
      return cuid::device_route(device_entry(existing)) == route ||
             same_gpu_object(existing, device);
    });
    if (found == devices_.end())
      devices_.push_back(device);
    else if (same_gpu_object(*found, device))
      route = cuid::device_route(device_entry(*found));
    else
      *found = device;
  }
  lock.unlock();
  // Reindex all routes together: refreshing one NIC function must not leave
  // its sibling under an old handle or change the canonical representative.
  auto status = build_cuid_index();
  lock.lock();
  if (status == AMDCUID_STATUS_SUCCESS) {
    const auto found = observations_.find(route);
    if (found == observations_.end() ||
        std::memcmp(found->second.derived_cuid.bytes, handle.bytes, 16) != 0)
      status = AMDCUID_STATUS_DEVICE_NOT_FOUND;
  }
  if (status != AMDCUID_STATUS_SUCCESS) {
    devices_ = std::move(old_devices);
    cuid_index_ = std::move(old_index);
    observations_ = std::move(old_observations);
  }
  return status;
}

amdcuid_status_t CuidDeviceManager::discover_devices() {
  std::lock_guard<std::recursive_mutex> operation(cuid_operation_mutex());
  const auto status = get_devices_on_system();
  if (status != AMDCUID_STATUS_SUCCESS) return status;
  return build_cuid_index();
}

amdcuid_status_t CuidDeviceManager::shutdown() {
  std::lock_guard<std::mutex> lock(manager_mutex_);
  devices_.clear();
  cuid_index_.clear();
  observations_.clear();
  return AMDCUID_STATUS_SUCCESS;
}

CuidDeviceManager& CuidDeviceManager::instance() {
  static CuidDeviceManager instance;
  return instance;
}

amdcuid_status_t CuidDeviceManager::build_cuid_index() {
  std::lock_guard<std::recursive_mutex> operation(cuid_operation_mutex());
  std::lock_guard<std::mutex> lock(manager_mutex_);
  cuid_index_.clear();
  observations_.clear();
  std::map<amdcuid_id_t, DevicePtr, CuidComparator> index;
  std::map<cuid::DeviceRoute, CuidDeviceEntry> observations;
  std::map<amdcuid_id_t, CuidDeviceEntry, CuidComparator> representatives;
  amdcuid_status_t unavailable = AMDCUID_STATUS_DEVICE_NOT_FOUND;
  for (const auto& device : devices_) {
    if (!device) return AMDCUID_STATUS_INVALID_ARGUMENT;
    // A component with no identity at all, such as an auxiliary one on a host
    // without a machine identity, is left out. It must not fail the refresh
    // for every other component.
    const auto skip = [&](amdcuid_status_t status) {
      LOG(INFO, "CUID reader: unavailable identity for component type "
                    << device->type() << " (" << amdcuid_status_to_string(status) << ")");
      unavailable = status;
    };
    amdcuid_primary_id primary{};
    // Only the NIC alias rule needs a primary.
    if (geteuid() == 0 && device->type() == AMDCUID_DEVICE_TYPE_NIC) {
      const auto status = device->get_primary_cuid(primary);
      if (status == AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND) {
        skip(status);
        continue;
      }
      // Root without CAP_SYS_ADMIN cannot read a driver-published primary;
      // the derived value below still comes from the driver.
      if (status != AMDCUID_STATUS_SUCCESS &&
          !(status == AMDCUID_STATUS_PERMISSION_DENIED && !CuidUtilities::has_cap_sys_admin()))
        return status;
    }
    amdcuid_derived_id derived{};
    const auto status = device->get_derived_cuid(derived, geteuid() == 0 ? hmac_ : nullptr);
    if (status != AMDCUID_STATUS_SUCCESS) {
      if (status != AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND &&
          status != AMDCUID_STATUS_UNSUPPORTED && status != AMDCUID_STATUS_PERMISSION_DENIED &&
          !(geteuid() != 0 && status == AMDCUID_STATUS_INVALID_ARGUMENT))
        return status;
      skip(status);
      continue;
    }
    auto entry = device_entry(device);
    entry.primary_cuid = primary.UUIDv8_representation;
    entry.derived_cuid = derived.UUIDv8_representation;
    entry.is_temporary =
        CuidUtilities::is_constructed(&entry.derived_cuid) && (derived.raw_bits[14] & 0x20);
    if (!observations.emplace(cuid::device_route(entry), entry).second)
      return AMDCUID_STATUS_INVALID_FORMAT;
    auto inserted = index.emplace(entry.derived_cuid, device);
    if (!inserted.second) {
      auto& previous = representatives.at(entry.derived_cuid);
      if (!cuid::nic_component_alias(previous, entry, geteuid() == 0))
        return AMDCUID_STATUS_INVALID_FORMAT;
      if (cuid::device_route(entry) < cuid::device_route(previous)) {
        previous = entry;
        inserted.first->second = device;
      }
    } else {
      representatives.emplace(entry.derived_cuid, entry);
    }
  }
  if (index.empty()) return unavailable;
  cuid_index_ = std::move(index);
  observations_ = std::move(observations);
  return AMDCUID_STATUS_SUCCESS;
}

DevicePtr CuidDeviceManager::lookup_by_handle(const amdcuid_id_t& handle) const {
  // cuid_index_ is rebuilt wholesale by build_cuid_index() under this mutex.
  // Reading it unlocked is a data race that ThreadSanitizer reports against an
  // ordinary multithreaded consumer of the public API; its caller,
  // amdcuid_query_device_property(), does not hold it, so this cannot deadlock.
  std::lock_guard<std::mutex> lock(manager_mutex_);

  // Since amdcuid_id_t is our handle, we can use it directly as key
  auto it = cuid_index_.find(handle);
  return (it != cuid_index_.end()) ? it->second : nullptr;
}

std::vector<amdcuid_id_t> CuidDeviceManager::get_all_handles() const {
  std::lock_guard<std::mutex> lock(manager_mutex_);

  std::vector<amdcuid_id_t> handles;
  handles.reserve(cuid_index_.size());

  for (const auto& pair : cuid_index_) {
    // amdcuid_id_t is the handle, so just copy directly
    handles.push_back(pair.first);
  }
  return handles;
}
