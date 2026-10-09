// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "smi_nic_subsystem.h"

#include <unistd.h>

#include <climits>
#include <filesystem>

#include "smi_sysfs.h"

namespace fs = std::filesystem;

std::pair<uint16_t, uint16_t> SmiNicSubsystem::read_pci_ids(
    const std::string& sysfs_bus_path) const {
  uint16_t vendor_id = 0, device_id = 0;

  std::string vendor_path = sysfs_bus_path + "/vendor";
  std::string device_path = sysfs_bus_path + "/device";

  SmiSysfsReader::SysfsValue vendor_val, device_val;
  // Discovery rescans every PCI device, so successful reads stay out of the debug log.
  constexpr bool kIsSuccessLogged = false;
  if ((SmiSysfsReader::readLine(vendor_path, vendor_val, kIsSuccessLogged) ==
       SmiSysfsReader::SysfsStatus::Success) &&
      (SmiSysfsReader::readLine(device_path, device_val, kIsSuccessLogged) ==
       SmiSysfsReader::SysfsStatus::Success)) {
    const auto parsed_vendor = parse_sysfs_uint(vendor_val);
    const auto parsed_device = parse_sysfs_uint(device_val);
    if (!parsed_vendor.has_value() || !parsed_device.has_value()) {
      // One malformed id must not abort discovery; {0,0} (unknown) beats a
      // half-parsed pair, which could only cause a false vendor match.
      return {0, 0};
    }
    vendor_id = static_cast<uint16_t>(*parsed_vendor);
    device_id = static_cast<uint16_t>(*parsed_device);
  }

  return {vendor_id, device_id};
}

bool SmiNicSubsystem::was_bdf_resolved(const std::string& symlink, std::string& bdf) const {
  char resolved_path[PATH_MAX];
  ssize_t len = readlink(symlink.c_str(), resolved_path, sizeof(resolved_path) - 1);

  if (len == -1) {
    return false;
  }
  resolved_path[len] = '\0';

  try {
    fs::path symlink_dir = fs::path(symlink).parent_path();
    fs::path target_path = symlink_dir / resolved_path;
    std::string full_path = fs::canonical(target_path);
    bdf = fs::path(full_path).filename();
    return true;
  } catch (const fs::filesystem_error&) {
    return false;
  }
}

bool SmiNicSubsystem::is_driver_bound_to_bdf(const std::string& driver_dir, const std::string& bdf,
                                             bool match_canonical) const {
  std::error_code ec;
  if (!fs::exists(driver_dir, ec) || !fs::is_directory(driver_dir, ec)) {
    return false;
  }

  try {
    for (const auto& entry : fs::directory_iterator(driver_dir, ec)) {
      if (ec || !fs::is_symlink(entry, ec)) {
        continue;
      }

      if (!match_canonical) {
        if (entry.path().filename().string() == bdf) {
          return true;
        }
        continue;
      }

      std::string target = fs::read_symlink(entry.path(), ec).string();
      if (ec) {
        continue;
      }
      std::string canonical_target =
          fs::canonical(entry.path().parent_path() / target, ec).string();
      if (ec) {
        continue;
      }
      if (canonical_target.find("/" + bdf + "/") != std::string::npos) {
        return true;
      }
    }
  } catch (const fs::filesystem_error&) {
    return false;
  }

  return false;
}
