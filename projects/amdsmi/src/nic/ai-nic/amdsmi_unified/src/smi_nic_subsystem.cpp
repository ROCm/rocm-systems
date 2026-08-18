// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "smi_nic_subsystem.h"

#include <unistd.h>

#include <climits>
#include <filesystem>
#include <stdexcept>

#include "smi_sysfs.h"

namespace fs = std::filesystem;

std::pair<uint16_t, uint16_t> SmiNicSubsystem::read_pci_ids(
    const std::string& sysfs_bus_path) const {
  uint16_t vendor_id = 0, device_id = 0;

  std::string vendor_path = sysfs_bus_path + "/vendor";
  std::string device_path = sysfs_bus_path + "/device";

  SmiSysfsReader::SysfsValue vendor_val, device_val;
  if (SmiSysfsReader::readLine(vendor_path, vendor_val) == SmiSysfsReader::SysfsStatus::Success &&
      SmiSysfsReader::readLine(device_path, device_val) == SmiSysfsReader::SysfsStatus::Success) {
    try {
      if (std::holds_alternative<int>(vendor_val)) {
        vendor_id = static_cast<uint16_t>(std::get<int>(vendor_val));
      } else if (std::holds_alternative<std::string>(vendor_val)) {
        vendor_id =
            static_cast<uint16_t>(std::stoul(std::get<std::string>(vendor_val), nullptr, 0));
      }

      if (std::holds_alternative<int>(device_val)) {
        device_id = static_cast<uint16_t>(std::get<int>(device_val));
      } else if (std::holds_alternative<std::string>(device_val)) {
        device_id =
            static_cast<uint16_t>(std::stoul(std::get<std::string>(device_val), nullptr, 0));
      }
    } catch (const std::invalid_argument&) {
      // One malformed id must not abort discovery; {0,0} (unknown) beats a
      // half-parsed pair, which could only cause a false vendor match.
      return {0, 0};
    } catch (const std::out_of_range&) {
      return {0, 0};
    }
  }

  return {vendor_id, device_id};
}

bool SmiNicSubsystem::resolve_bdf(const std::string& symlink, std::string& bdf) const {
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

bool SmiNicSubsystem::driver_binds_bdf(const std::string& driver_dir, const std::string& bdf,
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
