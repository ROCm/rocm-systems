// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ifoe_subsystem.h"

#include <filesystem>

namespace fs = std::filesystem;

SmiNicIfoe::SmiNicIfoe(const std::string& bdf, const std::string& sysfs_bus_path)
    : SmiNic("", bdf, NicType::Fabric, "", sysfs_bus_path, NicVendor::AMD, NicProduct::AINIC) {}

// Prefix rather than a fixed index: N is per-function and its base is not
// contractual, so pinning ifoe.cmd.0 would drop the bit on a card numbered
// from anything else.
uint32_t SmiNicIfoe::capabilities() const {
  const uint32_t caps = SmiNic::capabilities();
  std::error_code ec;  // non-throwing form: an unreadable path iterates empty
  for (const auto& entry : fs::directory_iterator(sysfs_bus_path_, ec)) {
    if (entry.path().filename().string().rfind(CMD_NODE_PREFIX, 0) == 0) {
      return caps | SMI_NIC_CAP_FWCTL;
    }
  }
  return caps;
}

NicVendor SmiNicSubsystemIfoe::vendor() const { return NicVendor::AMD; }

bool SmiNicSubsystemIfoe::is_driver_loaded(const std::string& bdf, DriverType driver_type) const {
  // A fabric endpoint carries no RDMA auxiliary device, so only Main resolves.
  if (driver_type != DriverType::Main) {
    return false;
  }
  return driver_binds_bdf(driver_sysfs_root_ + "/sys/bus/pci/drivers/ifoe", bdf,
                          /*match_canonical=*/false);
}

// PCI ids alone identify the endpoint: it exposes no netdev, so a net-class
// walk cannot see it, and it is worth listing whether or not ifoe is bound.
void SmiNicSubsystemIfoe::discover(const std::string& pci_path, const std::string&,
                                   std::shared_ptr<amd::smi::nic::transport::NicTransport>) {
  nics_.clear();
  std::error_code ec;

  for (const auto& entry : fs::directory_iterator(pci_path, ec)) {
    if (ec) {
      continue;
    }

    const std::string sysfs_bus_path = entry.path().string();
    auto [vendor_id, device_id] = read_pci_ids(sysfs_bus_path);
    if ((vendor_id == VENDOR_ID) && (device_id == DEVICE_ID)) {
      nics_.push_back(
          std::make_unique<SmiNicIfoe>(entry.path().filename().string(), sysfs_bus_path));
    }
  }
}

const std::vector<std::unique_ptr<SmiNic>>& SmiNicSubsystemIfoe::get_nics() const { return nics_; }
