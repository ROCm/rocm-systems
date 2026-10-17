// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_VENDORS_AMD_IFOE_SUBSYSTEM_H_
#define AMDSMI_UNIFIED_VENDORS_AMD_IFOE_SUBSYSTEM_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "smi_nic.h"
#include "smi_nic_subsystem.h"

/**
 * An Infinity Fabric over Ethernet endpoint: a function on an AMD GPU package,
 * not a discrete card. It exposes no netdev, no VPD and no hwmon, so identity
 * is its BDF and its firmware versions come from devlink.
 */
class SmiNicIfoe : public SmiNic {
 public:
  SmiNicIfoe(const std::string& bdf, const std::string& sysfs_bus_path);

  // Adds FWCTL when an ifoe.cmd.N node is present. That node is the endpoint's
  // only management surface, so its absence means the bit would promise nothing.
  uint32_t capabilities() const override;

 private:
  static constexpr const char* CMD_NODE_PREFIX = "ifoe.cmd.";
};

class SmiNicSubsystemIfoe : public SmiNicSubsystem {
 public:
  // driver_sysfs_root prefixes the driver sysfs dirs; "" means the real /sys.
  // Non-empty is used only by tests to redirect lookups to a tmpdir tree.
  explicit SmiNicSubsystemIfoe(std::string driver_sysfs_root = "")
      : driver_sysfs_root_(std::move(driver_sysfs_root)) {}
  ~SmiNicSubsystemIfoe() override = default;

  void discover(const std::string& pci_path, const std::string& net_path,
                std::shared_ptr<amd::smi::nic::transport::NicTransport> transport) override;
  NicVendor vendor() const override;
  const std::vector<std::unique_ptr<SmiNic>>& get_nics() const override;

 private:
  static constexpr uint16_t VENDOR_ID = 0x1022;
  static constexpr uint16_t DEVICE_ID = 0x1747;

  bool is_driver_loaded(const std::string& bdf, DriverType driver_type) const override;

  std::string driver_sysfs_root_;
  std::vector<std::unique_ptr<SmiNic>> nics_;
};

#endif  // AMDSMI_UNIFIED_VENDORS_AMD_IFOE_SUBSYSTEM_H_
