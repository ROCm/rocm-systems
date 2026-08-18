// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_VENDORS_BROADCOM_SUBSYSTEM_H_
#define AMDSMI_UNIFIED_VENDORS_BROADCOM_SUBSYSTEM_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "smi_nic.h"
#include "smi_nic_subsystem.h"

/**
 * Broadcom NIC support: discover() walks /sys/class/net and turns each
 * bnxt_en-bound function of PCI vendor 0x14e4 into one Ethernet SmiNic. Broadcom
 * exposes no grouping PCI bridge, so each netdev function maps to its own NIC.
 */
class SmiNicSubsystemBroadcom : public SmiNicSubsystem {
 public:
  // driver_sysfs_root prefixes the driver sysfs dirs; "" means the real /sys.
  // Non-empty is used only by tests to redirect lookups to a tmpdir tree.
  explicit SmiNicSubsystemBroadcom(std::string driver_sysfs_root = "")
      : driver_sysfs_root_(std::move(driver_sysfs_root)) {}
  ~SmiNicSubsystemBroadcom() override = default;

  void discover(const std::string& pci_path, const std::string& net_path,
                std::shared_ptr<amd::smi::nic::transport::NicTransport> transport) override;
  NicVendor vendor() const override;
  const std::vector<std::unique_ptr<SmiNic>>& get_nics() const override;

 private:
  static constexpr uint16_t VENDOR_ID = 0x14e4;

  bool is_driver_loaded(const std::string& bdf, DriverType driver_type) const override;
  bool bound_to_bnxt_en(const std::string& sysfs_bus_path) const;

  std::string driver_sysfs_root_;
  std::vector<std::unique_ptr<SmiNic>> nics_;
};

#endif  // AMDSMI_UNIFIED_VENDORS_BROADCOM_SUBSYSTEM_H_
