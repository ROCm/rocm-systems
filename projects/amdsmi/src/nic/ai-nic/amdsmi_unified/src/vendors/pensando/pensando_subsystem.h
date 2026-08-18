// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_VENDORS_PENSANDO_SUBSYSTEM_H_
#define AMDSMI_UNIFIED_VENDORS_PENSANDO_SUBSYSTEM_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "smi_nic.h"
#include "smi_nic_subsystem.h"

class SmiNicSubsystemPensando : public SmiNicSubsystem {
 public:
  // driver_sysfs_root prefixes the driver sysfs dirs; "" means the real /sys.
  // Non-empty is used only by tests to redirect lookups to a tmpdir tree.
  explicit SmiNicSubsystemPensando(std::string driver_sysfs_root = "")
      : driver_sysfs_root_(std::move(driver_sysfs_root)) {}
  ~SmiNicSubsystemPensando() override = default;

  void discover(const std::string& pci_path, const std::string& net_path,
                std::shared_ptr<amd::smi::nic::transport::NicTransport> transport) override;
  NicVendor vendor() const override;
  const std::vector<std::unique_ptr<SmiNic>>& get_nics() const override;

 private:
  static constexpr uint16_t VENDOR_ID = 0x1dd8;
  // The PCIe bridge above the ionic ports, not an ionic function itself; it
  // reports one of two device ids depending on the card.
  static constexpr std::array<uint16_t, 2> BRIDGE_DEVICE_IDS = {0x0008, 0x1008};
  static constexpr uint16_t PORT_ID = 0x1002;
  // pds_core: the card's management function, and the only one on the card that
  // registers a devlink health reporter.
  static constexpr uint16_t MGMT_ID = 0x100c;

  static bool is_bridge_device(uint16_t device_id);

  bool is_driver_loaded(const std::string& bdf, DriverType driver_type) const override;
  bool is_downstream_port(const std::string& port_bdf, const std::string& bridge_bdf,
                          const std::string& pci_path) const;
  void discover_ports(SmiNic& nic, const std::string& bridge_bdf, const std::string& pci_path,
                      const std::string& net_path,
                      const std::shared_ptr<amd::smi::nic::transport::NicTransport>& transport);
  void discover_mgmt_function(SmiNic& nic, const std::string& bridge_bdf,
                              const std::string& pci_path) const;

  std::string driver_sysfs_root_;
  std::vector<std::unique_ptr<SmiNic>> nics_;
};

#endif  // AMDSMI_UNIFIED_VENDORS_PENSANDO_SUBSYSTEM_H_
