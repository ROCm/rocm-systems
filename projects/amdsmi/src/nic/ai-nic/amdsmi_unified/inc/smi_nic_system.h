// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef __SMI_NIC_SYSTEM_H__
#define __SMI_NIC_SYSTEM_H__

#include <unistd.h>

#include <climits>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include "smi_nic.h"
#include "smi_nic_subsystem.h"

class SmiNicSystem {
 public:
  SmiNicSystem();
  // Test seam: point discovery at a fake sysfs tree instead of the real /sys.
  SmiNicSystem(const std::string& pci_path, const std::string& net_path);
  ~SmiNicSystem() = default;

  void register_subsystem(std::unique_ptr<SmiNicSubsystem> subsystem);
  // ainic_only=true drops non-AINIC NICs (product() != NicProduct::AINIC).
  void discover_nics(bool ainic_only = false);
  bool is_driver_loaded(const std::string& bdf, DriverType driver_type) const;

  std::vector<std::string> list_bdfs();
  bool interface_exists(const std::string& iface);
  const std::vector<const SmiNic*>& get_nics() const;
  const SmiNic* get_nic_by_interface(const std::string& iface) const;
  const SmiNic* get_nic_by_bdf(const std::string& bdf) const;
  const SmiNic* get_nic_by_bdf(uint64_t bdf) const;

  SmiNicSystem(const SmiNicSystem&) = delete;
  SmiNicSystem& operator=(const SmiNicSystem&) = delete;
  SmiNicSystem(SmiNicSystem&&) = delete;
  SmiNicSystem& operator=(SmiNicSystem&&) = delete;

 private:
  std::string net_path_;
  std::string pci_path_;
  /**
   * One transport shared by every port this system discovers, instead of one
   * backend per port. The netlink backend opens its socket eagerly on
   * construction, so per-port ownership would hold N sockets for N ports;
   * sharing holds one.
   */
  std::shared_ptr<amd::smi::nic::transport::NicTransport> transport_;
  std::vector<const SmiNic*> nics_;
  std::vector<std::unique_ptr<SmiNicSubsystem>> subsystems_;
};

#endif  // __SMI_NIC_SYSTEM_H__
