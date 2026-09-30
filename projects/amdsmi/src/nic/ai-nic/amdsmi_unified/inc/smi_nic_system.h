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

/**
 * @brief Convert BDF string format to uint64_t
 *
 * Converts a BDF string to uint64_t format:
 * (domain << 16) | (bus << 8) | (device << 3) | function
 *
 * @param bdf BDF string
 * @return uint64_t BDF value, or 0 if parsing fails
 */
uint64_t parse_bdf(const std::string& bdf);

class SmiNicSystem {
 public:
  SmiNicSystem();
  ~SmiNicSystem() = default;

  void register_subsystem(std::unique_ptr<SmiNicSubsystem> subsystem);
  void discover_nics();
  bool driver_loaded(const std::string& bdf, DriverType driver_type) const;

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
  std::vector<const SmiNic*> nics_;
  std::vector<std::unique_ptr<SmiNicSubsystem>> subsystems_;
};

#endif  // __SMI_NIC_SYSTEM_H__
