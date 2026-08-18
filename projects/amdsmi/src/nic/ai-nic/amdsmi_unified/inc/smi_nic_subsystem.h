// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_NIC_SUBSYSTEM_H_
#define AMDSMI_UNIFIED_NIC_SUBSYSTEM_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "smi_nic.h"

// Role of a NIC driver, resolved to a concrete driver path by each vendor
// subsystem. The interface layer names roles, never vendor drivers.
enum class DriverType {
  Main,  // primary netdev/PCI driver (Pensando ionic, Broadcom bnxt_en)
  Rdma,  // RDMA auxiliary driver (ionic_rdma, bnxt_en.rdma)
};

/**
 * Base interface for a per-vendor NIC discovery plugin. Concrete plugins live
 * under src/vendors/<name>/ and are wired in via make_default_vendor_plugins().
 */
class SmiNicSubsystem {
 public:
  virtual ~SmiNicSubsystem() = default;

  virtual void discover(const std::string& pci_path, const std::string& net_path,
                        std::shared_ptr<amd::smi::nic::transport::NicTransport> transport) = 0;
  // Descriptive only. Two plugins may report the same vendor (IFoE and Pensando
  // are both AMD), so this cannot route a query to a plugin; SmiNicSystem
  // resolves by which plugin owns the NIC.
  virtual NicVendor vendor() const = 0;
  virtual bool is_driver_loaded(const std::string& bdf, DriverType driver_type) const = 0;
  virtual const std::vector<std::unique_ptr<SmiNic>>& get_nics() const = 0;

 protected:
  std::pair<uint16_t, uint16_t> read_pci_ids(const std::string& sysfs_bus_path) const;
  bool resolve_bdf(const std::string& symlink, std::string& bdf) const;

  // True if the driver bound at driver_dir claims bdf. match_canonical=false:
  // a direct child entry named <bdf> counts (PCI driver dir). match_canonical=true:
  // any symlink whose canonical target path contains /<bdf>/ counts (auxiliary
  // driver dir). driver_dir already includes any test sysfs-root prefix.
  bool driver_binds_bdf(const std::string& driver_dir, const std::string& bdf,
                        bool match_canonical) const;
};

#endif  // AMDSMI_UNIFIED_NIC_SUBSYSTEM_H_
