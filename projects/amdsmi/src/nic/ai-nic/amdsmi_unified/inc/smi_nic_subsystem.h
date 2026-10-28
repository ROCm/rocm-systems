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
  Rdma,  // RDMA auxiliary driver (ionic_rdma, bnxt_re.rdma, bng_re.rdma)
};

/**
 * Base interface for a per-vendor NIC discovery plugin. Concrete plugins live
 * under src/vendors/<name>/ and are wired in via make_default_vendor_plugins().
 */
// Where the kernel lists bound drivers; a plugin appends the driver name.
constexpr auto kPciDriversDir = "/sys/bus/pci/drivers/";
constexpr auto kAuxDriversDir = "/sys/bus/auxiliary/drivers/";

class SmiNicSubsystem {
 public:
  virtual ~SmiNicSubsystem() = default;

  virtual void discover(const std::string& pci_path, const std::string& net_path,
                        std::shared_ptr<amd::smi::nic::transport::NicTransport> transport) = 0;
  // Descriptive only. Two plugins may report the same vendor (UALoE and Pensando
  // are both AMD), so this cannot route a query to a plugin; SmiNicSystem
  // resolves by which plugin owns the NIC.
  virtual NicVendor vendor() const = 0;
  virtual bool driver_loaded(const std::string& bdf, DriverType driver_type) const = 0;
  // True when the vendor's driver directory for driver_type exists. No device is checked, so a
  // loaded driver with nothing bound still counts. A plugin without a driver of that kind says no.
  virtual auto is_driver_present(DriverType /*driver_type*/) const -> bool { return false; }
  virtual const std::vector<std::unique_ptr<SmiNic>>& get_nics() const = 0;

 protected:
  std::pair<uint16_t, uint16_t> read_pci_ids(const std::string& sysfs_bus_path) const;
  bool resolve_bdf(const std::string& symlink, std::string& bdf) const;
  auto is_directory_present(const std::string& path) const -> bool;

  // True if the driver bound at driver_dir claims bdf. is_canonical_match=false:
  // a direct child entry named <bdf> counts (PCI driver dir). is_canonical_match=true:
  // any symlink whose resolved target path contains /<bdf>/ counts (auxiliary
  // driver dir). driver_dir already includes any test sysfs-root prefix.
  bool is_driver_bound_to_bdf(const std::string& driver_dir, const std::string& bdf,
                              bool is_canonical_match) const;
};

#endif  // AMDSMI_UNIFIED_NIC_SUBSYSTEM_H_
