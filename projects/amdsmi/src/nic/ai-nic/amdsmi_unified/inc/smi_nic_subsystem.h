/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

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
  virtual NicVendor vendor() const = 0;
  virtual bool driver_loaded(const std::string& bdf, DriverType driver_type) const = 0;
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
