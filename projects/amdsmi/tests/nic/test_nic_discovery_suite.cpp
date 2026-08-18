// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Hardware-independent unit tests for NIC vendor-subsystem driver detection.
 * Driver sysfs paths are redirected to a tmpdir tree, so these run root-free
 * with no live NIC and validate role->path mapping, not any real device.
 */

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "smi_nic.h"
#include "smi_nic_subsystem.h"
#include "smi_nic_system.h"
#include "vendors/broadcom/broadcom_subsystem.h"
#include "vendors/pensando/pensando_subsystem.h"

namespace fs = std::filesystem;

static int tests_run = 0;
static int tests_failed = 0;

static void check(const std::string& name, bool passed) {
  tests_run++;
  if (!passed) {
    tests_failed++;
  }
  std::cout << (passed ? "  PASS: " : "  FAIL: ") << name << "\n";
}

// Creates a unique tmp sysfs root; caller removes it.
static fs::path make_tmp_root() {
  fs::path base = fs::temp_directory_path() / "amdsmi_nic_disc_XXXXXX";
  std::string tmpl = base.string();
  char* buf = tmpl.data();
  if (!mkdtemp(buf)) {
    std::perror("mkdtemp");
    std::exit(2);
  }
  return fs::path(buf);
}

// Builds root/sys/{class/net/<iface>, bus/pci/devices/<bdf>} such that
// <iface>/device resolves to the pci device dir, the device advertises
// <vendor_hex>/<device_hex>, and its driver symlink points at <driver>.
static void make_fake_netdev(const fs::path& root, const std::string& iface, const std::string& bdf,
                             const std::string& vendor_hex, const std::string& device_hex,
                             const std::string& driver) {
  fs::path pci_dev = root / "sys/bus/pci/devices" / bdf;
  fs::create_directories(pci_dev);
  std::ofstream(pci_dev / "vendor") << vendor_hex << "\n";
  std::ofstream(pci_dev / "device") << device_hex << "\n";

  fs::path driver_dir = root / "sys/bus/pci/drivers" / driver;
  fs::create_directories(driver_dir);
  fs::create_symlink(fs::path("../../devices") / bdf, driver_dir / bdf);
  fs::create_symlink(fs::path("../drivers") / driver, pci_dev / "driver");

  fs::path net_dev = root / "sys/class/net" / iface;
  fs::create_directories(net_dev);
  fs::create_symlink(fs::path("../../../bus/pci/devices") / bdf, net_dev / "device");
}

// Builds root/sys/bus/pci/devices/<bdf> advertising <vendor_hex>/<device_hex>
// with no netdev and no downstream port -- models a fwctl-only card.
static void make_fake_pci_device(const fs::path& root, const std::string& bdf,
                                 const std::string& vendor_hex, const std::string& device_hex) {
  fs::path pci_dev = root / "sys/bus/pci/devices" / bdf;
  fs::create_directories(pci_dev);
  std::ofstream(pci_dev / "vendor") << vendor_hex << "\n";
  std::ofstream(pci_dev / "device") << device_hex << "\n";
}

int main() {
  const std::string bdf = "0000:c1:00.0";
  fs::path root = make_tmp_root();

  // Populate ONLY the ionic (Main) driver dir with a symlink named after the BDF.
  fs::path ionic_dir = root / "sys/bus/pci/drivers/ionic";
  fs::create_directories(ionic_dir);
  fs::create_symlink("../../../devices/pci/" + bdf, ionic_dir / bdf);

  SmiNicSubsystemPensando pensando(root.string());
  SmiNicSubsystem& sub = pensando;

  // Main role must resolve against the ionic dir we populated.
  check("pensando Main driver detected", sub.driver_loaded(bdf, DriverType::Main));
  // A different BDF must not match.
  check("pensando Main driver absent for other bdf",
        !sub.driver_loaded("0000:c1:00.1", DriverType::Main));
  // Rdma role hits a different (unpopulated) path -> proves the roles diverge.
  check("pensando Rdma driver absent (dir not present)", !sub.driver_loaded(bdf, DriverType::Rdma));

  // ---- Broadcom driver_loaded (mirrors the Pensando role->path checks) ----
  const std::string bnxt_bdf = "0000:e1:00.0";
  fs::path bnxt_dir = root / "sys/bus/pci/drivers/bnxt_en";
  fs::create_directories(bnxt_dir);
  fs::create_symlink("../../../devices/pci/" + bnxt_bdf, bnxt_dir / bnxt_bdf);

  SmiNicSubsystemBroadcom broadcom(root.string());
  SmiNicSubsystem& bsub = broadcom;
  check("broadcom Main driver detected", bsub.driver_loaded(bnxt_bdf, DriverType::Main));
  check("broadcom Main driver absent for other bdf",
        !bsub.driver_loaded("0000:e1:00.1", DriverType::Main));
  check("broadcom Rdma driver absent (dir not present)",
        !bsub.driver_loaded(bnxt_bdf, DriverType::Rdma));

  // ---- Broadcom discover(): netdev-walk bound to bnxt_en, vendor 0x14e4 ----
  fs::path disc_root = make_tmp_root();
  make_fake_netdev(disc_root, "bnxt_test0", "0000:e1:00.0", "0x14e4", "0x1750", "bnxt_en");
  // Broadcom vendor but wrong driver -> must be ignored.
  make_fake_netdev(disc_root, "bnxt_legacy", "0000:e1:00.1", "0x14e4", "0x16d7", "tg3");
  // Non-Broadcom vendor -> must be ignored.
  make_fake_netdev(disc_root, "intel0", "0000:e2:00.0", "0x8086", "0x1572", "i40e");

  SmiNicSubsystemBroadcom disc;
  disc.discover((disc_root / "sys/bus/pci/devices").string(),
                (disc_root / "sys/class/net").string(), nullptr);
  const auto& bnics = disc.get_nics();
  check("broadcom discover finds exactly one bnxt_en NIC", bnics.size() == 1);
  if (bnics.size() == 1) {
    check("broadcom NIC bdf correct", bnics[0]->bdf() == "0000:e1:00.0");
    check("broadcom NIC vendor is Broadcom", bnics[0]->vendor() == NicVendor::Broadcom);
    check("broadcom NIC has one port", bnics[0]->nic_ports_num() == 1);
    check("broadcom port iface correct", bnics[0]->nic_ports().at(0).interface() == "bnxt_test0");
  } else {
    check("broadcom NIC bdf correct", false);
    check("broadcom NIC vendor is Broadcom", false);
    check("broadcom NIC has one port", false);
    check("broadcom port iface correct", false);
  }
  fs::remove_all(disc_root);

  // ---- Broadcom Rdma positive: an aux-driver symlink whose canonical target
  //      passes through /<bdf>/ exercises the match_canonical=true branch of
  //      the shared driver_binds_bdf helper (its most intricate path). ----
  fs::path rdma_root = make_tmp_root();
  const std::string rdma_bdf = "0000:e1:00.0";
  fs::path aux_dev = rdma_root / "sys/bus/pci/devices" / rdma_bdf / "bnxt_en.rdma.0";
  fs::create_directories(aux_dev);
  fs::path aux_drv = rdma_root / "sys/bus/auxiliary/drivers/bnxt_re.rdma";
  fs::create_directories(aux_drv);
  fs::create_symlink("../../../../bus/pci/devices/" + rdma_bdf + "/bnxt_en.rdma.0",
                     aux_drv / "bnxt_en.rdma.0");
  SmiNicSubsystemBroadcom broadcom_rdma(rdma_root.string());
  SmiNicSubsystem& rdma_sub = broadcom_rdma;
  check("broadcom Rdma driver detected (aux symlink through bdf)",
        rdma_sub.driver_loaded(rdma_bdf, DriverType::Rdma));
  check("broadcom Rdma driver absent for other bdf",
        !rdma_sub.driver_loaded("0000:e1:00.1", DriverType::Rdma));
  fs::remove_all(rdma_root);

  // ---- Pensando discover(): fwctl-only pds_core (0x1dd8:0008), no netdev ----
  // The card exposes the pds_core management function with no downstream port
  // and no host netdev, so discovery must still register it with zero ports.
  fs::path pen_root = make_tmp_root();
  make_fake_pci_device(pen_root, "0000:a1:00.0", "0x1dd8", "0x0008");  // pds_core
  // Pensando vendor but wrong device id -> not the pds_core function, ignore.
  make_fake_pci_device(pen_root, "0000:a1:00.1", "0x1dd8", "0x1002");
  // Non-Pensando vendor -> ignore.
  make_fake_pci_device(pen_root, "0000:a2:00.0", "0x8086", "0x0008");

  SmiNicSubsystemPensando pen_disc;
  pen_disc.discover((pen_root / "sys/bus/pci/devices").string(),
                    (pen_root / "sys/class/net").string(), nullptr);
  const auto& pnics = pen_disc.get_nics();
  check("pensando discover finds exactly one pds_core NIC", pnics.size() == 1);
  if (pnics.size() == 1) {
    check("pensando NIC bdf correct", pnics[0]->bdf() == "0000:a1:00.0");
    check("pensando NIC vendor is AMD", pnics[0]->vendor() == NicVendor::AMD);
    check("pensando fwctl-only NIC has zero ports", pnics[0]->nic_ports_num() == 0);
  } else {
    check("pensando NIC bdf correct", false);
    check("pensando NIC vendor is AMD", false);
    check("pensando fwctl-only NIC has zero ports", false);
  }
  fs::remove_all(pen_root);

  // ---- SmiNicSystem discovery filter: ALL vs AINIC-only over a mixed tree ----
  // One tree holds a Pensando AINIC (fwctl-only) and a Broadcom bnxt_en netdev.
  // The default (ainic_only=false) keeps both; ainic_only=true drops non-AINIC.
  {
    fs::path mix_root = make_tmp_root();
    make_fake_pci_device(mix_root, "0000:a1:00.0", "0x1dd8", "0x0008");  // Pensando AINIC
    make_fake_netdev(mix_root, "bnxt_mix0", "0000:e1:00.0", "0x14e4", "0x1750", "bnxt_en");

    const std::string pci = (mix_root / "sys/bus/pci/devices").string();
    const std::string net = (mix_root / "sys/class/net").string();

    SmiNicSystem all_sys(pci, net);
    all_sys.discover_nics(/*ainic_only=*/false);
    check("filter ALL discovers both NICs", all_sys.get_nics().size() == 2);

    SmiNicSystem ainic_sys(pci, net);
    ainic_sys.discover_nics(/*ainic_only=*/true);
    const auto& only = ainic_sys.get_nics();
    check("filter AINIC-only discovers exactly one NIC", only.size() == 1);
    if (only.size() == 1) {
      check("filtered NIC product is AINIC", only[0]->product() == NicProduct::AINIC);
      check("filtered NIC vendor is AMD", only[0]->vendor() == NicVendor::AMD);
    } else {
      check("filtered NIC product is AINIC", false);
      check("filtered NIC vendor is AMD", false);
    }
    fs::remove_all(mix_root);
  }

  fs::remove_all(root);

  // ---- capabilities() bitmask: vendor-dependent FWCTL + port-derived NETDEV ----
  // Constructed in-memory (no sysfs), so this covers the ABI-visible bit logic
  // that the C accessor only forwards.
  {
    SmiNicPensando pen_fwctl("", "0000:a1:00.0", NicType::PCIBridge, "", "", NicVendor::AMD,
                             NicProduct::AINIC);
    check("pensando fwctl-only capabilities == FWCTL",
          pen_fwctl.capabilities() == SMI_NIC_CAP_FWCTL);
    check("pensando fwctl-only capabilities lacks NETDEV",
          (pen_fwctl.capabilities() & SMI_NIC_CAP_NETDEV) == 0);

    SmiNicPensando pen_netdev("", "0000:a1:00.0", NicType::PCIBridge, "", "", NicVendor::AMD,
                              NicProduct::AINIC);
    pen_netdev.add_nic_port(SmiNicPort("eth0", "0000:a1:00.1", "", ""));
    check("pensando netdev-backed capabilities == FWCTL|NETDEV",
          pen_netdev.capabilities() == (SMI_NIC_CAP_FWCTL | SMI_NIC_CAP_NETDEV));

    SmiNic bcm("", "0000:e1:00.0", NicType::Ethernet, "", "", NicVendor::Broadcom);
    bcm.add_nic_port(SmiNicPort("bnxt0", "0000:e1:00.0", "", ""));
    check("broadcom capabilities == NETDEV", bcm.capabilities() == SMI_NIC_CAP_NETDEV);
    check("broadcom capabilities lacks FWCTL", (bcm.capabilities() & SMI_NIC_CAP_FWCTL) == 0);

    SmiNic bare("", "0000:00:00.0");
    check("portless non-fwctl capabilities == 0", bare.capabilities() == 0);
  }

  std::cout << tests_run << " run, " << tests_failed << " failed\n";
  return tests_failed == 0 ? 0 : 1;
}
