// SPDX-License-Identifier: MIT
/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * Hardware-independent unit tests for NIC vendor-subsystem driver detection.
 * Driver sysfs paths are redirected to a tmpdir tree, so these run root-free
 * with no live NIC and validate role->path mapping, not any real device.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "smi_nic.h"
#include "smi_nic_interface.h"
#include "smi_nic_subsystem.h"
#include "smi_nic_system.h"
#include "vendors/amd/ifoe_subsystem.h"
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

// Builds the real Pensando shape: upstream bridge -> intermediate 0x1001 bridge
// -> ionic port, with sys/bus/pci/devices/<bdf> symlinked into sys/devices the
// way the kernel does it. is_downstream_port() proves ancestry off the canonical
// path, so the nesting has to be genuine for the port to attach.
static void make_fake_pci_tree(const fs::path& root, const std::string& domain,
                               const std::string& bridge_bdf, const std::string& bridge_dev,
                               const std::string& mid_bdf, const std::string& port_bdf,
                               const std::string& iface) {
  fs::path bridge = root / "sys/devices" / ("pci" + domain) / bridge_bdf;
  fs::path mid = bridge / mid_bdf;
  fs::path port = mid / port_bdf;
  fs::create_directories(port);

  auto write_ids = [](const fs::path& dir, const std::string& device_hex) {
    std::ofstream(dir / "vendor") << "0x1dd8\n";
    std::ofstream(dir / "device") << device_hex << "\n";
  };
  write_ids(bridge, bridge_dev);
  write_ids(mid, "0x1001");
  write_ids(port, "0x1002");

  fs::path bus = root / "sys/bus/pci/devices";
  fs::create_directories(bus);
  fs::create_symlink(bridge, bus / bridge_bdf);
  fs::create_symlink(mid, bus / mid_bdf);
  fs::create_symlink(port, bus / port_bdf);

  fs::path net = root / "sys/class/net" / iface;
  fs::create_directories(net);
  fs::create_symlink(port, net / "device");
}

// Emits a minimal PCI VPD image (identifier + VPD-R carrying PN/SN) into
// <dir>/vpd, using the resource-tag layout parse_pci_vpd walks.
static void write_fake_vpd(const fs::path& dir, const std::string& product, const std::string& part,
                           const std::string& serial) {
  auto put_large = [](std::vector<uint8_t>& v, uint8_t item, const std::vector<uint8_t>& data) {
    v.push_back(static_cast<uint8_t>(0x80 | item));
    v.push_back(static_cast<uint8_t>(data.size() & 0xff));
    v.push_back(static_cast<uint8_t>((data.size() >> 8) & 0xff));
    v.insert(v.end(), data.begin(), data.end());
  };
  auto put_keyword = [](std::vector<uint8_t>& v, const char* key, const std::string& val) {
    v.push_back(static_cast<uint8_t>(key[0]));
    v.push_back(static_cast<uint8_t>(key[1]));
    v.push_back(static_cast<uint8_t>(val.size()));
    v.insert(v.end(), val.begin(), val.end());
  };

  // An empty argument omits the field, so a caller can model a partial image.
  std::vector<uint8_t> vpd_r;
  if (!part.empty()) {
    put_keyword(vpd_r, "PN", part);
  }
  if (!serial.empty()) {
    put_keyword(vpd_r, "SN", serial);
  }

  std::vector<uint8_t> img;
  if (!product.empty()) {
    put_large(img, 0x02, std::vector<uint8_t>(product.begin(), product.end()));
  }
  put_large(img, 0x10, vpd_r);
  img.push_back(0x78);

  std::ofstream out(dir / "vpd", std::ios::binary);
  out.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size()));
}

// A stand-in plugin reporting AMD, the vendor Pensando already reports. It
// answers is_driver_loaded affirmatively only for the BDF it discovered, so a
// lookup that reaches the wrong plugin is observable as a false.
class StubAmdSubsystem : public SmiNicSubsystem {
 public:
  explicit StubAmdSubsystem(std::string bdf) : bdf_(std::move(bdf)) {}

  void discover(const std::string&, const std::string&,
                std::shared_ptr<amd::smi::nic::transport::NicTransport>) override {
    nics_.clear();
    nics_.push_back(std::make_unique<SmiNic>("", bdf_, NicType::Fabric, "", "", NicVendor::AMD,
                                             NicProduct::AINIC));
  }

  NicVendor vendor() const override { return NicVendor::AMD; }

  const std::vector<std::unique_ptr<SmiNic>>& get_nics() const override { return nics_; }

 private:
  bool is_driver_loaded(const std::string& bdf, DriverType) const override { return bdf == bdf_; }

  std::string bdf_;
  std::vector<std::unique_ptr<SmiNic>> nics_;
};

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
  check("pensando Main driver detected", sub.is_driver_loaded(bdf, DriverType::Main));
  // A different BDF must not match.
  check("pensando Main driver absent for other bdf",
        !sub.is_driver_loaded("0000:c1:00.1", DriverType::Main));
  // Rdma role hits a different (unpopulated) path -> proves the roles diverge.
  check("pensando Rdma driver absent (dir not present)",
        !sub.is_driver_loaded(bdf, DriverType::Rdma));

  // ---- Broadcom is_driver_loaded (mirrors the Pensando role->path checks) ----
  const std::string bnxt_bdf = "0000:e1:00.0";
  fs::path bnxt_dir = root / "sys/bus/pci/drivers/bnxt_en";
  fs::create_directories(bnxt_dir);
  fs::create_symlink("../../../devices/pci/" + bnxt_bdf, bnxt_dir / bnxt_bdf);

  SmiNicSubsystemBroadcom broadcom(root.string());
  SmiNicSubsystem& bsub = broadcom;
  check("broadcom Main driver detected", bsub.is_driver_loaded(bnxt_bdf, DriverType::Main));
  check("broadcom Main driver absent for other bdf",
        !bsub.is_driver_loaded("0000:e1:00.1", DriverType::Main));
  check("broadcom Rdma driver absent (dir not present)",
        !bsub.is_driver_loaded(bnxt_bdf, DriverType::Rdma));

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
        rdma_sub.is_driver_loaded(rdma_bdf, DriverType::Rdma));
  check("broadcom Rdma driver absent for other bdf",
        !rdma_sub.is_driver_loaded("0000:e1:00.1", DriverType::Rdma));
  fs::remove_all(rdma_root);

  // ---- Pensando discover(): a bridge with no port beneath it (0x1dd8:0008) ----
  // The bridge is the NIC handle. With no ionic port under it the card carries
  // no netdev, and discovery must still register it, with zero ports.
  fs::path pen_root = make_tmp_root();
  make_fake_pci_device(pen_root, "0000:a1:00.0", "0x1dd8", "0x0008");  // bridge
  // Pensando vendor but a port device id, not a bridge id -> not a handle, ignore.
  make_fake_pci_device(pen_root, "0000:a1:00.1", "0x1dd8", "0x1002");
  // Non-Pensando vendor -> ignore.
  make_fake_pci_device(pen_root, "0000:a2:00.0", "0x8086", "0x0008");

  SmiNicSubsystemPensando pen_disc;
  pen_disc.discover((pen_root / "sys/bus/pci/devices").string(),
                    (pen_root / "sys/class/net").string(), nullptr);
  const auto& pnics = pen_disc.get_nics();
  check("pensando discover finds exactly one bridge NIC", pnics.size() == 1);
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

  // ---- Pensando discover(): the second bridge device id (0x1dd8:1008) ----
  // Matching a single device id silently dropped every bridge reporting 0x1008,
  // which on a multi-domain host is most of them.
  {
    fs::path pen2_root = make_tmp_root();
    make_fake_pci_device(pen2_root, "0001:01:00.0", "0x1dd8", "0x1008");

    SmiNicSubsystemPensando pen2_disc;
    pen2_disc.discover((pen2_root / "sys/bus/pci/devices").string(),
                       (pen2_root / "sys/class/net").string(), nullptr);
    const auto& p2nics = pen2_disc.get_nics();
    check("pensando discover finds the 0x1008 bridge", p2nics.size() == 1);
    check("pensando 0x1008 bridge bdf correct",
          (p2nics.size() == 1) && (p2nics[0]->bdf() == "0001:01:00.0"));
    fs::remove_all(pen2_root);
  }

  // ---- Pensando discover(): port attaches two bridge levels below ----
  // The ionic function hangs off an intermediate 0x1001 bridge, so the bridge
  // is the port's grandparent, not its parent. Only the 0x1001 bridge must stay
  // unregistered, or every card would report three NICs.
  {
    fs::path nest_root = make_tmp_root();
    make_fake_pci_tree(nest_root, "0001:40", "0001:41:00.0", "0x1008", "0001:42:01.0",
                       "0001:44:00.0", "enP1p68s0");

    SmiNicSubsystemPensando nest_disc;
    nest_disc.discover((nest_root / "sys/bus/pci/devices").string(),
                       (nest_root / "sys/class/net").string(), nullptr);
    const auto& nnics = nest_disc.get_nics();
    check("nested topology registers exactly one NIC", nnics.size() == 1);
    check("nested NIC is the upstream bridge",
          (nnics.size() == 1) && (nnics[0]->bdf() == "0001:41:00.0"));
    check("nested NIC claims its grandchild ionic port",
          (nnics.size() == 1) && (nnics[0]->nic_ports_num() == 1));
    check("nested port iface correct",
          (nnics.size() == 1) && (nnics[0]->nic_ports_num() == 1) &&
              (nnics[0]->nic_ports().at(0).interface() == "enP1p68s0"));
    fs::remove_all(nest_root);
  }

  // ---- Identity falls back to the ionic port when the bridge carries no VPD ----
  // The registered NIC is the PCIe bridge, and on a Salina card only the ionic
  // function beneath it exposes a vpd node, so a bridge-only read reports N/A.
  {
    fs::path vpd_root = make_tmp_root();
    make_fake_pci_tree(vpd_root, "0000:00", "0000:01:00.0", "0x0008", "0000:02:01.0",
                       "0000:04:00.0", "enP0p4s0");
    write_fake_vpd(vpd_root / "sys/devices/pci0000:00/0000:01:00.0/0000:02:01.0/0000:04:00.0",
                   "Salina 2x400G QSFP112", "DSC3-2Q400-64R64E64P-O", "FPK2615006E");

    SmiNicSubsystemPensando vpd_disc;
    vpd_disc.discover((vpd_root / "sys/bus/pci/devices").string(),
                      (vpd_root / "sys/class/net").string(), nullptr);
    const auto& vnics = vpd_disc.get_nics();
    const bool has_one_nic = (vnics.size() == 1);
    check("bridge-without-vpd registers one NIC", has_one_nic);
    check("product_name falls back to the ionic port",
          has_one_nic && (vnics[0]->product_name() == std::string("Salina 2x400G QSFP112")));
    check("part_number falls back to the ionic port",
          has_one_nic && (vnics[0]->part_number() == std::string("DSC3-2Q400-64R64E64P-O")));
    check("serial_number falls back to the ionic port",
          has_one_nic && (vnics[0]->serial_number() == std::string("FPK2615006E")));
    fs::remove_all(vpd_root);
  }

  // ---- The NIC's own VPD wins when both it and the port carry one ----
  // Guards the prefer-self branch: a Vulcano bridge and its ionic report
  // identical images, so unconditional delegation would look correct on
  // hardware while silently changing which device the identity comes from.
  {
    fs::path pref_root = make_tmp_root();
    make_fake_pci_tree(pref_root, "0001:40", "0001:41:00.0", "0x1008", "0001:42:01.0",
                       "0001:44:00.0", "enP1p68s0");
    const fs::path bridge_dir = pref_root / "sys/devices/pci0001:40/0001:41:00.0";
    write_fake_vpd(bridge_dir, "BRIDGE-NAME", "BRIDGE-PN", "BRIDGE-SN");
    write_fake_vpd(bridge_dir / "0001:42:01.0/0001:44:00.0", "PORT-NAME", "PORT-PN", "PORT-SN");

    SmiNicSubsystemPensando pref_disc;
    pref_disc.discover((pref_root / "sys/bus/pci/devices").string(),
                       (pref_root / "sys/class/net").string(), nullptr);
    const auto& pnics2 = pref_disc.get_nics();
    const bool has_one_pref_nic = (pnics2.size() == 1);
    check("own VPD preferred over the port's",
          has_one_pref_nic && (pnics2[0]->product_name() == std::string("BRIDGE-NAME")));
    check("own VPD preference applies to serial_number",
          has_one_pref_nic && (pnics2[0]->serial_number() == std::string("BRIDGE-SN")));
    fs::remove_all(pref_root);
  }

  // ---- A partial bridge image still fills its gaps from the port ----
  // Preference is per field, not per device: a bridge carrying a part number but
  // no serial must not suppress the serial the ionic beneath it does carry.
  {
    fs::path part_root = make_tmp_root();
    make_fake_pci_tree(part_root, "0002:40", "0002:41:00.0", "0x1008", "0002:42:01.0",
                       "0002:44:00.0", "enP2p68s0");
    const fs::path part_bridge = part_root / "sys/devices/pci0002:40/0002:41:00.0";
    write_fake_vpd(part_bridge, "", "BRIDGE-PN", "");
    write_fake_vpd(part_bridge / "0002:42:01.0/0002:44:00.0", "PORT-NAME", "PORT-PN", "PORT-SN");

    SmiNicSubsystemPensando part_disc;
    part_disc.discover((part_root / "sys/bus/pci/devices").string(),
                       (part_root / "sys/class/net").string(), nullptr);
    const auto& partnics = part_disc.get_nics();
    const bool has_one_part_nic = (partnics.size() == 1);
    check("own part_number survives a partial image",
          has_one_part_nic && (partnics[0]->part_number() == std::string("BRIDGE-PN")));
    check("absent serial_number fills from the port",
          has_one_part_nic && (partnics[0]->serial_number() == std::string("PORT-SN")));
    check("absent product_name fills from the port",
          has_one_part_nic && (partnics[0]->product_name() == std::string("PORT-NAME")));
    fs::remove_all(part_root);
  }

  // ---- IFoE discovery: an AMD fabric endpoint (0x1022:0x1747) ----
  // The endpoint is a function on a GPU package, with no netdev, no vpd and no
  // hwmon, so its PCI ids are the only thing that identifies it.
  {
    fs::path ifoe_root = make_tmp_root();
    make_fake_pci_device(ifoe_root, "0001:01:00.1", "0x1022", "0x1747");
    // Same vendor, a different function on the package -> not an endpoint.
    make_fake_pci_device(ifoe_root, "0001:01:00.0", "0x1022", "0x14a0");
    // The endpoint device id under another vendor -> ignore.
    make_fake_pci_device(ifoe_root, "0001:02:00.1", "0x1dd8", "0x1747");

    SmiNicSubsystemIfoe ifoe_disc;
    ifoe_disc.discover((ifoe_root / "sys/bus/pci/devices").string(),
                       (ifoe_root / "sys/class/net").string(), nullptr);
    const auto& inics = ifoe_disc.get_nics();
    const bool has_one_ifoe = (inics.size() == 1);
    check("ifoe discover finds exactly one fabric endpoint", has_one_ifoe);
    check("ifoe endpoint bdf correct", has_one_ifoe && (inics[0]->bdf() == "0001:01:00.1"));
    check("ifoe endpoint vendor is AMD", has_one_ifoe && (inics[0]->vendor() == NicVendor::AMD));
    check("ifoe endpoint product is AINIC",
          has_one_ifoe && (inics[0]->product() == NicProduct::AINIC));
    fs::remove_all(ifoe_root);
  }

  // ---- a fabric endpoint survives the AINIC-only filter ----
  // NicProduct is what discover_nics() filters on, and AINIC is the only value
  // that passes. An endpoint labelled otherwise would vanish from the CLI, which
  // runs the filter, while still appearing in an unfiltered enumeration.
  {
    fs::path filter_root = make_tmp_root();
    make_fake_pci_device(filter_root, "0001:01:00.1", "0x1022", "0x1747");
    fs::create_directories(filter_root / "sys/class/net");

    SmiNicSystem filter_sys((filter_root / "sys/bus/pci/devices").string(),
                            (filter_root / "sys/class/net").string());
    filter_sys.discover_nics(/*ainic_only=*/true);
    check("fabric endpoint survives ainic_only", filter_sys.get_nics().size() == 1);
    fs::remove_all(filter_root);
  }

  // ---- IFoE is_driver_loaded maps Main to the ifoe PCI driver dir ----
  // There is no RDMA auxiliary driver for a fabric endpoint, so that role must
  // stay negative rather than aliasing onto the Main path.
  {
    fs::path ifoe_drv_root = make_tmp_root();
    const std::string ifoe_bdf = "0001:01:00.1";
    fs::path ifoe_dir = ifoe_drv_root / "sys/bus/pci/drivers/ifoe";
    fs::create_directories(ifoe_dir);
    fs::create_symlink("../../../devices/pci/" + ifoe_bdf, ifoe_dir / ifoe_bdf);

    SmiNicSubsystemIfoe ifoe_drv(ifoe_drv_root.string());
    SmiNicSubsystem& ifoe_sub = ifoe_drv;
    check("ifoe Main driver detected", ifoe_sub.is_driver_loaded(ifoe_bdf, DriverType::Main));
    check("ifoe Main driver absent for other bdf",
          !ifoe_sub.is_driver_loaded("0001:01:00.0", DriverType::Main));
    check("ifoe Rdma driver absent (no rdma aux driver)",
          !ifoe_sub.is_driver_loaded(ifoe_bdf, DriverType::Rdma));
    fs::remove_all(ifoe_drv_root);
  }

  // ---- hwmon resolves through the port, not the bridge ----
  // The ionic function registers the hwmon node; the bridge the NIC handle names
  // has no hwmon directory at all, so a bridge-only lookup reports no sensor.
  {
    fs::path hw_root = make_tmp_root();
    make_fake_pci_tree(hw_root, "0000:00", "0000:01:00.0", "0x0008", "0000:02:01.0", "0000:04:00.0",
                       "enP0p4s0");
    fs::create_directories(
        hw_root / "sys/devices/pci0000:00/0000:01:00.0/0000:02:01.0/0000:04:00.0/hwmon/hwmon3");
    std::ofstream(hw_root /
                  "sys/devices/pci0000:00/0000:01:00.0/0000:02:01.0/0000:04:00.0/hwmon/"
                  "hwmon3/temp1_input")
        << "46000\n";
    // hwmon_temp_path builds from the port's sys/bus/pci/devices path, so the
    // node it reports is reached through that symlink, not the canonical one.
    const fs::path expected = hw_root / "sys/bus/pci/devices/0000:04:00.0/hwmon/hwmon3/temp1_input";

    SmiNicSubsystemPensando hw_disc;
    hw_disc.discover((hw_root / "sys/bus/pci/devices").string(),
                     (hw_root / "sys/class/net").string(), nullptr);
    const auto& hnics = hw_disc.get_nics();
    std::optional<std::string> asic_path;
    if (hnics.size() == 1) {
      asic_path = hnics[0]->hwmon_temp_path(NicTempSensor::Asic);
    }
    check("asic hwmon path resolves through the port",
          asic_path.has_value() && (asic_path.value() == expected.string()));
    fs::remove_all(hw_root);
  }

  // ---- the management function is found on the far side of the switch ----
  // pds_core sits behind the other downstream port of the card's internal
  // switch, so it is not a sibling of the ionic and carries no netdev of its
  // own. It is the function that registers the devlink health reporter.
  {
    fs::path mgmt_root = make_tmp_root();
    make_fake_pci_tree(mgmt_root, "0000:00", "0000:01:00.0", "0x0008", "0000:02:01.0",
                       "0000:04:00.0", "enP0p4s0");
    const fs::path mgmt_mid = mgmt_root / "sys/devices/pci0000:00/0000:01:00.0/0000:02:00.0";
    const fs::path mgmt_fn = mgmt_mid / "0000:03:00.2";
    fs::create_directories(mgmt_fn);
    std::ofstream(mgmt_mid / "vendor") << "0x1dd8\n";
    std::ofstream(mgmt_mid / "device") << "0x1001\n";
    std::ofstream(mgmt_fn / "vendor") << "0x1dd8\n";
    std::ofstream(mgmt_fn / "device") << "0x100c\n";
    const fs::path mgmt_bus = mgmt_root / "sys/bus/pci/devices";
    fs::create_symlink(mgmt_mid, mgmt_bus / "0000:02:00.0");
    fs::create_symlink(mgmt_fn, mgmt_bus / "0000:03:00.2");

    SmiNicSubsystemPensando mgmt_disc;
    mgmt_disc.discover(mgmt_bus.string(), (mgmt_root / "sys/class/net").string(), nullptr);
    const auto& mnics = mgmt_disc.get_nics();
    std::string mgmt_bdf;
    if (mnics.size() == 1) {
      mgmt_bdf = mnics[0]->mgmt_bdf();
    }
    check("management BDF resolves to the pds_core function", mgmt_bdf == "0000:03:00.2");
    fs::remove_all(mgmt_root);
  }

  // ---- each card adopts its own management function, not a neighbour's ----
  // Every card on the host exposes a 0x100c function, so the ancestry test off
  // the bridge is the only thing keeping one card from reporting another's
  // health. Without it a flat scan hands both cards whichever it reaches first.
  {
    fs::path two_root = make_tmp_root();
    const fs::path two_bus = two_root / "sys/bus/pci/devices";

    auto add_mgmt_function = [&two_root, &two_bus](
                                 const std::string& domain, const std::string& bridge_bdf,
                                 const std::string& mid_bdf, const std::string& fn_bdf) {
      const fs::path mid = two_root / "sys/devices" / ("pci" + domain) / bridge_bdf / mid_bdf;
      const fs::path fn = mid / fn_bdf;
      fs::create_directories(fn);
      std::ofstream(mid / "vendor") << "0x1dd8\n";
      std::ofstream(mid / "device") << "0x1001\n";
      std::ofstream(fn / "vendor") << "0x1dd8\n";
      std::ofstream(fn / "device") << "0x100c\n";
      fs::create_symlink(mid, two_bus / mid_bdf);
      fs::create_symlink(fn, two_bus / fn_bdf);
    };

    make_fake_pci_tree(two_root, "0000:00", "0000:01:00.0", "0x0008", "0000:02:01.0",
                       "0000:04:00.0", "enP0p4s0");
    add_mgmt_function("0000:00", "0000:01:00.0", "0000:02:00.0", "0000:03:00.2");
    make_fake_pci_tree(two_root, "0001:00", "0001:41:00.0", "0x1008", "0001:42:01.0",
                       "0001:44:00.0", "enP1p44s0");
    add_mgmt_function("0001:00", "0001:41:00.0", "0001:42:00.0", "0001:43:00.2");
    // A third card with no management function of its own must not borrow one.
    make_fake_pci_tree(two_root, "0002:00", "0002:41:00.0", "0x1008", "0002:42:01.0",
                       "0002:44:00.0", "enP2p44s0");

    SmiNicSubsystemPensando two_disc;
    two_disc.discover(two_bus.string(), (two_root / "sys/class/net").string(), nullptr);
    const auto& two_nics = two_disc.get_nics();
    auto mgmt_of = [&two_nics](const std::string& bridge_bdf) {
      for (const auto& nic : two_nics) {
        if (nic->bdf() == bridge_bdf) {
          return nic->mgmt_bdf();
        }
      }
      return std::string{};
    };
    auto ports_of = [&two_nics](const std::string& bridge_bdf) {
      for (const auto& nic : two_nics) {
        if (nic->bdf() == bridge_bdf) {
          return nic->nic_ports_num();
        }
      }
      return static_cast<uint8_t>(0);
    };

    check("three cards discovered", two_nics.size() == 3);
    check("first card keeps its own management function",
          mgmt_of("0000:01:00.0") == "0000:03:00.2");
    check("second card keeps its own management function",
          mgmt_of("0001:41:00.0") == "0001:43:00.2");
    // No 0x100c under this bridge, so mgmt_bdf() falls back to its port.
    check("card without a management function falls back to its port",
          mgmt_of("0002:41:00.0") == "0002:44:00.0");
    // Ports are scoped by the same ancestry test: one card, one port, not three.
    check("each card claims only the port beneath its own bridge",
          (ports_of("0000:01:00.0") == 1) && (ports_of("0001:41:00.0") == 1) &&
              (ports_of("0002:41:00.0") == 1));
    fs::remove_all(two_root);
  }

  // ---- a portless NIC keeps reading hwmon from its own path ----
  // Delegating to port 0 must not cost a single-function NIC its own sensor.
  {
    fs::path own_root = make_tmp_root();
    const fs::path own_dev = own_root / "0000:c1:00.0";
    fs::create_directories(own_dev / "hwmon/hwmon7");
    std::ofstream(own_dev / "hwmon/hwmon7/temp1_input") << "51000\n";

    SmiNic portless("", "0000:c1:00.0", NicType::Ethernet, "", own_dev.string());
    const auto own_path = portless.hwmon_temp_path(NicTempSensor::Asic);
    check("portless NIC resolves its own hwmon node",
          own_path.has_value() &&
              (own_path.value() == (own_dev / "hwmon/hwmon7/temp1_input").string()));
    fs::remove_all(own_root);
  }

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

  // ---- a NIC resolves to the plugin that discovered it, not the first plugin
  //      reporting its vendor ----
  // Pensando and IFoE are both AMD, so keying the lookup on the vendor enum
  // hands every AMD query to the Pensando plugin and its ionic driver path.
  {
    fs::path own_root = make_tmp_root();
    fs::create_directories(own_root / "sys/bus/pci/devices");
    fs::create_directories(own_root / "sys/class/net");

    const std::string stub_bdf = "0003:01:00.1";
    SmiNicSystem own_sys((own_root / "sys/bus/pci/devices").string(),
                         (own_root / "sys/class/net").string());
    own_sys.register_subsystem(std::make_unique<StubAmdSubsystem>(stub_bdf));
    own_sys.discover_nics(/*ainic_only=*/false);
    check("only the stub plugin contributes a NIC to the empty tree",
          own_sys.get_nics().size() == 1);
    check("driver query resolves through the plugin that owns the NIC",
          own_sys.is_driver_loaded(stub_bdf, DriverType::Main));
    fs::remove_all(own_root);
  }

  // ---- BDF order weights the domain above the bus ----
  // Fabric endpoints sit on GPU-package buses numbered below the card buses, so
  // a key that dropped the domain would pool every bus 0x01 device together and
  // reorder the cards behind them.
  {
    fs::path sort_root = make_tmp_root();
    make_fake_pci_device(sort_root, "0001:01:00.1", "0x1022", "0x1747");  // fabric endpoint
    make_fake_pci_device(sort_root, "0002:01:00.1", "0x1022", "0x1747");  // fabric endpoint
    make_fake_pci_device(sort_root, "0001:41:00.0", "0x1dd8", "0x1008");  // card
    make_fake_pci_device(sort_root, "0002:41:00.0", "0x1dd8", "0x1008");  // card
    fs::create_directories(sort_root / "sys/class/net");

    SmiNicSystem sort_sys((sort_root / "sys/bus/pci/devices").string(),
                          (sort_root / "sys/class/net").string());
    sort_sys.discover_nics(/*ainic_only=*/false);
    const auto& sorted = sort_sys.get_nics();
    const bool has_four = (sorted.size() == 4);
    check("mixed tree discovers two cards and two fabric endpoints", has_four);
    check("a domain sorts as a whole, ahead of any bus within it",
          has_four && (sorted[0]->bdf() == "0001:01:00.1") &&
              (sorted[1]->bdf() == "0001:41:00.0") && (sorted[2]->bdf() == "0002:01:00.1") &&
              (sorted[3]->bdf() == "0002:41:00.0"));
    fs::remove_all(sort_root);
  }

  fs::remove_all(root);

  // ---- add_nic_port owns port order, so port 0 is the lowest BDF on insert ----
  // The netdev walk yields ports in readdir order, which is not BDF order.
  {
    SmiNic nic("", "0000:a1:00.0");
    nic.add_nic_port(SmiNicPort("eth1", "0000:a1:00.2", "", "/sys/dev/hi"));
    nic.add_nic_port(SmiNicPort("eth0", "0000:a1:00.1", "", "/sys/dev/lo"));
    check("out-of-order insert still yields BDF-ordered ports",
          (nic.nic_ports()[0].bdf() == "0000:a1:00.1") &&
              (nic.nic_ports()[1].bdf() == "0000:a1:00.2"));
    check("telemetry BDF is the lowest-BDF port", nic.telemetry_bdf() == "0000:a1:00.1");
    check("telemetry sysfs path follows the same port",
          nic.telemetry_sysfs_bus_path() == "/sys/dev/lo");
  }

  // ---- set_mgmt_bdf: an empty argument must not erase a discovered function ----
  {
    SmiNic nic("", "0000:a1:00.0");
    nic.set_mgmt_bdf("0000:a1:00.2");
    nic.set_mgmt_bdf("");
    check("empty set_mgmt_bdf leaves the discovered mgmt function intact",
          nic.mgmt_bdf() == "0000:a1:00.2");
  }

  // ---- capabilities() bitmask: vendor-dependent FWCTL + port-derived NETDEV ----
  // Constructed in-memory (no sysfs), so this covers the ABI-visible bit logic
  // that the C accessor only forwards.
  {
    SmiNicPensando pen_fwctl("", "0000:a1:00.0", NicType::PCIBridge, "", "", NicVendor::AMD,
                             NicProduct::AINIC);
    pen_fwctl.set_mgmt_bdf("0000:a1:00.2");
    check("pensando fwctl-only capabilities == FWCTL",
          pen_fwctl.capabilities() == SMI_NIC_CAP_FWCTL);
    check("pensando fwctl-only capabilities lacks NETDEV",
          (pen_fwctl.capabilities() & SMI_NIC_CAP_NETDEV) == 0);

    // A card whose pds_core function discovery never found cannot be driven
    // over fwctl, so claiming the bit would misreport what the row can do.
    SmiNicPensando pen_no_mgmt("", "0000:a1:00.0", NicType::PCIBridge, "", "", NicVendor::AMD,
                               NicProduct::AINIC);
    check("pensando without a mgmt function lacks FWCTL",
          (pen_no_mgmt.capabilities() & SMI_NIC_CAP_FWCTL) == 0);

    SmiNicPensando pen_netdev("", "0000:a1:00.0", NicType::PCIBridge, "", "", NicVendor::AMD,
                              NicProduct::AINIC);
    pen_netdev.set_mgmt_bdf("0000:a1:00.2");
    pen_netdev.add_nic_port(SmiNicPort("eth0", "0000:a1:00.1", "", ""));
    check("pensando netdev-backed capabilities == FWCTL|NETDEV",
          pen_netdev.capabilities() == (SMI_NIC_CAP_FWCTL | SMI_NIC_CAP_NETDEV));

    SmiNicPensando pen_netdev_no_mgmt("", "0000:a1:00.0", NicType::PCIBridge, "", "",
                                      NicVendor::AMD, NicProduct::AINIC);
    pen_netdev_no_mgmt.add_nic_port(SmiNicPort("eth0", "0000:a1:00.1", "", ""));
    check("pensando with ports but no mgmt function == NETDEV",
          pen_netdev_no_mgmt.capabilities() == SMI_NIC_CAP_NETDEV);

    SmiNic bcm("", "0000:e1:00.0", NicType::Ethernet, "", "", NicVendor::Broadcom);
    bcm.add_nic_port(SmiNicPort("bnxt0", "0000:e1:00.0", "", ""));
    check("broadcom capabilities == NETDEV", bcm.capabilities() == SMI_NIC_CAP_NETDEV);
    check("broadcom capabilities lacks FWCTL", (bcm.capabilities() & SMI_NIC_CAP_FWCTL) == 0);

    SmiNic bare("", "0000:00:00.0");
    check("portless non-fwctl capabilities == 0", bare.capabilities() == 0);

    // A fabric endpoint has no host port, so the ifoe.cmd.N nodes are the only
    // thing separating MODE: fwctl-only from MODE: unknown on the row.
    fs::path ifoe_root = make_tmp_root();
    fs::path with_cmd = ifoe_root / "0001:01:00.1";
    fs::create_directories(with_cmd);
    std::ofstream(with_cmd / "mcdi_logging") << "0\n";
    std::ofstream(with_cmd / "ifoe.cfg.0") << "\n";
    std::ofstream(with_cmd / "ifoe.cmd.0") << "\n";

    fs::path without_cmd = ifoe_root / "0002:01:00.1";
    fs::create_directories(without_cmd);
    std::ofstream(without_cmd / "mcdi_logging") << "0\n";
    std::ofstream(without_cmd / "ifoe.cfg.0") << "\n";

    SmiNicIfoe ifoe("0001:01:00.1", with_cmd.string());
    check("ifoe endpoint capabilities == FWCTL", ifoe.capabilities() == SMI_NIC_CAP_FWCTL);

    SmiNicIfoe ifoe_no_cmd("0002:01:00.1", without_cmd.string());
    check("ifoe endpoint without a command node lacks FWCTL",
          (ifoe_no_cmd.capabilities() & SMI_NIC_CAP_FWCTL) == 0);

    SmiNicIfoe ifoe_no_path("0003:01:00.1", "");
    check("ifoe endpoint with no sysfs path lacks FWCTL",
          (ifoe_no_path.capabilities() & SMI_NIC_CAP_FWCTL) == 0);

    fs::remove_all(ifoe_root);
    check("ifoe endpoint has zero ports", ifoe.nic_ports_num() == 0);
    check("ifoe endpoint port type is Fabric", ifoe.port_type() == "Fabric");

    // Identity falls back to port 0's VPD image. A fabric endpoint has no port,
    // so this is the only coverage of that empty case; without it the fallback
    // indexes an empty vector on every portless device.
    check("ifoe endpoint product name is absent", !ifoe.product_name().has_value());
    check("ifoe endpoint part number is absent", !ifoe.part_number().has_value());
    check("ifoe endpoint serial number is absent", !ifoe.serial_number().has_value());
  }

  std::cout << tests_run << " run, " << tests_failed << " failed\n";
  return tests_failed == 0 ? 0 : 1;
}
