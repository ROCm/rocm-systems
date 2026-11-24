// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cuid_nic.h"

#include <dirent.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "cuid_util.h"
#include "pci_util.h"

CuidNic::CuidNic(const amdcuid_nic_info& i) : m_info(i) {}

amdcuid_status_t CuidNic::discover(std::vector<DevicePtr>& nics) {
  return discover("/sys/bus/pci/devices", "/sys/class/net", nics);
}

namespace {

// The first network interface of a PCI function by name, or "" when the
// caller's network namespace shows none.
std::string first_interface(const std::string& function_path) {
  DIR* dir = opendir((function_path + "/net").c_str());
  if (!dir) return "";
  std::string first;
  struct dirent* entry;
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  while ((entry = readdir(dir)) != NULL) {
    const std::string name = entry->d_name;
    if (name[0] == '.') continue;
    if (first.empty() || name < first) first = name;
  }
  closedir(dir);
  return first;
}

bool is_network_class(uint16_t pci_class) {
  return (pci_class >> 8) == 0x02 || pci_class == 0x0d20 || pci_class == 0x0d21 ||
         pci_class == 0x0d80;
}

}  // namespace

amdcuid_status_t CuidNic::discover(const std::string& pci_base_path,
                                   const std::string& net_base_path, std::vector<DevicePtr>& nics) {
  DIR* dir = opendir(pci_base_path.c_str());
  if (!dir) return AMDCUID_STATUS_DEVICE_NOT_FOUND;
  // A NIC is a PCI function of a network class, whether or not an interface
  // for it is visible in the caller's network namespace.
  std::vector<std::string> bdfs;
  struct dirent* entry;
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  while ((entry = readdir(dir)) != NULL) {
    const std::string name = entry->d_name;
    if (!CuidUtilities::is_valid_bdf(name)) continue;
    uint16_t pci_class = 0;
    if (CuidUtilities::read_pci_class(pci_base_path + "/" + name, name, pci_class) !=
            AMDCUID_STATUS_SUCCESS ||
        !is_network_class(pci_class))
      continue;
    bdfs.push_back(name);
  }
  closedir(dir);
  std::sort(bdfs.begin(), bdfs.end());
  for (const auto& bdf : bdfs) {
    const std::string function_path = pci_base_path + "/" + bdf;
    amdcuid_nic_info info = {};
    const auto status = discover_single(&info, function_path);
    if (status != AMDCUID_STATUS_SUCCESS) {
      LOG(DEBUG, "CUID discovery: skipping NIC " << bdf << " (" << amdcuid_status_to_string(status)
                                                 << ")");
      continue;
    }
    const std::string iface = first_interface(function_path);
    info.network_interface = iface.empty() ? function_path : net_base_path + "/" + iface;
    nics.emplace_back(std::make_shared<CuidNic>(info));
  }
  if (nics.empty()) return AMDCUID_STATUS_DEVICE_NOT_FOUND;

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::discover_single(amdcuid_nic_info* nic_info,
                                          const std::string& device_path) {
  amdcuid_nic_info info = {};
  info.header.device_type = AMDCUID_DEVICE_TYPE_NIC;

  std::string bdf = CuidUtilities::readlink_bdf(device_path);
  if (bdf.empty()) {
    const size_t slash = device_path.rfind('/');
    const std::string last = device_path.substr(slash == std::string::npos ? 0 : slash + 1);
    if (CuidUtilities::is_valid_bdf(last)) bdf = last;
  }

  const auto identity_status =
      PciUtil::read_identity(device_path, bdf, info.header.fields.nic.vendor_id,
                             info.header.fields.nic.device_id, info.header.fields.nic.revision_id);
  if (identity_status != AMDCUID_STATUS_SUCCESS) return identity_status;

  uint16_t pci_class_integer = 0;
  const amdcuid_status_t class_status =
      CuidUtilities::read_pci_class(device_path, bdf, pci_class_integer);
  if (class_status != AMDCUID_STATUS_SUCCESS) return class_status;
  info.header.fields.nic.pci_class = pci_class_integer;
  // Network controllers (0x02xx) and 802.11 wireless (0x0d20, 0x0d21, 0x0d80)
  // only: a lookup by any other function's BDF, Bluetooth and IR included,
  // must not name it a NIC.
  if (!is_network_class(pci_class_integer)) return AMDCUID_STATUS_UNSUPPORTED;

  info.bdf = bdf;
  const std::string suffix = "/device";
  if (device_path.size() > suffix.size() &&
      device_path.compare(device_path.size() - suffix.size(), suffix.size(), suffix) == 0) {
    info.network_interface = device_path.substr(0, device_path.size() - suffix.size());
  } else {
    const std::string iface = first_interface(device_path);
    info.network_interface = iface.empty() ? device_path : "/sys/class/net/" + iface;
  }
  *nic_info = info;

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_hardware_fingerprint(uint64_t& fingerprint) const {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }

  uint16_t offset = 0;
  amdcuid_status_t status = PciUtil::get_pci_dsn_cap_offset(m_info.bdf, offset);
  if (status != AMDCUID_STATUS_SUCCESS) {
    // attempt to get fingerprint through VSEC fallback if DSN capability is not
    // found
    status =
        PciUtil::get_pci_vsec_cap_offset(m_info.bdf, m_info.header.fields.nic.vendor_id, offset);
  }
  if (status == AMDCUID_STATUS_SUCCESS) {
    const uint8_t fingerprint_size = 8;
    uint8_t fingerprint_bytes[fingerprint_size] = {0};
    status =
        PciUtil::read_pci_config_space(m_info.bdf, fingerprint_bytes, fingerprint_size, offset);
    if (status == AMDCUID_STATUS_SUCCESS) {
      fingerprint = PciUtil::load_le64(fingerprint_bytes);
      if (CuidUtilities::validate_fingerprint(fingerprint) == AMDCUID_STATUS_SUCCESS) {
        return AMDCUID_STATUS_SUCCESS;
      }
      // An all-zero DSN means the capability is present but unprogrammed. Fall
      // through and try the MAC address instead of claiming identity zero.
      fingerprint = 0;
    }
  }
  // TODO: serial ID could be found in FRU_ID so should attempt to look there
  // pci config space file does not exist or read failed, so create fingerprint
  // from MAC address
  std::string mac_address;
  status = get_mac_address(mac_address);
  if (status != AMDCUID_STATUS_SUCCESS) {
    // No DSN, no VSEC and no MAC leaves the device with no serial to report,
    // which is the auxiliary case. Any other failure is the caller's problem
    // and is returned, so a device is never called auxiliary on a bad read.
    if (status == AMDCUID_STATUS_UNSUPPORTED) {
      return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
    }
    return status;
  }
  return fingerprint_from_mac(mac_address, fingerprint);
}

amdcuid_status_t CuidNic::fingerprint_from_mac(const std::string& mac_address,
                                               uint64_t& fingerprint) {
  if (!mac_address.empty()) {
    // convert MAC address string to bytes
    uint8_t mac_bytes[6] = {0};
    // An unchecked sscanf leaves mac_bytes partly or wholly zero on a
    // malformed address, and the caller would then derive a CUID from that
    // silently. All six octets must convert.
    // cert-err34-c wants strtoul; the concern it encodes is a conversion that
    // fails silently, which the count check below rules out.
    // NOLINTNEXTLINE(cert-err34-c)
    const int converted = sscanf(  // NOLINT(cert-err34-c)
        mac_address.c_str(), "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx", &mac_bytes[0], &mac_bytes[1],
        &mac_bytes[2], &mac_bytes[3], &mac_bytes[4], &mac_bytes[5]);
    if (converted != 6) {
      fingerprint = 0;
      return AMDCUID_STATUS_INVALID_FORMAT;
    }
    // Octet 0 of the address occupies payload bits 0:7, the same orientation
    // as every other little-endian field, and not the host's byte order.
    uint64_t mac_fingerprint = 0;
    for (size_t i = 0; i < sizeof(mac_bytes); ++i) {
      mac_fingerprint |= static_cast<uint64_t>(mac_bytes[i]) << (8 * i);
    }
    fingerprint = mac_fingerprint;
    // An all-zero MAC is an unconfigured interface, not an identity.
    return CuidUtilities::validate_fingerprint(fingerprint);
  }

  return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
}

amdcuid_status_t CuidNic::get_primary_cuid(amdcuid_primary_id& id) const {
  // A device with no serial has no primary (HW_FINGERPRINT_NOT_FOUND) and
  // takes its temporary CUID; a serial this caller may not read is returned as
  // that error, since the identity a device presents must not depend on the
  // privilege of the caller asking.
  uint64_t fingerprint = 0;
  amdcuid_status_t status = get_hardware_fingerprint(fingerprint);
  if (status != AMDCUID_STATUS_SUCCESS) return status;

  uint16_t unit_id = 0;
  get_unit_id(unit_id);
  status = CuidUtilities::generate_primary_cuid(
      fingerprint, unit_id, m_info.header.fields.nic.revision_id,
      m_info.header.fields.nic.device_id, m_info.header.fields.nic.vendor_id,
      AMDCUID_DEVICE_TYPE_NIC, &id, false);
  if (status != AMDCUID_STATUS_SUCCESS) std::memset(&id, 0, sizeof(id));
  return status;
}

amdcuid_status_t CuidNic::get_temporary_input(CuidUtilities::AuxiliaryInput& aux) const {
  std::string bdf;
  const amdcuid_status_t status = this->get_bdf(bdf);
  if (status != AMDCUID_STATUS_SUCCESS) return status;

  aux = {};
  aux.format = CuidUtilities::kAuxFormatPcie;
  const amdcuid_status_t routing = CuidUtilities::set_pcie_routing(aux, bdf);
  if (routing != AMDCUID_STATUS_SUCCESS) return routing;
  aux.revision_id = m_info.header.fields.nic.revision_id;
  aux.device_id = m_info.header.fields.nic.device_id;
  aux.vendor_id = m_info.header.fields.nic.vendor_id;
  aux.component_type = static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_NIC);
  uint16_t unit_id = 0;
  get_unit_id(unit_id);
  aux.unit_id = unit_id;
  return AMDCUID_STATUS_SUCCESS;
}

const amdcuid_nic_info& CuidNic::get_info() const { return m_info; }

amdcuid_status_t CuidNic::get_vendor_id(uint16_t& vendor_id) const {
  vendor_id = m_info.header.fields.nic.vendor_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_device_id(uint16_t& device_id) const {
  device_id = m_info.header.fields.nic.device_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_pci_class(uint16_t& pci_class) const {
  pci_class = m_info.header.fields.nic.pci_class;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_revision_id(uint8_t& revision_id) const {
  revision_id = m_info.header.fields.nic.revision_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_unit_id(uint16_t& unit_id) const {
  const std::string& bdf = m_info.bdf;
  const bool has_function =
      bdf.size() >= 2 && bdf[bdf.size() - 2] == '.' && bdf.back() >= '0' && bdf.back() <= '7';
  unit_id = has_function ? static_cast<uint16_t>(bdf.back() - '0') : 0;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_bdf(std::string& bdf) const {
  if (m_info.bdf.empty()) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  bdf = m_info.bdf;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_device_path(std::string& path) const {
  if (m_info.network_interface.empty()) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  path = m_info.network_interface;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::get_mac_address(std::string& mac_address) const {
  // Extract interface name from the sysfs path (e.g. /sys/class/net/eth0 ->
  // eth0)
  std::string iface = m_info.network_interface;
  size_t last_slash = iface.rfind('/');
  if (last_slash != std::string::npos) iface = iface.substr(last_slash + 1);

  if (iface.empty() || iface.size() >= IFNAMSIZ) return AMDCUID_STATUS_UNSUPPORTED;

  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) return AMDCUID_STATUS_UNSUPPORTED;

  struct ethtool_perm_addr* epaddr = reinterpret_cast<struct ethtool_perm_addr*>(
      new uint8_t[sizeof(struct ethtool_perm_addr) + ETH_ALEN]);
  epaddr->cmd = ETHTOOL_GPERMADDR;
  epaddr->size = ETH_ALEN;

  struct ifreq ifr = {};
  strncpy(ifr.ifr_name, iface.c_str(), IFNAMSIZ - 1);
  ifr.ifr_data = reinterpret_cast<char*>(epaddr);

  if (ioctl(fd, SIOCETHTOOL, &ifr) < 0) {
    close(fd);
    delete[] reinterpret_cast<uint8_t*>(epaddr);
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  close(fd);

  // Six %02x plus five separators is exactly 17 characters plus the NUL.
  char buf[18];
  // NOLINTNEXTLINE(cert-err33-c)
  snprintf(buf, sizeof(buf), "%02x:%02x:%02x:%02x:%02x:%02x", epaddr->data[0], epaddr->data[1],
           epaddr->data[2], epaddr->data[3], epaddr->data[4], epaddr->data[5]);
  delete[] reinterpret_cast<uint8_t*>(epaddr);

  mac_address = buf;
  return AMDCUID_STATUS_SUCCESS;
}
