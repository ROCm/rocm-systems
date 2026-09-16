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

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "cuid_file.h"
#include "cuid_util.h"
#include "pci_util.h"

CuidNic::CuidNic(const amdcuid_nic_info& i) : m_info(i) {}

amdcuid_status_t CuidNic::discover(std::vector<DevicePtr>& nics) {
  std::string nic_base_path = "/sys/class/net";
  DIR* dir = opendir(nic_base_path.c_str());
  if (!dir) return AMDCUID_STATUS_DEVICE_NOT_FOUND;

  struct dirent* entry;
  // This call site owns its DIR*, which is all POSIX requires for readdir to be
  // safe; readdir_r is deprecated and must not be adopted.
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  while ((entry = readdir(dir)) != NULL) {
    // grab everything except the loopback device and hidden entries
    if (strncmp(entry->d_name, "lo", 2) != 0 && entry->d_name[0] != '.') {
      amdcuid_nic_info info = {};
      std::string device_path = std::string(nic_base_path) + "/" + entry->d_name + "/device";
      // filter out virtual devices by checking device symlink
      if (CuidUtilities::readlink_bdf(device_path).empty()) {
        continue;
      }
      discover_single(&info, device_path);

      nics.emplace_back(std::make_shared<CuidNic>(info));
    }
  }
  closedir(dir);
  if (nics.size() == 0) return AMDCUID_STATUS_DEVICE_NOT_FOUND;

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidNic::discover_single(amdcuid_nic_info* nic_info,
                                          const std::string& device_path) {
  amdcuid_nic_info info = {};
  info.header.device_type = AMDCUID_DEVICE_TYPE_NIC;

  std::string bdf = CuidUtilities::readlink_bdf(device_path);

  std::string vendor = CuidUtilities::read_sysfs_file(device_path + "/vendor");
  if (vendor.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    uint8_t vendor_id_bytes[2] = {0};
    const uint16_t offset = 0x0;
    amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, vendor_id_bytes, 2, offset);
    uint16_t vendor_id_int = PciUtil::load_le16(vendor_id_bytes);
    info.header.fields.nic.vendor_id = (status == AMDCUID_STATUS_SUCCESS) ? vendor_id_int : 0;
  } else {
    info.header.fields.nic.vendor_id = (uint16_t)strtol(vendor.c_str(), nullptr, 16);
  }

  std::string device = CuidUtilities::read_sysfs_file(device_path + "/device");
  if (device.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    uint8_t device_id_bytes[2] = {0};
    const uint16_t offset = 0x2;
    amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, device_id_bytes, 2, offset);
    uint16_t device_id_int = PciUtil::load_le16(device_id_bytes);
    info.header.fields.nic.device_id = (status == AMDCUID_STATUS_SUCCESS) ? device_id_int : 0;
  } else {
    info.header.fields.nic.device_id = (uint16_t)strtol(device.c_str(), nullptr, 16);
  }

  std::string pci_class = CuidUtilities::read_sysfs_file(device_path + "/class");
  uint16_t pci_class_integer = 0;
  if (pci_class.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    uint8_t class_id_bytes[2] = {0};
    const uint16_t offset = 0xa;
    amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, class_id_bytes, 2, offset);
    uint16_t class_id_int = PciUtil::load_le16(class_id_bytes);
    pci_class_integer = (status == AMDCUID_STATUS_SUCCESS) ? class_id_int : 0;
  } else {
    // sysfs class file returns 24-bit value (class:subclass:prog_if), shift
    // right by 8 to get 16-bit class:subclass
    pci_class_integer = (uint16_t)(strtol(pci_class.c_str(), nullptr, 16) >> 8);
  }
  info.header.fields.nic.pci_class = pci_class_integer;

  std::string revision_id = CuidUtilities::read_sysfs_file(device_path + "/revision");
  if (revision_id.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    // RevisionID is the single byte at 0x08. The byte at 0x09 is prog-if and
    // must not be folded in: revision_id is a uint8_t, so a two-byte load left
    // prog-if in the low half and the revision was discarded by the narrowing.
    uint8_t revision_id_byte = 0;
    const uint16_t offset = 0x8;
    amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, &revision_id_byte, 1, offset);
    info.header.fields.nic.revision_id = (status == AMDCUID_STATUS_SUCCESS) ? revision_id_byte : 0;
  } else {
    info.header.fields.nic.revision_id = (uint16_t)strtol(revision_id.c_str(), nullptr, 16);
  }
  info.bdf = bdf;
  std::string full_device_node;
  size_t last_slash = device_path.rfind('/');
  if (last_slash != std::string::npos && last_slash > 0) {
    full_device_node = device_path.substr(0, last_slash);
  } else {
    full_device_node = device_path;
  }
  info.network_interface = full_device_node;
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
    status = PciUtil::get_pci_vsec_cap_offset(m_info.bdf, offset);
  }
  if (status == AMDCUID_STATUS_SUCCESS) {
    const uint8_t fingerprint_size = 8;
    uint8_t fingerprint_bytes[fingerprint_size] = {0};
    status =
        PciUtil::read_pci_config_space(m_info.bdf, fingerprint_bytes, fingerprint_size, offset);
    if (status == AMDCUID_STATUS_SUCCESS) {
      uint64_t fingerprint_value = 0;
      std::memcpy(&fingerprint_value, fingerprint_bytes, sizeof(fingerprint_bytes));
      fingerprint = PciUtil::le64_to_be64(fingerprint_value);
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
    return status;
  }
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
    uint64_t mac_fingerprint = 0;
    std::memcpy(&mac_fingerprint, mac_bytes, sizeof(mac_bytes));
    fingerprint = mac_fingerprint;
    // An all-zero MAC is an unconfigured interface, not an identity.
    return CuidUtilities::validate_fingerprint(fingerprint);
  }

  return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
}

amdcuid_status_t CuidNic::get_primary_cuid(amdcuid_primary_id& id) const {
  bool temp = false;
  uint64_t fingerprint = 0;
  amdcuid_status_t status = AMDCUID_STATUS_SUCCESS;

  if (geteuid() == 0) {
    // attempt to read the CUID from the file first
    std::string cuid_file_path = CuidUtilities::priv_cuid_file();
    CuidFile primary_file(cuid_file_path, false);
    primary_file.load();
    std::vector<CuidFileEntry> entries = primary_file.get_entries();

    CuidFileEntry entry;
    status = primary_file.find_by_device_node(m_info.network_interface, entry);
    if (status == AMDCUID_STATUS_SUCCESS && entry.is_temporary == false) {
      id.UUIDv8_representation = entry.primary_cuid;
      CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);
      return AMDCUID_STATUS_SUCCESS;
    }

    status = get_hardware_fingerprint(fingerprint);
  }

  if (geteuid() != 0 || status != AMDCUID_STATUS_SUCCESS) {
    std::string bdf;
    status = this->get_bdf(bdf);
    if (status != AMDCUID_STATUS_SUCCESS) {
      return status;
    }
    status = CuidUtilities::make_fallback_fingerprint(bdf, fingerprint);
    if (status != AMDCUID_STATUS_SUCCESS) {
      return status;
    }
    temp = true;
  }

  status = CuidUtilities::generate_primary_cuid(
      fingerprint, 0, m_info.header.fields.nic.revision_id, m_info.header.fields.nic.device_id,
      m_info.header.fields.nic.vendor_id, static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_NIC), &id, temp);
  if (status != AMDCUID_STATUS_SUCCESS) {
    std::memset(&id, 0, sizeof(id));
    return status;
  }

  return status;
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
