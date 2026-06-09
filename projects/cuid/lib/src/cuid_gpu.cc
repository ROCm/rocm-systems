/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "cuid_gpu.h"
#include "cuid_file.h"
#include "cuid_util.h"
#include "gim_util.h"
#include "pci_util.h"
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <sys/types.h>
#include <unistd.h>

CuidGpu::CuidGpu(const amdcuid_gpu_info &i) : m_info(i) {}

// Helper to check if a /sys/class/drm entry name is a card device (e.g.,
// "card0", "card1"). Excludes connector entries like "card0-DP-1" or
// "card0-HDMI-A-1".
static bool is_card_entry(const char *name) {
  if (strncmp(name, "card", 4) != 0 || !isdigit(name[4]))
    return false;
  for (size_t i = 4; name[i] != '\0'; ++i) {
    if (!isdigit(name[i]))
      return false;
  }
  return true;
}

// Resolve the renderD node path for a given card path.
// If a renderD node exists under the card's device/drm directory, returns
// "/sys/class/drm/renderDXXX". Otherwise, returns the original card path.
static std::string resolve_render_node(const std::string &card_path) {
  std::string drm_dir = card_path + "/device/drm";
  DIR *dir = opendir(drm_dir.c_str());
  if (dir) {
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
      if (strncmp(entry->d_name, "renderD", 7) == 0 &&
          isdigit(entry->d_name[7])) {
        std::string result = "/sys/class/drm/" + std::string(entry->d_name);
        closedir(dir);
        return result;
      }
    }
    closedir(dir);
  }
  return card_path;
}

amdcuid_status_t CuidGpu::discover(std::vector<DevicePtr> &gpus) {
  // Track BDFs we've already added so the GIM enumeration below doesn't
  // create duplicates of GPUs that are also visible via /sys/class/drm.
  std::set<std::string> seen_bdfs;

  // Create one GimClient up front and share it across every discover_single
  // call as well as the GIM enumeration loop below. This avoids the per-GPU
  // ioctl handshake overhead that a fresh client incurs on construction.
  std::unique_ptr<cuid::gim::GimClient> gim_client;
  if (cuid::gim::GimClient::is_available()) {
    gim_client.reset(new cuid::gim::GimClient());
  }

  const char *drm_path = "/sys/class/drm";
  DIR *dir = opendir(drm_path);
  if (dir != nullptr) {
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
      // Use card entries (e.g., card0, card1) which are always present for DRM
      // devices, unlike renderD nodes which may be absent with certain drivers
      // (e.g., GIM) or for non-AMD GPUs.
      if (is_card_entry(entry->d_name)) {
        std::string card_name(entry->d_name);
        std::string device_path =
            std::string(drm_path) + "/" + card_name + "/device";
        amdcuid_gpu_info info = {};
        discover_single(&info, device_path, gim_client.get());
        if (!info.bdf.empty()) {
          seen_bdfs.insert(info.bdf);
        }
        gpus.emplace_back(std::make_shared<CuidGpu>(info));
      }
    }
    closedir(dir);
  }

  // When the GIM driver is loaded, GPUs may not show up under /sys/class/drm
  // at all. Enumerate them via the GIM SMI ioctl interface and add any BDFs
  // we have not already discovered.
  if (gim_client) {
    std::vector<cuid::gim::GimDeviceEntry> gim_devices;
    if (gim_client->get_devices(gim_devices) == AMDCUID_STATUS_SUCCESS) {
      for (const auto &dev : gim_devices) {
        if (dev.bdf.empty() || seen_bdfs.count(dev.bdf) > 0) {
          continue;
        }
        // Skip devices the GIM driver reports as failed: such devices are
        // either not present, not initialized, or otherwise unable to
        // provide useful identifying information, so they should not be
        // surfaced to consumers as discoverable GPUs.
        if (dev.failed) {
          LOG(DEBUG, "GIM: skipping failed device at BDF " << dev.bdf);
          continue;
        }
        std::string sys_device_path = "/sys/bus/pci/devices/" + dev.bdf;
        amdcuid_gpu_info info = {};
        discover_single(&info, sys_device_path, gim_client.get());
        // Ensure BDF is populated even if the sysfs node is missing entirely;
        // discover_single relies on the device symlink which may not exist
        // for GIM-only devices.
        if (info.bdf.empty()) {
          info.bdf = dev.bdf;
        }
        // discover_single() may have synthesized an unrelated render_node
        // (e.g. by trimming the input path) for inputs that are not the
        // canonical /sys/class/drm/<card>/device form used elsewhere. For
        // GIM-only devices the per-device PCI sysfs directory is the
        // stable identifier we want persisted in the CUID file, so always
        // overwrite it here.
        info.render_node = sys_device_path;
        info.header.device_type = AMDCUID_DEVICE_TYPE_GPU;
        seen_bdfs.insert(dev.bdf);
        gpus.emplace_back(std::make_shared<CuidGpu>(info));
      }
    }
  }

  if (dir == nullptr && seen_bdfs.empty()) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::discover_single(amdcuid_gpu_info *gpu_info,
                                          const std::string &device_path,
                                          cuid::gim::GimClient *gim_client) {

  amdcuid_gpu_info info = {};
  std::string bdf = CuidUtilities::readlink_bdf(device_path);

  // Determine unit_id from SR-IOV VF (Virtual Function) status via ioctl.
  // In bare metal or passthrough, unit_id is 0.
  // For VFs, unit_id is the 1-based VF index.
  // Falls back to 0 if VF detection is unavailable.
  info.header.fields.gpu.unit_id = CuidUtilities::get_gpu_vf_id(device_path);

  std::string vendor = CuidUtilities::read_sysfs_file(device_path + "/vendor");
  if (vendor.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    uint8_t vendor_id_bytes[2] = {0};
    const uint16_t offset = 0x0;
    amdcuid_status_t status =
        PciUtil::read_pci_config_space(bdf, vendor_id_bytes, 2, offset);
    uint16_t vendor_id_int =
        PciUtil::le16_to_be16(*reinterpret_cast<uint16_t *>(vendor_id_bytes));
    info.header.fields.gpu.vendor_id =
        (status == AMDCUID_STATUS_SUCCESS) ? vendor_id_int : 0;
  } else {
    info.header.fields.gpu.vendor_id =
        (uint16_t)strtol(vendor.c_str(), nullptr, 16);
  }

  std::string device = CuidUtilities::read_sysfs_file(device_path + "/device");
  if (device.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    uint8_t device_id_bytes[2] = {0};
    const uint16_t offset = 0x2;
    amdcuid_status_t status =
        PciUtil::read_pci_config_space(bdf, device_id_bytes, 2, offset);
    uint16_t device_id_int =
        PciUtil::le16_to_be16(*reinterpret_cast<uint16_t *>(device_id_bytes));
    info.header.fields.gpu.device_id =
        (status == AMDCUID_STATUS_SUCCESS) ? device_id_int : 0;
  } else {
    info.header.fields.gpu.device_id =
        (uint16_t)strtol(device.c_str(), nullptr, 16);
  }

  std::string pci_class =
      CuidUtilities::read_sysfs_file(device_path + "/class");
  uint16_t pci_class_integer = 0;
  if (pci_class.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    uint8_t class_id_bytes[2] = {0};
    const uint16_t offset = 0xa;
    amdcuid_status_t status =
        PciUtil::read_pci_config_space(bdf, class_id_bytes, 2, offset);
    uint16_t class_id_int =
        PciUtil::le16_to_be16(*reinterpret_cast<uint16_t *>(class_id_bytes));
    pci_class_integer = (status == AMDCUID_STATUS_SUCCESS) ? class_id_int : 0;
  } else {
    // sysfs class file returns 24-bit value (class:subclass:prog_if), shift
    // right by 8 to get 16-bit class:subclass
    pci_class_integer = (uint16_t)(strtol(pci_class.c_str(), nullptr, 16) >> 8);
  }
  info.header.fields.gpu.pci_class = pci_class_integer;

  std::string revision_id =
      CuidUtilities::read_sysfs_file(device_path + "/revision");
  if (revision_id.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    uint8_t revision_id_bytes[2] = {0};
    const uint16_t offset = 0x8;
    amdcuid_status_t status =
        PciUtil::read_pci_config_space(bdf, revision_id_bytes, 2, offset);
    uint16_t revision_id_int =
        PciUtil::le16_to_be16(*reinterpret_cast<uint16_t *>(revision_id_bytes));
    info.header.fields.gpu.revision_id =
        (status == AMDCUID_STATUS_SUCCESS) ? revision_id_int : 0;
  } else {
    info.header.fields.gpu.revision_id =
        (uint16_t)strtol(revision_id.c_str(), nullptr, 16);
  }

  // Determine the device node path. We prefer the renderD node for backward
  // compatibility (CUID files may reference renderD paths). If no renderD node
  // exists (e.g., GIM driver), the card path is used instead.
  //
  // Callers from /sys/class/drm enumeration pass paths of the form
  // "/sys/class/drm/<card>/device" and expect the trailing "/device"
  // component to be stripped so that lookups under render_node still work
  // (the resolver re-appends "/device/..." itself). Other callers (e.g. the
  // GIM enumeration path which passes "/sys/bus/pci/devices/<bdf>") must
  // not be trimmed -- doing so would collapse every device to the parent
  // directory and corrupt the CUID-file identifier.
  std::string full_device_node = device_path;
  const std::string kDeviceSuffix = "/device";
  if (full_device_node.size() > kDeviceSuffix.size() &&
      full_device_node.compare(full_device_node.size() - kDeviceSuffix.size(),
                               kDeviceSuffix.size(), kDeviceSuffix) == 0) {
    full_device_node.resize(full_device_node.size() - kDeviceSuffix.size());
  }

  // If discovered via card entry, try to resolve the associated renderD node
  if (full_device_node.find("/card") != std::string::npos) {
    std::string render_node = resolve_render_node(full_device_node);
    if (render_node != full_device_node) {
      full_device_node = render_node;
    }
  }

  info.header.device_type = AMDCUID_DEVICE_TYPE_GPU;
  info.bdf = bdf;
  info.render_node = full_device_node;

  // Final fallback: when sysfs and PCI config space were both unable to
  // populate the core PCI identifiers (typical of hosts running the GIM
  // SR-IOV driver where the PCI device files are not exposed to userspace),
  // query the same fields via the GIM SMI ioctl interface. The caller is
  // expected to share a single GimClient across all discover_single calls
  // so the ioctl handshake only happens once per discovery pass.
  const bool needs_gim_fallback = !info.bdf.empty() && gim_client != nullptr &&
                                  (info.header.fields.gpu.vendor_id == 0 ||
                                   info.header.fields.gpu.device_id == 0);
  if (needs_gim_fallback) {
    cuid::gim::GimAsicInfo asic;
    if (gim_client->get_asic_info_for_bdf(info.bdf, asic) ==
        AMDCUID_STATUS_SUCCESS) {
      if (info.header.fields.gpu.vendor_id == 0) {
        info.header.fields.gpu.vendor_id =
            static_cast<uint16_t>(asic.vendor_id);
      }
      if (info.header.fields.gpu.device_id == 0) {
        info.header.fields.gpu.device_id =
            static_cast<uint16_t>(asic.device_id);
      }
      if (info.header.fields.gpu.revision_id == 0) {
        info.header.fields.gpu.revision_id =
            static_cast<uint8_t>(asic.rev_id);
      }
      // pci_class is not exposed by the GIM ASIC info; default to the
      // standard PCI display-controller class (0x0300) so consumers do not
      // see an all-zero class identifier for GIM-only GPUs.
      if (info.header.fields.gpu.pci_class == 0) {
        info.header.fields.gpu.pci_class = 0x0300;
      }
    }
  }

  *gpu_info = info;

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t
CuidGpu::get_hardware_fingerprint(uint64_t &fingerprint) const {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }

  // The render_node may be either a DRM character-device directory
  // ("/sys/class/drm/renderD<N>" or "/sys/class/drm/card<N>"), under which
  // the PCI device attributes live at "<render_node>/device/...", or a PCI
  // device directory directly ("/sys/bus/pci/devices/<bdf>") in which case
  // the attributes are at "<render_node>/...". Detect the latter so the
  // sysfs unique_id lookup works for GIM-only devices too.
  const bool render_node_is_pci_dir =
      m_info.render_node.find("/sys/bus/pci/devices/") == 0;
  const std::string device_attr_prefix =
      render_node_is_pci_dir ? m_info.render_node
                             : (m_info.render_node + "/device");

  std::string unique_id_path = device_attr_prefix + "/unique_id";

  // Try to read the unique_id from the device sysfs file
  std::ifstream fin(unique_id_path);

  // If not available and this is a VF, try the PF's unique_id via physfn
  if (!fin.is_open() && m_info.header.fields.gpu.unit_id != 0) {
    std::string physfn_path = device_attr_prefix + "/physfn/unique_id";
    fin.open(physfn_path);
  }
  if (fin.is_open()) {
    std::string hex_str;
    std::getline(fin, hex_str);
    fin.close();
    if (hex_str.empty()) {
      fingerprint = 0;
      return AMDCUID_STATUS_UNSUPPORTED;
    }
    // Parse as 64-bit hex value (if possible)
    try {
      fingerprint = std::stoull(hex_str, nullptr, 16);
    } catch (...) {
      fingerprint = 0;
      return AMDCUID_STATUS_UNSUPPORTED;
    }
  } else if (m_info.header.fields.gpu.unit_id == 0) {
    // attempt to get fingerprint through PCI Config Space if not a VF
    uint16_t offset = 0;
    amdcuid_status_t status =
        PciUtil::get_pci_dsn_cap_offset(m_info.bdf, offset);
    if (status != AMDCUID_STATUS_SUCCESS) {
      // attempt to get fingerprint through VSEC fallback if DSN capability is
      // not found
      status = PciUtil::get_pci_vsec_cap_offset(m_info.bdf, offset);
      if (status != AMDCUID_STATUS_SUCCESS) {
        fingerprint = 0;
        return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
      }
    }

    const uint8_t fingerprint_size = 8;
    uint8_t fingerprint_bytes[fingerprint_size] = {0};
    status = PciUtil::read_pci_config_space(m_info.bdf, fingerprint_bytes,
                                            fingerprint_size, offset);
    if (status != AMDCUID_STATUS_SUCCESS) {
      fingerprint = 0;
      return status;
    }
    // pcie config file is little endian, so need to convert to big endian
    uint64_t fingerprint_value = 0;
    std::memcpy(&fingerprint_value, fingerprint_bytes,
                sizeof(fingerprint_value));
    fingerprint = PciUtil::le64_to_be64(fingerprint_value);
  } else {
    // partitioned device without unique_id file or pci config cannot get
    // fingerprint
    fingerprint = 0;
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_primary_cuid(amdcuid_primary_id &id) const {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }

  // attempt to read the CUID from the file first
  std::string cuid_file_path = CuidUtilities::priv_cuid_file();
  CuidFile primary_file(cuid_file_path, false);
  primary_file.load();
  std::vector<CuidFileEntry> entries = primary_file.get_entries();

  CuidFileEntry entry;
  amdcuid_status_t status =
      primary_file.find_by_device_node(m_info.render_node, entry);
  if (status == AMDCUID_STATUS_SUCCESS) {
    id.UUIDv8_representation = entry.primary_cuid;
    CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, id.raw_bits);
    return AMDCUID_STATUS_SUCCESS;
  }

  // primary CUID not found in file so generate it
  uint64_t fingerprint = 0;
  status = get_hardware_fingerprint(fingerprint);
  if (status != AMDCUID_STATUS_SUCCESS) {
    std::memset(&id, 0, sizeof(id));
    return status;
  }
  // Use header fields for the rest
  amdcuid_primary_id result = {};
  const auto &h = m_info.header;
  CuidUtilities::generate_primary_cuid(
      fingerprint, h.fields.gpu.unit_id, h.fields.gpu.revision_id,
      h.fields.gpu.device_id, h.fields.gpu.vendor_id,
      static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_GPU), &result);

  id = result;
  return AMDCUID_STATUS_SUCCESS;
}

const amdcuid_gpu_info &CuidGpu::get_info() const { return m_info; }

amdcuid_status_t CuidGpu::get_vendor_id(uint16_t &vendor_id) const {
  vendor_id = m_info.header.fields.gpu.vendor_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_device_id(uint16_t &device_id) const {
  device_id = m_info.header.fields.gpu.device_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_pci_class(uint16_t &pci_class) const {
  pci_class = m_info.header.fields.gpu.pci_class;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_revision_id(uint8_t &revision_id) const {
  revision_id = m_info.header.fields.gpu.revision_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_unit_id(uint16_t &unit_id) const {
  unit_id = m_info.header.fields.gpu.unit_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_bdf(std::string &bdf) const {
  if (m_info.bdf.empty()) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  bdf = m_info.bdf;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_device_path(std::string &path) const {
  if (m_info.render_node.empty()) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  path = m_info.render_node;
  return AMDCUID_STATUS_SUCCESS;
}
