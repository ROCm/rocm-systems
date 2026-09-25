// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cuid_gpu.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>

#include "cuid_util.h"
#include "gim_util.h"
#include "pci_util.h"

namespace {

constexpr uint16_t kAmdGpuVendorId = 0x1002;

std::string resolved_path(const std::string& path) {
  char* real = realpath(path.c_str(), nullptr);
  if (!real) return {};
  std::string result(real);
  free(real);
  return result;
}

bool last_component(const std::string& path, const char* name) {
  return path.substr(path.find_last_of('/') + 1) == name;
}

bool numbered_node(const std::string& name, const char* prefix) {
  const auto length = std::strlen(prefix);
  return name.size() > length && name.compare(0, length, prefix) == 0 &&
         name.find_first_not_of("0123456789", length) == std::string::npos;
}

bool pci_bdf(const std::string& name) {
  if (name.size() != 12 || name[4] != ':' || name[7] != ':' || name[10] != '.' || name[8] > '1' ||
      name[11] < '0' || name[11] > '7')
    return false;
  for (size_t i = 0; i < name.size(); ++i) {
    if (i == 4 || i == 7 || i == 10) continue;
    if (!std::isxdigit(static_cast<unsigned char>(name[i]))) return false;
  }
  return true;
}

std::string drm_node_for_device(const std::string& device) {
  DIR* dir = opendir((device + "/drm").c_str());
  if (!dir) return {};
  std::string render, card;
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  while (const auto* entry = readdir(dir)) {
    const std::string name(entry->d_name);
    const bool is_render = numbered_node(name, "renderD");
    if (!is_render && !numbered_node(name, "card")) continue;
    const auto candidate = "/sys/class/drm/" + name;
    // The name alone is not evidence: check both the DRM object and its device
    // link. Other XCPs can have the same parent BDF but a different device link.
    const auto real = resolved_path(candidate);
    if (real.empty() || real != resolved_path(device + "/drm/" + name) ||
        resolved_path(candidate + "/device") != device)
      continue;
    auto& chosen = is_render ? render : card;
    if (chosen.empty() || candidate < chosen) chosen = candidate;
  }
  closedir(dir);
  return render.empty() ? card : render;
}

amdcuid_status_t read_partition_metadata(amdcuid_gpu_info& info) {
  if (info.unnamed_vf) {
    info.partition_metadata_valid = false;
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  amdcuid_id_t primary{};
  const auto status = CuidUtilities::read_driver_cuid_from_path(
      info.partition_attr_dir + "/" + CuidUtilities::kDriverPrimaryAttribute, &primary);
  if (status != AMDCUID_STATUS_SUCCESS) return status;
  uint8_t raw[16]{};
  CuidUtilities::remove_UUIDv8_bits(&primary, raw);
  if (((raw[14] >> 6) | ((raw[15] & 3) << 2)) != AMDCUID_DEVICE_TYPE_GPU)
    return AMDCUID_STATUS_INVALID_FORMAT;
  info.header.fields.gpu.unit_id = raw[8] | ((raw[14] & 0x1f) << 8);
  info.header.fields.gpu.revision_id = raw[9];
  info.header.fields.gpu.device_id = raw[10] | (raw[11] << 8);
  info.header.fields.gpu.vendor_id = raw[12] | (raw[13] << 8);
  info.partition_metadata_valid = true;
  return AMDCUID_STATUS_SUCCESS;
}

// Helper to check if a /sys/class/drm entry name is a card device (e.g.,
// "card0", "card1"). Excludes connector entries like "card0-DP-1" or
// "card0-HDMI-A-1".
bool is_card_entry(const char* name) {
  if (strncmp(name, "card", 4) != 0 || !isdigit(name[4])) return false;
  for (size_t i = 4; name[i] != '\0'; ++i) {
    if (!isdigit(name[i])) return false;
  }
  return true;
}

// Read the eight serial octets at `offset` in `bdf`'s configuration space.
// SUCCESS only for a non-zero value: a capability that is present but
// unprogrammed reads back all-zero, which is no serial.
amdcuid_status_t read_config_space_serial_at(const std::string& bdf, uint16_t offset,
                                             uint64_t& fingerprint) {
  const uint8_t fingerprint_size = 8;
  uint8_t fingerprint_bytes[fingerprint_size] = {0};
  const amdcuid_status_t status =
      PciUtil::read_pci_config_space(bdf, fingerprint_bytes, fingerprint_size, offset);
  if (status != AMDCUID_STATUS_SUCCESS) {
    fingerprint = 0;
    return status;
  }
  // No byte swap; see PciUtil::load_le64().
  fingerprint = PciUtil::load_le64(fingerprint_bytes);
  return CuidUtilities::validate_fingerprint(fingerprint);
}

}  // namespace

amdcuid_status_t CuidGpu::resolve_path(const std::string& path, CuidGpuRoute& route) {
  route = {};
  std::string input = path;
  while (input.size() > 1 && input.back() == '/') input.pop_back();
  route.partition = last_component(input, "xcp");
  struct stat st{};
  if (stat(input.c_str(), &st) != 0) {
    if (errno == EACCES || errno == EPERM) return AMDCUID_STATUS_PERMISSION_DENIED;
    return route.partition ? AMDCUID_STATUS_UNSUPPORTED : AMDCUID_STATUS_FILE_NOT_FOUND;
  }
  if (S_ISCHR(st.st_mode)) {
    input = "/sys/dev/char/" + std::to_string(major(st.st_rdev)) + ":" +
            std::to_string(minor(st.st_rdev));
  } else if (!S_ISDIR(st.st_mode)) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  if (last_component(input, "device")) {
    const auto parent = input.substr(0, input.find_last_of('/'));
    const auto real_parent = resolved_path(parent);
    const auto name = real_parent.substr(real_parent.find_last_of('/') + 1);
    if (numbered_node(name, "renderD") || numbered_node(name, "card")) input = parent;
  }
  auto real = resolved_path(input);
  if (real.empty())
    return (errno == EACCES || errno == EPERM) ? AMDCUID_STATUS_PERMISSION_DENIED
                                               : AMDCUID_STATUS_FILE_NOT_FOUND;
  route.partition = route.partition || last_component(real, "xcp");
  if (route.partition) {
    route.node = real;
    route.device = real;
  } else {
    const auto leaf = real.substr(real.find_last_of('/') + 1);
    bool drm_input = false;
    if (numbered_node(leaf, "renderD") || numbered_node(leaf, "card")) {
      // Require the class link to identify this DRM node as well.
      if (resolved_path("/sys/class/drm/" + leaf) != real) return AMDCUID_STATUS_UNSUPPORTED;
      const auto drm_object = real;
      real = resolved_path(real + "/device");
      if (real.empty())
        return (errno == EACCES || errno == EPERM) ? AMDCUID_STATUS_PERMISSION_DENIED
                                                   : AMDCUID_STATUS_FILE_NOT_FOUND;
      if (resolved_path(real + "/drm/" + leaf) != drm_object) return AMDCUID_STATUS_INVALID_FORMAT;
      drm_input = true;
    }
    const auto bdf = real.substr(real.find_last_of('/') + 1);
    if (pci_bdf(bdf) && resolved_path("/sys/bus/pci/devices/" + bdf) == real) {
      route.device = real;
      route.bdf = bdf;
      route.node = drm_node_for_device(real);
      if (route.node.empty()) {
        const auto text = CuidUtilities::read_sysfs_file(real + "/class");
        char* end = nullptr;
        const auto pci_class = std::strtoul(text.c_str(), &end, 0);
        if (text.empty() || *end != '\0' || (pci_class >> 16) != 0x03)
          return AMDCUID_STATUS_UNSUPPORTED;
        route.node = "/sys/bus/pci/devices/" + bdf;
      }
      return AMDCUID_STATUS_SUCCESS;
    }
    // A non-PCI DRM device may be an XCP platform device. Never fall back to
    // its parent's PCI identity, including when its CUID is not published yet.
    if (!drm_input && drm_node_for_device(real).empty() &&
        access((real + "/xcp").c_str(), F_OK) != 0)
      return AMDCUID_STATUS_UNSUPPORTED;
    route.partition = true;
    route.device = resolved_path(real + "/xcp");
    if (route.device.empty()) route.device = real + "/xcp";
    route.node = route.device;
  }
  if (route.node.empty()) return AMDCUID_STATUS_UNSUPPORTED;
  if (access((route.node + "/" + CuidUtilities::kDriverDerivedAttribute).c_str(), F_OK) != 0)
    return (errno == EACCES || errno == EPERM) ? AMDCUID_STATUS_PERMISSION_DENIED
                                               : AMDCUID_STATUS_UNSUPPORTED;
  return AMDCUID_STATUS_SUCCESS;
}

CuidGpu::CuidGpu(const amdcuid_gpu_info& i) : m_info(i) {
  // An xcp node names a partition even when partition_attr_dir is not set.
  if (m_info.partition_attr_dir.empty() && last_component(m_info.render_node, "xcp"))
    m_info.partition_attr_dir = m_info.render_node;
  if (!m_info.partition_attr_dir.empty()) m_info.bdf.clear();
  CuidGpuRoute route;
  const auto path =
      m_info.partition_attr_dir.empty() ? m_info.render_node : m_info.partition_attr_dir;
  const auto status = resolve_path(path, route);
  if (route.partition) {
    m_info.partition_attr_dir = route.node.empty() ? path : route.node;
    m_info.render_node = m_info.partition_attr_dir;
    m_info.bdf.clear();
  }
  if (status != AMDCUID_STATUS_SUCCESS) return;
  if (!m_info.bdf.empty() && m_info.bdf != route.bdf) return;
  if (!m_info.partition_attr_dir.empty() && !route.partition) return;
  m_info.render_node = route.node;
  m_info.bdf = route.bdf;
  if (route.partition) {
    m_info.partition_attr_dir = route.node;
    const auto metadata = read_partition_metadata(m_info);
    if (metadata != AMDCUID_STATUS_SUCCESS && metadata != AMDCUID_STATUS_FILE_NOT_FOUND &&
        metadata != AMDCUID_STATUS_PERMISSION_DENIED)
      m_info.partition_metadata_valid = false;
  }
}

bool CuidGpu::matches_route(const CuidGpuRoute& route) const {
  if (route.partition != !m_info.partition_attr_dir.empty()) return false;
  if (!route.partition && !m_info.bdf.empty() && m_info.bdf != route.bdf) return false;
  CuidGpuRoute own;
  const auto path =
      m_info.partition_attr_dir.empty() ? m_info.render_node : m_info.partition_attr_dir;
  return resolve_path(path, own) == AMDCUID_STATUS_SUCCESS && own.partition == route.partition &&
         own.device == route.device;
}

bool CuidGpu::same_device(const CuidGpu& other) const {
  CuidGpuRoute route;
  const auto& info = other.get_info();
  const auto path = info.partition_attr_dir.empty() ? info.render_node : info.partition_attr_dir;
  return resolve_path(path, route) == AMDCUID_STATUS_SUCCESS && other.matches_route(route) &&
         matches_route(route);
}

// Paths that do not resolve are kept as given (minus a /device suffix). In
// particular, a GIM-only PCI path must not collapse to its parent directory.
std::string CuidGpu::normalize_render_node(const std::string& device_path) {
  CuidGpuRoute route;
  if (resolve_path(device_path, route) == AMDCUID_STATUS_SUCCESS) return route.node;
  std::string full_device_node = device_path;
  const std::string kDeviceSuffix = "/device";
  if (full_device_node.size() > kDeviceSuffix.size() &&
      full_device_node.compare(full_device_node.size() - kDeviceSuffix.size(), kDeviceSuffix.size(),
                               kDeviceSuffix) == 0) {
    full_device_node.resize(full_device_node.size() - kDeviceSuffix.size());
  }

  return full_device_node;
}

amdcuid_status_t CuidGpu::discover(std::vector<DevicePtr>& gpus) {
  // Track BDFs we've already added so the GIM enumeration below doesn't
  // create duplicates of GPUs that are also visible via /sys/class/drm.
  std::set<std::string> seen_bdfs;
  std::set<std::string> seen_nodes;

  // Share one GimClient across all discover_single calls and the GIM
  // enumeration below to avoid repeating the ioctl handshake per GPU.
  std::unique_ptr<cuid::gim::GimClient> gim_client;
  if (cuid::gim::GimClient::is_available()) {
    gim_client.reset(new cuid::gim::GimClient());
  }

  const char* drm_path = "/sys/class/drm";
  DIR* dir = opendir(drm_path);
  if (dir != nullptr) {
    struct dirent* entry;
    // This call site owns its DIR*, which is all POSIX requires for readdir to be
    // safe; readdir_r is deprecated and must not be adopted.
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    while ((entry = readdir(dir)) != NULL) {
      // Use card entries (e.g., card0, card1) which are always present for DRM
      // devices, unlike renderD nodes which may be absent with certain drivers
      // (e.g., GIM) or for non-AMD GPUs.
      if (is_card_entry(entry->d_name)) {
        std::string card_name(entry->d_name);
        std::string device_path = std::string(drm_path) + "/" + card_name + "/device";
        amdcuid_gpu_info info = {};
        // Skip partitioned or otherwise unsupported cards; discover_single
        // leaves `info` untouched in that case, so emplacing it would add a
        // zero-filled GPU entry.
        const auto status = discover_single(&info, device_path, gim_client.get());
        if (status == AMDCUID_STATUS_UNSUPPORTED) {
          continue;
        }
        if (status != AMDCUID_STATUS_SUCCESS) {
          closedir(dir);
          return status;
        }
        if (!seen_nodes.insert(info.render_node).second) continue;
        if (!info.bdf.empty()) {
          seen_bdfs.insert(info.bdf);
        }
        gpus.emplace_back(std::make_shared<CuidGpu>(info));

        // Partition 0 has no card node of its own, so it shares this one and
        // would otherwise be missed; it is a different component from the GPU
        // as a whole. The rest have their own card entries and reach this loop.
        if (info.partition_attr_dir.empty()) {
          amdcuid_gpu_info partition = {};
          if (discover_partition(&partition, device_path) == AMDCUID_STATUS_SUCCESS) {
            gpus.emplace_back(std::make_shared<CuidGpu>(partition));
          }
        }
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
      for (const auto& dev : gim_devices) {
        if (dev.bdf.empty() || seen_bdfs.count(dev.bdf) > 0) {
          continue;
        }
        // Skip devices GIM reports as failed; they cannot provide useful
        // identifying information.
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
        // A GIM-only device is named by its PCI sysfs directory.
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

amdcuid_status_t CuidGpu::discover_single(amdcuid_gpu_info* gpu_info, const std::string& input_path,
                                          cuid::gim::GimClient* gim_client) {
  if (!gpu_info) return AMDCUID_STATUS_INVALID_ARGUMENT;
  CuidGpuRoute route;
  const auto resolved = resolve_path(input_path, route);
  if (route.partition) {
    if (resolved != AMDCUID_STATUS_SUCCESS) return resolved;
    return discover_partition(gpu_info, route.node);
  }
  if (resolved == AMDCUID_STATUS_UNSUPPORTED && !gim_client) return resolved;
  if (resolved != AMDCUID_STATUS_SUCCESS && resolved != AMDCUID_STATUS_FILE_NOT_FOUND &&
      resolved != AMDCUID_STATUS_UNSUPPORTED)
    return resolved;
  const std::string device_path = resolved == AMDCUID_STATUS_SUCCESS ? route.device : input_path;
  amdcuid_gpu_info info = {};
  std::string bdf =
      resolved == AMDCUID_STATUS_SUCCESS ? route.bdf : CuidUtilities::readlink_bdf(device_path);

  // A VF we could not name is recorded as such rather than rounded to unit_id 0,
  // which would say "the whole card". See CuidUtilities::VfIdentity.
  const CuidUtilities::VfIdentity vf = CuidUtilities::get_gpu_vf_identity(
      resolved == AMDCUID_STATUS_SUCCESS ? route.node + "/device" : device_path);
  info.header.fields.gpu.unit_id = vf.unit_id;
  info.unnamed_vf = vf.is_vf && !vf.index_known;

  // No PCI configuration space under a card whose unit_id is 0 means a spatial
  // partition rather than a whole GPU.
  std::string config_file = device_path + "/config";
  if (resolved != AMDCUID_STATUS_SUCCESS && access(config_file.c_str(), F_OK) == -1 &&
      info.header.fields.gpu.unit_id == 0) {
    return discover_partition(gpu_info, device_path);
  }

  std::string vendor = CuidUtilities::read_sysfs_file(device_path + "/vendor");
  if (vendor.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    uint8_t vendor_id_bytes[2] = {0};
    const uint16_t offset = 0x0;
    amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, vendor_id_bytes, 2, offset);
    uint16_t vendor_id_int = PciUtil::load_le16(vendor_id_bytes);
    info.header.fields.gpu.vendor_id = (status == AMDCUID_STATUS_SUCCESS) ? vendor_id_int : 0;
  } else {
    info.header.fields.gpu.vendor_id = (uint16_t)strtol(vendor.c_str(), nullptr, 16);
  }

  std::string device = CuidUtilities::read_sysfs_file(device_path + "/device");
  if (device.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    uint8_t device_id_bytes[2] = {0};
    const uint16_t offset = 0x2;
    amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, device_id_bytes, 2, offset);
    uint16_t device_id_int = PciUtil::load_le16(device_id_bytes);
    info.header.fields.gpu.device_id = (status == AMDCUID_STATUS_SUCCESS) ? device_id_int : 0;
  } else {
    info.header.fields.gpu.device_id = (uint16_t)strtol(device.c_str(), nullptr, 16);
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
  info.header.fields.gpu.pci_class = pci_class_integer;

  std::string revision_id = CuidUtilities::read_sysfs_file(device_path + "/revision");
  if (revision_id.empty() && !bdf.empty()) {
    // if file read fails, attempt to get from pci config
    // RevisionID is the single byte at 0x08. The byte at 0x09 is prog-if and
    // must not be folded in: revision_id is a uint8_t, so a two-byte load left
    // prog-if in the low half and the revision was discarded by the narrowing.
    uint8_t revision_id_byte = 0;
    const uint16_t offset = 0x8;
    amdcuid_status_t status = PciUtil::read_pci_config_space(bdf, &revision_id_byte, 1, offset);
    info.header.fields.gpu.revision_id = (status == AMDCUID_STATUS_SUCCESS) ? revision_id_byte : 0;
  } else {
    info.header.fields.gpu.revision_id = (uint16_t)strtol(revision_id.c_str(), nullptr, 16);
  }

  // Prefer the renderD node; without one (e.g., GIM driver), use the card path.
  std::string full_device_node = normalize_render_node(device_path);

  info.header.device_type = AMDCUID_DEVICE_TYPE_GPU;
  info.bdf = bdf;
  info.render_node = full_device_node;

  // For GIM SR-IOV hosts, query the GIM SMI ioctl interface for every GIM
  // device. When PCI device files are not exposed to userspace this fills the
  // core identifiers (vendor/device/revision). It also captures the ASIC
  // serial for AMDCUID_QUERY_HARDWARE_FINGERPRINT: the SR-IOV PF has no sysfs
  // unique_id or usable PCI config space.
  const bool needs_gim_fallback = !info.bdf.empty() && gim_client != nullptr;
  if (needs_gim_fallback) {
    cuid::gim::GimAsicInfo asic;
    if (gim_client->get_asic_info_for_bdf(info.bdf, asic) == AMDCUID_STATUS_SUCCESS) {
      if (info.header.fields.gpu.vendor_id == 0) {
        info.header.fields.gpu.vendor_id = static_cast<uint16_t>(asic.vendor_id);
      }
      if (info.header.fields.gpu.device_id == 0) {
        info.header.fields.gpu.device_id = static_cast<uint16_t>(asic.device_id);
      }
      if (info.header.fields.gpu.revision_id == 0) {
        info.header.fields.gpu.revision_id = static_cast<uint8_t>(asic.rev_id);
      }
      // GIM ASIC info omits pci_class; default to the PCI display-controller
      // class so GIM-only GPUs do not report an all-zero class.
      if (info.header.fields.gpu.pci_class == 0) {
        info.header.fields.gpu.pci_class = 0x0300;
      }
      // GIM-only devices expose no sysfs unique_id or PCI config space, so
      // carry the ASIC serial as the hardware fingerprint source.
      uint64_t parsed_serial = 0;
      if (cuid::gim::GimClient::parse_asic_serial(asic.asic_serial, parsed_serial)) {
        info.gim_fingerprint = parsed_serial;
        info.gim_fingerprint_valid = true;
      }
    }
  }

  if (info.header.fields.gpu.vendor_id != kAmdGpuVendorId) {
    LOG(DEBUG, "not an AMD GPU: " << device_path);
    return AMDCUID_STATUS_UNSUPPORTED;
  }

  *gpu_info = info;

  return AMDCUID_STATUS_SUCCESS;
}

std::string CuidGpu::partition_attr_dir_for_device(const std::string& device_path) {
  if (device_path.empty()) return "";

  // Idempotent: callers hand this either the card's device node or the
  // partition node itself, and amd-smi resolves by the latter.
  const std::string dir =
      (device_path.size() > 4 && device_path.compare(device_path.size() - 4, 4, "/xcp") == 0)
          ? device_path
          : device_path + "/xcp";
  // The derived value is the one every consumer reads and is world-readable,
  // so its presence is what says the kernel is publishing here. cuid_primary
  // is 0400 and an ordinary user cannot see it at all.
  if (access((dir + "/" + CuidUtilities::kDriverDerivedAttribute).c_str(), F_OK) == -1) {
    return "";
  }
  return dir;
}

amdcuid_status_t CuidGpu::discover_partition(amdcuid_gpu_info* gpu_info,
                                             const std::string& device_path) {
  if (!gpu_info) return AMDCUID_STATUS_INVALID_ARGUMENT;

  std::string attr_dir = partition_attr_dir_for_device(device_path);
  if (attr_dir.empty()) {
    // No kernel-published identifier: an older driver, amdgpu.cuid=0, or not a
    // partition at all. Skipped: a fabricated one would be shared by every
    // partition of the GPU.
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  const auto real = resolved_path(attr_dir);
  if (!real.empty()) attr_dir = real;

  amdcuid_gpu_info info = {};
  info.header.device_type = AMDCUID_DEVICE_TYPE_GPU;
  // Named by its own node: its parent's BDF is the whole GPU's, shared by
  // every partition.
  info.render_node = attr_dir;
  info.partition_attr_dir = attr_dir;
  info.header.fields.gpu.unit_id = std::numeric_limits<uint16_t>::max();
  const auto primary_status = read_partition_metadata(info);
  if (primary_status != AMDCUID_STATUS_SUCCESS && primary_status != AMDCUID_STATUS_FILE_NOT_FOUND &&
      primary_status != AMDCUID_STATUS_PERMISSION_DENIED) {
    return primary_status;
  }

  *gpu_info = info;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::driver_attribute_path(const std::string& attribute,
                                                std::string& path) const {
  // A reachable attribute does not identify a VF whose index is unknown.
  if (m_info.unnamed_vf) return AMDCUID_STATUS_UNSUPPORTED;
  if (!m_info.partition_attr_dir.empty()) {
    path = m_info.partition_attr_dir + "/" + attribute;
    return AMDCUID_STATUS_SUCCESS;
  }
  return CuidDevice::driver_attribute_path(attribute, path);
}

amdcuid_status_t CuidGpu::get_derived_cuid(amdcuid_derived_id& id, cuid_hmac* hmac) const {
  if (m_info.partition_attr_dir.empty()) return CuidDevice::get_derived_cuid(id, hmac);
  const auto status = read_driver_published(CuidUtilities::kDriverDerivedAttribute,
                                            id.UUIDv8_representation, id.raw_bits);
  if (status == AMDCUID_STATUS_SUCCESS) cuid::get_hash_from_raw(id.raw_bits, id.hash);
  last_source_ = status == AMDCUID_STATUS_SUCCESS ? AMDCUID_SOURCE_DRIVER : AMDCUID_SOURCE_UNKNOWN;
  return status;
}

amdcuid_status_t CuidGpu::get_hardware_fingerprint(uint64_t& fingerprint) const {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }
  // Neither PCI nor partition attributes can supply this VF's own serial.
  if (m_info.unnamed_vf) {
    fingerprint = 0;
    return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
  }
  if (!m_info.partition_attr_dir.empty()) {
    amdcuid_primary_id primary{};
    const auto status = driver_primary_cuid(primary);
    if (status != AMDCUID_STATUS_SUCCESS) return status;
    fingerprint = PciUtil::load_le64(primary.raw_bits);
    return AMDCUID_STATUS_SUCCESS;
  }

  // For DRM render nodes the PCI attributes live at "<render_node>/device";
  // for GIM-only PCI directories they live directly under render_node.
  const bool render_node_is_pci_dir = m_info.render_node.find("/sys/bus/pci/devices/") == 0;

  // GIM-only devices have no sysfs unique_id or PCI config space; use the
  // ASIC serial captured during discovery as the fingerprint.
  if (render_node_is_pci_dir && m_info.gim_fingerprint_valid) {
    fingerprint = m_info.gim_fingerprint;
    return CuidUtilities::validate_fingerprint(fingerprint);
  }

  const std::string device_attr_prefix =
      render_node_is_pci_dir ? m_info.render_node : (m_info.render_node + "/device");

  // Precedence: the driver's serial, then the PCIe Device Serial Number, then a
  // vendor-specific capability, stopping at the first source that yields a
  // non-zero value rather than the first that exists. A source reading back
  // zero is an unimplemented capability or an unpopulated register.
  // CuidNic::get_hardware_fingerprint() differs: it has no driver serial, tries
  // the VSEC only where the DSN capability is absent, and falls back to the MAC.

  // Source 1: the serial amdgpu publishes as sysfs unique_id. For a VF the PF
  // holds it, so a VF that does not answer for itself is asked via physfn.
  amdcuid_status_t status = read_unique_id(device_attr_prefix + "/unique_id", fingerprint);
  if (status != AMDCUID_STATUS_SUCCESS && m_info.header.fields.gpu.unit_id != 0) {
    status = read_unique_id(device_attr_prefix + "/physfn/unique_id", fingerprint);
  }
  if (status == AMDCUID_STATUS_SUCCESS) {
    return AMDCUID_STATUS_SUCCESS;
  }

  if (m_info.header.fields.gpu.unit_id != 0) {
    // A VF's own configuration space does not carry the ASIC serial, so with
    // neither its own unique_id nor the PF's there is no next source to try.
    fingerprint = 0;
    return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
  }

  // Sources 2 and 3: the PCIe Device Serial Number extended capability, then
  // the vendor-specific capability.
  return read_config_space_serial(m_info.bdf, m_info.header.fields.gpu.vendor_id, fingerprint);
}

amdcuid_status_t CuidGpu::read_unique_id(const std::string& path, uint64_t& fingerprint) {
  fingerprint = 0;

  std::ifstream fin(path);
  if (!fin.is_open()) {
    return AMDCUID_STATUS_FILE_NOT_FOUND;
  }
  std::string hex_str;
  std::getline(fin, hex_str);
  fin.close();
  if (hex_str.empty()) {
    return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
  }
  // Parse as 64-bit hex value (if possible)
  try {
    fingerprint = std::stoull(hex_str, nullptr, 16);
  } catch (...) {
    fingerprint = 0;
    return AMDCUID_STATUS_INVALID_FORMAT;
  }
  // An attribute reading back zero is an absent serial, not a serial of zero.
  return CuidUtilities::validate_fingerprint(fingerprint);
}

amdcuid_status_t CuidGpu::read_config_space_serial(const std::string& bdf, uint16_t vendor_id,
                                                   uint64_t& fingerprint) {
  fingerprint = 0;

  amdcuid_status_t status = AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;

  // Source 2: the PCIe Device Serial Number extended capability.
  uint16_t offset = 0;
  if (PciUtil::get_pci_dsn_cap_offset(bdf, offset) == AMDCUID_STATUS_SUCCESS) {
    status = read_config_space_serial_at(bdf, offset, fingerprint);
    if (status == AMDCUID_STATUS_SUCCESS) {
      return AMDCUID_STATUS_SUCCESS;
    }
  }

  // Source 3: the vendor-specific capability. Reached when the DSN capability
  // is absent and when it is present but unprogrammed.
  offset = 0;
  if (PciUtil::get_pci_vsec_cap_offset(bdf, vendor_id, offset) == AMDCUID_STATUS_SUCCESS) {
    const amdcuid_status_t vsec = read_config_space_serial_at(bdf, offset, fingerprint);
    if (vsec == AMDCUID_STATUS_SUCCESS) {
      return AMDCUID_STATUS_SUCCESS;
    }
    status = vsec;
  }

  // No source yielded a value. A read that failed outright is reported as
  // itself; anything else is the absence of a serial.
  fingerprint = 0;
  return status;
}

amdcuid_status_t CuidGpu::get_primary_cuid(amdcuid_primary_id& id) const {
  // The driver first: where amdgpu publishes cuid_primary it has read the
  // device serial out of privileged storage userspace cannot reach, so that
  // value is the identity. The reconstruction below is for devices whose
  // driver publishes nothing.
  {
    const amdcuid_status_t drv = driver_primary_cuid(id);
    if (drv != AMDCUID_STATUS_UNSUPPORTED) {
      return drv;
    }
  }

  // A partition exists as a device only because the driver named it, and there
  // is nothing here to reconstruct it from. Falling through would build an
  // auxiliary identifier out of the parent's routing id -- shared by every
  // partition of that GPU, which is the collision this path exists to avoid.
  if (!m_info.partition_attr_dir.empty()) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }

  // Refused whatever the fingerprint source, so every host reports it alike.
  if (m_info.header.fields.gpu.unit_id > CuidUtilities::kMaxUnitId) {
    LOG(ERROR, "UnitID " << m_info.header.fields.gpu.unit_id << " exceeds the 13-bit field (max "
                         << CuidUtilities::kMaxUnitId << ")");
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }

  // A whole GPU the driver does not name is temporary, whether or not its
  // serial is readable here: the driver is the only source of a GPU's
  // permanent identity.
  std::string bdf;
  amdcuid_status_t status = this->get_bdf(bdf);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }
  CuidUtilities::AuxiliaryInput aux;
  aux.format = CuidUtilities::kAuxFormatPcie;
  aux.routing_id = CuidUtilities::routing_id_from_bdf(bdf);
  aux.revision_id = m_info.header.fields.gpu.revision_id;
  aux.device_id = m_info.header.fields.gpu.device_id;
  aux.vendor_id = m_info.header.fields.gpu.vendor_id;
  aux.component_type = static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_GPU);
  uint64_t fingerprint = 0;
  status = CuidUtilities::make_fallback_fingerprint(aux, fingerprint);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }

  // Use header fields for the rest
  amdcuid_primary_id result = {};
  const auto& h = m_info.header;
  // Can fail: an out-of-range UnitID is refused rather than masked onto 0.
  const amdcuid_status_t pack_status = CuidUtilities::generate_primary_cuid(
      fingerprint, h.fields.gpu.unit_id, h.fields.gpu.revision_id, h.fields.gpu.device_id,
      h.fields.gpu.vendor_id, AMDCUID_DEVICE_TYPE_GPU, &result, true);
  if (pack_status != AMDCUID_STATUS_SUCCESS) {
    return pack_status;
  }

  id = result;
  return AMDCUID_STATUS_SUCCESS;
}

const amdcuid_gpu_info& CuidGpu::get_info() const { return m_info; }

amdcuid_status_t CuidGpu::get_vendor_id(uint16_t& vendor_id) const {
  if (!m_info.partition_attr_dir.empty() && !m_info.partition_metadata_valid)
    return AMDCUID_STATUS_UNSUPPORTED;
  vendor_id = m_info.header.fields.gpu.vendor_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_device_id(uint16_t& device_id) const {
  if (!m_info.partition_attr_dir.empty() && !m_info.partition_metadata_valid)
    return AMDCUID_STATUS_UNSUPPORTED;
  device_id = m_info.header.fields.gpu.device_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_pci_class(uint16_t& pci_class) const {
  if (!m_info.partition_attr_dir.empty() && m_info.header.fields.gpu.pci_class == 0)
    return AMDCUID_STATUS_UNSUPPORTED;
  pci_class = m_info.header.fields.gpu.pci_class;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_revision_id(uint8_t& revision_id) const {
  if (!m_info.partition_attr_dir.empty() && !m_info.partition_metadata_valid)
    return AMDCUID_STATUS_UNSUPPORTED;
  revision_id = m_info.header.fields.gpu.revision_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_unit_id(uint16_t& unit_id) const {
  if (!m_info.partition_attr_dir.empty() && !m_info.partition_metadata_valid)
    return AMDCUID_STATUS_UNSUPPORTED;
  unit_id = m_info.header.fields.gpu.unit_id;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_bdf(std::string& bdf) const {
  if (m_info.bdf.empty() || !m_info.partition_attr_dir.empty()) {
    bdf.clear();
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  bdf = m_info.bdf;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::get_device_path(std::string& path) const {
  if (m_info.render_node.empty()) {
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  path = m_info.render_node;
  return AMDCUID_STATUS_SUCCESS;
}
