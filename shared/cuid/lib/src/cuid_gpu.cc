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
  if (!CuidUtilities::is_valid_bdf(name)) return false;
  const size_t tail = name.size() - 8;
  return name[tail + 4] <= '1' && name[tail + 7] >= '0' && name[tail + 7] <= '7';
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

std::string parent_dir(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

// The minor of the renderD node under `device`/drm, or -1.
int render_minor_of(const std::string& device) {
  DIR* dir = opendir((device + "/drm").c_str());
  if (!dir) return -1;
  int minor = -1;
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  while (const auto* entry = readdir(dir)) {
    const std::string name(entry->d_name);
    if (numbered_node(name, "renderD")) {
      char* end = nullptr;
      errno = 0;
      const long value = std::strtol(name.c_str() + 7, &end, 10);
      if (errno != ERANGE && *end == '\0' && value <= std::numeric_limits<int>::max())
        minor = static_cast<int>(value);
      break;
    }
  }
  closedir(dir);
  return minor;
}

struct KfdNode {
  long long render_minor = -1;
  long long domain = -1;
  long long location_id = -1;
};

std::vector<KfdNode> kfd_nodes_under(const std::string& root) {
  std::vector<KfdNode> nodes;
  DIR* dir = opendir(root.c_str());
  if (!dir) return nodes;
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  while (const auto* entry = readdir(dir)) {
    if (entry->d_name[0] == '.') continue;
    std::ifstream properties(root + "/" + entry->d_name + "/properties");
    KfdNode node;
    std::string key;
    std::string text;
    // Values are read as text: unique_id is a 64-bit serial that exceeds a
    // signed long long for half of all parts, and a failed numeric read would
    // stop the scan.
    while (properties >> key >> text) {
      long long* field = key == "drm_render_minor" ? &node.render_minor
                         : key == "domain"         ? &node.domain
                         : key == "location_id"    ? &node.location_id
                                                   : nullptr;
      if (!field) continue;
      char* end = nullptr;
      errno = 0;
      const long long value = std::strtoll(text.c_str(), &end, 10);
      if (errno != ERANGE && end != text.c_str() && *end == '\0') *field = value;
    }
    if (node.render_minor > 0 && node.domain >= 0 && node.location_id >= 0) nodes.push_back(node);
  }
  closedir(dir);
  return nodes;
}

// KFD renumbers a GPU's nodes when its partition mode changes; a read that
// straddles the change is retried rather than mixed.
std::vector<KfdNode> kfd_snapshot(const std::string& nodes_dir) {
  const std::string generation = parent_dir(nodes_dir) + "/generation_id";
  for (int attempt = 0; attempt < 3; ++attempt) {
    const std::string before = CuidUtilities::read_sysfs_file(generation);
    auto nodes = kfd_nodes_under(nodes_dir);
    if (CuidUtilities::read_sysfs_file(generation) == before) return nodes;
  }
  return {};
}

std::string bdf_of(long long domain, long long pci_dev_id) {
  if (domain < 0 || domain > UINT32_MAX || pci_dev_id < 0 || pci_dev_id > UINT16_MAX) return {};
  char text[32];
  const int length =
      std::snprintf(text, sizeof(text), "%04llx:%02llx:%02llx.%llx", domain,
                    (pci_dev_id >> 8) & 0xFF, (pci_dev_id >> 3) & 0x1F, pci_dev_id & 0x7);
  if (length < 0 || static_cast<size_t>(length) >= sizeof(text)) return {};
  return text;
}

// What KFD says of the partition whose render node has `minor`: the PCI
// function it belongs to, and, when that GPU has more than one partition, its
// UnitID. KFD puts the parent's bus<<8|devfn in location_id with the
// partition index k ORed into the function bits, so the parent is function 0
// of that device and n is the number of GPU nodes that share it. Partition k
// of n has UnitID n*(n-1)/2 + 1 + k.
struct KfdPartition {
  std::string parent_bdf;
  uint16_t unit_id = 0;  // 0 when the GPU has a single partition
};

bool kfd_partition(const std::vector<KfdNode>& nodes, long long minor, KfdPartition& out) {
  const KfdNode* self = nullptr;
  for (const auto& node : nodes)
    if (node.render_minor == minor) {
      if (self) return false;
      self = &node;
    }
  if (!self) return false;
  const long long parent = self->location_id & ~0x7LL;
  out = {};
  out.parent_bdf = bdf_of(self->domain, parent);
  if (out.parent_bdf.empty()) return false;
  std::set<long long> indices;
  for (const auto& node : nodes) {
    if (node.domain != self->domain || (node.location_id & ~0x7LL) != parent) continue;
    // Two nodes with one index are not a partition layout this rule knows.
    if (!indices.insert(node.location_id & 0x7).second) return false;
  }
  // The indices must be exactly 0..n-1; a gap means a node is missing.
  const long long n = static_cast<long long>(indices.size());
  if (*indices.rbegin() != n - 1) return false;
  if (n < 2) return true;
  const long long k = self->location_id & 0x7;
  out.unit_id =
      CuidUtilities::partition_unit_id(static_cast<unsigned>(n), static_cast<unsigned>(k));
  return true;
}

constexpr const char kKfdNodes[] = "/sys/class/kfd/kfd/topology/nodes";
constexpr const char kPciDevices[] = "/sys/bus/pci/devices";

// The UnitID KFD gives the partition rendered by `device` (a PCI GPU for
// partition 0, an amdgpu_xcp_<n> platform device for the rest), or 0 where
// its GPU is not partitioned, has no KFD node, or is not the parent KFD names.
uint16_t kfd_unit_id(const std::string& device, const std::string& pci_root,
                     const std::string& kfd_nodes) {
  const int minor = render_minor_of(device);
  if (minor <= 0) return 0;
  KfdPartition partition;
  if (!kfd_partition(kfd_snapshot(kfd_nodes), minor, partition) || partition.unit_id == 0) return 0;
  const std::string name = device.substr(device.find_last_of('/') + 1);
  if (pci_bdf(name) && name != partition.parent_bdf) return 0;
  if (access((pci_root + "/" + partition.parent_bdf).c_str(), F_OK) != 0) return 0;
  return partition.unit_id;
}

// A PCI GPU in a mode with more than one partition. Its own render node then
// serves partition 0, not the device as a whole. Only a device that can be
// partitioned has current_compute_partition.
bool partitioned_gpu(const std::string& device) {
  if (access((device + "/current_compute_partition").c_str(), F_OK) != 0) return false;
  if (access((device + "/xcp/" + CuidUtilities::kDriverUnitIdAttribute).c_str(), F_OK) == 0)
    return true;
  if (kfd_unit_id(device, kPciDevices, kKfdNodes) != 0) return true;
  const auto mode = CuidUtilities::read_sysfs_file(device + "/current_compute_partition");
  return mode != "SPX";
}

// The UnitID amdgpu publishes in `dir`'s cuid_unit_id. False when there is no
// such file; true with 0 when the value is not a partition's.
bool published_unit_id(const std::string& dir, uint16_t& unit_id) {
  const std::string path = dir + "/" + CuidUtilities::kDriverUnitIdAttribute;
  if (access(path.c_str(), F_OK) != 0) return false;
  const std::string text = CuidUtilities::read_sysfs_file(path);
  char* end = nullptr;
  errno = 0;
  const unsigned long value = std::strtoul(text.c_str(), &end, 10);
  // UnitID 0 is the whole GPU and 1 the whole GPU in SLC mode; a partition
  // claiming either would take the device's identity.
  unit_id = (text.empty() || errno == ERANGE || *end != '\0' || value < 2 || value > 8128)
                ? 0
                : static_cast<uint16_t>(value);
  return true;
}

// A partition's UnitID, from cuid_unit_id or KFD, and its parent's IDs: what
// any caller can read.
amdcuid_status_t read_public_partition_metadata(amdcuid_gpu_info& info) {
  CuidPartitionParent parent;
  const auto status = CuidGpu::partition_parent(
      info.partition_attr_dir, info.pci_root.empty() ? kPciDevices : info.pci_root,
      info.kfd_nodes.empty() ? kKfdNodes : info.kfd_nodes, parent);
  if (status != AMDCUID_STATUS_SUCCESS) return status;
  uint16_t unit_id = 0;
  if (!published_unit_id(info.partition_attr_dir, unit_id)) unit_id = parent.kfd_unit_id;
  if (unit_id == 0) return AMDCUID_STATUS_UNSUPPORTED;
  info.header.fields.gpu.unit_id = unit_id;
  info.header.fields.gpu.revision_id = parent.revision_id;
  info.header.fields.gpu.device_id = parent.device_id;
  info.header.fields.gpu.vendor_id = parent.vendor_id;
  info.partition_metadata_valid = true;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t read_partition_metadata(amdcuid_gpu_info& info) {
  if (info.unnamed_vf) {
    info.partition_metadata_valid = false;
    return AMDCUID_STATUS_UNSUPPORTED;
  }
  return read_public_partition_metadata(info);
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
      // A partitioned GPU's DRM nodes serve its partition 0, so the device as
      // a whole is named by its PCI directory alone.
      if (partitioned_gpu(real)) {
        if (drm_input) {
          route.partition = true;
          route.device = resolved_path(real + "/xcp");
          if (route.device.empty()) route.device = real + "/xcp";
          route.node = route.device;
          return AMDCUID_STATUS_SUCCESS;
        }
        route.device = real;
        route.bdf = bdf;
        route.node = "/sys/bus/pci/devices/" + bdf;
        return AMDCUID_STATUS_SUCCESS;
      }
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
  if (access((route.node + "/" + CuidUtilities::kDriverUnitIdAttribute).c_str(), F_OK) != 0) {
    const int err = errno;
    // A stock amdgpu publishes no cuid_unit_id; KFD still names the partition.
    if (kfd_unit_id(parent_dir(route.node), kPciDevices, kKfdNodes) != 0)
      return AMDCUID_STATUS_SUCCESS;
    return (err == EACCES || err == EPERM) ? AMDCUID_STATUS_PERMISSION_DENIED
                                           : AMDCUID_STATUS_UNSUPPORTED;
  }
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

// A GIM-only PCI path that does not resolve must not collapse to its parent
// directory.
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
        std::vector<std::pair<amdcuid_gpu_info, std::string>> found{{info, device_path}};
        // A partitioned GPU's card is its partition 0's; the GPU as a whole is
        // named by its PCI directory.
        if (!info.partition_attr_dir.empty()) {
          const auto real = resolved_path(info.partition_attr_dir);
          const auto device = parent_dir(real.empty() ? info.partition_attr_dir : real);
          const auto bdf = device.substr(device.find_last_of('/') + 1);
          amdcuid_gpu_info whole = {};
          const auto pci = std::string(kPciDevices) + "/" + bdf;
          if (pci_bdf(bdf) &&
              discover_single(&whole, pci, gim_client.get()) == AMDCUID_STATUS_SUCCESS &&
              whole.partition_attr_dir.empty())
            found.emplace_back(whole, pci);
        }
        for (const auto& candidate : found) {
          const auto& gpu = candidate.first;
          if (!seen_nodes.insert(gpu.render_node).second) continue;
          if (!gpu.bdf.empty()) seen_bdfs.insert(gpu.bdf);
          gpus.emplace_back(std::make_shared<CuidGpu>(gpu));

          // Partition 0 has no card node of its own where the GPU's card names
          // the GPU, so it shares this one and would otherwise be missed; it is
          // a different component from the GPU as a whole. The rest have their
          // own card entries and reach this loop.
          if (gpu.partition_attr_dir.empty()) {
            amdcuid_gpu_info partition = {};
            if (discover_partition(&partition, candidate.second) == AMDCUID_STATUS_SUCCESS &&
                seen_nodes.insert(partition.render_node).second) {
              gpus.emplace_back(std::make_shared<CuidGpu>(partition));
            }
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
        if (discover_single(&info, sys_device_path, gim_client.get(), dev.bdf) !=
            AMDCUID_STATUS_SUCCESS) {
          LOG(DEBUG, "GIM: skipping unidentified device at BDF " << dev.bdf);
          continue;
        }
        // A GIM-only device is named by its PCI sysfs directory.
        info.render_node = sys_device_path;
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
                                          cuid::gim::GimClient* gim_client,
                                          const std::string& gim_bdf) {
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
  if (bdf.empty()) bdf = gim_bdf;

  // A VF we could not name is recorded as such rather than rounded to unit_id 0,
  // which would say "the whole card". See CuidUtilities::VfIdentity.
  const CuidUtilities::VfIdentity vf = CuidUtilities::get_gpu_vf_identity(
      resolved == AMDCUID_STATUS_SUCCESS ? route.node + "/device" : device_path);
  info.header.fields.gpu.unit_id = vf.unit_id;
  info.unnamed_vf = vf.is_vf && !vf.index_known;

  // No PCI configuration space under a card whose unit_id is 0 means a spatial
  // partition rather than a whole GPU.
  std::string config_file = device_path + "/config";
  if (resolved != AMDCUID_STATUS_SUCCESS && gim_bdf.empty() &&
      access(config_file.c_str(), F_OK) == -1 && info.header.fields.gpu.unit_id == 0) {
    return discover_partition(gpu_info, device_path);
  }

  auto identity_status =
      PciUtil::read_identity(device_path, bdf, info.header.fields.gpu.vendor_id,
                             info.header.fields.gpu.device_id, info.header.fields.gpu.revision_id);
  if (identity_status == AMDCUID_STATUS_INVALID_FORMAT) return identity_status;

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
      if (identity_status != AMDCUID_STATUS_SUCCESS) {
        if (asic.vendor_id > UINT16_MAX || asic.device_id > UINT16_MAX || asic.rev_id > UINT8_MAX)
          return AMDCUID_STATUS_INVALID_FORMAT;
        info.header.fields.gpu.vendor_id = static_cast<uint16_t>(asic.vendor_id);
        info.header.fields.gpu.device_id = static_cast<uint16_t>(asic.device_id);
        info.header.fields.gpu.revision_id = static_cast<uint8_t>(asic.rev_id);
        identity_status = AMDCUID_STATUS_SUCCESS;
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

  if (identity_status != AMDCUID_STATUS_SUCCESS) return identity_status;
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
  // cuid_unit_id is world-readable, so its presence is what says the kernel
  // is publishing here. A stock amdgpu publishes none, and the partition is
  // the one KFD describes.
  if (access((dir + "/" + CuidUtilities::kDriverUnitIdAttribute).c_str(), F_OK) == 0) return dir;
  const auto real = resolved_path(dir);
  if (!real.empty() && kfd_unit_id(parent_dir(real), kPciDevices, kKfdNodes) != 0) return dir;
  return "";
}

amdcuid_status_t CuidGpu::discover_partition(amdcuid_gpu_info* gpu_info,
                                             const std::string& device_path) {
  if (!gpu_info) return AMDCUID_STATUS_INVALID_ARGUMENT;

  std::string attr_dir = partition_attr_dir_for_device(device_path);
  if (attr_dir.empty()) {
    // Neither cuid_unit_id nor a split KFD topology: not a partition, or one
    // whose UnitID cannot be told. Skipped: a fabricated one would be shared
    // by every partition of the GPU.
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
  // A partition whose UnitID cannot be told is still listed; it has no CUID.
  const auto metadata = read_partition_metadata(info);
  if (metadata != AMDCUID_STATUS_SUCCESS && metadata != AMDCUID_STATUS_FILE_NOT_FOUND &&
      metadata != AMDCUID_STATUS_PERMISSION_DENIED && metadata != AMDCUID_STATUS_UNSUPPORTED) {
    return metadata;
  }

  *gpu_info = info;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidGpu::partition_parent(const std::string& attr_dir, const std::string& pci_root,
                                           const std::string& kfd_nodes,
                                           CuidPartitionParent& parent) {
  parent = {};
  const std::string real = resolved_path(attr_dir);
  const std::string device = parent_dir(real.empty() ? attr_dir : real);
  const std::string name = device.substr(device.find_last_of('/') + 1);
  if (pci_bdf(name)) {
    parent.bdf = name;
  } else {
    // An amdgpu_xcp_<n> platform device has no link to its GPU: KFD names it.
    KfdPartition partition;
    const int minor = render_minor_of(device);
    if (minor <= 0 || !kfd_partition(kfd_snapshot(kfd_nodes), minor, partition))
      return AMDCUID_STATUS_UNSUPPORTED;
    parent.bdf = partition.parent_bdf;
  }
  parent.kfd_unit_id = kfd_unit_id(device, pci_root, kfd_nodes);

  const std::string pci = pci_root + "/" + parent.bdf;
  return PciUtil::read_identity(pci, parent.bdf, parent.vendor_id, parent.device_id,
                                parent.revision_id);
}

// The UnitID of a partition: cuid_unit_id where the driver publishes it,
// otherwise the one KFD implies on a stock kernel. 0 when neither names one.
uint16_t CuidGpu::partition_unit_id(const CuidPartitionParent& parent) const {
  uint16_t unit_id = 0;
  return published_unit_id(m_info.partition_attr_dir, unit_id) ? unit_id : parent.kfd_unit_id;
}

// A partition's temporary input: its parent's routing and IDs plus its own
// UnitID, which is what keeps it apart from the whole GPU and its siblings.
amdcuid_status_t CuidGpu::partition_temporary_input(CuidUtilities::AuxiliaryInput& aux) const {
  if (m_info.unnamed_vf) return AMDCUID_STATUS_UNSUPPORTED;
  CuidPartitionParent parent;
  const auto status = partition_parent(
      m_info.partition_attr_dir, m_info.pci_root.empty() ? kPciDevices : m_info.pci_root,
      m_info.kfd_nodes.empty() ? kKfdNodes : m_info.kfd_nodes, parent);
  if (status != AMDCUID_STATUS_SUCCESS) return status;
  const uint16_t unit_id = partition_unit_id(parent);
  if (unit_id == 0) return AMDCUID_STATUS_UNSUPPORTED;

  aux = {};
  aux.format = CuidUtilities::kAuxFormatPcie;
  CuidUtilities::set_pcie_routing(aux, parent.bdf);
  aux.revision_id = parent.revision_id;
  aux.device_id = parent.device_id;
  aux.vendor_id = parent.vendor_id;
  aux.component_type = static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_GPU);
  aux.unit_id = unit_id;
  return AMDCUID_STATUS_SUCCESS;
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
  // A partition's serial is its parent's, which names the whole GPU.
  if (!m_info.partition_attr_dir.empty()) {
    fingerprint = 0;
    return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
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

amdcuid_status_t CuidGpu::get_primary_cuid(amdcuid_primary_id& /*id*/) const {
  // A GPU's permanent identity comes from amdgpu, not from a serial read
  // here, so a GPU has no primary in this library and takes its temporary
  // CUID.
  return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
}

amdcuid_status_t CuidGpu::get_temporary_input(CuidUtilities::AuxiliaryInput& aux) const {
  if (m_info.unnamed_vf) return AMDCUID_STATUS_UNSUPPORTED;
  // A partition shares its parent's routing id, so only its UnitID keeps it
  // apart; that path reads it from cuid_unit_id or KFD.
  if (!m_info.partition_attr_dir.empty()) return partition_temporary_input(aux);

  // Refused whatever the fingerprint source, so every host reports it alike.
  if (m_info.header.fields.gpu.unit_id > CuidUtilities::kMaxUnitId) {
    LOG(ERROR, "UnitID " << m_info.header.fields.gpu.unit_id << " exceeds the 13-bit field (max "
                         << CuidUtilities::kMaxUnitId << ")");
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }

  std::string bdf;
  const amdcuid_status_t status = this->get_bdf(bdf);
  if (status != AMDCUID_STATUS_SUCCESS) {
    return status;
  }
  aux = {};
  aux.format = CuidUtilities::kAuxFormatPcie;
  CuidUtilities::set_pcie_routing(aux, bdf);
  aux.revision_id = m_info.header.fields.gpu.revision_id;
  aux.device_id = m_info.header.fields.gpu.device_id;
  aux.vendor_id = m_info.header.fields.gpu.vendor_id;
  aux.component_type = static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_GPU);
  aux.unit_id = m_info.header.fields.gpu.unit_id;
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
