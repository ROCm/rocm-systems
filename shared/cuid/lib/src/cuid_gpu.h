// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_GPU_H
#define CUID_GPU_H

#include <memory>
#include <string>
#include <vector>

#include "include/amd_cuid.h"
#include "src/cuid_device.h"
#include "src/cuid_internal.h"

namespace cuid {
namespace gim {
class GimClient;
}  // namespace gim
}  // namespace cuid

struct amdcuid_gpu_info {
  amdcuid_cuid_public_fields header;
  // DRM device node: /sys/class/drm/renderDXXX or /sys/class/drm/cardN
  std::string render_node;
  std::string bdf;
  // An SR-IOV virtual function whose index could not be determined, which is
  // what a guest sees: there is no physfn link there, so nothing says which
  // share of the card this is. Everything else reachable belongs to the card,
  // so this device has no serial of its own and takes the auxiliary path.
  bool unnamed_vf = false;
  // Set only for a spatial partition, whose driver-published CUID attributes
  // live under its own node (/sys/class/drm/cardN/xcp) because its BDF names
  // the whole GPU. Empty for an ordinary GPU, which is answered by BDF.
  std::string partition_attr_dir;
  bool partition_metadata_valid = false;
  // The GIM SMI ASIC serial, reported as the hardware fingerprint of a
  // GIM-only device whose sysfs unique_id and PCI config space are not exposed
  // to userspace.
  uint64_t gim_fingerprint = 0;
  bool gim_fingerprint_valid = false;
};

// Verified routing information, not an identity inferred from a CUID or BDF.
struct CuidGpuRoute {
  std::string node;
  std::string device;
  std::string bdf;
  bool partition = false;
};

class CuidGpu : public CuidDevice {
 public:
  CuidGpu(const amdcuid_gpu_info& i);
  amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_GPU; }
  amdcuid_status_t get_primary_cuid(amdcuid_primary_id& id) const override;
  amdcuid_status_t get_derived_cuid(amdcuid_derived_id& id,
                                    cuid_hmac* hmac = nullptr) const override;
  amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const override;
  amdcuid_status_t driver_attribute_path(const std::string& attribute,
                                         std::string& path) const override;

  // Where the kernel publishes a partition's CUID, given the node a card points
  // at (/sys/class/drm/cardN/device: the PCI device for the first partition, an
  // amdgpu_xcp platform device for the rest). Empty when nothing is published
  // there. Path-based so a test can point it at a fabricated tree.
  static std::string partition_attr_dir_for_device(const std::string& device_path);

  // Build the device entry for the spatial partition published under
  // `device_path`, or UNSUPPORTED when there is none.
  static amdcuid_status_t discover_partition(amdcuid_gpu_info* gpu_info,
                                             const std::string& device_path);
  static amdcuid_status_t discover(std::vector<DevicePtr>& gpus);
  // discover_single populates `gpu_info` from `device_path`. When the host
  // runs the GIM SR-IOV driver, sysfs/PCI config space may not expose the
  // device, in which case the caller can pass an already-initialized
  // GimClient via `gim_client` to be used as a fallback. Passing nullptr
  // disables the GIM fallback for that call (used by code paths that have
  // no GimClient handy, e.g. amdcuid_get_handle_by_dev_path).
  static amdcuid_status_t discover_single(amdcuid_gpu_info* gpu_info,
                                          const std::string& device_path,
                                          cuid::gim::GimClient* gim_client = nullptr);

  // Source 1 of the serial precedence: amdgpu's sysfs `unique_id` attribute.
  // Succeeds only for a well-formed, non-zero value; absent, empty, malformed
  // or all-zero is a failure, so the caller carries on to the PCIe Device
  // Serial Number rather than stopping at a source that yields nothing. `path`
  // is a full path so a test can point it at a file.
  static amdcuid_status_t read_unique_id(const std::string& path, uint64_t& fingerprint);

  // Sources 2 and 3: the PCIe Device Serial Number extended capability, then
  // the vendor-specific capability of a function whose Vendor ID is
  // `vendor_id`, each accepted only if it yields a non-zero value. Exposed for
  // testing.
  static amdcuid_status_t read_config_space_serial(const std::string& bdf, uint16_t vendor_id,
                                                   uint64_t& fingerprint);

  // Prefer a verified render node, then a card node, then the PCI directory.
  // Paths that do not resolve are kept as given (minus a /device suffix).
  // Path equivalence itself always requires successful resolve_path evidence.
  static std::string normalize_render_node(const std::string& device_path);
  static amdcuid_status_t resolve_path(const std::string& path, CuidGpuRoute& route);
  bool matches_route(const CuidGpuRoute& route) const;
  bool same_device(const CuidGpu& other) const;

  // Virtual accessor overrides
  amdcuid_status_t get_vendor_id(uint16_t& vendor_id) const override;
  amdcuid_status_t get_device_id(uint16_t& device_id) const override;
  amdcuid_status_t get_pci_class(uint16_t& pci_class) const override;
  amdcuid_status_t get_revision_id(uint8_t& revision_id) const override;
  amdcuid_status_t get_unit_id(uint16_t& unit_id) const override;
  amdcuid_status_t get_bdf(std::string& bdf) const override;
  amdcuid_status_t get_device_path(std::string& path) const override;

  const amdcuid_gpu_info& get_info() const;

 private:
  amdcuid_gpu_info m_info;
};

#endif  // CUID_GPU_H
