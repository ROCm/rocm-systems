// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CUID_GPU_H
#define CUID_GPU_H

#include "include/amd_cuid.h"
#include "src/cuid_device.h"
#include "src/cuid_internal.h"
#include <memory>
#include <string>
#include <vector>

namespace cuid {
namespace gim {
class GimClient;
} // namespace gim
} // namespace cuid

struct amdcuid_gpu_info {
  amdcuid_cuid_public_fields header;
  // DRM device node: /sys/class/drm/renderDXXX or /sys/class/drm/cardN
  std::string render_node;
  std::string bdf;
  // Hardware fingerprint derived from the GIM SMI ASIC serial. Used as a
  // fallback for GIM-only devices whose sysfs unique_id and PCI config space
  // are not exposed to userspace.
  uint64_t gim_fingerprint = 0;
  bool gim_fingerprint_valid = false;
};

class CuidGpu : public CuidDevice {
public:
  CuidGpu(const amdcuid_gpu_info &i);
  amdcuid_device_type_t type() const override {
    return AMDCUID_DEVICE_TYPE_GPU;
  }
  amdcuid_status_t get_primary_cuid(amdcuid_primary_id &id) const override;
  amdcuid_status_t
  get_hardware_fingerprint(uint64_t &fingerprint) const override;
  static amdcuid_status_t discover(std::vector<DevicePtr> &gpus);
  // discover_single populates `gpu_info` from `device_path`. When the host
  // runs the GIM SR-IOV driver, sysfs/PCI config space may not expose the
  // device, in which case the caller can pass an already-initialized
  // GimClient via `gim_client` to be used as a fallback. Passing nullptr
  // disables the GIM fallback for that call (used by code paths that have
  // no GimClient handy, e.g. amdcuid_get_handle_by_dev_path).
  static amdcuid_status_t
  discover_single(amdcuid_gpu_info *gpu_info, const std::string &device_path,
                  cuid::gim::GimClient *gim_client = nullptr);

  // Derive the render_node from an enumeration `device_path`. Strips a
  // trailing "/device" (as passed by /sys/class/drm enumeration) and, for
  // card paths, resolves the associated renderD node when one exists. Paths
  // that are neither (e.g. the GIM "/sys/bus/pci/devices/<bdf>" form) are
  // returned verbatim. Exposed for testing.
  static std::string normalize_render_node(const std::string &device_path);

  // Virtual accessor overrides
  amdcuid_status_t get_vendor_id(uint16_t &vendor_id) const override;
  amdcuid_status_t get_device_id(uint16_t &device_id) const override;
  amdcuid_status_t get_pci_class(uint16_t &pci_class) const override;
  amdcuid_status_t get_revision_id(uint8_t &revision_id) const override;
  amdcuid_status_t get_unit_id(uint16_t &unit_id) const override;
  amdcuid_status_t get_bdf(std::string &bdf) const override;
  amdcuid_status_t get_device_path(std::string &path) const override;

  const amdcuid_gpu_info &get_info() const;

private:
  amdcuid_gpu_info m_info;
};

#endif // CUID_GPU_H
