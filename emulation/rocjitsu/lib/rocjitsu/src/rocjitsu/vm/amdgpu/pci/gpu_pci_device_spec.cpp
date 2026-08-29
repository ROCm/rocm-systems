// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device_spec.h"

#include "rocjitsu/vm/amdgpu/pci/ip_discovery_profile.h"
#include "util/log.h"

#include <algorithm>
#include <format>

namespace rocjitsu {
namespace {

/// @brief Largest power of two no larger than @p limit, or zero if there is
/// none.
///
/// @details A limit of zero has no answer, and returning one would turn a
/// missing memory size into a one-byte aperture that looks valid.
uint64_t largest_power_of_two_within(uint64_t limit) {
  if (limit == 0) {
    return 0;
  }
  uint64_t size = 1;
  while (size <= limit / 2) {
    size *= 2;
  }
  return size;
}

/// @brief Largest window onto video memory chosen when a config asks for none.
///
/// @details Memory capacities are rarely powers of two and are often enormous,
/// so exposing all of one as a BAR is neither legal nor necessary: the indirect
/// window reaches whatever the aperture does not.
constexpr uint64_t kDefaultVramApertureBytes = 256 * 1024 * 1024;

/// @brief KFD target version of the one part a discovery profile exists for.
constexpr uint32_t kGfx1250TargetVersion = 120500;

/// @brief Choose the IP blocks to describe for @p device.
///
/// @details gfx1250 is the only part modelled well enough to publish. Any other
/// target, including one left unset, gets no blocks: publishing gfx1250's table
/// for a configuration that models a different part would have the guest driver
/// bind support for hardware the rest of the simulation is not, which fails
/// later and further away than refusing here. An empty profile makes the device
/// refuse to become usable, and says why.
/// @param[in] device The configured device.
/// @returns The blocks to describe, empty if this part has no profile.
[[nodiscard]] IpDiscoverySpec discovery_spec_for(const config::KfdDeviceConfig &device) {
  if (device.gfx_target_version != kGfx1250TargetVersion) {
    util::Logger::warn(std::format(
        "gfx target {} has no IP discovery profile, so this device cannot describe itself to a "
        "guest driver; only gfx{} is modelled",
        device.gfx_target_version, kGfx1250TargetVersion));
    return {};
  }
  return gfx1250_discovery_spec();
}

} // namespace

GpuPciDeviceSpec gpu_pci_spec_from_config(const config::KfdDeviceConfig &device,
                                          const config::PciDeviceConfig &pci) {
  GpuPciDeviceSpec spec;
  spec.id.vendor = static_cast<uint16_t>(device.vendor_id);
  spec.id.device = static_cast<uint16_t>(device.device_id);
  // A subsystem that names itself after the device is the common case for a
  // reference board, so an unset value follows the device rather than reading
  // as an unrelated vendor.
  spec.id.subsys_vendor = static_cast<uint16_t>(
      pci.subsystem_vendor_id != 0 ? pci.subsystem_vendor_id : device.vendor_id);
  spec.id.subsys =
      static_cast<uint16_t>(pci.subsystem_id != 0 ? pci.subsystem_id : device.device_id);
  spec.id.cls = static_cast<uint8_t>((pci.class_code >> 16) & 0xff);
  spec.id.subcls = static_cast<uint8_t>((pci.class_code >> 8) & 0xff);
  spec.id.prog_if = static_cast<uint8_t>(pci.class_code & 0xff);
  spec.id.revision = static_cast<uint8_t>(device.pci_revision_id);

  spec.vram_bytes = device.local_mem_size;
  // An unset aperture is resolved here rather than left for each consumer to
  // interpret, so everything downstream sees the same explicit window.
  spec.vram_aperture_bytes =
      pci.vram_aperture_bytes != 0
          ? pci.vram_aperture_bytes
          : largest_power_of_two_within(std::min(device.local_mem_size, kDefaultVramApertureBytes));
  spec.doorbell_aperture_bytes = pci.doorbell_aperture_bytes;
  spec.register_aperture_bytes = pci.register_aperture_bytes;
  // The blocks are chosen here, next to the identity, so the two describe one
  // GPU.
  spec.discovery = discovery_spec_for(device);
  return spec;
}

} // namespace rocjitsu
