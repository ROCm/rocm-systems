// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device_spec.h"

#include <algorithm>

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
  return spec;
}

} // namespace rocjitsu
