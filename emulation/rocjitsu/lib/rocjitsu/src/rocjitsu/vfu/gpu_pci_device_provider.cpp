// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vfu/gpu_pci_device_provider.h"

namespace rocjitsu::vfu {

GpuPciDeviceProvider::GpuPciDeviceProvider(SimulatedKfd *kfd,
                                             uint32_t guest_pid,
                                             uint64_t vram_bar_size) {
  // One device per guest process (single-GPU case today; extend here for MIG).
  devices_.push_back(
      std::make_unique<GpuPciDevice>(kfd, guest_pid, vram_bar_size));
}

std::vector<simdojo::PciDevice *> GpuPciDeviceProvider::pci_devices() {
  std::vector<simdojo::PciDevice *> out;
  out.reserve(devices_.size());
  for (auto &d : devices_)
    out.push_back(d.get());
  return out;
}

} // namespace rocjitsu::vfu
