// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_pci_device_provider.h
/// @brief Provider that yields one GpuPciDevice per GPU in the rocjitsu VM.

#ifndef ROCJITSU_VFU_GPU_PCI_DEVICE_PROVIDER_H_
#define ROCJITSU_VFU_GPU_PCI_DEVICE_PROVIDER_H_

#include "simdojo/components/pci_device.h"
#include "rocjitsu/vfu/gpu_pci_device.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace rocjitsu {
class SimulatedKfd;
} // namespace rocjitsu

namespace rocjitsu::vfu {

/// @brief Factory that owns one GpuPciDevice per rocjitsu GPU.
///
/// At construction the provider calls rj_vm_device_open() for each GPU and
/// creates the corresponding GpuPciDevice. The caller retains ownership of
/// the rj_vm_t handle; the provider borrows the SimulatedKfd pointer.
class GpuPciDeviceProvider : public simdojo::PciDeviceProvider {
public:
  /// @param kfd           SimulatedKfd backing the GPU(s).
  /// @param guest_pid     Guest process ID (from rj_vm_device_open).
  /// @param vram_bar_size VRAM BAR size in bytes.
  GpuPciDeviceProvider(SimulatedKfd *kfd, uint32_t guest_pid,
                        uint64_t vram_bar_size);

  GpuPciDeviceProvider(const GpuPciDeviceProvider &) = delete;
  GpuPciDeviceProvider &operator=(const GpuPciDeviceProvider &) = delete;

  std::vector<simdojo::PciDevice *> pci_devices() override;

private:
  std::vector<std::unique_ptr<GpuPciDevice>> devices_;
};

} // namespace rocjitsu::vfu

#endif // ROCJITSU_VFU_GPU_PCI_DEVICE_PROVIDER_H_
