// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_pci_device.h
/// @brief Transport-neutral AMD GPU device model for the rocjitsu vfio-user server.
///
/// GpuPciDevice implements simdojo::PciDevice for the MI350P/GFX950 GPU.
/// It owns the BAR memory regions (VRAM memfd, doorbell memfd, MMIO shadow)
/// and delegates guest events to SimulatedKfd. It contains no libvfio-user
/// calls; the transport (VfioDeviceHost) handles all vfu_* interactions.

#ifndef ROCJITSU_VFU_GPU_PCI_DEVICE_H_
#define ROCJITSU_VFU_GPU_PCI_DEVICE_H_

#include "simdojo/components/pci_device.h"
#include "rocjitsu/vfu/bar0_vram.h"
#include "rocjitsu/vfu/bar2_doorbell.h"
#include "rocjitsu/vfu/bar5_mmio.h"
#include "rocjitsu/vfu/pci_config.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace rocjitsu {
class SimulatedKfd;
} // namespace rocjitsu

namespace rocjitsu::vfu {

/// @brief AMD GPU PCI device model (MI350P / GFX950).
///
/// Exposes three BARs:
///   BAR0/1: 64-bit prefetchable VRAM (memfd-backed, mmap-able, no callback).
///   BAR2/3: 64-bit prefetchable doorbell window (memfd-backed + access callback).
///   BAR5:   32-bit non-prefetchable MMIO register window (callback only).
class GpuPciDevice : public simdojo::PciDevice {
public:
  /// @param kfd           Non-owning pointer to the rocjitsu KFD backend.
  /// @param guest_pid     Process ID for the guest (from rj_vm_device_open).
  /// @param vram_bar_size VRAM BAR size in bytes (256 MB default or 144 GB ReBAR).
  GpuPciDevice(SimulatedKfd *kfd, uint32_t guest_pid,
               uint64_t vram_bar_size = BarSizes::kBar0VramDefault);

  GpuPciDevice(const GpuPciDevice &) = delete;
  GpuPciDevice &operator=(const GpuPciDevice &) = delete;

  // --- simdojo::PciDevice interface ---

  simdojo::PciId pci_id() const override;
  std::vector<simdojo::BarSpec> bars() const override;
  uint32_t msix_vectors() const override { return kMsiXVectors; }

  /// Returns the backing memfd for BAR0 and BAR2; -1 for BAR5 (register-trapped).
  int bar_fd(int bar_index) const override;

  ssize_t bar_access(int bar_index, std::span<std::byte> buf,
                     uint64_t offset, bool is_write) override;

  void dma_map(const simdojo::DmaRegion &region) override;
  void dma_unmap(const simdojo::DmaRegion &region) override;

private:
  SimulatedKfd *kfd_;       ///< Non-owning; lifetime managed by the caller.
  uint32_t      guest_pid_;

  Bar0Vram     bar0_;
  Bar2Doorbell bar2_;
  MmioModel    bar5_;
};

} // namespace rocjitsu::vfu

#endif // ROCJITSU_VFU_GPU_PCI_DEVICE_H_
