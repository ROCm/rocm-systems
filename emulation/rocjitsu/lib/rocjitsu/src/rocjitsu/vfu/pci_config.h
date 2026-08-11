// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file pci_config.h
/// @brief PCI identity and capability setup for the rocjitsu vfio-user server.
///
/// Sets up the libvfio-user context with the correct PCI identifiers,
/// BAR sizes, and capabilities to match an AMD Instinct MI350P (GFX950).

#ifndef ROCJITSU_VFU_PCI_CONFIG_H_
#define ROCJITSU_VFU_PCI_CONFIG_H_

#include <cstdint>

// Forward-declare libvfio-user context to avoid pulling in the header here.
typedef struct vfu_ctx vfu_ctx_t;

namespace rocjitsu::vfu {

/// @brief PCI identity constants for AMD Instinct MI350P.
struct PciIdentity {
  static constexpr uint16_t kVendorId         = 0x1002; ///< AMD
  static constexpr uint16_t kDeviceId         = 0x75C8; ///< MI350P (physical chip_id 30120)
  static constexpr uint16_t kSubsystemVendor  = 0x1002;
  static constexpr uint16_t kSubsystemDevice  = 0x0000; ///< Placeholder until AMD publishes it
  static constexpr uint8_t  kRevisionId       = 0x00;   ///< Updated once ext_rev_id confirmed
  // PCI_CLASS_ACCELERATOR_PROCESSING (0x1200) is the class the amdgpu driver
  // wildcard PCI ID entry matches for CHIP_IP_DISCOVERY devices. Using 0x030200
  // (3D controller) does not match any amdgpu wildcard entry and results in
  // flags == 0 → "Unsupported asic" probe rejection.
  static constexpr uint32_t kClassCode        = 0x120000; ///< Accelerator (matches CHIP_IP_DISCOVERY)
};

/// @brief BAR size constants matching real MI350P hardware.
///
/// BAR0/1: 64-bit prefetchable VRAM aperture.
///   - Without ReBAR: 256 MB visible (hardware default without BIOS "Above 4G" + ReBAR).
///   - With ReBAR:    144 GB (full HBM3E capacity).
/// BAR2/3: 64-bit prefetchable doorbell region (2 MB, fixed for all CDNA4).
/// BAR5:   32-bit non-prefetchable MMIO register window (256 KB, fixed for BONAIRE+).
struct BarSizes {
  static constexpr uint64_t kBar0VramDefault  = 512ULL * 1024 * 1024;  ///< 512 MB — GFX9.4.4 needs 280 MB TMR reserve + headroom
  static constexpr uint64_t kBar0VramFull     = 144ULL * 1024 * 1024 * 1024; ///< 144 GB (ReBAR)
  static constexpr uint64_t kBar2Doorbell     = 2ULL * 1024 * 1024;    ///< 2 MB
  static constexpr uint64_t kBar5Mmio        = 256ULL * 1024;          ///< 256 KB
};

/// @brief Number of MSI-X vectors to advertise.
///
/// Matches the interrupt vector count expected by the amdgpu driver for GFX950.
/// The driver allocates vectors per-ring (GFX, compute, SDMA, display, etc.).
inline constexpr uint32_t kMsiXVectors = 256;

/// @brief Configure the vfu_ctx PCI identity and capabilities.
///
/// Must be called after BAR regions have been registered and before
/// vfu_realize_ctx(). Sets PCI IDs, class code, revision, PCIe/PM/MSI-X
/// capabilities, and interrupt counts.
///
/// @param ctx         The libvfio-user context.
/// @param vram_fd     Unused (BAR0 is registered by Bar0Vram::setup).
/// @param vram_size   Unused (BAR0 is registered by Bar0Vram::setup).
/// @param doorbell_fd Unused (BAR2 is registered by Bar2Doorbell::setup).
/// @returns 0 on success, -1 on failure (errno is set).
int setup_pci_config(vfu_ctx_t *ctx, int vram_fd, uint64_t vram_size, int doorbell_fd);

} // namespace rocjitsu::vfu

#endif // ROCJITSU_VFU_PCI_CONFIG_H_
