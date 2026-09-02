// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file pci_device_config.h
/// @brief How a simulated GPU presents itself on a PCI bus.
///
/// @details These are the values a front end needs that describe the bus rather
/// than the GPU: the class a guest driver matches on, and the apertures it maps.
/// Identity proper is not repeated here, because the vendor, device and revision
/// a driver reads are the same values KFD reports and already live in
/// @ref rocjitsu::config::KfdDeviceConfig.

#pragma once

#include <cstdint>

namespace rocjitsu::config {

/// @brief PCI bus shape from the simulation config.
struct PciDeviceConfig {
  /// @brief 24-bit PCI class, subclass and programming interface.
  ///
  /// @details The default presents a processing accelerator, which is what a
  /// compute GPU without a display is, and is one of the classes amdgpu's PCI
  /// table binds on.
  uint32_t class_code = 0x120000;

  uint32_t subsystem_vendor_id = 0; ///< Zero means the same as the vendor id.
  uint32_t subsystem_id = 0;        ///< Zero means the same as the device id.

  /// @brief Window onto video memory to expose as a BAR.
  ///
  /// @details Zero asks for an implementation-chosen legal window, currently the
  /// largest power of two within 256 MiB and within local memory. Exposing all
  /// of memory is usually neither legal as a BAR nor useful, and the indirect
  /// register window reaches whatever the aperture does not.
  uint64_t vram_aperture_bytes = 0;
  uint64_t doorbell_aperture_bytes = 2 * 1024 * 1024; ///< Doorbell aperture size.
  uint64_t register_aperture_bytes = 512 * 1024;      ///< Register aperture size.
};

} // namespace rocjitsu::config
