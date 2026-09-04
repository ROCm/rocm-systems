// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_pci_device_spec.h
/// @brief Builds a device's bus presentation from the simulation config.
///
/// @details The identity a guest driver reads and the identity KFD reports are
/// the same GPU, so they come from one place in the config rather than two that
/// could disagree. The bus-only values, such as the class code a driver matches
/// on and the aperture sizes, come from the config's PCI section.

#pragma once

#include "rocjitsu/config/kfd_device_config.h"
#include "rocjitsu/config/pci_device_config.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device.h"

namespace rocjitsu {

/// @brief Combine the configured GPU identity and bus shape into a device spec.
/// @param[in] device The GPU's identity, as KFD also reports it.
/// @param[in] pci The bus shape to present it with.
/// @returns The spec to construct a @ref GpuPciDevice from.
[[nodiscard]] GpuPciDeviceSpec gpu_pci_spec_from_config(const config::KfdDeviceConfig &device,
                                                        const config::PciDeviceConfig &pci);

} // namespace rocjitsu
