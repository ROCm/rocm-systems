// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file graphics_block_model.h
/// @brief Graphics-core identity and per-VMID shader aperture registers.

#pragma once

#include "rocjitsu/vm/amdgpu/pci/ip_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"

#include <array>
#include <cstdint>
#include <memory>

namespace rocjitsu {

/// @brief One graphics block's control registers and selected shader VMID bank.
class GraphicsBlockModel final : public IpBlockModel {
public:
  [[nodiscard]] static std::unique_ptr<GraphicsBlockModel>
  create(const IpBlock &block, const GraphicsDiscoveryInfo &graphics, IpRegisterWindow registers);

  [[nodiscard]] std::vector<RegisterClaim> claims() const override;
  bool reset() override;
  void observe_register_write(uint64_t byte_offset, PciMemoryAccess &memory) override;

private:
  GraphicsBlockModel(IpRegisterWindow registers, uint32_t gb_addr_config);
  void publish_selected_apertures(PciMemoryAccess &memory);

  uint32_t gb_addr_config_ = 0;
  std::array<uint32_t, 16> sh_mem_bases_{};
  std::array<uint32_t, 16> sh_mem_config_{};
  uint32_t selected_vmid_ = 0;
};

} // namespace rocjitsu
