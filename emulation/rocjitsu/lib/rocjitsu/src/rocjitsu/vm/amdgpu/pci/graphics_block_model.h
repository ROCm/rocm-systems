// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file graphics_block_model.h
/// @brief Fixed graphics-core registers needed before ring startup.

#pragma once

#include "rocjitsu/vm/amdgpu/pci/ip_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"

#include <cstdint>
#include <memory>

namespace rocjitsu {

/// @brief The non-VM register surface of one graphics block.
class GraphicsBlockModel final : public IpBlockModel {
public:
  [[nodiscard]] static std::unique_ptr<GraphicsBlockModel>
  create(const IpBlock &block, const GraphicsDiscoveryInfo &graphics, IpRegisterWindow registers);

  [[nodiscard]] std::vector<RegisterClaim> claims() const override;
  bool reset() override;

private:
  GraphicsBlockModel(IpRegisterWindow registers, uint32_t gb_addr_config);

  uint32_t gb_addr_config_ = 0;
};

} // namespace rocjitsu
