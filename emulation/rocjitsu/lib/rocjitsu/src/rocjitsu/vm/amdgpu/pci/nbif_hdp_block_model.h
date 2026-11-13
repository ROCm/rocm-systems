// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file nbif_hdp_block_model.h
/// @brief NBIF's BAR-remapped host data path flush register.

#pragma once

#include "rocjitsu/vm/amdgpu/pci/ip_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"

#include <atomic>
#include <memory>

namespace rocjitsu {

class SoC;

/// The NBIF profile selects the driver's HDP remap. A write completes in the
/// BAR callback before a later VM-hub invalidation can be acknowledged. The
/// model has no hidden CPU-write buffer: DMA and the guest use the same mapped
/// backing. The flush retires clean translated local cache lines before its
/// completion is visible. It does not authorize a cacheable peer alias.
class NbifHdpBlockModel final : public IpBlockModel {
public:
  [[nodiscard]] static std::unique_ptr<NbifHdpBlockModel> create(const IpBlock &block,
                                                                 IpRegisterWindow registers);

  [[nodiscard]] std::vector<RegisterClaim> claims() const override;
  bool reset() override;
  void attach_soc(SoC *soc) { soc_ = soc; }
  void observe_register_write(uint64_t byte_offset, PciMemoryAccess &memory) override;
  [[nodiscard]] uint64_t completed_flushes() const {
    return completed_flushes_.load(std::memory_order_acquire);
  }

private:
  explicit NbifHdpBlockModel(IpRegisterWindow registers, uint32_t flush_dword);

  uint32_t flush_dword_ = 0;
  SoC *soc_ = nullptr;
  std::atomic<uint64_t> completed_flushes_{0};
};

} // namespace rocjitsu
