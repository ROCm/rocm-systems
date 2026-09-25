// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file memory_hub_block_model.h
/// @brief A memory hub's VMID-0 configuration and invalidation engines.
///
/// @details A flush writes an engine's request register and polls its
/// acknowledge register for the bit of the VMID being flushed. Nothing else
/// reports the flush finishing, so an unanswered step is not a missing register
/// but a stall of the driver's whole timeout, once per flush -- about nine
/// seconds each, and eighty-eight seconds of one boot before it was answered.
/// VMID-0 requests publish the pending page-table root and aperture into the
/// shared GPU VM before the acknowledge bit becomes visible.
///
/// Both hubs are the same model at different versions, because they are the
/// same hardware: `gfxhub_v12_1` and `mmhub_v4_1_0` differ in where their
/// registers sit and in nothing else this device models. Keying on the version
/// is what stops a GC 12.0 profile being answered with 12.1's addresses, which
/// are 0x10 apart and would leave the driver polling registers this device
/// never defined.

#pragma once

#include "rocjitsu/vm/amdgpu/pci/ip_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rocjitsu {

class SoC;

/// @brief One hub's invalidation-register layout, and the version it describes.
///
/// @details The handshake and VMID-0 configuration are described together
/// because their offsets move with the hub version and invalidation is the
/// operation that publishes one coherent configuration snapshot.
struct HubInvalidationLayout {
  IpHardwareId id = IpHardwareId::Gc; ///< Block the layout belongs to.
  uint16_t major = 0;                 ///< Block major version it was read from.
  uint16_t minor = 0;                 ///< Block minor version it was read from.
  uint16_t revision = 0;              ///< Block revision it was read from.
  uint32_t semaphore = 0;   ///< Engine 0's semaphore, in dwords from the block's first segment.
  uint32_t request = 0;     ///< Engine 0's request, likewise.
  uint32_t acknowledge = 0; ///< Engine 0's acknowledge, likewise.
  uint32_t page_table_base_low = 0;   ///< VMID-0 root low dword.
  uint32_t page_table_base_high = 0;  ///< VMID-0 root high dword.
  uint32_t page_table_start_low = 0;  ///< VMID-0 aperture start page low dword.
  uint32_t page_table_start_high = 0; ///< VMID-0 aperture start page high dword.
  uint32_t page_table_end_low = 0;    ///< VMID-0 aperture end page low dword.
  uint32_t page_table_end_high = 0;   ///< VMID-0 aperture end page high dword.
};

/// @brief The invalidation engines of one published hub record.
class MemoryHubBlockModel final : public IpBlockModel {
public:
  /// @brief Model @p block, when its version has a known register layout.
  ///
  /// @param[in] block The published GC or MMHUB record.
  /// @param[in] registers Window onto that record's registers.
  /// @returns The model, or nullptr for a version with no known layout, which
  ///          has been reported. A device whose flush handshakes cannot be
  ///          answered looks correct right up until the driver waits on one, so
  ///          the caller is expected to refuse rather than carry on.
  [[nodiscard]] static std::unique_ptr<MemoryHubBlockModel> create(const IpBlock &block,
                                                                   IpRegisterWindow registers);

  [[nodiscard]] std::vector<RegisterClaim> claims() const override;
  bool reset() override;
  void attach_soc(SoC *soc) { soc_ = soc; }
  void observe_register_write(uint64_t byte_offset, PciMemoryAccess &memory) override;

private:
  MemoryHubBlockModel(std::string what, IpRegisterWindow registers,
                      const HubInvalidationLayout &layout);

  const HubInvalidationLayout *layout_;
  SoC *soc_ = nullptr;
};

} // namespace rocjitsu
