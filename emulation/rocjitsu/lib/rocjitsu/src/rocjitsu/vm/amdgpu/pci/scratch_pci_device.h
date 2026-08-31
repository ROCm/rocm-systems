// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file scratch_pci_device.h
/// @brief A PCI function whose only behavior is to remember what was written.
///
/// @details Bringing up a transport and bringing up a device model are separate
/// problems, and debugging them at the same time is what makes PCI emulation
/// slow to start. This device exists so the transport can be exercised on its
/// own: a guest can enumerate it, map its BAR, write bytes, and read them back,
/// and any failure is the transport's.
///
/// It is deliberately not a model of anything. It has one small trapped BAR, so
/// every access reaches @ref rocjitsu::ScratchPciDevice::bar_access and can be
/// observed, which also makes it the fixture for tests of the access
/// diagnostics.

#pragma once

#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"
#include "simdojo/components/pci_device.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

/// @brief A single-BAR PCI function backed by a byte buffer.
///
class ScratchPciDevice final : public simdojo::PciDevice {
public:
  /// @brief BAR index the scratch region is exposed through.
  static constexpr int kBarIndex = 0;

  /// @brief Size of the scratch region in bytes.
  static constexpr uint64_t kBarSize = 4096;

  /// @brief Construct a scratch function.
  /// @param[in] name Component name, used in diagnostics.
  /// @param[in] id Identity to present in configuration space.
  /// @param[in] trace Access diagnostics to feed, or nullptr for none. Must
  ///                  outlive this device.
  ScratchPciDevice(std::string name, simdojo::PciId id, BarAccessTrace *trace);

  [[nodiscard]] std::vector<simdojo::BarSpec> bars() const override;

  [[nodiscard]] int64_t bar_access(int bar, std::span<std::byte> buf, uint64_t offset,
                                   bool write) override;
  void dma_map(const simdojo::DmaRegion &region) override;
  void dma_unmap(const simdojo::DmaRegion &region) override;
  void reset(simdojo::ResetKind kind) override;

  /// @brief Return how many guest memory windows are currently mapped.
  [[nodiscard]] std::size_t mapped_regions() const { return mapped_regions_; }

private:
  BarAccessTrace *trace_;
  std::vector<std::byte> storage_;
  std::size_t mapped_regions_ = 0;
};

} // namespace rocjitsu
