// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/fabric_backing_kfd.h"

#include <algorithm>
#include <utility>

namespace rocjitsu {
namespace {

class KfdBackingOwner final : public FabricBacking::Owner {
public:
  explicit KfdBackingOwner(KfdProcess::GpuBackingLease lease) : lease_(std::move(lease)) {}

  [[nodiscard]] uint64_t base_offset() const noexcept override { return lease_.file_offset(); }
  [[nodiscard]] size_t extent() const noexcept override { return lease_.size(); }
  [[nodiscard]] uint64_t backing_id() const noexcept override { return lease_.backing_id(); }

  [[nodiscard]] bool read(uint64_t offset, std::span<std::byte> result) noexcept override {
    if (!contains(offset, result.size()))
      return false;
    const auto bytes = lease_.bytes();
    const size_t relative = static_cast<size_t>(offset - base_offset());
    std::copy_n(bytes.begin() + relative, result.size(), result.begin());
    return true;
  }

  [[nodiscard]] bool write(uint64_t offset, std::span<const std::byte> bytes) noexcept override {
    if (!contains(offset, bytes.size()))
      return false;
    const auto backing = lease_.bytes();
    const size_t relative = static_cast<size_t>(offset - base_offset());
    std::copy(bytes.begin(), bytes.end(), backing.begin() + relative);
    return true;
  }

private:
  [[nodiscard]] bool contains(uint64_t offset, size_t size) const noexcept {
    if (!lease_ || offset < base_offset())
      return false;
    const uint64_t relative = offset - base_offset();
    return relative <= extent() && size <= extent() - relative;
  }

  KfdProcess::GpuBackingLease lease_;
};

} // namespace

FabricBacking make_fabric_backing(KfdProcess::GpuBackingLease lease) {
  if (!lease)
    return {};
  const uint32_t owner_gpu_id = lease.owner_gpu_id();
  const uint64_t offset = lease.file_offset();
  const size_t extent = lease.size();
  return FabricBacking(std::make_shared<KfdBackingOwner>(std::move(lease)), owner_gpu_id, offset,
                       extent);
}

} // namespace rocjitsu
