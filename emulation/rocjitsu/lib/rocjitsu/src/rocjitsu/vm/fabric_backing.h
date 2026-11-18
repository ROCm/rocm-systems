// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file fabric_backing.h
/// @brief Retained, request-bounded backing for a local fabric transfer.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>

namespace rocjitsu {

/// A transfer retains its owner until the request and any committed response
/// are released. Offsets are in the owner's coordinate space, never a guest
/// virtual address. The owner implements access to its own backing source.
class FabricBacking {
public:
  class Owner {
  public:
    virtual ~Owner() = default;

    /// The coordinate range stays fixed for every live FabricBacking view.
    /// Revocation must use a separate permit or generation, never shrink it.
    [[nodiscard]] virtual uint64_t base_offset() const noexcept = 0;
    [[nodiscard]] virtual size_t extent() const noexcept = 0;
    /// Canonical backing identity when the owner has one. Zero is anonymous.
    [[nodiscard]] virtual uint64_t backing_id() const noexcept { return 0; }
    /// Called on the fabric owner thread inside the VM effect gate and the
    /// endpoint drain permit. These methods must not reenter GPU VM, enqueue
    /// topology work, or wait for an engine worker. Access only the supplied
    /// range. A write must commit every byte on success and change no backing
    /// bytes on failure. Stage fallible writes or reject before touching the
    /// backing; never report failure after a partial write.
    [[nodiscard]] virtual bool read(uint64_t offset, std::span<std::byte> result) noexcept = 0;
    [[nodiscard]] virtual bool write(uint64_t offset,
                                     std::span<const std::byte> bytes) noexcept = 0;
  };

  FabricBacking() = default;

  FabricBacking(std::shared_ptr<Owner> owner, uint32_t owner_gpu_id, uint64_t offset, size_t extent)
      : owner_(std::move(owner)), owner_gpu_id_(owner_gpu_id), offset_(offset), extent_(extent) {
    if (!owner_ || offset_ < owner_->base_offset() ||
        (extent_ != 0 && extent_ - 1 > std::numeric_limits<uint64_t>::max() - offset_)) {
      owner_.reset();
      return;
    }
    const uint64_t relative = offset_ - owner_->base_offset();
    if (relative > owner_->extent() || extent_ > owner_->extent() - relative)
      owner_.reset();
  }

  [[nodiscard]] explicit operator bool() const noexcept { return owner_ != nullptr; }
  [[nodiscard]] uint32_t owner_gpu_id() const noexcept { return owner_ ? owner_gpu_id_ : 0; }
  [[nodiscard]] uint64_t backing_id() const noexcept { return owner_ ? owner_->backing_id() : 0; }
  [[nodiscard]] uint64_t offset() const noexcept { return offset_; }
  [[nodiscard]] size_t extent() const noexcept { return extent_; }

  /// @brief Retain a smaller view in this backing's coordinate space.
  /// @details A child cannot escape the request bounds even when the owner
  /// has a larger live allocation. An invalid or overflowing range refuses.
  [[nodiscard]] FabricBacking subview(uint64_t offset, size_t extent) const noexcept {
    if (!owner_ || offset < offset_)
      return {};
    const uint64_t relative = offset - offset_;
    if (relative > extent_ || extent > extent_ - relative)
      return {};
    // The parent was validated against its owner at construction. A child
    // inside that fixed parent needs no owner callback under cache locks.
    return FabricBacking(owner_, owner_gpu_id_, offset, extent, BoundedSubviewTag{});
  }

  /// Read or write exactly the request's extent. The owner must also enforce
  /// its physical bounds, since it alone knows how its offset maps to storage.
  [[nodiscard]] bool read(std::span<std::byte> result) const noexcept {
    return owner_ && result.size() == extent_ && owner_->read(offset_, result);
  }
  [[nodiscard]] bool write(std::span<const std::byte> bytes) const noexcept {
    return owner_ && bytes.size() == extent_ && owner_->write(offset_, bytes);
  }

private:
  class BoundedSubviewTag {};
  FabricBacking(std::shared_ptr<Owner> owner, uint32_t owner_gpu_id, uint64_t offset, size_t extent,
                BoundedSubviewTag)
      : owner_(std::move(owner)), owner_gpu_id_(owner_gpu_id), offset_(offset), extent_(extent) {}

  std::shared_ptr<Owner> owner_;
  uint32_t owner_gpu_id_ = 0;
  uint64_t offset_ = 0;
  size_t extent_ = 0;
};

} // namespace rocjitsu
