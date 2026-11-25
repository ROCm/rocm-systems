// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file retained_backing_range.h
/// @brief Retain a byte range in existing storage for a deferred transfer.
///
/// SparseMemory owns the pages of a simulated address space. A
/// RetainedBackingRange refers to bytes that a caller has already resolved,
/// for example a range of a shared file mapping. Copies keep the same owner
/// alive and access the same bytes.

#pragma once

#include "simdojo/components/backing_store_id.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <variant>

namespace simdojo {

/// Optional access to storage through owner-supplied read/write operations.
/// A contiguous mapping can use RetainedBackingRange::mapped() directly.
/// Offsets are measured from the storage origin. The ID, base and extent
/// must stay fixed while a range retains this object.
class BackingRangeAccess {
public:
  virtual ~BackingRangeAccess() = default;
  [[nodiscard]] virtual uint64_t base_offset() const noexcept = 0;
  [[nodiscard]] virtual size_t extent() const noexcept = 0;
  [[nodiscard]] virtual BackingStoreId backing_id() const noexcept { return 0; }
  [[nodiscard]] virtual bool read(uint64_t store_offset, std::span<std::byte> result) noexcept = 0;
  /// A failed write leaves every byte unchanged.
  [[nodiscard]] virtual bool write(uint64_t store_offset,
                                   std::span<const std::byte> bytes) noexcept = 0;
};

/// A copyable view of a fixed interval in an existing storage object. A
/// queued operation can keep the selected bytes alive after the mapping used
/// to find them is retired. The caller checks access rights and revocation
/// before using the range and synchronizes conflicting accesses.
class RetainedBackingRange {
public:
  RetainedBackingRange() = default;
  RetainedBackingRange(std::shared_ptr<BackingRangeAccess> access, uint64_t store_offset,
                       size_t extent) noexcept {
    if (access && valid_extent(store_offset, extent) &&
        contains(access->base_offset(), access->extent(), store_offset, extent)) {
      id_ = access->backing_id();
      offset_ = store_offset;
      extent_ = extent;
      storage_ = std::move(access);
    }
  }

  /// first_byte points at the start of the requested range and retains its
  /// mapping. The caller must establish that extent bytes are accessible
  /// there. store_offset identifies that byte relative to the storage origin
  /// (for example a file offset).
  [[nodiscard]] static RetainedBackingRange mapped(std::shared_ptr<std::byte> first_byte,
                                                   BackingStoreId id, uint64_t store_offset,
                                                   size_t extent) noexcept {
    RetainedBackingRange view;
    if (first_byte && valid_extent(store_offset, extent)) {
      view.storage_ = std::move(first_byte);
      view.id_ = id;
      view.offset_ = store_offset;
      view.extent_ = extent;
    }
    return view;
  }
  [[nodiscard]] explicit operator bool() const noexcept {
    return std::visit([](const auto &owner) { return static_cast<bool>(owner); }, storage_);
  }
  [[nodiscard]] BackingStoreId backing_id() const noexcept { return id_; }
  [[nodiscard]] uint64_t offset() const noexcept { return offset_; }
  [[nodiscard]] size_t extent() const noexcept { return extent_; }

  /// Select a child by absolute store offset without calling the provider or
  /// allocating. It must fit inside this view.
  [[nodiscard]] RetainedBackingRange subview(uint64_t store_offset, size_t extent) const noexcept {
    if (!*this || !contains(offset_, extent_, store_offset, extent)) {
      return {};
    }
    RetainedBackingRange child = *this;
    if (const auto *first_byte = std::get_if<std::shared_ptr<std::byte>>(&storage_)) {
      child.storage_ =
          std::shared_ptr<std::byte>(*first_byte, first_byte->get() + (store_offset - offset_));
    }
    child.offset_ = store_offset;
    child.extent_ = extent;
    return child;
  }

  /// Copy exactly extent() bytes into the caller's buffer.
  [[nodiscard]] bool read(std::span<std::byte> result) const noexcept {
    if (!*this || result.size() != extent_) {
      return false;
    }
    if (const auto *first_byte = std::get_if<std::shared_ptr<std::byte>>(&storage_)) {
      if (!result.empty()) {
        std::memmove(result.data(), first_byte->get(), extent_);
      }
      return true;
    }
    return std::get<std::shared_ptr<BackingRangeAccess>>(storage_)->read(offset_, result);
  }

  /// Replace exactly extent() bytes in the retained storage.
  [[nodiscard]] bool write(std::span<const std::byte> bytes) const noexcept {
    if (!*this || bytes.size() != extent_) {
      return false;
    }
    if (const auto *first_byte = std::get_if<std::shared_ptr<std::byte>>(&storage_)) {
      if (!bytes.empty()) {
        std::memmove(first_byte->get(), bytes.data(), extent_);
      }
      return true;
    }
    return std::get<std::shared_ptr<BackingRangeAccess>>(storage_)->write(offset_, bytes);
  }

private:
  static bool valid_extent(uint64_t offset, size_t extent) noexcept {
    return extent == 0 || extent - 1 <= std::numeric_limits<uint64_t>::max() - offset;
  }

  static bool contains(uint64_t base, size_t length, uint64_t offset, size_t extent) noexcept {
    return offset >= base && offset - base <= length && extent <= length - (offset - base);
  }

  std::variant<std::shared_ptr<BackingRangeAccess>, std::shared_ptr<std::byte>> storage_;
  BackingStoreId id_ = 0;
  uint64_t offset_ = 0;
  size_t extent_ = 0;
};

} // namespace simdojo
