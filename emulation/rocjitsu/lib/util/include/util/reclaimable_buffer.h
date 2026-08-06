// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file reclaimable_buffer.h
/// @brief Stable zero-filled storage with best-effort physical-page reclamation.

#ifndef UTIL_RECLAIMABLE_BUFFER_H_
#define UTIL_RECLAIMABLE_BUFFER_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace util {

/// @brief Move-only owner of aligned, zero-initialized byte storage.
///
/// @details On Linux, the storage uses a private anonymous mapping so untouched
/// pages do not consume physical memory. zero_and_reclaim() restores the
/// requested range to zero and returns wholly covered pages to the operating
/// system when possible. Other platforms use aligned heap storage and clear the
/// requested range eagerly. Reclamation is always best-effort and never affects
/// correctness. The address returned by data() remains stable until ownership
/// is transferred or the buffer is destroyed.
class ReclaimableBuffer {
public:
  ReclaimableBuffer() noexcept = default;
  ~ReclaimableBuffer() noexcept { release(); }

  ReclaimableBuffer(const ReclaimableBuffer &) = delete;
  ReclaimableBuffer &operator=(const ReclaimableBuffer &) = delete;

  ReclaimableBuffer(ReclaimableBuffer &&other) noexcept { move_from(other); }
  ReclaimableBuffer &operator=(ReclaimableBuffer &&other) noexcept {
    if (this != &other) {
      release();
      move_from(other);
    }
    return *this;
  }

  /// @brief Allocate stable, zero-initialized storage.
  /// @param bytes Number of usable bytes.
  /// @param alignment Required alignment of data(); must be a power of two.
  /// @pre The buffer is empty and @p alignment is nonzero and a power of two.
  /// @throws std::bad_alloc if the requested size overflows or allocation fails.
  void allocate(size_t bytes, size_t alignment) {
    assert(data_ == nullptr && "ReclaimableBuffer already allocated");
    assert(alignment != 0 && (alignment & (alignment - 1)) == 0 &&
           "alignment must be a nonzero power of two");
    if (bytes == 0)
      return;

    alignment_ = std::max(alignment, alignof(std::max_align_t));
#if defined(__linux__)
    allocate_anonymous_mapping(bytes);
#else
    void *allocation = ::operator new(bytes, std::align_val_t(alignment_));
    allocation_ = static_cast<std::byte *>(allocation);
    data_ = allocation_;
    size_ = bytes;
    std::memset(data_, 0, size_);
#endif
  }

  /// @brief Restore a byte range to zero and reclaim its physical pages.
  /// @param offset Offset from data() to the first byte to clear.
  /// @param bytes Number of bytes to clear.
  /// @pre The requested range is wholly contained in the buffer.
  ///
  /// @details Bytes outside the requested range are preserved. Physical-page
  /// reclamation is best-effort; if it is unavailable or fails, the range is
  /// still cleared eagerly.
  void zero_and_reclaim(size_t offset, size_t bytes) noexcept {
    assert(offset <= size_ && bytes <= size_ - offset && "range exceeds buffer");
    if (bytes == 0)
      return;

    std::byte *first = data_ + offset;
#if defined(__linux__)
    std::byte *last = first + bytes;
    const size_t page_size = system_page_size();
    const uintptr_t first_address = reinterpret_cast<uintptr_t>(first);
    const uintptr_t last_address = reinterpret_cast<uintptr_t>(last);
    const size_t first_remainder = first_address % page_size;
    const uintptr_t page_first_address =
        first_address + (first_remainder == 0 ? 0 : page_size - first_remainder);
    const uintptr_t page_last_address = last_address - last_address % page_size;

    if (page_first_address >= page_last_address) {
      std::memset(first, 0, bytes);
      return;
    }

    std::byte *page_first = reinterpret_cast<std::byte *>(page_first_address);
    std::byte *page_last = reinterpret_cast<std::byte *>(page_last_address);
    std::memset(first, 0, static_cast<size_t>(page_first - first));
    const size_t full_page_bytes = static_cast<size_t>(page_last - page_first);
    if (madvise(page_first, full_page_bytes, MADV_DONTNEED) != 0)
      std::memset(page_first, 0, full_page_bytes);
    std::memset(page_last, 0, static_cast<size_t>(last - page_last));
#else
    std::memset(first, 0, bytes);
#endif
  }

  /// @brief Return the first usable byte, or nullptr for an empty buffer.
  [[nodiscard]] std::byte *data() noexcept { return data_; }

  /// @brief Return the first usable byte, or nullptr for an empty buffer.
  [[nodiscard]] const std::byte *data() const noexcept { return data_; }

  /// @brief Return the number of usable bytes.
  [[nodiscard]] size_t size() const noexcept { return size_; }

private:
#if defined(__linux__)
  void allocate_anonymous_mapping(size_t bytes) {
    const size_t padding = checked_padding(alignment_);
    if (bytes > std::numeric_limits<size_t>::max() - padding)
      throw std::bad_alloc();

    const size_t required_bytes = bytes + padding;
    const size_t page_size = system_page_size();
    const size_t page_remainder = required_bytes % page_size;
    const size_t page_padding = page_remainder == 0 ? 0 : page_size - page_remainder;
    if (required_bytes > std::numeric_limits<size_t>::max() - page_padding)
      throw std::bad_alloc();
    allocation_bytes_ = required_bytes + page_padding;

    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    // Sparse consumers can reserve many buffers. Avoid charging untouched
    // virtual pages under strict overcommit policies.
    flags |= MAP_NORESERVE;
#endif
    void *mapping = mmap(nullptr, allocation_bytes_, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (mapping == MAP_FAILED) {
      allocation_bytes_ = 0;
      alignment_ = 0;
      throw std::bad_alloc();
    }

    // Starting a byte-array lifetime permits implicit-lifetime objects to be
    // created in this storage without initializing and faulting in its pages.
    allocation_ = ::new (mapping) std::byte[allocation_bytes_];
    const uintptr_t allocation_address = reinterpret_cast<uintptr_t>(allocation_);
    // Keep ordinary buffers at the allocator-like max-aligned offset instead
    // of placing every one at the mapping's page-aligned cache index.
    const uintptr_t candidate_address = allocation_address + alignment_;
    const size_t candidate_remainder = candidate_address % alignment_;
    const uintptr_t data_address =
        candidate_address + (candidate_remainder == 0 ? 0 : alignment_ - candidate_remainder);
    data_ = reinterpret_cast<std::byte *>(data_address);
    size_ = bytes;

#ifdef MADV_NOHUGEPAGE
    // Sparse access should not turn one touched base page into a huge-page
    // commitment under transparent huge-page policies.
    (void)madvise(allocation_, allocation_bytes_, MADV_NOHUGEPAGE);
#endif
  }

  static size_t checked_padding(size_t alignment) {
    if (alignment > std::numeric_limits<size_t>::max() / 2 + size_t{1})
      throw std::bad_alloc();
    return alignment + (alignment - 1);
  }

  static size_t system_page_size() noexcept {
    static const size_t value = [] {
      const long result = sysconf(_SC_PAGESIZE);
      return result > 0 ? static_cast<size_t>(result) : size_t{4096};
    }();
    return value;
  }
#endif

  void move_from(ReclaimableBuffer &other) noexcept {
    allocation_ = std::exchange(other.allocation_, nullptr);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    alignment_ = std::exchange(other.alignment_, 0);
    allocation_bytes_ = std::exchange(other.allocation_bytes_, 0);
  }

  void release() noexcept {
    if (allocation_ == nullptr)
      return;
#if defined(__linux__)
    (void)munmap(allocation_, allocation_bytes_);
#else
    ::operator delete(allocation_, std::align_val_t(alignment_));
#endif
    allocation_ = nullptr;
    data_ = nullptr;
    size_ = 0;
    alignment_ = 0;
    allocation_bytes_ = 0;
  }

  std::byte *allocation_ = nullptr;
  std::byte *data_ = nullptr;
  size_t size_ = 0;
  size_t alignment_ = 0;
  size_t allocation_bytes_ = 0;
};

} // namespace util

#endif // UTIL_RECLAIMABLE_BUFFER_H_
