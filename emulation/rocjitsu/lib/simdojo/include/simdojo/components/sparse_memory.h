// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sparse_memory.h
/// @brief Sparse page-table memory model with on-demand page allocation.

#ifndef SIMDOJO_COMPONENTS_SPARSE_MEMORY_H_
#define SIMDOJO_COMPONENTS_SPARSE_MEMORY_H_

#include "simdojo/sim/component.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simdojo {

/// @brief Sparse page-table memory model as a simulation Component.
///
/// Little-endian, 4KB pages allocated on first access. Provides byte, word,
/// and doubleword access plus bulk image loading and page iteration for
/// serialization.
///
class SparseMemory : public Component {
public:
  static constexpr size_t PAGE_SIZE = 4096;
  static constexpr size_t PAGE_MASK = PAGE_SIZE - 1;
  static constexpr size_t PAGE_SHIFT = 12;
  using Page = std::array<uint8_t, PAGE_SIZE>;

  /// @brief Construct a sparse memory component.
  /// @param name Human-readable component name.
  explicit SparseMemory(std::string name) : Component(std::move(name)) {}

  /// @brief Read an 8-bit value from the given address.
  /// @param addr Memory address to read from.
  /// @returns The byte at the given address (0 if page not yet allocated).
  uint8_t read8(uint64_t addr) const {
    uint8_t val = 0;
    read_block(addr, std::span<uint8_t>(&val, 1));
    return val;
  }

  /// @brief Read a 16-bit value from the given address (little-endian).
  /// @param addr Memory address to read from.
  /// @returns The 16-bit value at the given address.
  uint16_t read16(uint64_t addr) const {
    uint16_t val = 0;
    read_block(addr, std::span<uint8_t>(reinterpret_cast<uint8_t *>(&val), sizeof(val)));
    return val;
  }

  /// @brief Read a 32-bit value from the given address (little-endian).
  /// @param addr Memory address to read from.
  /// @returns The 32-bit value at the given address.
  uint32_t read32(uint64_t addr) const {
    uint32_t val = 0;
    read_block(addr, std::span<uint8_t>(reinterpret_cast<uint8_t *>(&val), sizeof(val)));
    return val;
  }

  /// @brief Read a 64-bit value from the given address (little-endian).
  /// @param addr Memory address to read from.
  /// @returns The 64-bit value at the given address.
  uint64_t read64(uint64_t addr) const {
    uint64_t val = 0;
    read_block(addr, std::span<uint8_t>(reinterpret_cast<uint8_t *>(&val), sizeof(val)));
    return val;
  }

  /// @brief Read a block of bytes from sparse memory.
  /// @details Thread-safe with page-granular synchronization. A read contained
  /// within one sparse page is atomic with respect to writers of that page. A
  /// read spanning multiple pages is not atomic as one transaction and may
  /// observe different pages before and after a concurrent multi-page write.
  /// @param addr Starting memory address.
  /// @param dst Destination buffer.
  void read_block(uint64_t addr, std::span<uint8_t> dst) const {
    size_t copied = 0;
    while (copied < dst.size()) {
      const uint64_t ea = addr + copied;
      const size_t page_off = ea & PAGE_MASK;
      const size_t chunk = std::min(dst.size() - copied, PAGE_SIZE - page_off);
      read_sparse_chunk(ea, dst.data() + copied, chunk);
      copied += chunk;
    }
  }

  /// @brief Write an 8-bit value to the given address.
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write8(uint64_t addr, uint8_t val) { write_block(addr, std::span<const uint8_t>(&val, 1)); }

  /// @brief Write a 16-bit value to the given address (little-endian).
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write16(uint64_t addr, uint16_t val) {
    write_block(addr,
                std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&val), sizeof(val)));
  }

  /// @brief Write a 32-bit value to the given address (little-endian).
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write32(uint64_t addr, uint32_t val) {
    write_block(addr,
                std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&val), sizeof(val)));
  }

  /// @brief Write a 64-bit value to the given address (little-endian).
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write64(uint64_t addr, uint64_t val) {
    write_block(addr,
                std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&val), sizeof(val)));
  }

  /// @brief Write a block of bytes to sparse memory.
  /// @details Thread-safe with page-granular synchronization. A write contained
  /// within one sparse page is atomic with respect to other accesses to that
  /// page. A write spanning multiple pages is not atomic as one transaction;
  /// concurrent overlapping multi-page operations may interleave by page.
  /// @param addr Starting memory address.
  /// @param src Source buffer.
  void write_block(uint64_t addr, std::span<const uint8_t> src) {
    size_t copied = 0;
    while (copied < src.size()) {
      const uint64_t ea = addr + copied;
      const size_t page_off = ea & PAGE_MASK;
      const size_t chunk = std::min(src.size() - copied, PAGE_SIZE - page_off);
      write_sparse_chunk(ea, src.data() + copied, chunk);
      copied += chunk;
    }
  }

  /// @brief Instruction fetch - read a 32-bit word (little-endian).
  /// @param addr Memory address to fetch from.
  /// @returns The 32-bit instruction word at the given address.
  uint32_t fetch32(uint64_t addr) const { return read32(addr); }

  /// @brief Load a raw binary image into memory at the given base address.
  /// @param data Pointer to the image data.
  /// @param size Size of the image in bytes.
  /// @param base_addr Starting address to load the image at.
  void load_image(const uint8_t *data, size_t size, uint64_t base_addr) {
    write_block(base_addr, std::span<const uint8_t>(data, size));
  }

  /// @brief Iterate over all allocated pages for checkpoint serialization.
  /// @details Copies one stripe at a time before invoking callbacks, so callbacks
  /// run without page-stripe locks held. The iteration is thread-safe but is not
  /// a globally atomic snapshot across all stripes.
  /// @tparam F Callable with signature void(uint64_t page_addr, const Page&).
  /// @param fn Callback invoked for each allocated page.
  template <typename F> void for_each_page(F &&fn) const {
    for (const auto &stripe : page_stripes_) {
      std::vector<std::pair<uint64_t, Page>> pages;
      {
        std::shared_lock<std::shared_mutex> lock(stripe.mutex);
        pages.reserve(stripe.pages.size());
        for (const auto &[addr, page] : stripe.pages)
          pages.emplace_back(addr, page);
      }
      for (const auto &[addr, page] : pages)
        fn(addr, page);
    }
  }

  /// @brief Return the number of allocated pages.
  /// @details Counts one stripe at a time. The result is thread-safe but is not
  /// a globally atomic snapshot if pages are allocated concurrently.
  /// @returns Count of allocated pages.
  size_t num_pages() const {
    size_t count = 0;
    for (const auto &stripe : page_stripes_) {
      std::shared_lock<std::shared_mutex> lock(stripe.mutex);
      count += stripe.pages.size();
    }
    return count;
  }

private:
  static constexpr size_t NUM_PAGE_STRIPES = 1024;

  struct PageStripe {
    mutable std::shared_mutex mutex;
    mutable std::unordered_map<uint64_t, Page> pages;
  };

  static size_t stripe_index(uint64_t addr) {
    constexpr uint64_t kMul = 11400714819323198485ull;
    const uint64_t page_number = addr >> PAGE_SHIFT;
    const uint64_t product = page_number * kMul;
    const uint64_t mixed = product ^ (product >> 32);
    return static_cast<size_t>(mixed & (NUM_PAGE_STRIPES - 1));
  }

  PageStripe &page_stripe(uint64_t addr) const { return page_stripes_[stripe_index(addr)]; }

  void read_sparse_chunk(uint64_t addr, uint8_t *dst, size_t size) const {
    auto &stripe = page_stripe(addr);
    std::shared_lock<std::shared_mutex> lock(stripe.mutex);
    auto it = stripe.pages.find(addr & ~PAGE_MASK);
    if (it != stripe.pages.end()) {
      std::memcpy(dst, &it->second[addr & PAGE_MASK], size);
    } else {
      std::memset(dst, 0, size);
    }
  }

  void write_sparse_chunk(uint64_t addr, const uint8_t *src, size_t size) {
    auto &stripe = page_stripe(addr);
    std::unique_lock<std::shared_mutex> lock(stripe.mutex);
    std::memcpy(&get_page_locked(stripe, addr)[addr & PAGE_MASK], src, size);
  }

  mutable std::array<PageStripe, NUM_PAGE_STRIPES> page_stripes_;

  // Returns a reference to the 4KB page containing addr, allocating if needed.
  // Caller must hold exclusive lock on the page stripe.
  Page &get_page_locked(PageStripe &stripe, uint64_t addr) const {
    uint64_t page_addr = addr & ~PAGE_MASK;
    auto it = stripe.pages.find(page_addr);
    if (it == stripe.pages.end()) {
      auto [inserted, _] = stripe.pages.emplace(page_addr, Page{});
      return inserted->second;
    }
    return it->second;
  }
};

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_SPARSE_MEMORY_H_
