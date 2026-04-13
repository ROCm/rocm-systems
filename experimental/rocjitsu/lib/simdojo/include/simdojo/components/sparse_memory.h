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
#include <string>
#include <unordered_map>
#include <utility>

namespace simdojo {

/// @brief Sparse page-table memory model as a simulation Component.
///
/// Little-endian, 4KB pages allocated on first access. Provides byte, word,
/// and doubleword access plus bulk image loading and page iteration for
/// serialization.
class SparseMemory : public Component {
public:
  static constexpr size_t PAGE_SIZE = 4096;
  static constexpr size_t PAGE_MASK = PAGE_SIZE - 1;
  using Page = std::array<uint8_t, PAGE_SIZE>;

  /// @brief Construct a sparse memory component.
  /// @param name Human-readable component name.
  explicit SparseMemory(std::string name) : Component(std::move(name)) {}

  /// @brief Read an 8-bit value from the given address.
  /// @param addr Memory address to read from.
  /// @returns The byte at the given address (0 if page not yet allocated).
  uint8_t read8(uint64_t addr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return get_page(addr)[addr & PAGE_MASK];
  }

  /// @brief Read a 16-bit value from the given address (little-endian).
  /// @param addr Memory address to read from.
  /// @returns The 16-bit value at the given address.
  uint16_t read16(uint64_t addr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint16_t val = 0;
    read_bytes(addr, &val, sizeof(val));
    return val;
  }

  /// @brief Read a 32-bit value from the given address (little-endian).
  /// @param addr Memory address to read from.
  /// @returns The 32-bit value at the given address.
  uint32_t read32(uint64_t addr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t val = 0;
    read_bytes(addr, &val, sizeof(val));
    return val;
  }

  /// @brief Read a 64-bit value from the given address (little-endian).
  /// @param addr Memory address to read from.
  /// @returns The 64-bit value at the given address.
  uint64_t read64(uint64_t addr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t val = 0;
    read_bytes(addr, &val, sizeof(val));
    return val;
  }

  /// @brief Write an 8-bit value to the given address.
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write8(uint64_t addr, uint8_t val) {
    std::lock_guard<std::mutex> lock(mutex_);
    get_page(addr)[addr & PAGE_MASK] = val;
  }

  /// @brief Write a 16-bit value to the given address (little-endian).
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write16(uint64_t addr, uint16_t val) {
    std::lock_guard<std::mutex> lock(mutex_);
    write_bytes(addr, &val, sizeof(val));
  }

  /// @brief Write a 32-bit value to the given address (little-endian).
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write32(uint64_t addr, uint32_t val) {
    std::lock_guard<std::mutex> lock(mutex_);
    write_bytes(addr, &val, sizeof(val));
  }

  /// @brief Write a 64-bit value to the given address (little-endian).
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write64(uint64_t addr, uint64_t val) {
    std::lock_guard<std::mutex> lock(mutex_);
    write_bytes(addr, &val, sizeof(val));
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
    std::lock_guard<std::mutex> lock(mutex_);
    size_t offset = 0;
    while (offset < size) {
      uint64_t addr = base_addr + offset;
      size_t page_off = addr & PAGE_MASK;
      size_t chunk = std::min(PAGE_SIZE - page_off, size - offset);
      std::memcpy(&get_page(addr)[page_off], data + offset, chunk);
      offset += chunk;
    }
  }

  /// @brief Iterate over all allocated pages for checkpoint serialization.
  /// @tparam F Callable with signature void(uint64_t page_addr, const Page&).
  /// @param fn Callback invoked for each allocated page.
  template <typename F> void for_each_page(F &&fn) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &[addr, page] : pages_)
      fn(addr, page);
  }

  /// @brief Return the number of allocated pages.
  /// @returns Count of allocated pages.
  size_t num_pages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pages_.size();
  }

private:
  mutable std::mutex mutex_;
  mutable std::unordered_map<uint64_t, Page> pages_;

  Page &get_page(uint64_t addr) const {
    uint64_t page_addr = addr & ~PAGE_MASK;
    auto it = pages_.find(page_addr);
    if (it == pages_.end()) {
      auto [inserted, _] = pages_.emplace(page_addr, Page{});
      return inserted->second;
    }
    return it->second;
  }

  /// @brief Read N bytes, handling cross-page boundary access.
  /// @note Caller must hold mutex_.
  void read_bytes(uint64_t addr, void *dst, size_t n) const {
    size_t offset = addr & PAGE_MASK;
    if (offset + n <= PAGE_SIZE) {
      std::memcpy(dst, &get_page(addr)[offset], n);
    } else {
      auto *out = static_cast<uint8_t *>(dst);
      for (size_t i = 0; i < n; ++i)
        out[i] = get_page(addr + i)[(addr + i) & PAGE_MASK];
    }
  }

  /// @brief Write N bytes, handling cross-page boundary access.
  /// @note Caller must hold mutex_.
  void write_bytes(uint64_t addr, const void *src, size_t n) {
    size_t offset = addr & PAGE_MASK;
    if (offset + n <= PAGE_SIZE) {
      std::memcpy(&get_page(addr)[offset], src, n);
    } else {
      const auto *in = static_cast<const uint8_t *>(src);
      for (size_t i = 0; i < n; ++i)
        get_page(addr + i)[(addr + i) & PAGE_MASK] = in[i];
    }
  }
};

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_SPARSE_MEMORY_H_
