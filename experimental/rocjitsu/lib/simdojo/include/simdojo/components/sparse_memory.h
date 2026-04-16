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
#include <string>
#include <unordered_map>
#include <utility>

namespace simdojo {

/// @brief Sparse page-table memory model as a simulation Component.
///
/// Little-endian, 4KB pages allocated on first access. Provides byte, word,
/// and doubleword access plus bulk image loading and page iteration for
/// serialization.
///
/// Host ranges are tracked as a page-indexed map (page_number → host_ptr)
/// for O(1) lookup per access, mirroring real IOMMU page table structure.
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
    {
      std::shared_lock<std::shared_mutex> lock(host_range_mutex_);
      auto it = host_page_map_.find(addr >> PAGE_SHIFT);
      if (it != host_page_map_.end())
        return it->second[addr & PAGE_MASK];
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = pages_.find(addr & ~PAGE_MASK);
    return (it != pages_.end()) ? it->second[addr & PAGE_MASK] : 0;
  }

  /// @brief Read a 16-bit value from the given address (little-endian).
  /// @param addr Memory address to read from.
  /// @returns The 16-bit value at the given address.
  uint16_t read16(uint64_t addr) const {
    {
      std::shared_lock<std::shared_mutex> lock(host_range_mutex_);
      auto it = host_page_map_.find(addr >> PAGE_SHIFT);
      if (it != host_page_map_.end() && (addr & PAGE_MASK) + 2 <= PAGE_SIZE) {
        uint16_t val = 0;
        std::memcpy(&val, it->second + (addr & PAGE_MASK), 2);
        return val;
      }
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    uint16_t val = 0;
    read_bytes(addr, &val, 2);
    return val;
  }

  /// @brief Read a 32-bit value from the given address (little-endian).
  /// @param addr Memory address to read from.
  /// @returns The 32-bit value at the given address.
  uint32_t read32(uint64_t addr) const {
    {
      std::shared_lock<std::shared_mutex> lock(host_range_mutex_);
      auto it = host_page_map_.find(addr >> PAGE_SHIFT);
      if (it != host_page_map_.end() && (addr & PAGE_MASK) + 4 <= PAGE_SIZE) {
        uint32_t val = 0;
        std::memcpy(&val, it->second + (addr & PAGE_MASK), 4);
        return val;
      }
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    uint32_t val = 0;
    read_bytes(addr, &val, 4);
    return val;
  }

  /// @brief Read a 64-bit value from the given address (little-endian).
  /// @param addr Memory address to read from.
  /// @returns The 64-bit value at the given address.
  uint64_t read64(uint64_t addr) const {
    {
      std::shared_lock<std::shared_mutex> lock(host_range_mutex_);
      auto it = host_page_map_.find(addr >> PAGE_SHIFT);
      if (it != host_page_map_.end() && (addr & PAGE_MASK) + 8 <= PAGE_SIZE) {
        uint64_t val = 0;
        std::memcpy(&val, it->second + (addr & PAGE_MASK), 8);
        return val;
      }
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    uint64_t val = 0;
    read_bytes(addr, &val, 8);
    return val;
  }

  /// @brief Write an 8-bit value to the given address.
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write8(uint64_t addr, uint8_t val) {
    {
      std::shared_lock<std::shared_mutex> lock(host_range_mutex_);
      auto it = host_page_map_.find(addr >> PAGE_SHIFT);
      if (it != host_page_map_.end()) {
        it->second[addr & PAGE_MASK] = val;
        return;
      }
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    get_page(addr)[addr & PAGE_MASK] = val;
  }

  /// @brief Write a 16-bit value to the given address (little-endian).
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write16(uint64_t addr, uint16_t val) {
    {
      std::shared_lock<std::shared_mutex> lock(host_range_mutex_);
      auto it = host_page_map_.find(addr >> PAGE_SHIFT);
      if (it != host_page_map_.end() && (addr & PAGE_MASK) + 2 <= PAGE_SIZE) {
        std::memcpy(it->second + (addr & PAGE_MASK), &val, 2);
        return;
      }
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    write_bytes(addr, &val, sizeof(val));
  }

  /// @brief Write a 32-bit value to the given address (little-endian).
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write32(uint64_t addr, uint32_t val) {
    {
      std::shared_lock<std::shared_mutex> lock(host_range_mutex_);
      auto it = host_page_map_.find(addr >> PAGE_SHIFT);
      if (it != host_page_map_.end() && (addr & PAGE_MASK) + 4 <= PAGE_SIZE) {
        std::memcpy(it->second + (addr & PAGE_MASK), &val, 4);
        return;
      }
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    write_bytes(addr, &val, sizeof(val));
  }

  /// @brief Write a 64-bit value to the given address (little-endian).
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write64(uint64_t addr, uint64_t val) {
    {
      std::shared_lock<std::shared_mutex> lock(host_range_mutex_);
      auto it = host_page_map_.find(addr >> PAGE_SHIFT);
      if (it != host_page_map_.end() && (addr & PAGE_MASK) + 8 <= PAGE_SIZE) {
        std::memcpy(it->second + (addr & PAGE_MASK), &val, 8);
        return;
      }
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
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
    size_t offset = 0;
    while (offset < size) {
      uint64_t addr = base_addr + offset;
      uint64_t page_off = addr & PAGE_MASK;
      size_t chunk = std::min(PAGE_SIZE - page_off, size - offset);
      {
        std::shared_lock<std::shared_mutex> lock(host_range_mutex_);
        auto it = host_page_map_.find(addr >> PAGE_SHIFT);
        if (it != host_page_map_.end()) {
          std::memcpy(it->second + page_off, data + offset, chunk);
          offset += chunk;
          continue;
        }
      }
      {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        std::memcpy(&get_page(addr)[page_off], data + offset, chunk);
      }
      offset += chunk;
    }
  }

  /// @brief Iterate over all allocated pages for checkpoint serialization.
  /// @tparam F Callable with signature void(uint64_t page_addr, const Page&).
  /// @param fn Callback invoked for each allocated page.
  template <typename F> void for_each_page(F &&fn) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (const auto &[addr, page] : pages_)
      fn(addr, page);
  }

  /// @brief Return the number of allocated pages.
  /// @returns Count of allocated pages.
  size_t num_pages() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return pages_.size();
  }

  /// @brief Map host pages as the backing store for a GPU VA range.
  /// @details After this call, read/write to addresses in [gpu_va, gpu_va+size)
  /// access the host memory at [host_ptr, host_ptr+size) directly. Both the host
  /// CPU and the simulated GPU see the same data — no copy needed.
  ///
  /// Internally stored as one entry per 4KB page (page_number → host_ptr),
  /// giving O(1) lookup per memory access instead of O(n) range scan.
  /// @param gpu_va Start of the GPU virtual address range (page-aligned).
  /// @param host_ptr Host pointer to the backing pages (page-aligned).
  /// @param size Size of the mapping in bytes (page-aligned).
  void map_host_pages(uint64_t gpu_va, void *host_ptr, size_t size) {
    auto *hp = static_cast<uint8_t *>(host_ptr);
    std::lock_guard<std::shared_mutex> lock(host_range_mutex_);
    for (uint64_t off = 0; off < size; off += PAGE_SIZE)
      host_page_map_[(gpu_va + off) >> PAGE_SHIFT] = hp + off;
  }

  bool is_host_mapped(uint64_t gpu_va) const {
    std::shared_lock<std::shared_mutex> lock(host_range_mutex_);
    return host_page_map_.count(gpu_va >> PAGE_SHIFT) > 0;
  }

  void unmap_host_pages(uint64_t gpu_va, size_t size) {
    std::lock_guard<std::shared_mutex> lock(host_range_mutex_);
    for (uint64_t off = 0; off < size; off += PAGE_SIZE)
      host_page_map_.erase((gpu_va + off) >> PAGE_SHIFT);
  }

private:
  mutable std::shared_mutex mutex_;
  mutable std::unordered_map<uint64_t, Page> pages_;

  /// @brief Page-indexed host range map. Key = page_number (gpu_va >> 12),
  /// value = host_ptr for the start of that page. O(1) lookup per access.
  mutable std::shared_mutex host_range_mutex_;
  std::unordered_map<uint64_t, uint8_t *> host_page_map_;

  // Returns a reference to the 4KB page containing addr, allocating if needed.
  // Caller must hold exclusive lock on mutex_.
  Page &get_page(uint64_t addr) const {
    uint64_t page_addr = addr & ~PAGE_MASK;
    auto it = pages_.find(page_addr);
    if (it == pages_.end()) {
      auto [inserted, _] = pages_.emplace(page_addr, Page{});
      return inserted->second;
    }
    return it->second;
  }

  uint8_t *get_byte_ptr(uint64_t addr) const { return &get_page(addr)[addr & PAGE_MASK]; }

  /// @brief Read N bytes from sparse pages, handling cross-page boundary access.
  /// @note Caller must hold shared_lock on mutex_. Does NOT allocate pages;
  ///       unallocated addresses read as zero.
  void read_bytes(uint64_t addr, void *dst, size_t n) const {
    // Note: caller holds shared_lock on mutex_.
    size_t off = addr & PAGE_MASK;
    if (off + n <= PAGE_SIZE) {
      auto it = pages_.find(addr & ~PAGE_MASK);
      if (it != pages_.end())
        std::memcpy(dst, &it->second[off], n);
      else
        std::memset(dst, 0, n);
    } else {
      auto *out = static_cast<uint8_t *>(dst);
      for (size_t i = 0; i < n; ++i) {
        uint64_t a = addr + i;
        auto it = pages_.find(a & ~PAGE_MASK);
        out[i] = (it != pages_.end()) ? it->second[a & PAGE_MASK] : 0;
      }
    }
  }

  /// @brief Write N bytes, handling cross-page boundary access.
  /// @note Caller must hold unique_lock on mutex_. Allocates pages as needed.
  void write_bytes(uint64_t addr, const void *src, size_t n) {
    size_t offset = addr & PAGE_MASK;
    if (offset + n <= PAGE_SIZE) {
      std::memcpy(&get_page(addr)[offset], src, n);
    } else {
      const auto *in = static_cast<const uint8_t *>(src);
      for (size_t i = 0; i < n; ++i)
        *get_byte_ptr(addr + i) = in[i];
    }
  }
};

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_SPARSE_MEMORY_H_
