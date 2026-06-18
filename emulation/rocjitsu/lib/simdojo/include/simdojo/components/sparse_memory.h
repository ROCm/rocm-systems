// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sparse_memory.h
/// @brief Sparse page-table memory model with on-demand page allocation.

#ifndef SIMDOJO_COMPONENTS_SPARSE_MEMORY_H_
#define SIMDOJO_COMPONENTS_SPARSE_MEMORY_H_

#include "simdojo/sim/component.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <shared_mutex>
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
    uint8_t val = 0;
    read_block(addr, &val, sizeof(val));
    return val;
  }

  /// @brief Read a 16-bit value from the given address (little-endian).
  /// @param addr Memory address to read from.
  /// @returns The 16-bit value at the given address.
  uint16_t read16(uint64_t addr) const {
    uint16_t val = 0;
    read_block(addr, reinterpret_cast<uint8_t *>(&val), sizeof(val));
    return val;
  }

  /// @brief Read a 32-bit value from the given address (little-endian).
  /// @param addr Memory address to read from.
  /// @returns The 32-bit value at the given address.
  uint32_t read32(uint64_t addr) const {
    uint32_t val = 0;
    read_block(addr, reinterpret_cast<uint8_t *>(&val), sizeof(val));
    return val;
  }

  /// @brief Read a 64-bit value from the given address (little-endian).
  /// @param addr Memory address to read from.
  /// @returns The 64-bit value at the given address.
  uint64_t read64(uint64_t addr) const {
    uint64_t val = 0;
    read_block(addr, reinterpret_cast<uint8_t *>(&val), sizeof(val));
    return val;
  }

  /// @brief Read a block of bytes from sparse memory.
  /// @param addr Starting memory address.
  /// @param dst Destination buffer.
  /// @param size Number of bytes to read.
  void read_block(uint64_t addr, uint8_t *dst, size_t size) const {
    size_t copied = 0;
    while (copied < size) {
      const uint64_t ea = addr + copied;
      const size_t page_off = ea & PAGE_MASK;
      const size_t chunk = std::min(size - copied, PAGE_SIZE - page_off);
      if (!read_host_chunk(ea, dst + copied, chunk))
        read_sparse_chunk(ea, dst + copied, chunk);
      copied += chunk;
    }
  }

  /// @brief Write an 8-bit value to the given address.
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write8(uint64_t addr, uint8_t val) { write_block(addr, &val, sizeof(val)); }

  /// @brief Write a 16-bit value to the given address (little-endian).
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write16(uint64_t addr, uint16_t val) {
    write_block(addr, reinterpret_cast<const uint8_t *>(&val), sizeof(val));
  }

  /// @brief Write a 32-bit value to the given address (little-endian).
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write32(uint64_t addr, uint32_t val) {
    write_block(addr, reinterpret_cast<const uint8_t *>(&val), sizeof(val));
  }

  /// @brief Write a 64-bit value to the given address (little-endian).
  /// @param addr Memory address to write to.
  /// @param val Value to write.
  void write64(uint64_t addr, uint64_t val) {
    write_block(addr, reinterpret_cast<const uint8_t *>(&val), sizeof(val));
  }

  /// @brief Write a block of bytes to sparse memory.
  /// @param addr Starting memory address.
  /// @param src Source buffer.
  /// @param size Number of bytes to write.
  void write_block(uint64_t addr, const uint8_t *src, size_t size) {
    size_t copied = 0;
    while (copied < size) {
      const uint64_t ea = addr + copied;
      const size_t page_off = ea & PAGE_MASK;
      const size_t chunk = std::min(size - copied, PAGE_SIZE - page_off);
      if (!write_host_chunk(ea, src + copied, chunk))
        write_sparse_chunk(ea, src + copied, chunk);
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
    size_t offset = 0;
    while (offset < size) {
      uint64_t addr = base_addr + offset;
      uint64_t page_off = addr & PAGE_MASK;
      size_t chunk = std::min(PAGE_SIZE - page_off, size - offset);
      if (!write_host_chunk(addr, data + offset, chunk))
        write_sparse_chunk(addr, data + offset, chunk);
      offset += chunk;
    }
  }

  /// @brief Iterate over all allocated pages for checkpoint serialization.
  /// @tparam F Callable with signature void(uint64_t page_addr, const Page&).
  /// @param fn Callback invoked for each allocated page.
  template <typename F> void for_each_page(F &&fn) const {
    auto locks = lock_all_page_stripes_shared();
    for (const auto &stripe : page_stripes_)
      for (const auto &[addr, page] : stripe.pages)
        fn(addr, page);
  }

  /// @brief Return the number of allocated pages.
  /// @returns Count of allocated pages.
  size_t num_pages() const {
    auto locks = lock_all_page_stripes_shared();
    size_t count = 0;
    for (const auto &stripe : page_stripes_)
      count += stripe.pages.size();
    return count;
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
    if (size > 0)
      host_pages_mapped_.store(true, std::memory_order_release);
    auto *hp = static_cast<uint8_t *>(host_ptr);
    std::lock_guard<std::shared_mutex> lock(host_range_mutex_);
    for (uint64_t off = 0; off < size; off += PAGE_SIZE)
      host_page_map_[(gpu_va + off) >> PAGE_SHIFT] = hp + off;
    host_pages_mapped_.store(!host_page_map_.empty(), std::memory_order_release);
  }

  bool is_host_mapped(uint64_t gpu_va) const {
    if (!host_pages_mapped_.load(std::memory_order_acquire))
      return false;
    std::shared_lock<std::shared_mutex> lock(host_range_mutex_);
    return host_page_map_.count(gpu_va >> PAGE_SHIFT) > 0;
  }

  /// @brief Find the base and size of the contiguous host-mapped region containing addr.
  /// @details Scans backward and forward through host_page_map_ to find the
  /// full contiguous range. Returns {0, 0} if addr is not host-mapped.
  std::pair<uint64_t, uint64_t> find_host_range(uint64_t addr) const {
    if (!host_pages_mapped_.load(std::memory_order_acquire))
      return {0, 0};
    std::shared_lock<std::shared_mutex> lock(host_range_mutex_);
    uint64_t page = addr >> PAGE_SHIFT;
    if (host_page_map_.count(page) == 0)
      return {0, 0};
    uint64_t lo = page;
    while (lo > 0 && host_page_map_.count(lo - 1))
      --lo;
    uint64_t hi = page;
    while (host_page_map_.count(hi + 1))
      ++hi;
    return {lo << PAGE_SHIFT, ((hi - lo) + 1) << PAGE_SHIFT};
  }

  void unmap_host_pages(uint64_t gpu_va, size_t size) {
    std::lock_guard<std::shared_mutex> lock(host_range_mutex_);
    for (uint64_t off = 0; off < size; off += PAGE_SIZE)
      host_page_map_.erase((gpu_va + off) >> PAGE_SHIFT);
    host_pages_mapped_.store(!host_page_map_.empty(), std::memory_order_release);
  }

private:
  static constexpr size_t NUM_PAGE_STRIPES = 1024;

  struct PageStripe {
    mutable std::shared_mutex mutex;
    mutable std::unordered_map<uint64_t, Page> pages;
  };

  static size_t stripe_index(uint64_t addr) {
    constexpr uint64_t kMul = 11400714819323198485ull;
    uint64_t page_number = addr >> PAGE_SHIFT;
    return static_cast<size_t>((page_number * kMul) & (NUM_PAGE_STRIPES - 1));
  }

  PageStripe &page_stripe(uint64_t addr) const { return page_stripes_[stripe_index(addr)]; }

  std::vector<std::shared_lock<std::shared_mutex>> lock_all_page_stripes_shared() const {
    std::vector<std::shared_lock<std::shared_mutex>> locks;
    locks.reserve(NUM_PAGE_STRIPES);
    for (const auto &stripe : page_stripes_)
      locks.emplace_back(stripe.mutex);
    return locks;
  }

  bool read_host_chunk(uint64_t addr, uint8_t *dst, size_t size) const {
    if (!host_pages_mapped_.load(std::memory_order_acquire))
      return false;
    std::shared_lock<std::shared_mutex> lock(host_range_mutex_);
    auto it = host_page_map_.find(addr >> PAGE_SHIFT);
    if (it == host_page_map_.end())
      return false;
    std::memcpy(dst, it->second + (addr & PAGE_MASK), size);
    return true;
  }

  bool write_host_chunk(uint64_t addr, const uint8_t *src, size_t size) {
    if (!host_pages_mapped_.load(std::memory_order_acquire))
      return false;
    std::shared_lock<std::shared_mutex> lock(host_range_mutex_);
    auto it = host_page_map_.find(addr >> PAGE_SHIFT);
    if (it == host_page_map_.end())
      return false;
    std::memcpy(it->second + (addr & PAGE_MASK), src, size);
    return true;
  }

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

  /// @brief Page-indexed host range map. Key = page_number (gpu_va >> 12),
  /// value = host_ptr for the start of that page. O(1) lookup per access.
  mutable std::shared_mutex host_range_mutex_;
  std::unordered_map<uint64_t, uint8_t *> host_page_map_;
  std::atomic<bool> host_pages_mapped_{false};

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
