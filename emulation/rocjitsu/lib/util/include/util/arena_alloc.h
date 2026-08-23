// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file arena_alloc.h
/// @brief Fixed-size block pool allocators (free-list based).
///
/// ArenaAlloc reserves one fixed-capacity inline buffer and falls back to the
/// global allocator when it is exhausted. GrowingArenaAlloc starts empty and
/// adds fixed-size slabs as demand grows. Both recycle eligible blocks in O(1)
/// and use the global allocator for oversized requests.
///
/// Intended for use as a class-specific operator new/delete override to
/// eliminate malloc/free from hot paths. Thread-safety: NOT thread-safe.
/// Use one pool per thread.

#ifndef UTIL_ARENA_ALLOC_H_
#define UTIL_ARENA_ALLOC_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <new>
#include <vector>

namespace util {

/// @brief Fixed-size block pool allocator.
/// @tparam BlockSize  Size of each block in bytes.
/// @tparam NumBlocks  Number of pre-allocated blocks.
/// @tparam BlockAlign Alignment of each block.
template <size_t BlockSize, size_t NumBlocks = 64, size_t BlockAlign = alignof(std::max_align_t)>
class ArenaAlloc {
  static_assert(BlockSize >= sizeof(void *), "Block must hold a free-list pointer");
  static_assert(NumBlocks > 0, "Must have at least one block");

public:
  ArenaAlloc() { init_free_list(); }

  /// @brief Allocate one block.  O(1) from free-list; falls back to heap.
  void *allocate(size_t size) {
    if (size > BlockSize)
      return ::operator new(size);

    if (void *ptr = try_allocate(size))
      return ptr;
    // Pool exhausted — fall back to heap.
    return ::operator new(size);
  }

  /// @brief Allocate one block without falling back to the global allocator.
  ///
  /// @details Use this in C ABI paths that must report allocation failure as an
  /// error code instead of allowing `operator new` to throw.
  /// @returns Pointer to a block, or nullptr when the free-list is exhausted.
  void *try_allocate(size_t size) {
    if (size > BlockSize)
      return nullptr;
    if (free_head_ == nullptr)
      return nullptr;
    void *ptr = free_head_;
    free_head_ = free_head_->next;
    return ptr;
  }

  /// @brief Return a block to the free-list.  O(1).
  /// Heap-allocated overflow or oversized blocks are returned to the global
  /// allocator because they are not backed by this pool's fixed-size buffer.
  void deallocate(void *ptr) {
    if (!owns(ptr)) {
      ::operator delete(ptr);
      return;
    }

    auto *node = static_cast<FreeNode *>(ptr);
    node->next = free_head_;
    free_head_ = node;
  }

  /// @brief Check if a pointer was allocated from the pre-allocated buffer.
  bool owns(const void *ptr) const {
    auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    auto begin = reinterpret_cast<std::uintptr_t>(buffer_);
    auto end = begin + sizeof(buffer_);
    return addr >= begin && addr < end && ((addr - begin) % BlockSize) == 0;
  }

  static constexpr size_t BLOCK_SIZE = BlockSize;
  static constexpr size_t NUM_BLOCKS = NumBlocks;

private:
  struct FreeNode {
    FreeNode *next;
  };

  void init_free_list() {
    free_head_ = nullptr;
    for (size_t i = NumBlocks; i > 0; --i) {
      auto *node = reinterpret_cast<FreeNode *>(buffer_ + (i - 1) * BlockSize);
      node->next = free_head_;
      free_head_ = node;
    }
  }

  alignas(BlockAlign) uint8_t buffer_[BlockSize * NumBlocks];
  FreeNode *free_head_ = nullptr;
};

/// @brief Grow-on-demand fixed-size block allocator.
/// @tparam BlockSize Size of each reusable block in bytes.
/// @tparam BlocksPerSlab Number of blocks added whenever the free-list is empty.
/// @tparam BlockAlign Alignment of each block.
///
/// Unlike ArenaAlloc, this allocator does not reserve inline storage and does
/// not return eligible overflow blocks to the global allocator. It adds slabs
/// at peak demand and recycles every block until the allocator is destroyed.
template <size_t BlockSize, size_t BlocksPerSlab = 64,
          size_t BlockAlign = alignof(std::max_align_t)>
class GrowingArenaAlloc {
  static_assert(BlockSize >= sizeof(void *), "Block must hold a free-list pointer");
  static_assert(BlocksPerSlab > 0, "A slab must contain at least one block");
  static_assert(BlockSize % BlockAlign == 0, "Block stride must preserve block alignment");

public:
  GrowingArenaAlloc() = default;
  GrowingArenaAlloc(const GrowingArenaAlloc &) = delete;
  GrowingArenaAlloc &operator=(const GrowingArenaAlloc &) = delete;

  ~GrowingArenaAlloc() {
    for (void *slab : slabs_)
      ::operator delete(slab, std::align_val_t(BlockAlign));
  }

  /// @brief Allocate one block, growing by one slab when necessary.
  void *allocate(size_t size) {
    if (size > BlockSize)
      return ::operator new(size);
    if (free_head_ == nullptr)
      grow();
    void *ptr = free_head_;
    free_head_ = free_head_->next;
    return ptr;
  }

  /// @brief Return a block to the free-list or free an oversized allocation.
  void deallocate(void *ptr) {
    if (!owns(ptr)) {
      ::operator delete(ptr);
      return;
    }
    auto *node = static_cast<FreeNode *>(ptr);
    node->next = free_head_;
    free_head_ = node;
  }

  /// @brief Check whether a pointer belongs to one of this allocator's slabs.
  bool owns(const void *ptr) const {
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    const auto slab_it = std::upper_bound(slabs_.begin(), slabs_.end(), addr,
                                          [](std::uintptr_t value, const void *slab) {
                                            return value < reinterpret_cast<std::uintptr_t>(slab);
                                          });
    if (slab_it == slabs_.begin())
      return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(*std::prev(slab_it));
    return addr < begin + kSlabBytes && ((addr - begin) % BlockSize) == 0;
  }

  static constexpr size_t BLOCK_SIZE = BlockSize;
  static constexpr size_t BLOCKS_PER_SLAB = BlocksPerSlab;

private:
  struct FreeNode {
    FreeNode *next;
  };

  static constexpr size_t kSlabBytes = BlockSize * BlocksPerSlab;

  void grow() {
    void *slab = ::operator new(kSlabBytes, std::align_val_t(BlockAlign));
    try {
      const auto insertion = std::lower_bound(
          slabs_.begin(), slabs_.end(), slab, [](const void *lhs, const void *rhs) {
            return reinterpret_cast<std::uintptr_t>(lhs) < reinterpret_cast<std::uintptr_t>(rhs);
          });
      slabs_.insert(insertion, slab);
    } catch (...) {
      ::operator delete(slab, std::align_val_t(BlockAlign));
      throw;
    }
    auto *bytes = static_cast<uint8_t *>(slab);
    for (size_t i = BlocksPerSlab; i > 0; --i) {
      auto *node = reinterpret_cast<FreeNode *>(bytes + (i - 1) * BlockSize);
      node->next = free_head_;
      free_head_ = node;
    }
  }

  std::vector<void *> slabs_;
  FreeNode *free_head_ = nullptr;
};

} // namespace util

#endif // UTIL_ARENA_ALLOC_H_
