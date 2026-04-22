// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file arena_alloc.h
/// @brief Fixed-size block pool allocator (free-list based).
///
/// Pre-allocates a contiguous buffer of N fixed-size blocks. Allocations
/// pop from a free-list in O(1); deallocations push back in O(1). When
/// the pool is exhausted, falls back to the global allocator.
///
/// Intended for use as a class-specific operator new/delete override to
/// eliminate malloc/free from hot paths. Thread-safety: NOT thread-safe.
/// Use one pool per thread.

#ifndef UTIL_ARENA_ALLOC_H_
#define UTIL_ARENA_ALLOC_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>

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
  void *allocate([[maybe_unused]] size_t size) {
    assert(size <= BlockSize && "Allocation exceeds block size");
    if (free_head_) {
      void *ptr = free_head_;
      free_head_ = free_head_->next;
      return ptr;
    }
    // Pool exhausted — fall back to heap.
    return ::operator new(BlockSize);
  }

  /// @brief Return a block to the free-list.  O(1).
  /// Heap-allocated overflow blocks are also accepted — they become part
  /// of the free-list and will be reused before more heap allocations.
  void deallocate(void *ptr) {
    auto *node = static_cast<FreeNode *>(ptr);
    node->next = free_head_;
    free_head_ = node;
  }

  /// @brief Check if a pointer was allocated from the pre-allocated buffer.
  bool owns(const void *ptr) const { return ptr >= buffer_ && ptr < buffer_ + sizeof(buffer_); }

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

} // namespace util

#endif // UTIL_ARENA_ALLOC_H_
