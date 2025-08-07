////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

// A simple best fit memory allocator with eager compaction.  Manages block sub-allocation.
// For use when memory efficiency is more important than allocation speed.
// O(log n) time.

#ifndef HSA_RUNTME_CORE_UTIL_SIMPLE_HEAP_H_
#define HSA_RUNTME_CORE_UTIL_SIMPLE_HEAP_H_

#include <map>
#include <deque>
#include <utility>

#include "core/util/utils.h"

namespace rocr {

template <typename Allocator> class SimpleHeap {
 private:
  struct Fragment_T {
    typedef std::multimap<size_t, uintptr_t>::iterator ptr_t;
    ptr_t free_list_entry_;
    struct {
      size_t size : 62;
      bool discard : 1;
      bool free : 1;
    };

    Fragment_T(ptr_t Iterator, size_t Len, bool Free)
        : free_list_entry_(Iterator), size(Len), discard(false), free(Free) {}
    Fragment_T() = default;
  };

  struct Block {
    uintptr_t base_ptr_;
    size_t length_;

    Block(uintptr_t base, size_t length) : base_ptr_(base), length_(length) {}
    Block() = default;
  };

  Allocator block_allocator_;

  std::multimap<size_t, uintptr_t> free_list_;
  std::map<uintptr_t, std::map<uintptr_t, Fragment_T>> block_list_;
  std::deque<Block> block_cache_;

  // Size of blocks that are at least partially in use.
  size_t in_use_size_;
  // Total size of block cache
  size_t cache_size_;

  __forceinline bool isFree(const Fragment_T& node) { return node.free; }
  __forceinline void setUsed(Fragment_T& node) {
    node.free = false;
    node.free_list_entry_ = free_list_.end();
  }
  __forceinline void setFree(Fragment_T& node, typename Fragment_T::ptr_t Iterator) {
    node.free_list_entry_ = Iterator;
    node.free = true;
  }
  __forceinline Fragment_T makeFragment(size_t Len) {
    return Fragment_T(free_list_.end(), Len, false);
  }
  __forceinline Fragment_T makeFragment(typename Fragment_T::ptr_t Iterator, size_t Len) {
    return Fragment_T(Iterator, Len, true);
  }
  __forceinline void removeFreeListEntry(Fragment_T& node) {
    if (node.free_list_entry_ != free_list_.end()) {
      free_list_.erase(node.free_list_entry_);
      node.free_list_entry_ = free_list_.end();
    }
  }
  __forceinline void discard(Fragment_T& node) {
    removeFreeListEntry(node);
    node.discard = true;
  }

 public:
  explicit SimpleHeap(const Allocator& BlockAllocator = Allocator())
      : block_allocator_(BlockAllocator), in_use_size_(0), cache_size_(0) {}
  ~SimpleHeap() {
    trim();
    // Leak here may be due to the user.  Check is for debugging only.
    // assert(in_use_size_ == 0 && "Leak in SimpleHeap.");
  }

  SimpleHeap(const SimpleHeap& rhs) = delete;
  SimpleHeap(SimpleHeap&& rhs) = delete;
  SimpleHeap& operator=(const SimpleHeap& rhs) = delete;
  SimpleHeap& operator=(SimpleHeap&& rhs) = delete;

  void* alloc(size_t bytes) {
    // Find best fit.
    auto free_fragment = free_list_.lower_bound(bytes);
    uintptr_t base;
    size_t size;

    if (free_fragment != free_list_.end()) {
      base = free_fragment->second;
      size = free_fragment->first;
      free_list_.erase(free_fragment);

      assert(size >= bytes && "SimpleHeap: map lower_bound failure.");

      // Find the containing block and fragment
      auto it = block_list_.upper_bound(base);
      it--;
      auto& frag_map = it->second;
      const auto& fragment = frag_map.find(base);

      assert(fragment != frag_map.end() && "Inconsistency in SimpleHeap.");
      assert(size == fragment->second.size && "Inconsistency in SimpleHeap.");

      // Sub-allocate from fragment.
      fragment->second.size = bytes;
      setUsed(fragment->second);
      // Record remaining free space.
      if (size > bytes) {
        free_fragment = free_list_.insert(std::make_pair(size - bytes, base + bytes));
        frag_map[base + bytes] = makeFragment(free_fragment, size - bytes);
      }
      return reinterpret_cast<void*>(base);
    }

    // No usable fragment, check block cache
    if (bytes < default_block_size() && !block_cache_.empty()) {
      const auto& block = block_cache_.back();
      base = block.base_ptr_;
      size = block.length_;
      block_cache_.pop_back();
      cache_size_ -= size;
    } else {  // Alloc new block - new block may be larger than default.
      void* ptr = block_allocator_.alloc(bytes, size);
      base = reinterpret_cast<uintptr_t>(ptr);
      assert(ptr != nullptr && "Block allocation failed, Allocator is expected to throw.");
    }

    in_use_size_ += size;
    assert(size >= bytes && "Alloc exceeds block size.");
    // Sub alloc and insert free region.
    if (size > bytes) {
      free_fragment = free_list_.insert(std::make_pair(size - bytes, base + bytes));
      block_list_[base][base + bytes] = makeFragment(free_fragment, size - bytes);
    }
    // Track used region
    block_list_[base][base] = makeFragment(bytes);

    // Disallow multiple suballocation from large blocks.
    // Prevents a small allocation from retaining a large block.
    if (bytes > default_block_size()) {
      bool err = discardBlock(reinterpret_cast<void*>(base));
      assert(err && "Large block discard failed.");
    }

    return reinterpret_cast<void*>(base);
  }

  bool free(void* ptr) {
    if (ptr == nullptr) return true;

    uintptr_t base = reinterpret_cast<uintptr_t>(ptr);

    // Find fragment and validate.
    auto frag_map_it = block_list_.upper_bound(base);
    if (frag_map_it == block_list_.begin()) return false;
    frag_map_it--;
    auto& frag_map = frag_map_it->second;
    auto fragment = frag_map.find(base);
    if (fragment == frag_map.end() || isFree(fragment->second)) return false;

    bool discard = fragment->second.discard;

    // Merge lower
    if (fragment != frag_map.begin()) {
      auto lower = fragment;
      lower--;
      if (isFree(lower->second)) {
        removeFreeListEntry(lower->second);
        lower->second.size += fragment->second.size;
        frag_map.erase(fragment);
        fragment = lower;
      }
    }

    // Merge upper
    {
      auto upper = fragment;
      upper++;
      if ((upper != frag_map.end()) && isFree(upper->second)) {
        removeFreeListEntry(upper->second);
        fragment->second.size += upper->second.size;
        frag_map.erase(upper);
      }
    }

    // Release whole free blocks.
    if (frag_map.size() == 1) {
      Block block(fragment->first, fragment->second.size);
      block_list_.erase(frag_map_it);

      // Discard or add to the block cache.
      if (discard) {
        block_allocator_.free(reinterpret_cast<void*>(block.base_ptr_), block.length_);
      } else {
        block_cache_.push_back(block);
        cache_size_ += block.length_;
        in_use_size_ -= block.length_;
      }

      balance();

      // Don't publish free space since block was moved to the cache.
      return true;
    }

    // Don't report free memory if discarding the fragment.
    if (discard) return true;

    // Report free fragment
    const auto& freeEntry =
        free_list_.insert(std::make_pair(size_t(fragment->second.size), fragment->first));
    setFree(fragment->second, freeEntry);

    return true;
  }

  void balance() {
    // Release old blocks when over cache limit.
    while ((block_cache_.size() > 1) && (cache_size_ > in_use_size_ * 2)) {
      const auto& block = block_cache_.front();
      block_allocator_.free(reinterpret_cast<void*>(block.base_ptr_), block.length_);
      cache_size_ -= block.length_;
      block_cache_.pop_front();
    }
  }

  void trim() {
    for (const auto& block : block_cache_)
      block_allocator_.free(reinterpret_cast<void*>(block.base_ptr_), block.length_);
    block_cache_.clear();
    cache_size_ = 0;
  }

  size_t cache_size() const { return cache_size_; }

  size_t default_block_size() const { return block_allocator_.block_size(); }

  // Prevent reuse of the block containing ptr.  No further fragments will be allocated from the
  // block and the block will not be added to the block cache when it is free.
  bool discardBlock(void* ptr) {
    if (ptr == nullptr) return true;

    uintptr_t base = reinterpret_cast<uintptr_t>(ptr);

    // Find block validate.
    auto frag_map_it = block_list_.upper_bound(base);
    if (frag_map_it == block_list_.begin()) return false;
    frag_map_it--;
    auto& frag_map = frag_map_it->second;
    if ((base < frag_map.begin()->first) ||
        (frag_map.rbegin()->first + frag_map.rbegin()->second.size <= base))
      return false;

    // Is block already discarded?
    if (frag_map.begin()->second.discard) return true;

    // Mark all fragments for discard and compute block size.  Removes freelist records for all
    // fragments in the block.
    size_t size = 0;
    for (auto& frag : frag_map) {
      discard(frag.second);
      size += frag.second.size;
    }

    // Remove discarded block from in-use tracking and rebalance the block cache.
    in_use_size_ -= size;
    balance();

    return true;
  }
};

}  // namespace rocr

#endif  // HSA_RUNTME_CORE_UTIL_SIMPLE_HEAP_H_
