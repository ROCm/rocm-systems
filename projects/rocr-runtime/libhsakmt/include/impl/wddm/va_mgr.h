/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

 #pragma once

#include <mutex>
#include <map>
#include "util/utils.h"

namespace wsl {
namespace thunk {

class VaMgr {
public:
  VaMgr(uint64_t start, uint64_t size, uint64_t min_align);
  ~VaMgr();

  /* Allocate `bytes` VA, if `align` is not zero, the returned address is aligned by `align`.
   * If `addr` parameter is not zero, try best to allocate VA from fixed address `addr`.
   */
  uint64_t Alloc(uint64_t bytes, uint64_t align, uint64_t addr = 0);

  void Free(uint64_t addr);

private:
  uint64_t AllocImpl(uint64_t bytes, uint64_t align);

  struct Fragment {
    using ptr = std::multimap<uint64_t, uint64_t>::iterator;
    ptr free_list_entry_;

    struct {
      uint64_t size : 63;
      bool is_free : 1;
    };

    Fragment() : size(0), is_free(false) {}
    Fragment(ptr iterator, uint64_t len, bool is_free)
        : free_list_entry_(iterator), size(len), is_free(is_free) {}
  };

  static inline Fragment make_fragment(typename Fragment::ptr iter, uint64_t len) {
    return {iter, len, true};
  }

  inline Fragment make_fragment(uint64_t len) { return {free_list_.end(), len, false}; }

  static inline bool is_free(const Fragment& f) { return f.is_free; }
  void set_used(Fragment& f) {
    f.is_free = false;
    f.free_list_entry_ = free_list_.end();
  }
  static void set_free(Fragment& f, typename Fragment::ptr iter) {
    f.free_list_entry_ = iter;
    f.is_free = true;
  }

  inline void remove_free_list_entry(Fragment& frag) {
    if (frag.free_list_entry_ != free_list_.end()) {
      free_list_.erase(frag.free_list_entry_);
      frag.free_list_entry_ = free_list_.end();
    }
  }

  inline void add_free_fragment(uint64_t size, uint64_t base) {
    auto it = free_list_.insert(std::make_pair(size, base));
    frag_map_[base] = make_fragment(it, size);
  }

  inline void add_used_fragment(uint64_t size, uint64_t base) {
    frag_map_[base] = make_fragment(size);
  }
  // Indexed by size
  std::multimap<uint64_t, uint64_t> free_list_;
  // Indexed by VA, each fragment has no overlap
  std::map<uint64_t, Fragment> frag_map_;

  uint64_t min_align_;

  std::mutex lock_;  // Mutex protecting allocation and free of va


  DISALLOW_COPY_AND_ASSIGN(VaMgr);
};

} // namespace thunk
} // namespace wsl

