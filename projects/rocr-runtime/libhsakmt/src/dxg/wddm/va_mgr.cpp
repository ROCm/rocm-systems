/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cassert>
#include <map>
#include <algorithm>
#include "impl/wddm/va_mgr.h"

using namespace std;

namespace wsl {
namespace thunk {

VaMgr::VaMgr(uint64_t start, uint64_t size, uint64_t min_align) {
  min_align_ = min_align;
  auto free_it = free_list_.insert(make_pair(size, start));
  frag_map_[start] = make_fragment(free_it, size);
}

VaMgr::~VaMgr() {

  if (free_list_.size() != 1)
    pr_warn("free_list_ size:%ld which should be 1.\n", free_list_.size());
  if (frag_map_.size() != 1)
    pr_warn("frag_map_ size:%ld which should be 1.\n", frag_map_.size());

  free_list_.clear();
  frag_map_.clear();
}

uint64_t VaMgr::Alloc(uint64_t bytes, uint64_t align, uint64_t addr) {

  if (addr > 0 &&
      (align == 0 || (addr % align) == 0)) {

    lock_guard<mutex> gard(lock_);
    auto frag_it = frag_map_.upper_bound(addr);
    assert(frag_it != frag_map_.begin());
    --frag_it;

    while (frag_it != frag_map_.begin()) {
      const uint64_t base = frag_it->first;
      const uint64_t size = frag_it->second.size;

      // Cannot find free fragment contains the target `addr`
      if (bytes > size || addr < base || addr + bytes > base + size ||
          !is_free(frag_it->second)) {
        --frag_it;
        continue;
      } else if (addr >= base + size)
        break;


      // Try to allocate target `addr` from this free fragment
      auto free_it = frag_it->second.free_list_entry_;
      assert(free_it != free_list_.end());

      free_list_.erase(free_it);
      frag_it->second.size = bytes;
      set_used(frag_it->second);

      // [base, addr)
      if (addr > base) add_free_fragment(addr - base, base);

      // [addr, addr + bytes) is used

      // [addr + bytes, base + size)
      if (base + size > addr + bytes) add_free_fragment(base + size - addr - bytes, addr + bytes);

      return addr;
    }
  }

  // Allocate not fixed address
  return AllocImpl(bytes, align);
}

uint64_t VaMgr::AllocImpl(const uint64_t bytes, const uint64_t align) {
  uint64_t addr = 0;
  uint64_t align_bytes = bytes;
  const int retry = align == 0 ? 0 : 1;
  const uint64_t new_align = align == 0 ? min_align_ : rocr::AlignUp(align, min_align_);

  lock_guard<mutex> gard(lock_);
  for (int i = 0; i <= retry; i++) {
    auto free_it = free_list_.lower_bound(align_bytes);
    if (free_it == free_list_.end()) break;

    uint64_t base = free_it->second;
    uint64_t size = free_it->first;

    assert(size >= align_bytes);

    auto fragment = frag_map_.find(base);

    assert(fragment != frag_map_.end());
    assert(size == fragment->second.size);

    uint64_t delta = align == 0 ? 0 : base % align;
    if (delta == 0) {
      // already find aligned address
      addr = base;

      free_list_.erase(free_it);
      fragment->second.size = bytes;
      set_used(fragment->second);

      if (size > bytes) add_free_fragment(size - bytes, base + bytes);

      break;
    } else if (i == 0) {
      align_bytes += new_align;
      continue;
    } else {
      uint64_t aligned_base = base + align - delta;
      addr = aligned_base;

      free_list_.erase(free_it);

      add_used_fragment(bytes, aligned_base);
      add_free_fragment(aligned_base - base, base);

      if (size > aligned_base - base + bytes)
        add_free_fragment(size - (aligned_base - base) - bytes, aligned_base + bytes);

      break;
    }
  }
  return addr;
}

void VaMgr::Free(uint64_t addr) {
  if (addr == 0) return;

  lock_guard<mutex> gard(lock_);
  auto frag_it = frag_map_.find(addr);
  if (frag_it == frag_map_.end() || is_free(frag_it->second)) return;

  uint64_t base = addr;
  // Merge lower
  if (frag_it != frag_map_.begin()) {
    auto lower = frag_it;
    --lower;
    if (is_free(lower->second)) {
      remove_free_list_entry(lower->second);
      base -= lower->second.size;
      lower->second.size += frag_it->second.size;
      frag_map_.erase(frag_it);
      frag_it = lower;
    }
  }
  // Merge upper
  {
    auto upper = frag_it;
    ++upper;
    if (upper != frag_map_.end() && is_free(upper->second)) {
      remove_free_list_entry(upper->second);
      frag_it->second.size += upper->second.size;
      frag_map_.erase(upper);
    }
  }
  uint64_t size = frag_it->second.size;
  auto it = free_list_.insert(make_pair(size, base));
  set_free(frag_it->second, it);
}

} // namespace thunk
} // namespace wsl
