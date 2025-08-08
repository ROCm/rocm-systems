////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2020-2020, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HSA_RUNTIME_CORE_INC_SCRATCH_CACHE_H_
#define HSA_RUNTIME_CORE_INC_SCRATCH_CACHE_H_

#include "core/util/locks.h"
#include "core/util/utils.h"

#include <map>
#include <functional>

namespace rocr {
namespace AMD {

class ScratchCache {
 public:
  struct node {
    enum STATE { FREE = 0, ALLOC = 1, TRIM = 2, STEAL = 4 };
    void* base;
    bool large;
    uint32_t state;

    node() : base(nullptr), large(false), state(FREE) {}

    bool isFree() const { return state == FREE; }
    bool trimPending() const { return state == (ALLOC | TRIM); }

    void trim() {
      assert(!isFree() && "Trim of free scratch node.");
      state |= TRIM;
    }
    void free() {
      assert(!isFree() && "Free of free scratch node.");
      state = FREE;
    }
    void alloc() {
      assert(isFree() && "Alloc of non-free scratch node.");
      state = ALLOC;
    }
  };

  typedef ::std::multimap<size_t, node> map_t;
  typedef map_t::iterator ref_t;
  typedef ::std::function<void(void*, size_t, bool)> deallocator_t;

  // @brief Contains scratch memory information.
  struct ScratchInfo {
    // Size to satisfy the present dispatch without throttling.
    size_t dispatch_size;
    uint64_t dispatch_slots;

    bool large;
    size_t use_once_limit;
    size_t use_alt_limit;
    bool async_reclaim;  // This version of CP FW supports async_reclaim
    bool retry;
    uint32_t mem_alignment_size;  // Populated into SRD
    bool cooperative;
    hsa_signal_t queue_retry;

    // Size to fill the main_scratch with size_per_thread
    size_t main_size;
    size_t main_size_per_thread;    // Populated into SRD
    uint32_t main_lanes_per_wave;   // Populated into SRD
    uint32_t main_waves_per_group;  // Used during waves reduction
    void* main_queue_base;
    ptrdiff_t main_queue_process_offset;
    ScratchCache::ref_t main_scratch_node;

    size_t alt_size;
    size_t alt_size_per_thread;    // Populated into SRD
    uint32_t alt_lanes_per_wave;   // Populated into SRD
    uint32_t alt_waves_per_group;  // Used during waves reduction

    uint64_t alt_dispatch_limit_x;
    uint64_t alt_dispatch_limit_y;
    uint64_t alt_dispatch_limit_z;
    void* alt_queue_base;
    ptrdiff_t alt_queue_process_offset;
    ScratchCache::ref_t alt_scratch_node;
  };

  ScratchCache(const ScratchCache& rhs) = delete;
  ScratchCache(ScratchCache&& rhs) = delete;
  ScratchCache& operator=(const ScratchCache& rhs) = delete;
  ScratchCache& operator=(ScratchCache&& rhs) = delete;

  ScratchCache(deallocator_t deallocator) : dealloc(std::move(deallocator)), available_bytes_(0) {}

  ~ScratchCache() { assert(map.empty() && "ScratchCache not empty at shutdown."); }

  bool allocMain(ScratchInfo& info) {
    ref_t it = map.upper_bound(info.main_size - 1);
    if (it == map.end()) return false;

    // Small requests must have an exact size match and be small.
    if (!info.large) {
      while ((it != map.end()) && (it->first == info.main_size)) {
        if (it->second.isFree() && (!it->second.large)) {
          it->second.alloc();
          info.main_queue_base = it->second.base;
          info.main_scratch_node = it;
          available_bytes_ -= it->first;
          return true;
        }
        it++;
      }
      return false;
    }

    // Large requests may use a small allocation and do not require an exact size match.
    while (it != map.end()) {
      if (it->second.isFree()) {
        it->second.alloc();
        info.main_queue_base = it->second.base;
        info.main_scratch_node = it;
        available_bytes_ -= it->first;
        return true;
      }
      it++;
    }
    return false;
  }

  void freeMain(ScratchInfo& info) {
    if (info.main_scratch_node == map.end()) {
      // This is reserved scratch memory. Do not de-allocate, just mark it as free.
      assert(!reserved_.second.isFree() && "free called when reserved node already free.");
      reserved_.second.free();
      available_bytes_ += reserved_.first;
      return;
    }

    assert(!info.main_scratch_node->second.isFree() && "free called on free scratch node.");
    auto it = info.main_scratch_node;
    if (it->second.trimPending()) {
      dealloc(it->second.base, it->first, it->second.large);
      map.erase(it);
      return;
    }
    it->second.free();
    available_bytes_ += it->first;
  }

  void insertMain(ScratchInfo& info) {
    node n;
    n.base = info.main_queue_base;
    n.large = info.large;
    n.alloc();

    auto it = map.insert(std::make_pair(info.main_size, n));
    info.main_scratch_node = it;
  }

  bool trim(bool trim_nodes_in_use) {
    bool ret = !map.empty();
    auto it = map.begin();
    while (it != map.end()) {
      if (it->second.isFree()) {
        available_bytes_ -= it->first;
        dealloc(it->second.base, it->first, it->second.large);
        auto temp = it;
        it++;
        map.erase(temp);
      } else {
        if (trim_nodes_in_use) it->second.trim();
        it++;
      }
    }
    return ret;
  }

  bool allocAlt(ScratchInfo& info) {
    ref_t it = map.upper_bound(info.alt_size - 1);
    if (it == map.end()) return false;

    // Alt requests should have exact size
    while ((it != map.end()) && (it->first == info.alt_size)) {
      if (it->second.isFree() && (!it->second.large)) {
        it->second.alloc();
        info.alt_queue_base = it->second.base;
        info.alt_scratch_node = it;
        available_bytes_ -= it->first;
        return true;
      }
      it++;
    }
    return false;
  }

  void freeAlt(ScratchInfo& info) {
    assert(!info.alt_scratch_node->second.isFree() && "free called on free scratch node.");
    auto it = info.alt_scratch_node;
    if (it->second.trimPending()) {
      dealloc(it->second.base, it->first, it->second.large);
      map.erase(it);
      return;
    }
    it->second.free();
    available_bytes_ += it->first;
  }

  void insertAlt(ScratchInfo& info) {
    node n;
    n.base = info.alt_queue_base;
    n.large = false;
    n.alloc();

    auto it = map.insert(std::make_pair(info.alt_size, n));
    info.alt_scratch_node = it;
  }

  size_t free_bytes() const { return available_bytes_; }
  size_t reserved_bytes() const { return reserved_.first; }

  void reserve(size_t bytes, void* base) {
    assert(!reserved_.first && "Already reserved memory.");

    node n;
    n.base = base;
    n.large = 0;

    available_bytes_ += bytes;

    reserved_ = std::make_pair(bytes, n);
  }

  bool use_reserved(ScratchInfo& info) {
    if (!reserved_.second.isFree() || info.main_size > reserved_.first) {
      debug_print("reserved node is already in use or too small (requested:%ld reserved:%ld)\n",
                  info.main_size, reserved_.first);
      return false;
    }
    reserved_.second.large = info.large;
    reserved_.second.alloc();
    info.main_queue_base = reserved_.second.base;
    // Special case to indicate that this node is reserved memory
    info.main_scratch_node = map.end();
    available_bytes_ -= reserved_.first;
    return true;
  }

  void free_reserve() {
    available_bytes_ -= reserved_.first;
    if (reserved_.first) dealloc(reserved_.second.base, reserved_.first, reserved_.second.large);

    reserved_.first = 0;
    reserved_.second.base = NULL;
    reserved_.second.large = 0;
  }

 private:
  map_t map;
  deallocator_t dealloc;
  size_t available_bytes_;

  std::pair<size_t, node> reserved_;
};

}  // namespace AMD
}  // namespace rocr

#endif  // header guard
