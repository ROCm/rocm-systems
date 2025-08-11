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

// A simple first fit memory allocator with eager compaction.  For use with few
// items (where list iteration is faster than trees).
// Not thread safe!

#ifndef HSA_RUNTME_CORE_UTIL_SMALL_HEAP_H_
#define HSA_RUNTME_CORE_UTIL_SMALL_HEAP_H_

#include <map>
#include <set>

#include "utils.h"

namespace rocr {

class SmallHeap {
 private:
  struct Node;
  typedef std::map<void*, Node> memory_t;
  typedef memory_t::iterator iterator_t;

  struct Node {
    size_t len;
    iterator_t next;
    iterator_t prior;
  };

  SmallHeap(const SmallHeap& rhs) = delete;
  SmallHeap& operator=(const SmallHeap& rhs) = delete;

  void* const pool;
  const size_t length;

  size_t total_free;
  memory_t memory;
  std::set<void*> high;

  __forceinline bool isfree(const Node& node) const { return node.next != memory.begin(); }
  __forceinline bool islastfree(const Node& node) const { return node.next == memory.end(); }
  __forceinline bool isfirstfree(const Node& node) const { return node.prior == memory.end(); }
  __forceinline void setlastfree(Node& node) { node.next = memory.end(); }
  __forceinline void setfirstfree(Node& node) { node.prior = memory.end(); }
  __forceinline void setused(Node& node) { node.next = memory.begin(); }

  __forceinline iterator_t firstfree() { return memory.begin()->second.next; }
  __forceinline iterator_t lastfree() { return memory.rbegin()->second.prior; }
  void insertafter(iterator_t place, iterator_t node);
  void remove(iterator_t node);
  iterator_t merge(iterator_t low, iterator_t high);

 public:
  SmallHeap() : pool(nullptr), length(0), total_free(0) {}
  SmallHeap(void* base, size_t length)
      : pool(base), length(length), total_free(length) {
    assert(pool != nullptr && "Invalid base address.");
    assert(pool != (void*)0xFFFFFFFFFFFFFFFFull && "Invalid base address.");
    assert((char*)pool + length != (char*)0xFFFFFFFFFFFFFFFFull && "Invalid pool bounds.");

    Node& start = memory[0];
    Node& node = memory[pool];
    Node& end = memory[(void*)0xFFFFFFFFFFFFFFFFull];

    start.len = 0;
    start.next = memory.find(pool);
    setfirstfree(start);

    node.len = length;
    node.prior = memory.begin();
    node.next = --memory.end();

    end.len = 0;
    end.prior = start.next;
    setlastfree(end);

    high.insert((void*)0xFFFFFFFFFFFFFFFFull);
  }

  void* alloc(size_t bytes);
  void* alloc_high(size_t bytes);
  void free(void* ptr);

  void* base() const { return pool; }
  size_t size() const { return length; }
  size_t remaining() const { return total_free; }
  void* high_split() const { return *high.begin(); }
};

}  // namespace rocr

#endif
