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

#include "small_heap.h"

namespace rocr {

// Inserts node into freelist after place.
// Assumes node will not be an end of the list (list has guard nodes).
void SmallHeap::insertafter(SmallHeap::iterator_t place, SmallHeap::iterator_t node) {
  assert(place->first < node->first && "Order violation");
  assert(isfree(place->second) && "Freelist operation error.");
  iterator_t next = place->second.next;
  node->second.next = next;
  node->second.prior = place;
  place->second.next = node;
  next->second.prior = node;
}

// Removes node from freelist.
// Assumes node will not be an end of the list (list has guard nodes).
void SmallHeap::remove(SmallHeap::iterator_t node) {
  assert(isfree(node->second) && "Freelist operation error.");
  node->second.prior->second.next = node->second.next;
  node->second.next->second.prior = node->second.prior;
  setused(node->second);
}

// Returns high if merge failed or the merged node.
SmallHeap::memory_t::iterator SmallHeap::merge(SmallHeap::memory_t::iterator low,
                                               SmallHeap::memory_t::iterator high) {
  assert(isfree(low->second) && "Merge with allocated block");
  assert(isfree(high->second) && "Merge with allocated block");

  if ((char*)low->first + low->second.len != (char*)high->first) return high;

  assert(!islastfree(high->second) && "Illegal merge.");

  low->second.len += high->second.len;
  low->second.next = high->second.next;
  high->second.next->second.prior = low;

  memory.erase(high);
  return low;
}

void SmallHeap::free(void* ptr) {
  if (ptr == nullptr) return;

  auto iterator = memory.find(ptr);

  // Check for illegal free
  if (iterator == memory.end()) {
    assert(false && "Illegal free.");
    return;
  }

  // Return memory to total and link node into free list
  total_free += iterator->second.len;

  // Could also traverse the free list which might be faster in some cases.
  auto before = iterator;
  before--;
  while (!isfree(before->second)) before--;
  assert(before->second.next->first > iterator->first && "Inconsistency in small heap.");
  insertafter(before, iterator);

  // Attempt compaction
  iterator = merge(before, iterator);
  merge(iterator, iterator->second.next);

  // Update lowHighBondary
  high.erase(ptr);
}

void* SmallHeap::alloc(size_t bytes) {
  // Is enough memory available?
  if ((bytes > total_free) || (bytes == 0)) return nullptr;

  iterator_t current;

  // Walk the free list and allocate at first fitting location
  current = firstfree();
  while (!islastfree(current->second)) {
    if (bytes <= current->second.len) {
      // Decrement from total
      total_free -= bytes;

      // Split node
      if (bytes != current->second.len) {
        void* remaining = (char*)current->first + bytes;
        Node& node = memory[remaining];
        node.len = current->second.len - bytes;
        current->second.len = bytes;
        insertafter(current, memory.find(remaining));
      }

      remove(current);
      return current->first;
    }
    current = current->second.next;
  }
  assert(current->second.len == 0 && "Freelist corruption.");

  // Can't service the request due to fragmentation
  return nullptr;
}

void* SmallHeap::alloc_high(size_t bytes) {
  // Is enough memory available?
  if ((bytes > total_free) || (bytes == 0)) return nullptr;

  iterator_t current;

  // Walk the free list and allocate at first fitting location
  current = lastfree();
  while (!isfirstfree(current->second)) {
    if (bytes <= current->second.len) {
      // Decrement from total
      total_free -= bytes;

      void* alloc;
      // Split node
      if (bytes != current->second.len) {
        alloc = (char*)current->first + current->second.len - bytes;
        current->second.len -= bytes;
        Node& node = memory[alloc];
        node.len = bytes;
        setused(node);
      } else {
        alloc = current->first;
        remove(current);
      }

      high.insert(alloc);
      return alloc;
    }
    current = current->second.prior;
  }
  assert(current->second.len == 0 && "Freelist corruption.");

  // Can't service the request due to fragmentation
  return nullptr;
}

}  // namespace rocr
