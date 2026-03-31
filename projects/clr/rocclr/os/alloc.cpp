/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "os/alloc.hpp"
#include "os/os.hpp"
#include "utils/util.hpp"

#include <cstdlib>

namespace amd {

void* AlignedMemory::allocate(size_t size, size_t alignment) {
  return Os::alignedMalloc(size, alignment);
}

void* GuardedMemory::allocate(size_t size, size_t alignment, size_t guardSize) {
  size_t sizeToAllocate = guardSize + alignment;
  sizeToAllocate += size + guardSize + Os::pageSize();

  sizeToAllocate = amd::alignUp(sizeToAllocate, Os::pageSize());
  address userHostMemGuarded = Os::reserveMemory(NULL, sizeToAllocate);
  if (!userHostMemGuarded ||
      !Os::commitMemory(userHostMemGuarded, sizeToAllocate, Os::MEM_PROT_RW)) {
    return NULL;
  }

  address userHostMem = userHostMemGuarded + sizeToAllocate;
  userHostMem = amd::alignDown(userHostMem - guardSize, Os::pageSize());

  // Protect the guard pages after the end of the users's buffer.
  if (!Os::protectMemory(userHostMem, guardSize, Os::MEM_PROT_NONE)) {
    fatal("Protect memory (up) failed");
  }

  userHostMem = userHostMem - size;
  userHostMem = amd::alignDown(userHostMem, alignment);
  // Write the actual size allocated including all the guard pages,
  // alignment, page file size... as well as the size of guarded byte
  // count before the beginning of the user's buffer.
  size_t* temp = reinterpret_cast<size_t*>(userHostMem);
  *--temp = sizeToAllocate;
  *--temp = userHostMem - userHostMemGuarded;

  // Protect the guard pages before the beginning of the user's buffer.
  if (!Os::protectMemory(userHostMemGuarded, guardSize, Os::MEM_PROT_NONE)) {
    fatal("Protect memory (down) failed");
  }

  return userHostMem;
}

void AlignedMemory::deallocate(void* ptr) { Os::alignedFree(ptr); }

void GuardedMemory::deallocate(void* ptr) {
  size_t* userHostMem = static_cast<size_t*>(ptr);

  size_t size = *--userHostMem;
  size_t offset = *--userHostMem;

  Os::releaseMemory(static_cast<address>(ptr) - offset, size);
}

void* HeapObject::operator new(size_t size) { return malloc(size); }

void HeapObject::operator delete(void* obj) { free(obj); }


}  // namespace amd
