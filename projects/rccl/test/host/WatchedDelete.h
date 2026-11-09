/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Replaces every operator new/delete form with malloc/free, the matched pair ASan expects, and counts deletes
// of one watched pointer. Include it from one TU per binary, below any #undef of malloc/free.

#ifndef RCCL_TEST_HOST_WATCHEDDELETE_H_
#define RCCL_TEST_HOST_WATCHEDDELETE_H_

#include <cstddef>
#include <cstdlib>
#include <new>

static const void* g_deleteWatchPtr = nullptr;
static int g_deleteWatchHits = 0;
// Leaves the watched block allocated, so a second delete of it is counted rather than a double free.
static bool g_deleteWatchKeeps = false;

static void MicroOperatorDelete(void* p) noexcept {
  if (p != nullptr && p == g_deleteWatchPtr) {
    ++g_deleteWatchHits;
    if (g_deleteWatchKeeps) return;
  }
  std::free(p);
}
// Replaced as a matched pair and in every form: `delete p` emits the SIZED overload, so void* alone counts nothing.
void* operator new(std::size_t n) {
  void* p = std::malloc(n != 0 ? n : 1);
  if (p == nullptr) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { return std::malloc(n != 0 ? n : 1); }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { return std::malloc(n != 0 ? n : 1); }
void operator delete(void* p) noexcept { MicroOperatorDelete(p); }
void operator delete[](void* p) noexcept { MicroOperatorDelete(p); }
void operator delete(void* p, std::size_t) noexcept { MicroOperatorDelete(p); }
void operator delete[](void* p, std::size_t) noexcept { MicroOperatorDelete(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { MicroOperatorDelete(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { MicroOperatorDelete(p); }

#endif  // RCCL_TEST_HOST_WATCHEDDELETE_H_
