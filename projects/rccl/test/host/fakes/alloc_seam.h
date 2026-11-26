/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Installs a free()/calloc() seam over a unit under test, for binaries that do
// not link fakes/libc_fakes.cc (fakes/libc_seam.h renames far more than these
// two). Include it immediately BEFORE the #include <UNIT>_CC_PATH and
// fakes/alloc_seam_undef.h straight after, so the test body still sees real
// libc. The define/undef pair is positional, so only the slots are guarded.
//
// The slots are TU-local and default to real libc, so installing the seam
// without driving it changes nothing. Swap them per case with ScopedHook, or
// install a fixture-wide recorder in SetUp and call ResetAllocSeam() in TearDown.
//
// TRAP: the renames are textual and TU-wide while the unit is included, so
// every free()/calloc() the unit's own headers reach goes through them too.

#ifndef RCCL_TEST_HOST_FAKES_ALLOC_SEAM_SLOTS
#define RCCL_TEST_HOST_FAKES_ALLOC_SEAM_SLOTS

#include <cstdlib>
#include <functional>

static std::function<void(void*)> g_microFree = [](void* p) { ::free(p); };
static std::function<void*(size_t, size_t)> g_microCalloc = [](size_t n, size_t size) { return ::calloc(n, size); };

static inline void MicroSeamFree(void* p) { g_microFree(p); }
static inline void* MicroSeamCalloc(size_t n, size_t size) { return g_microCalloc(n, size); }

static inline void ResetAllocSeam() {
  g_microFree = [](void* p) { ::free(p); };
  g_microCalloc = [](size_t n, size_t size) { return ::calloc(n, size); };
}

#endif  // RCCL_TEST_HOST_FAKES_ALLOC_SEAM_SLOTS

#define free(p) MicroSeamFree(p)
#define calloc(n, size) MicroSeamCalloc(n, size)
