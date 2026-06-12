/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// When RCCL is built with ENABLE_ROCSHMEM_GIN but not ENABLE_ROCSHMEM,
// librccl.so holds an unresolved reference to rocshmem::anvil::anvil.
// Tests link librocshmem.a and rely on the dynamic loader to bind that
// symbol from the executable. The linker can still drop the anvil object
// from the archive (no TU in the tests references it). This file keeps
// the symbol linked into the binary.

#include "anvil.hpp"

namespace {

void* rcclGinRocshmemAnvilAddr() { return (void*)&rocshmem::anvil::anvil; }

#if defined(__GNUC__) || defined(__clang__)
__attribute__((used))
#endif
static void* volatile kRcclGinRocshmemAnvilAnchor = rcclGinRocshmemAnvilAddr();

}  // namespace
