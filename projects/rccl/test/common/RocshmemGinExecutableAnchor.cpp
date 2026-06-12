/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// When RCCL is built with ENABLE_ROCSHMEM_GIN but not ENABLE_ROCSHMEM,
// librccl.so holds an unresolved reference to rocshmem::anvil::anvil.
// Tests link librocshmem.a; the linker can drop anvil.o if nothing in the
// test objects references it. We only need the symbol address — do not
// include rocshmem's anvil.hpp here (it pulls HSA/HIP/device headers that
// are not set up like the rocshmem library build).

namespace rocshmem {
namespace anvil {
class AnvilLib;
extern AnvilLib& anvil;
}  // namespace anvil
}  // namespace rocshmem

namespace {

void* rcclGinRocshmemAnvilAddr() { return (void*)&rocshmem::anvil::anvil; }

#if defined(__GNUC__) || defined(__clang__)
__attribute__((used))
#endif
static void* volatile kRcclGinRocshmemAnvilAnchor = rcclGinRocshmemAnvilAddr();

}  // namespace
