/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "common.h"

#ifndef RCCL_DEVICE_LINKER
__shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ < 700
__shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize() * (NCCL_MAX_NTHREADS / WARP_SIZE) / sizeof(ulong2)];
#endif
#endif

struct RunWorkNop {
  __device__ void run() {}
};

// RCCL_NT_SYM() renames these to *_512 when compiling the alternate gfx950
// 512-thread device dispatcher (-DRCCL_NTHREADS_512), giving distinct __global__
// entry points and a distinct 512-thread func table (see device.h / common.h).
//
// The 512-thread set is built for unroll 2 only (single-node/unroll-1 is capped
// at 256 threads and unroll 4 is never selected on gfx950), so the unroll-1 and
// unroll-4 generic dispatchers are emitted only for the default 256-thread set.
#ifndef RCCL_NTHREADS_512
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__
  void RCCL_NT_SYM(ncclDevKernel_Generic_1)(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, /*Unroll*/ 1>(&argsStorage.args);
}
#endif
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__
  void RCCL_NT_SYM(ncclDevKernel_Generic_2)(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, /*Unroll*/ 2>(&argsStorage.args);
}
#ifndef RCCL_NTHREADS_512
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__
  void RCCL_NT_SYM(ncclDevKernel_Generic_4)(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, /*Unroll*/ 4>(&argsStorage.args);
}
#endif

// [RCCL] Host-side stubs for the 512-thread generic kernels. The host launch
// path (enqueue.cc) takes the address of these symbols to launch the 512-thread
// device kernels that live in the gfx950 device ELF. They are emitted only when
// compiling common.cu.cpp for the host with the embedded fat binary
// (RCCL_HOST_FATBIN_COMPILE) so the device dispatcher compiles above emit exactly
// one kernel set each (avoiding duplicate device symbols at link time).
// Only the unroll-2 stub is emitted: the 512-thread set is built for unroll 2
// only (see the device dispatcher above and enqueue.cc/ncclKerns512).
#if defined(RCCL_ENABLE_GFX950_512) && defined(RCCL_HOST_FATBIN_COMPILE)
__launch_bounds__(512, 1) __global__
  void ncclDevKernel_Generic_2_512(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, /*Unroll*/ 2>(&argsStorage.args);
}
#endif

// RCCL_NT_SYM() keeps this in the same symbol namespace as the rest of the set
// (base or _512), so the two gfx950 dispatchers don't emit a duplicate definition.
#if defined(USE_INDIRECT_FUNCTION_CALL) || defined(RCCL_DEVICE_LINKER)
__device__ void RCCL_NT_SYM(ncclDevFunc_Nop)();
#else
__device__ __attribute__((noinline)) void RCCL_NT_SYM(ncclDevFunc_Nop)();
#endif

// [RCCL] Body for the no-op device func. RCCL's common.cu declares ncclDevFunc_Nop
// above (mode-aware attributes); generate.py excludes "Nop" from the generated
// per-impl/specialized files, so the definition must live here (as in upstream and
// the v2.29.7-1 base). The generated ncclDevFuncTable references this symbol.
__device__ void RCCL_NT_SYM(ncclDevFunc_Nop)() {}
