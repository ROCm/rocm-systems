/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "alloc.h"
#include "collectives.h"
#include "common_kernel.h"
#include "common.h"
#include <cuda_runtime.h>

#if defined(__gfx950__)
#define COLL_UNROLL 1
#elif defined(__gfx908__) || defined(__gfx942__)
#define COLL_UNROLL 2
#else
#define COLL_UNROLL 4
#endif

namespace {
  template<typename RedOp>
  __global__ __launch_bounds__(512, 1)
  void oneRankReduce(void* dst, void* src, void* acc, size_t nElts, uint64_t redOpArg, bool redOpArgIsPtr) {
    using T = typename RedOp::EltType;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int bid = blockIdx.x;
    int bn = gridDim.x;

    // each block/channel gets a roughly equal segment of 16 byte packs
    constexpr int EltPerPack = 16/sizeof(T);
    intptr_t i0 = (bid+0)*alignUp(nElts/bn, EltPerPack);
    intptr_t i1 = (bid+1)*alignUp(nElts/bn, EltPerPack);
    i0 = min(i0, nElts);
    i1 = min(i1, nElts);

    if (redOpArgIsPtr) {
      if (redOpArg%2 != 0) {
        redOpArg = *reinterpret_cast<uint8_t*>(redOpArg);
      } else if (redOpArg%4 != 0) {
        redOpArg = *reinterpret_cast<uint16_t*>(redOpArg);
      } else if (redOpArg%8 != 0) {
        redOpArg = *reinterpret_cast<uint32_t*>(redOpArg);
      } else {
        redOpArg = *reinterpret_cast<uint64_t*>(redOpArg);
      }
    }

    if (acc != nullptr) {
      void* srcPtr = (T*)src + i0;
      void* accPtr = (T*)acc + i0;
      void* dstPtr = (T*)dst + i0;
      void* srcs[2] = {srcPtr, accPtr};
      void* dsts[1] = {dstPtr};
      reduceCopy<COLL_UNROLL, 0, RedOp, T, 0,2,2, 0,1,1, /*PreOpSrcs=*/1>
        (tid, tn, redOpArg, &redOpArg, true, 2, srcs, 1, dsts, i1-i0);
    } else {
      src = (T*)src + i0;
      dst = (T*)dst + i0;
      reduceCopy<COLL_UNROLL, 0, RedOp, T, 0,1,1, 0,1,1, /*PreOpSrcs=*/1>
        (tid, tn, redOpArg, &redOpArg, true, 1, &src, 1, &dst, i1-i0);
    }
  }
}

ncclResult_t ncclLaunchOneRank(void* dst, void const* src, size_t nElts, struct ncclDevRedOpFull redOp, ncclDataType_t eltType, cudaStream_t stream, void const* acc) {
  size_t eltSize = ncclTypeSize(eltType);

  // handles all_reduce for non-PreMulSum ops
  // for 1 rank, out-of-place is memcpy to self, and in-place is nop
  if (acc == nullptr && redOp.op != ncclDevPreMulSum) {
    if (dst != src) {
      NCCLCHECK(ncclCudaMemcpyAsync((char*)dst, (char*)src, nElts*eltSize, stream));
    }
    return ncclSuccess;
  }

  // handles all_reduce (both in-place and out-of-place) for PreMulSum
  //
  // handles all_reduce_bias (both in-place and out-of-place) for all ops
  // bias is applied as op(allreduce_result, bias), using the same reduction op:
  //   Sum/Avg/PreMulSum -> result + bias  (applyReduce delegates to FuncSum)
  //   Prod              -> result * bias
  //   Min/Max           -> min/max(result, bias)
  // SumPostDiv (ncclAvg on integers) maps to FuncSum here because:
  //   1. applyReduce for FuncSumPostDiv delegates to FuncSum (addition)
  //   2. for 1-rank, post-divide by nRanks=1 is identity
  //   3. FuncSumPostDiv has a static_assert restricting it to unsigned types
  void const* kernel;

#define CASE(ncclDataType, dataType) \
  case ncclDataType: \
    switch (redOp.op) { \
    case ncclDevSum: kernel = (void const*)&oneRankReduce<FuncSum<dataType>>; break; \
    case ncclDevProd: kernel = (void const*)&oneRankReduce<FuncProd<dataType>>; break; \
    case ncclDevMinMax: kernel = (void const*)&oneRankReduce<FuncMinMax<dataType>>; break; \
    case ncclDevPreMulSum: kernel = (void const*)&oneRankReduce<FuncPreMulSum<dataType>>; break; \
    case ncclDevSumPostDiv: kernel = (void const*)&oneRankReduce<FuncSum<dataType>>; break; \
    default: return ncclInvalidArgument; \
    } break;

  switch (eltType) {
  CASE(ncclInt8, int8_t)
  CASE(ncclUint8, uint8_t)
  CASE(ncclInt32, int32_t)
  CASE(ncclUint32, uint32_t)
  CASE(ncclInt64, int64_t)
  CASE(ncclUint64, uint64_t)
#if defined(RCCL_FLOAT8)
  CASE(ncclFloat8e4m3, rccl_float8)
  CASE(ncclFloat8e5m2, rccl_bfloat8)
#endif
  CASE(ncclFloat16, half)
#if defined(RCCL_BFLOAT16)
  CASE(ncclBfloat16, hip_bfloat16)
#endif
  CASE(ncclFloat32, float)
  CASE(ncclFloat64, double)
  default: return ncclInvalidArgument;
  }
#undef CASE

  dim3 grid = {0, 1, 1};
  grid.x = std::min(32, (int)divUp(nElts*eltSize, 16<<10));
  dim3 block = {512, 1, 1};
  void* mutableSrc = const_cast<void*>(src);
  void* mutableAcc = const_cast<void*>(acc);
  void* args[6] = {&dst, &mutableSrc, &mutableAcc, &nElts, &redOp.scalarArg, &redOp.scalarArgIsPtr};
  CUDACHECK(cudaLaunchKernel(kernel, grid, block, args, 0, stream));
  return ncclSuccess;
}
