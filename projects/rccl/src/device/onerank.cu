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
  // handles all_reduce_bias (both in-place and out-of-place) for all ops
  // for 1-rank, Sum/Min/Max/Avg can be represented as Sum
  // however, PreMulSum for all_reduce_bias needs to be handled separately 
  void const* kernel;
  bool isPMS = (redOp.op == ncclDevPreMulSum);
  switch (eltType) {
  case ncclInt8:     kernel = isPMS ? (void const*)&oneRankReduce<FuncPreMulSum<int8_t>>
                                    : (void const*)&oneRankReduce<FuncSum<int8_t>>; break;
  case ncclUint8:    kernel = isPMS ? (void const*)&oneRankReduce<FuncPreMulSum<uint8_t>>
                                    : (void const*)&oneRankReduce<FuncSum<uint8_t>>; break;
  case ncclInt32:    kernel = isPMS ? (void const*)&oneRankReduce<FuncPreMulSum<int32_t>>
                                    : (void const*)&oneRankReduce<FuncSum<int32_t>>; break;
  case ncclUint32:   kernel = isPMS ? (void const*)&oneRankReduce<FuncPreMulSum<uint32_t>>
                                    : (void const*)&oneRankReduce<FuncSum<uint32_t>>; break;
  case ncclInt64:    kernel = isPMS ? (void const*)&oneRankReduce<FuncPreMulSum<int64_t>>
                                    : (void const*)&oneRankReduce<FuncSum<int64_t>>; break;
  case ncclUint64:   kernel = isPMS ? (void const*)&oneRankReduce<FuncPreMulSum<uint64_t>>
                                    : (void const*)&oneRankReduce<FuncSum<uint64_t>>; break;
#if defined(RCCL_FLOAT8)
  case ncclFloat8e4m3: kernel = isPMS ? (void const*)&oneRankReduce<FuncPreMulSum<rccl_float8>>
                                      : (void const*)&oneRankReduce<FuncSum<rccl_float8>>; break;
  case ncclFloat8e5m2: kernel = isPMS ? (void const*)&oneRankReduce<FuncPreMulSum<rccl_bfloat8>>
                                      : (void const*)&oneRankReduce<FuncSum<rccl_bfloat8>>; break;
#endif
  case ncclFloat16:  kernel = isPMS ? (void const*)&oneRankReduce<FuncPreMulSum<half>>
                                    : (void const*)&oneRankReduce<FuncSum<half>>; break;
#if defined(RCCL_BFLOAT16)
  case ncclBfloat16: kernel = isPMS ? (void const*)&oneRankReduce<FuncPreMulSum<hip_bfloat16>>
                                    : (void const*)&oneRankReduce<FuncSum<hip_bfloat16>>; break;
#endif
  case ncclFloat32:  kernel = isPMS ? (void const*)&oneRankReduce<FuncPreMulSum<float>>
                                    : (void const*)&oneRankReduce<FuncSum<float>>; break;
  case ncclFloat64:  kernel = isPMS ? (void const*)&oneRankReduce<FuncPreMulSum<double>>
                                    : (void const*)&oneRankReduce<FuncSum<double>>; break;
  default: return ncclInvalidArgument;
  }

  dim3 grid = {0, 1, 1};
  grid.x = std::min(32, (int)divUp(nElts*eltSize, 16<<10));
  dim3 block = {512, 1, 1};
  void* mutableSrc = const_cast<void*>(src);
  void* mutableAcc = const_cast<void*>(acc);
  void* args[6] = {&dst, &mutableSrc, &mutableAcc, &nElts, &redOp.scalarArg, &redOp.scalarArgIsPtr};
  CUDACHECK(cudaLaunchKernel(kernel, grid, block, args, 0, stream));
  return ncclSuccess;
}
