/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef __ASYNCCOPY_H
#define __ASYNCCOPY_H

#include "tdm.h"

// Higher productivity async copy wrappers on top of TDM header, which is lower level.
// The TDM API does not have a function to set the transfer size, so we need to do it manually.
__device__ static void setTransferSize(gfx1250_TDM_GROUP1& group1, int numElements){
  group1.tensorDim0(numElements);
  group1.tensorDim0Stride(numElements);
  group1.tileDim0(numElements);
}

// Warp-level data copier.  Moves whole 1D tiles in between global memory and LDS using the TDM.  Both pointers ideally should be a multiple of 128-byte aligned 
// for maximum performance.
template<typename T>
struct TileMover{
  gfx1250_TDM_GROUP0 group0;
  gfx1250_TDM_GROUP1 group1;
  TileMover(){
    constexpr int log2DataSize = __builtin_ctzll(sizeof(T));
    static_assert(log2DataSize <= 2, "Datatype must be 1, 2, or 4 bytes in width"); // TODO(breslow): Replace with C++ concepts when we migrate to C++20
    group1.dataSize(log2DataSize);
  }

  // An entire warp makes this call.  The shmemPtr and srcPtr should be the same across all lanes in the warp.  Do not pass in per-lane offsets for addresses.
  __device__ void loadTile(T* shmemPtr, const T* srcPtr, int numElementsToProcess) {
    group0.ldsAddr((uintptr_t)shmemPtr);
    group0.globalAddr((uintptr_t)srcPtr);
    setTransferSize(group1, numElementsToProcess);
    __builtin_amdgcn_tensor_load_to_lds(group0.m_bitfield, group1.m_bitfield, __hip_uint32x4{}, __hip_uint32x4{}, __hip_uint32x8{}, 0);
  }

  // An entire warp makes this call.  The dstPtr should be the same across all lanes in the warp.  Do not pass in per-lane offsets for addresses.
  __device__ void storeTile(T* dstPtr) {
    group0.globalAddr((uintptr_t)dstPtr);
    __builtin_amdgcn_tensor_store_from_lds(group0.m_bitfield, group1.m_bitfield, __hip_uint32x4{}, __hip_uint32x4{}, __hip_uint32x8{}, 0);
  }

  __device__ void waitTile(){
    __builtin_amdgcn_s_wait_tensorcnt(0);
  }
};


#endif // __ASYNCCOPY_H
