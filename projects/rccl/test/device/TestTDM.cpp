/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for asyncCopy.h
// 
// Includes tests for tensor data mover logic

#include "DeviceTestBase.hpp"
#include <limits>

#include "tdm/asyncCopy.h"

namespace RcclUnitTesting
{
  constexpr int warpSize = 32;

 // Naive TDM copy kernel: each warp copies one 1-D tile of data from global memory to LDS at a time and then writes back to global memory.
 // Exercises tdm.h
  template<typename T>
__global__ void kernelNaiveTDMCopy(const T* __restrict__ src, T* __restrict__ dst, size_t numElements, int numElementsPerTile) {
  extern __shared__ __align__(128) unsigned char sharedBytes[];
  T* shmem = reinterpret_cast<T*>(sharedBytes);
  int waveId = threadIdx.x / warpSize;
  int numWavesPerBlock = blockDim.x / warpSize;
  size_t itemsProcessedPerGridIteration = numElementsPerTile * numWavesPerBlock * gridDim.x;

  T* shmemPtr = shmem + numElementsPerTile * waveId;
  // Local per-wave source and destination pointers
  const T* srcPtr = src + numElementsPerTile * (waveId + blockIdx.x * numWavesPerBlock);
  T* dstPtr = dst + numElementsPerTile * (waveId + blockIdx.x * numWavesPerBlock);

  gfx1250_TDM_GROUP0 group0;

  group0.ldsAddr((uintptr_t)shmemPtr);

  gfx1250_TDM_GROUP1 group1;
  group1.dataSize(__builtin_ctzll(sizeof(T))); // Log2 of the element size in bytes, so 2 for float
  setTransferSize(group1, numElementsPerTile);

  constexpr __hip_uint32x4 empty_x4{};
  constexpr __hip_uint32x8 empty_x8{};
  while(srcPtr < src + numElements){
    // Handle the last tile of the block, which may be less than num_elements_per_tile.
    if(src + numElements - srcPtr < numElementsPerTile){
      size_t remainingElements = src + numElements - srcPtr;
      setTransferSize(group1, remainingElements);
    }
    // Copy from global memory to LDS
    group0.globalAddr((uintptr_t)srcPtr);
    __builtin_amdgcn_tensor_load_to_lds(group0.m_bitfield, group1.m_bitfield, empty_x4, empty_x4, empty_x8, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    // In practice, here we might do some computation on the data we've loaded, but just a copy here for simplicity
    // Write back from LDS to global
    group0.globalAddr((uintptr_t)dstPtr);
    __builtin_amdgcn_tensor_store_from_lds(group0.m_bitfield, group1.m_bitfield, empty_x4, empty_x4, empty_x8, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    srcPtr += itemsProcessedPerGridIteration;
    dstPtr += itemsProcessedPerGridIteration;
  }
}

 // Naive TDM copy kernel: each warp copies one 1-D tile of data from global memory to LDS at a time and then writes back to global memory.
 // Exercises TileMover API
 template<typename T>
 __global__ void kernelNaiveTDMCopyTileApi(const T* __restrict__ src, T* __restrict__ dst, size_t numElements, int numElementsPerTile) {
   extern __shared__ __align__(128) unsigned char sharedBytes[];
   T* shmem = reinterpret_cast<T*>(sharedBytes);
   int waveId = threadIdx.x / warpSize;
   int numWavesPerBlock = blockDim.x / warpSize;
   size_t itemsProcessedPerGridIteration = numElementsPerTile * numWavesPerBlock * gridDim.x;
 
   T* shmemPtr = shmem + numElementsPerTile * waveId;
   // Local per-wave source and destination pointers
   const T* srcPtr = src + numElementsPerTile * (waveId + blockIdx.x * numWavesPerBlock);
   T* dstPtr = dst + numElementsPerTile * (waveId + blockIdx.x * numWavesPerBlock);
 
   TileMover<T> tileMover;
   
   while(srcPtr < src + numElements){
     // Handle the last tile of the block, which may be less than num_elements_per_tile.
     if(src + numElements - srcPtr < numElementsPerTile){
       numElementsPerTile = static_cast<int>(src + numElements - srcPtr);
     }
     // Copy from global memory to LDS
     tileMover.loadTile(shmemPtr, srcPtr, numElementsPerTile);
     tileMover.waitTile();
     // In practice, here we might do some computation on the data we've loaded, but just a copy here for simplicity
     // Write back from LDS to global
     tileMover.storeTile(dstPtr);
     tileMover.waitTile();
     srcPtr += itemsProcessedPerGridIteration;
     dstPtr += itemsProcessedPerGridIteration;
   }
 }

class TDMTest : public DeviceTestBase { 
protected: 
  template<typename T> 
  void TestRoundTrip(const std::vector<T>& h_in) { 
    const int N = static_cast<int>(h_in.size());
    const int numBlocks = 4;
    DeviceBuffer<T> d_in(N), d_out(N); 
    d_in.copyFrom(h_in); 
    const int numElementsPerTile = 1024 * 4; 
    int minSharedMemorySize = numElementsPerTile * sizeof(T) * kDefaultBlockSize / warpSize;
    kernelNaiveTDMCopyTileApi<<<numBlocks, kDefaultBlockSize, minSharedMemorySize>>>(d_in.ptr, d_out.ptr, N, numElementsPerTile); 
    syncAndCheck(); 
  
    auto h_out = d_out.copyTo(); 
    for (int i = 0; i < N; i++) 
      EXPECT_EQ(h_in[i], h_out[i]) << "at index " << i; 
  } 
}; 

TEST_F(TDMTest, Float) {
  const int N = 1024*1024 + 128;
  std::vector<float> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = 1.0f / (i + 1);
  TestRoundTrip(h_in);
}

TEST_F(TDMTest, Double) {
  const int N = 1024*1024 + 128;
  std::vector<double> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = static_cast<double>(i) * 3.14159;
  TestRoundTrip(h_in);
}

}
