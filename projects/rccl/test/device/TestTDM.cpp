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

// Async-to/from-LDS round trip kernel: each lane copies one element of type T from global memory
// into its own LDS slot and then writes it back to global memory.  Exercises asyncLoadToLDS /
// asyncStoreFromLDS across the supported element widths (1, 2, 4, 8, 16 bytes).
template<typename T>
__global__ void kernelAsyncCopyRoundTrip(const T* __restrict__ src, T* __restrict__ dst, size_t numElements) {
  extern __shared__ __align__(128) unsigned char sharedBytes[];
  T* shmem = reinterpret_cast<T*>(sharedBytes);
  size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= numElements) return;

  // Global -> LDS: each lane brings in its own element.
  asyncLoadToLDS<T>(&src[gid], &shmem[threadIdx.x]);
  asyncWait<0>();
  // LDS -> global: write the element back out.
  asyncStoreFromLDS<T>(&shmem[threadIdx.x], &dst[gid]);
  asyncWait<0>();
}

// 16-byte element type to exercise the b128 path.
struct alignas(16) Vec16 {
  uint32_t v[4];
  __host__ __device__ bool operator==(const Vec16& o) const {
    return v[0] == o.v[0] && v[1] == o.v[1] && v[2] == o.v[2] && v[3] == o.v[3];
  }
};
static_assert(sizeof(Vec16) == 16, "Vec16 must be 16 bytes");

inline std::ostream& operator<<(std::ostream& os, const Vec16& x) {
  return os << "{" << x.v[0] << "," << x.v[1] << "," << x.v[2] << "," << x.v[3] << "}";
}

// Launcher policies select which kernel a fixture exercises. Each provides a
// templated operator() so it works for any element type under test.
struct NaiveTDMLauncher {
  template<typename T>
  void operator()(int numBlocks, int blockSize, int sharedMem, const T* in, T* out, size_t numElements, int numElementsPerTile) const {
    kernelNaiveTDMCopy<<<numBlocks, blockSize, sharedMem>>>(in, out, numElements, numElementsPerTile);
  }
};

struct TileApiTDMLauncher {
  template<typename T>
  void operator()(int numBlocks, int blockSize, int sharedMem, const T* in, T* out, size_t numElements, int numElementsPerTile) const {
    kernelNaiveTDMCopyTileApi<<<numBlocks, blockSize, sharedMem>>>(in, out, numElements, numElementsPerTile);
  }
};

template<typename Launcher>
class TDMTestBase : public DeviceTestBase { 
protected: 
  template<typename T> 
  void TestRoundTrip(const std::vector<T>& h_in) { 
    const int N = static_cast<int>(h_in.size());
    const int numBlocks = 4;
    DeviceBuffer<T> d_in(N), d_out(N); 
    d_in.copyFrom(h_in); 
    const int numElementsPerTile = 1024 * 4; 
    int minSharedMemorySize = numElementsPerTile * sizeof(T) * kDefaultBlockSize / warpSize;
    Launcher{}(numBlocks, kDefaultBlockSize, minSharedMemorySize, d_in.ptr, d_out.ptr, N, numElementsPerTile); 
    syncAndCheck(); 
  
    auto h_out = d_out.copyTo(); 
    for (int i = 0; i < N; i++) 
      EXPECT_EQ(h_in[i], h_out[i]) << "at index " << i; 
  } 
}; 

using TDMTest = TDMTestBase<NaiveTDMLauncher>;
using TDMTestTileApi = TDMTestBase<TileApiTDMLauncher>;

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

TEST_F(TDMTestTileApi, Float) {
  const int N = 1024*1024 + 128;
  std::vector<float> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = 1.0f / (i + 1);
  TestRoundTrip(h_in);
}

TEST_F(TDMTestTileApi, Double) {
  const int N = 1024*1024 + 128;
  std::vector<double> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = static_cast<double>(i) * 3.14159;
  TestRoundTrip(h_in);
}

// Round-trips data through LDS using asyncLoadToLDS / asyncStoreFromLDS, one element per lane.
class AsyncCopyTest : public DeviceTestBase {
protected:
  template<typename T>
  void TestRoundTrip(const std::vector<T>& h_in) {
    const size_t N = h_in.size();
    DeviceBuffer<T> d_in(N), d_out(N);
    d_in.copyFrom(h_in);
    d_out.zero();

    const int blockSize = kDefaultBlockSize;
    const int numBlocks = static_cast<int>((N + blockSize - 1) / blockSize);
    const int sharedMem = blockSize * sizeof(T);
    kernelAsyncCopyRoundTrip<<<numBlocks, blockSize, sharedMem>>>(d_in.ptr, d_out.ptr, N);
    syncAndCheck();

    auto h_out = d_out.copyTo();
    for (size_t i = 0; i < N; i++)
      EXPECT_EQ(h_in[i], h_out[i]) << "at index " << i;
  }
};

// 1-byte elements -> b8 path.
TEST_F(AsyncCopyTest, Int8) {
  const int N = 256 * 100 + 17;  // intentionally not a multiple of the block size
  std::vector<int8_t> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = static_cast<int8_t>(i * 7 - 3);
  TestRoundTrip(h_in);
}

// 4-byte elements -> b32 path.
TEST_F(AsyncCopyTest, Float) {
  const int N = 256 * 100 + 17;
  std::vector<float> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = 1.0f / (i + 1);
  TestRoundTrip(h_in);
}

// 8-byte elements -> b64 path.
TEST_F(AsyncCopyTest, Double) {
  const int N = 256 * 100 + 17;
  std::vector<double> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = static_cast<double>(i) * 3.14159;
  TestRoundTrip(h_in);
}

// 16-byte elements -> b128 path.
TEST_F(AsyncCopyTest, Vec16) {
  const int N = 256 * 100 + 17;
  std::vector<Vec16> h_in(N);
  for (int i = 0; i < N; i++)
    h_in[i] = Vec16{{static_cast<uint32_t>(i), static_cast<uint32_t>(i * 2 + 1),
                     static_cast<uint32_t>(i * 3 + 2), static_cast<uint32_t>(i * 5 + 4)}};
  TestRoundTrip(h_in);
}

}  // namespace RcclUnitTesting
