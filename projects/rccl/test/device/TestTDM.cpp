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
#include <hip/hip_bfloat16.h>

#include "tdm/asyncCopy.h"

namespace RcclUnitTesting
{
  constexpr int warpSize = 32;

// The kernels and fixtures below name the gfx1250 TDM descriptor types and drive the
// tensor load/store instructions directly, so they only exist when the SDK ships
// hip/amd_detail/amd_gfx1250_TDM.h. Without it asyncCopy.h leaves TileMover undeclared
// and this coverage compiles out; test/CMakeLists.txt warns when that happens. The
// AsyncDataCopier cases at the bottom need no descriptor and always build.
#if TDM_TOOLCHAIN_AVAILABLE

 // Naive TDM copy kernel: each warp copies one 1-D tile of data from global memory to LDS at a time and then writes back to global memory.
 // Exercises the SDK's TDM descriptor header
  template<typename T>
__global__ void kernelNaiveTDMCopy([[maybe_unused]] const T* __restrict__ src, [[maybe_unused]] T* __restrict__ dst,
                                   [[maybe_unused]] size_t numElements, [[maybe_unused]] int numElementsPerTile,
                                   [[maybe_unused]] int offAlignmentLDS) {
// The TDM instructions below only exist on gfx1250, and this file is compiled once per
// target in a multi-arch build, so the body has to compile out on every other device
// pass (and in the host pass). The fixture skips at runtime when support is absent.
#if TDM_SUPPORTED
  extern __shared__ __align__(128) unsigned char sharedBytes[];
  T* shmem = reinterpret_cast<T*>(sharedBytes + offAlignmentLDS);
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
#endif // TDM_SUPPORTED
}
#endif // TDM_TOOLCHAIN_AVAILABLE

 // Naive TDM copy kernel: each warp copies one 1-D tile of data from global memory to LDS at a time and then writes back to global memory.
 // Allows for injecting misalignment of the LDS pointer to test the tile mover's ability to handle it.
 // Exercises whichever tile mover is supplied (TileMover or AsyncDataCopier), so the
 // mover is named explicitly by each launcher rather than defaulted here, keeping this
 // kernel usable when TileMover is unavailable.
 template<typename T, typename TileMoverType>
 __global__ void kernelNaiveTDMCopyTileApi([[maybe_unused]] const T* __restrict__ src, [[maybe_unused]] T* __restrict__ dst,
                                           [[maybe_unused]] size_t numElements, [[maybe_unused]] int numElementsPerTile,
                                           [[maybe_unused]] int offAlignmentLDS) {
// Both movers issue gfx1250-only instructions, and their members are deleted on a device
// pass for any other arch, so the body -- which is what instantiates them -- compiles out
// there rather than failing to build a target that never runs these tests.
#if TDM_SUPPORTED || ASYNC_COPY_SUPPORTED
   extern __shared__ __align__(128) unsigned char sharedBytes[];
   T* shmem = reinterpret_cast<T*>(sharedBytes + offAlignmentLDS); // 
   int waveId = threadIdx.x / warpSize;
   int numWavesPerBlock = blockDim.x / warpSize;
   size_t itemsProcessedPerGridIteration = numElementsPerTile * numWavesPerBlock * gridDim.x;
 
   T* shmemPtr = shmem + numElementsPerTile * waveId;
   // Local per-wave source and destination pointers
   const T* srcPtr = src + numElementsPerTile * (waveId + blockIdx.x * numWavesPerBlock);
   T* dstPtr = dst + numElementsPerTile * (waveId + blockIdx.x * numWavesPerBlock);
 
   TileMoverType tileMover;
   
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
#endif // TDM_SUPPORTED || ASYNC_COPY_SUPPORTED
 }

// Launcher policies select which kernel a fixture exercises. Each provides a
// templated operator() so it works for any element type under test.
#if TDM_TOOLCHAIN_AVAILABLE
struct NaiveTDMLauncher {
  template<typename T>
  void operator()(int numBlocks, int blockSize, int sharedMem, const T* in, T* out, size_t numElements, int numElementsPerTile, int offAlignmentLDS) const {
    kernelNaiveTDMCopy<<<numBlocks, blockSize, sharedMem>>>(in, out, numElements, numElementsPerTile, offAlignmentLDS);
  }
};

struct TileApiTDMLauncher {
  template<typename T>
  void operator()(int numBlocks, int blockSize, int sharedMem, const T* in, T* out, size_t numElements, int numElementsPerTile, int offAlignmentLDS) const {
    kernelNaiveTDMCopyTileApi<T, TileMover<T>><<<numBlocks, blockSize, sharedMem>>>(in, out, numElements, numElementsPerTile, offAlignmentLDS);
  }
};
#endif // TDM_TOOLCHAIN_AVAILABLE

// Same kernel as TileApiTDMLauncher, but drives it with the AsyncDataCopier tile mover, which is implemented
// on top of the async-to/from-LDS builtins rather than the TDM tensor load/store instructions.
struct AsyncDataCopierTileApiLauncher {
  template<typename T>
  void operator()(int numBlocks, int blockSize, int sharedMem, const T* in, T* out, size_t numElements, int numElementsPerTile, int offAlignmentLDS) const {
    kernelNaiveTDMCopyTileApi<T, AsyncDataCopier<T>><<<numBlocks, blockSize, sharedMem>>>(in, out, numElements, numElementsPerTile, offAlignmentLDS);
  }
};

template<typename Launcher>
class AsyncCopyTestBase : public DeviceTestBase { 
protected: 
  void SetUp() override {
    DeviceTestBase::SetUp();
    // These kernels reach the tensor data mover and the async-to/from-LDS builtins,
    // whose bodies compile away on any device that does not support them. Skip rather
    // than compare against a copy that never ran.
    if (!async::IsTdmCopySupported(0))
      GTEST_SKIP() << "async copy / TDM not supported on this device";
  }

  template<typename T> 
  void TestRoundTrip(const std::vector<T>& h_in) { 
    const int N = static_cast<int>(h_in.size());
    const int numBlocks = 4;
    DeviceBuffer<T> d_in(N), d_out(N); 
    d_in.copyFrom(h_in); 
    const int numElementsPerTile = 1024 * 4 - 1; 
    const int offAlignmentLDS = 3 * sizeof(T);
    int minSharedMemorySize = numElementsPerTile * sizeof(T) * kDefaultBlockSize / warpSize + offAlignmentLDS;
    Launcher{}(numBlocks, kDefaultBlockSize, minSharedMemorySize, d_in.ptr, d_out.ptr, N, numElementsPerTile, offAlignmentLDS); 
    syncAndCheck(); 
  
    auto h_out = d_out.copyTo(); 
    for (int i = 0; i < N; i++) 
      EXPECT_EQ(h_in[i], h_out[i]) << "at index " << i; 
  } 
}; 

using TestAsyncDataCopierTileApi = AsyncCopyTestBase<AsyncDataCopierTileApiLauncher>;

#if TDM_TOOLCHAIN_AVAILABLE
using TestNaiveTDM = AsyncCopyTestBase<NaiveTDMLauncher>;
using TestTileApiTDM = AsyncCopyTestBase<TileApiTDMLauncher>;

TEST_F(TestNaiveTDM, char) {
  const int N = 314159;;
  std::vector<char> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = 1.0f / (i + 1);
  TestRoundTrip(h_in);
}

TEST_F(TestNaiveTDM, BFloat16) {
  const int N = 314159;
  std::vector<hip_bfloat16> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = hip_bfloat16(1.0f / (i + 1));
  TestRoundTrip(h_in);
}

TEST_F(TestNaiveTDM, Float) {
  const int N = 314159;
  std::vector<float> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = 1.0f / (i + 1);
  TestRoundTrip(h_in);
}

TEST_F(TestNaiveTDM, Double) {
  const int N = 314159;
  std::vector<double> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = static_cast<double>(i) * 3.14159;
  TestRoundTrip(h_in);
}

TEST_F(TestTileApiTDM, Float) {
  const int N = 314159;
  std::vector<float> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = 1.0f / (i + 1);
  TestRoundTrip(h_in);
}

TEST_F(TestTileApiTDM, Double) {
  const int N = 314159;
  std::vector<double> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = static_cast<double>(i) * 3.14159;
  TestRoundTrip(h_in);
}
#endif // TDM_TOOLCHAIN_AVAILABLE

// Same tile-copy kernel as TDMTestTileApi, but driven by the AsyncDataCopier tile mover.
TEST_F(TestAsyncDataCopierTileApi, Byte) {
  const int N = 314159;
  std::vector<uint8_t> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = rand() % 256;
  TestRoundTrip(h_in);
}

TEST_F(TestAsyncDataCopierTileApi, Double) {
  const int N = 314159;
  std::vector<double> h_in(N);
  for (int i = 0; i < N; i++) h_in[i] = static_cast<double>(i) * 3.14159;
  TestRoundTrip(h_in);
}

}  // namespace RcclUnitTesting
