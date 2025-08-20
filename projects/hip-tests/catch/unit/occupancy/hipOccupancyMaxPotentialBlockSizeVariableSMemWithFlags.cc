/*
Copyright (c) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip_test_common.hh>

#define SHARED_MEM_CONST 256
#define UNUSED(expr)                                                                               \
  do {                                                                                             \
    (void)(expr);                                                                                  \
  } while (0)
// global variables
static int gArrSize = 0;

// sample global functions
static __global__ void f1(float* a) { *a = 1.0; }

// Dynamic shared
static __global__ void copyKerDyn(int* out, int* in) {
  extern __shared__ int sharedMem[];
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  sharedMem[tid] = in[tid];
  __syncthreads();
  out[tid] = sharedMem[tid];
}

// Without Dynamic shared
static __global__ void copyKer(int* out, int* in) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  out[tid] = in[tid];
}

// sample function
static size_t blockSizeToDynamicSMemSize(int blocksize) {
  return (static_cast<size_t>(blocksize * SHARED_MEM_CONST));
}

// sample functor
class functorBlockSizeToDynamicSMemSize {
  int myconst;

 public:
  explicit functorBlockSizeToDynamicSMemSize(int n) : myconst(n) {}
  int operator()(int blocksize) const { return (static_cast<size_t>(blocksize * myconst)); }
};

/**
  Local function to check hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
  functionality for different block_size_limit.
*/
void hipOccupancyMaxPotBlkSizeVariableSMemWithFlags_chkRange(int block_size_limit,
                                                             int maxThreadsPerBlock) {
  int minGridSize = 0, blockSize = 0;
  hipError_t ret;
  // Get potential blocksize
  ret = hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
      &minGridSize, &blockSize, f1, blockSizeToDynamicSMemSize, block_size_limit, 0);
  REQUIRE(ret == hipSuccess);
  REQUIRE(minGridSize > 0);
  REQUIRE(blockSize > 0);
  REQUIRE(blockSize <= maxThreadsPerBlock);
}

/**
  Check the basic functionality of hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
  - for block_size_limit = 0
  - for 0 < block_size_limit < attr.maxThreadsPerBlock
  - for block_size_limit > attr.maxThreadsPerBlock
*/
TEST_CASE("Unit_hipOccupancyMaxPotBlkSizeVariableSMemWithFlags_chkRange") {
  hipDeviceProp_t devProp;
  // Get current device property
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  SECTION("block_size_limit = 0") {
    hipOccupancyMaxPotBlkSizeVariableSMemWithFlags_chkRange(0, devProp.maxThreadsPerBlock);
  }
  SECTION("block_size_limit < maxThreadsPerBlock") {
    hipOccupancyMaxPotBlkSizeVariableSMemWithFlags_chkRange((devProp.maxThreadsPerBlock - 1),
                                                            devProp.maxThreadsPerBlock);
  }
  SECTION("block_size_limit = maxThreadsPerBlock") {
    hipOccupancyMaxPotBlkSizeVariableSMemWithFlags_chkRange(devProp.maxThreadsPerBlock,
                                                            devProp.maxThreadsPerBlock);
  }
  SECTION("block_size_limit > maxThreadsPerBlock") {
    hipOccupancyMaxPotBlkSizeVariableSMemWithFlags_chkRange((devProp.maxThreadsPerBlock + 1),
                                                            devProp.maxThreadsPerBlock);
  }
}

/**
  Check range of minGridSize and blockSize for multiple GPU
  - for block_size_limit = 0
  - for 0 < block_size_limit < attr.maxThreadsPerBlock
  - for block_size_limit > attr.maxThreadsPerBlock
*/
TEST_CASE("Unit_hipOccupancyMaxPotBlkSizeVariableSMemWithFlags_mgpu") {
  int devcount = 0;
  HIP_CHECK(hipGetDeviceCount(&devcount));
  // If only single GPU is detected then return
  if (devcount < 2) {
    SUCCEED("Skipping the test as number of Devices found less than 2");
    return;
  }
  // Get current device property
  for (int dev = 0; dev < devcount; dev++) {
    hipDeviceProp_t devProp;
    HIP_CHECK(hipGetDeviceProperties(&devProp, dev));
    HIP_CHECK(hipSetDevice(dev));
    hipOccupancyMaxPotBlkSizeVariableSMemWithFlags_chkRange(0, devProp.maxThreadsPerBlock);
    hipOccupancyMaxPotBlkSizeVariableSMemWithFlags_chkRange((devProp.maxThreadsPerBlock - 1),
                                                            devProp.maxThreadsPerBlock);
    hipOccupancyMaxPotBlkSizeVariableSMemWithFlags_chkRange(devProp.maxThreadsPerBlock,
                                                            devProp.maxThreadsPerBlock);
    HIP_CHECK(hipSetDevice(0));
  }
}

/**
  Check the basic functionality of hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
  by passing a functor as 4th parameter.
*/
TEST_CASE("Unit_hipOccupancyMaxPotBlkSizeVariableSMemWithFlags_Functor") {
  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  functorBlockSizeToDynamicSMemSize testFunc(SHARED_MEM_CONST);
  // Get current device property
  int minGridSize = 0, blockSize = 0;
  hipError_t ret;
  // Get potential blocksize
  ret = hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(&minGridSize, &blockSize, f1,
                                                               testFunc, 0, 0);
  REQUIRE(ret == hipSuccess);
  REQUIRE(minGridSize > 0);
  REQUIRE(blockSize > 0);
  REQUIRE(blockSize <= devProp.maxThreadsPerBlock);
}

/**
  Check the basic functionality of hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
  by passing a lambda function as 4th parameter.
*/
TEST_CASE("Unit_hipOccupancyMaxPotBlkSizeVariableSMemWithFlags_Lambda") {
  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  auto testFunc = [](const int blockSize) {
    return (static_cast<size_t>(blockSize * SHARED_MEM_CONST));
  };
  // Get current device property
  int minGridSize = 0, blockSize = 0;
  hipError_t ret;
  // Get potential blocksize
  ret = hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(&minGridSize, &blockSize, f1,
                                                               testFunc, 0, 0);
  REQUIRE(ret == hipSuccess);
  REQUIRE(minGridSize > 0);
  REQUIRE(blockSize > 0);
  REQUIRE(blockSize <= devProp.maxThreadsPerBlock);
  // Test again by passing the lamda function directly
  ret = hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
      &minGridSize, &blockSize, f1,
      [](const int blockSize) { return (static_cast<size_t>(blockSize * SHARED_MEM_CONST)); }, 0,
      0);
  REQUIRE(ret == hipSuccess);
  REQUIRE(minGridSize > 0);
  REQUIRE(blockSize > 0);
  REQUIRE(blockSize <= devProp.maxThreadsPerBlock);
}

/**
  Negative tests hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
  - null min_grid_size
  - null block_size
  - null func
  - Invalid flag
*/
TEST_CASE("Unit_hipOccupancyMaxPotBlkSizeVariableSMemWithFlags_NegTst") {
  hipError_t ret;
  int minGridSize = 0, blockSize = 0;

  SECTION("null min_grid_size") {
    ret = hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(nullptr, &blockSize, f1,
                                                                 blockSizeToDynamicSMemSize, 0, 0);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("null block_size") {
    ret = hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(&minGridSize, nullptr, f1,
                                                                 blockSizeToDynamicSMemSize, 0, 0);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("null func") {
    ret = hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags<size_t (*)(int), void (*)(float*)>(
        &minGridSize, &blockSize, nullptr, blockSizeToDynamicSMemSize, 0, 0);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("invalid flag") {
    ret = hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
        &minGridSize, &blockSize, f1, blockSizeToDynamicSMemSize, 0, 0xffff);
    REQUIRE(ret == hipErrorInvalidValue);
  }
}

/**
  Local function to launch kernel with gridsize and blocksize derived from
  hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags.
*/
static void checkFunc(void (*kerFn)(int*, int*), int num, int sharedMemBytes, int blockSize) {
  int SIZE = num * sizeof(int);
  int *inpArr_h, *outArr_h;
  int *inpArr_d, *outArr_d;
  // allocate host matrix
  inpArr_h = reinterpret_cast<int*>(malloc(SIZE));
  REQUIRE(inpArr_h != nullptr);
  outArr_h = reinterpret_cast<int*>(malloc(SIZE));
  REQUIRE(outArr_h != nullptr);
  // initialize the input data
  for (int i = 0; i < num; i++) {
    inpArr_h[i] = i;
  }
  // allocate the memory on the device side
  HIP_CHECK(hipMalloc(&inpArr_d, SIZE));
  HIP_CHECK(hipMalloc(&outArr_d, SIZE));
  // Memory transfer from host to device
  HIP_CHECK(hipMemcpy(inpArr_d, inpArr_h, SIZE, hipMemcpyHostToDevice));
  // Lauching kernel from host
  dim3 gridsize = dim3(num / blockSize);
  dim3 blocksize = dim3(blockSize);
  hipLaunchKernelGGL(kerFn, gridsize, blocksize, sharedMemBytes, 0, outArr_d, inpArr_d);
  // Memory transfer from device to host
  HIP_CHECK(hipMemcpy(outArr_h, outArr_d, SIZE, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  // verify the results
  for (int i = 0; i < num; i++) {
    REQUIRE(outArr_h[i] == inpArr_h[i]);
  }
  // free the resources on device side
  HIP_CHECK(hipFree(inpArr_d));
  HIP_CHECK(hipFree(outArr_d));
  // free the resources on host side
  free(inpArr_h);
  free(outArr_h);
}

/**
  Local function to return appropriate array size which consumes
  memory less than the maximum allowed shared memory per block.
*/
static int getAppropriateDynShMemSize(int sharedMemPerBlock) {
  int size = 1;
  while (static_cast<int>(size * size * sizeof(int)) < sharedMemPerBlock) {
    size = size * 2;
  }
  return (size / 2);
}

// functor to return 0 dynamic shared memory
static size_t getZeroDynShMem(int blocksize) {
  UNUSED(blocksize);
  return 0;
}

// functor to return maximum possible dynamic shared memory.
static size_t getMaxDynShMem(int blocksize) {
  UNUSED(blocksize);
  return static_cast<size_t>(gArrSize * gArrSize * sizeof(int));
}

/**
  Functional tests for hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags.
  Scenario1:
  Calculate the gridsize and blocksize that give theoretical maximum potential
  occupancy for a kernel function that does not use dynamic shared memory.
  Using the derived gridsize and blocksize launch the kernel and validate its
  output.
  Scenario2:
  Calculate the gridsize and blocksize that give theoretical maximum potential
  occupancy for a kernel function that uses dynamic shared memory. Ensure that
  allocated dynamic shared memory is less than the maximum allowed by system.
  Using the derived gridsize and blocksize launch the kernel and validate its
  output.
*/
TEST_CASE("Unit_hipOccupancyMaxPotBlkSizeVariableSMemWithFlags_Functional") {
  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  SECTION("Non Dynamic Shared Kernel") {
    int arrSize;
    int minGridSize = 0, blockSize = 0;
    hipError_t ret;
    // Get potential blocksize
    ret = hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(&minGridSize, &blockSize, copyKer,
                                                                 getZeroDynShMem, 0, 0);
    REQUIRE(ret == hipSuccess);
    REQUIRE(minGridSize > 0);
    REQUIRE(blockSize > 0);
    REQUIRE(blockSize <= devProp.maxThreadsPerBlock);
    arrSize = minGridSize * blockSize;
    checkFunc(copyKer, arrSize, 0, blockSize);
  }
  SECTION("Dynamic Shared Kernel") {
    int arrSize = getAppropriateDynShMemSize(devProp.sharedMemPerBlock);
    gArrSize = arrSize;
    int minGridSize = 0, blockSize = 0;
    hipError_t ret;
    // Get potential blocksize
    ret = hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(&minGridSize, &blockSize,
                                                                 copyKerDyn, getMaxDynShMem, 0, 0);
    REQUIRE(ret == hipSuccess);
    REQUIRE(minGridSize > 0);
    REQUIRE(blockSize > 0);
    REQUIRE(blockSize <= devProp.maxThreadsPerBlock);
    int totalThreads;
    totalThreads = minGridSize * blockSize;
    // allow launching kernel with occupancy derived blocksize and gridsize
    // only if allocated dynamic memory is less than system limit.
    if ((totalThreads * sizeof(int)) < devProp.sharedMemPerBlock) {
      checkFunc(copyKerDyn, totalThreads, (totalThreads * sizeof(int)), blockSize);
    } else {
      totalThreads = arrSize * arrSize;
      // allow launching kernel only if blockSize is a multiple of
      // totalThreads
      if (((totalThreads % blockSize) == 0) && ((totalThreads / blockSize) > 0)) {
        checkFunc(copyKerDyn, totalThreads, (totalThreads * sizeof(int)), blockSize);
      }
    }
  }
}
