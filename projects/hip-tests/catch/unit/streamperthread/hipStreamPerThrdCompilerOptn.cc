/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* This file will be compiled using the compiler option given below.
   The functions will be calculating the time taken to execute under
   the influence of the compiler flag:
   -fgpu-default-stream=per-thread
*/

#include <ctime>
#include <hip_test_common.hh>
#include "hip/hip_cooperative_groups.h"
#include <utils.hh>
namespace cg = cooperative_groups;

namespace DefltStrmPT {
int64_t N = 1024 * 1024 * 100;
int64_t Sz = N * sizeof(int64_t);
int64_t *DevA, *HstA, *HstRes;
int64_t OneMB = 1024 * 1024;
int64_t OneMBSz = OneMB * sizeof(int64_t);
hipStream_t Strm;
int clockrate, CONST = 123;
size_t numH = 1024, numW = 1024;
size_t pitch_A, width = numW * sizeof(int64_t);
size_t sizeElements = width * numH;
size_t elements = numW * numH;
}  // namespace DefltStrmPT

__device__ int64_t globalInDStrmPT[1024 * 1024];
__managed__ int SigComplte = 0;

// Kernel codes
__global__ void DefltStrmPT_Square(int64_t* C_d, int64_t N) {
  int64_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
  int64_t stride = hipBlockDim_x * hipGridDim_x;

  for (int64_t i = offset; i < N; i += stride) {
    C_d[i] = C_d[i] * C_d[i];
  }
}

__global__ void Wait_Kernel3(int clockrate, uint64_t WaitSecs, int PassSignal = 0) {
  uint64_t num_cycles = WaitSecs * clockrate * 1000;
  uint64_t start = clock64(), cycles = 0;
  while (cycles < num_cycles) {
    cycles = clock64() - start;
  }
  if (PassSignal) {
    SigComplte = 1;
  }
}

static __global__ void notifiedKernel(volatile unsigned int* notified, int PassSignal = 0) {
  while (*notified == 0) {
  }  // wait until notified to exit.
  if (PassSignal) {
    SigComplte = 1;
  }
}
__global__ void DefltStrmPT_Test_gws(uint* buf, uint bufSize, int64_t* tmpBuf, int64_t* result) {
  extern __shared__ int64_t tmp[];
  uint offset = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = gridDim.x * blockDim.x;
  cg::grid_group gg = cg::this_grid();

  int64_t sum = 0;
  for (uint i = offset; i < bufSize; i += stride) {
    sum += buf[i];
  }
  tmp[threadIdx.x] = sum;

  __syncthreads();

  if (threadIdx.x == 0) {
    sum = 0;
    for (uint i = 0; i < blockDim.x; i++) {
      sum += tmp[i];
    }
    tmpBuf[blockIdx.x] = sum;
  }

  gg.sync();

  if (offset == 0) {
    for (uint i = 1; i < gridDim.x; ++i) {
      sum += tmpBuf[i];
    }
    *result = sum;
  }
}

float DefaultPT2_Memcpy_MemSet(int CpyAsync, int MemSetAsync) {
  bool IfTstPassed = true;
  DefltStrmPT::HstA = reinterpret_cast<int64_t*>(malloc(DefltStrmPT::Sz));
  DefltStrmPT::HstRes = reinterpret_cast<int64_t*>(malloc(DefltStrmPT::Sz));
  HIP_CHECK(hipDeviceGetAttribute(&(DefltStrmPT::clockrate), hipDeviceAttributeMemoryClockRate, 0));
  HIP_CHECK(hipMalloc(&(DefltStrmPT::DevA), DefltStrmPT::Sz));
  for (int64_t i = 0; i < DefltStrmPT::N; ++i) {
    DefltStrmPT::HstA[i] = DefltStrmPT::CONST;
  }
  HIP_CHECK(hipStreamCreate(&(DefltStrmPT::Strm)));
  if (CpyAsync) {
    HIP_CHECK(hipMemcpyAsync(DefltStrmPT::DevA, DefltStrmPT::HstA, DefltStrmPT::Sz,
                             hipMemcpyHostToDevice, DefltStrmPT::Strm));
    HIP_CHECK(hipStreamSynchronize(DefltStrmPT::Strm));
  } else {
    HIP_CHECK(
        hipMemcpy(DefltStrmPT::DevA, DefltStrmPT::HstA, DefltStrmPT::Sz, hipMemcpyHostToDevice));
  }
  DefltStrmPT_Square<<<(DefltStrmPT::N / 256 + 1), 256, 0, DefltStrmPT::Strm>>>(DefltStrmPT::DevA,
                                                                                DefltStrmPT::N);
  HIP_CHECK(hipStreamSynchronize(DefltStrmPT::Strm));
  HIP_CHECK(
      hipMemcpy(DefltStrmPT::HstRes, DefltStrmPT::DevA, DefltStrmPT::Sz, hipMemcpyDeviceToHost));
  // Verifying the result
  for (int64_t i = 0; i < DefltStrmPT::N; ++i) {
    if (DefltStrmPT::HstRes[i] != (DefltStrmPT::HstA[i] * DefltStrmPT::HstA[i])) {
      IfTstPassed = false;
    }
  }
  if (MemSetAsync) {
    HIP_CHECK(hipMemsetAsync(DefltStrmPT::DevA, 0, DefltStrmPT::Sz, DefltStrmPT::Strm));
    HIP_CHECK(hipStreamSynchronize(DefltStrmPT::Strm));
  } else {
    HIP_CHECK(hipMemset(DefltStrmPT::DevA, 0, DefltStrmPT::Sz));
  }
  // Copying the device memory to host to check if Memset is successful
  HIP_CHECK(
      hipMemcpy(DefltStrmPT::HstA, DefltStrmPT::DevA, DefltStrmPT::Sz, hipMemcpyDeviceToHost));
  // verifying if memset was successful
  for (int64_t i = 0; i < DefltStrmPT::N; ++i) {
    if (DefltStrmPT::HstA[i] != 0) {
      IfTstPassed = false;
    }
  }
  HIP_CHECK(hipStreamDestroy(DefltStrmPT::Strm));
  HIP_CHECK(hipFree(DefltStrmPT::DevA));
  free(DefltStrmPT::HstA);
  free(DefltStrmPT::HstRes);
  return IfTstPassed;
}

void DefaultPT2_Memset2D(int Async) {
  constexpr int memsetval = 0x24;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t pitch_A;
  size_t width = numW * sizeof(char);
  size_t sizeElements = width * numH;
  size_t elements = numW * numH;
  char *A_d, *A_h;

  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, numH));
  A_h = reinterpret_cast<char*>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);

  for (size_t i = 0; i < elements; i++) {
    A_h[i] = 1;
  }

  if (Async) {
    hipStream_t Strm;
    HIP_CHECK(hipStreamCreate(&Strm));
    HIP_CHECK(hipMemset2DAsync(A_d, pitch_A, memsetval, numW, numH, Strm));
    HIP_CHECK(hipStreamSynchronize(Strm));
    HIP_CHECK(hipStreamDestroy(Strm));
  } else {
    HIP_CHECK(hipMemset2D(A_d, pitch_A, memsetval, numW, numH));
  }
  HIP_CHECK(hipMemcpy2D(A_h, width, A_d, pitch_A, numW, numH, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      INFO("Memset2D mismatch at index:" << i << " computed:" << A_h[i]
                                         << " memsetval:" << memsetval);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipFree(A_d));
  free(A_h);
}


void PerThrdDefltStrm_Memset3D(int Async) {
  constexpr int memsetval = 0x22;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  constexpr size_t depth = 10;
  size_t width = numW * sizeof(char);
  size_t sizeElements = width * numH * depth;
  size_t elements = numW * numH * depth;
  char* A_h;

  hipExtent extent = make_hipExtent(width, numH, depth);
  hipPitchedPtr devPitchedPtr;

  HIP_CHECK(hipMalloc3D(&devPitchedPtr, extent));
  A_h = reinterpret_cast<char*>(malloc(sizeElements));
  if (A_h == nullptr) REQUIRE(false);

  for (size_t i = 0; i < elements; i++) {
    A_h[i] = 1;
  }

  if (Async) {
    hipStream_t Strm;
    HIP_CHECK(hipStreamCreate(&Strm));
    HIP_CHECK(hipMemset3DAsync(devPitchedPtr, memsetval, extent, Strm));
    HIP_CHECK(hipStreamSynchronize(Strm));
    HIP_CHECK(hipStreamDestroy(Strm));
  } else {
    HIP_CHECK(hipMemset3D(devPitchedPtr, memsetval, extent));
  }
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.srcPtr = devPitchedPtr;
  myparms.extent = extent;
#if HT_NVIDIA
  myparms.kind = hipMemcpyKindToCudaMemcpyKind(hipMemcpyDeviceToHost);
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif
  HIP_CHECK(hipMemcpy3D(&myparms));

  for (size_t i = 0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      INFO("Memset3D mismatch at index:" << i << " computed:" << A_h[i]
                                         << " memsetval:" << memsetval);
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipFree(devPitchedPtr.ptr));
  free(A_h);
}


void DefaultPT2_StrmQuery() {
  unsigned int* notified = nullptr;
  HIP_CHECK(hipHostMalloc(&notified, sizeof(unsigned int)));
  *notified = 0;
  HIP_CHECK(hipStreamCreate(&(DefltStrmPT::Strm)));
  // StreamQuery with user created stream
  notifiedKernel<<<1, 1, 0, DefltStrmPT::Strm>>>(notified);
  REQUIRE((hipErrorNotReady == hipStreamQuery(DefltStrmPT::Strm)));

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  *notified = 1;
  HIP_CHECK(hipStreamDestroy(DefltStrmPT::Strm));
  HIP_CHECK(hipHostFree(notified));
}


void DefaultPT2_StreamSync() {
  HIP_CHECK(hipDeviceGetAttribute(&(DefltStrmPT::clockrate), hipDeviceAttributeMemoryClockRate, 0));
  HIP_CHECK(hipStreamCreate(&(DefltStrmPT::Strm)));
  // Calling hipStreamSync on user created stream object
  Wait_Kernel3<<<1, 1, 0, DefltStrmPT::Strm>>>(DefltStrmPT::clockrate, 1);
  HIP_CHECK(hipStreamSynchronize(DefltStrmPT::Strm));
  // Calling hipStreamSync on null stream
  Wait_Kernel3<<<1, 1>>>(DefltStrmPT::clockrate, 1);
  HIP_CHECK(hipStreamSynchronize(0));
  HIP_CHECK(hipStreamDestroy(DefltStrmPT::Strm));
}


void DefaultPT2_StrmWaitEvent() {
  int device;
  HIP_CHECK(hipGetDevice(&device));
  if (!DeviceAttributesSupport(device, hipDeviceAttributeManagedMemory)) {
    HipTest::HIP_SKIP_TEST("Managed memory is not supported");
    return;
  }

  hipEvent_t evt;
  hipStream_t Strm1;
  unsigned int* notified = nullptr;
  HIP_CHECK(hipHostMalloc(&notified, sizeof(unsigned int)));
  *notified = 0;
  HIP_CHECK(hipStreamCreate(&(DefltStrmPT::Strm)));
  HIP_CHECK(hipStreamCreate(&Strm1));
  HIP_CHECK(hipEventCreate(&evt));
  notifiedKernel<<<1, 1, 0, DefltStrmPT::Strm>>>(notified, 1);
  HIP_CHECK(hipEventRecord(evt, DefltStrmPT::Strm));
  HIP_CHECK(hipStreamWaitEvent(Strm1, evt, 0));
  notifiedKernel<<<1, 1, 0, Strm1>>>(notified);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  // By the time control reaches the below point SigComplte is expected
  // to be still zero
  if (SigComplte) {
    REQUIRE(false);
  }
  *notified = 1;
  HIP_CHECK(hipStreamSynchronize(Strm1));
  HIP_CHECK(hipStreamDestroy(DefltStrmPT::Strm));
  if (SigComplte == 0) {
    REQUIRE(false);
  }
  HIP_CHECK(hipStreamDestroy(Strm1));
  HIP_CHECK(hipEventDestroy(evt));
  HIP_CHECK(hipHostFree(notified));
}

void DefaultPT2_EvtQuery() {
  hipEvent_t evt, evt1;
  hipError_t err;
  unsigned int* notified = nullptr;
  HIP_CHECK(hipHostMalloc(&notified, sizeof(unsigned int)));
  *notified = 0;
  HIP_CHECK(hipStreamCreate(&(DefltStrmPT::Strm)));
  HIP_CHECK(hipEventCreate(&evt));
  HIP_CHECK(hipEventCreate(&evt1));
  notifiedKernel<<<1, 1, 0, DefltStrmPT::Strm>>>(notified);
  HIP_CHECK(hipEventRecord(evt, DefltStrmPT::Strm));
  err = hipEventQuery(evt);
  if (err != hipErrorNotReady) {
    REQUIRE(false);
  }
  // Testing for Null or default stream
  HIP_CHECK(hipEventRecord(evt1, 0));
  int Got_hipSuccess = 0;  // 0 for no, 1 for yes
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  *notified = 1;  // notify to exit
  std::chrono::time_point start = std::chrono::steady_clock::now();
  while (true) {
    err = hipEventQuery(evt1);
    if (err == hipSuccess) {
      Got_hipSuccess = 1;
      break;
    }
    if (std::chrono::steady_clock::now() - start > std::chrono::seconds(60)) {
      break;
    }
  }
  if (!Got_hipSuccess) {
    REQUIRE(false);
  }
  HIP_CHECK(hipStreamDestroy(DefltStrmPT::Strm));
  HIP_CHECK(hipEventDestroy(evt));
  HIP_CHECK(hipEventDestroy(evt1));
  HIP_CHECK(hipHostFree(notified));
}


void Default_LaunchKernel(int NullStrm) {
  DefltStrmPT::N = DefltStrmPT::N / 4;
  DefltStrmPT::Sz = DefltStrmPT::N * sizeof(int64_t);
  DefltStrmPT::HstA = reinterpret_cast<int64_t*>(malloc(DefltStrmPT::Sz));
  HIP_CHECK(hipMalloc(&(DefltStrmPT::DevA), DefltStrmPT::Sz));
  for (int64_t i = 0; i < DefltStrmPT::N; ++i) {
    DefltStrmPT::HstA[i] = DefltStrmPT::CONST;
  }
  HIP_CHECK(
      hipMemcpy(DefltStrmPT::DevA, DefltStrmPT::HstA, DefltStrmPT::Sz, hipMemcpyHostToDevice));
  HIP_CHECK(hipStreamCreate(&(DefltStrmPT::Strm)));
  unsigned ThrdsPerBlk = 32;
  unsigned Blocks = ((DefltStrmPT::N + ThrdsPerBlk - 1) / ThrdsPerBlk);
  void* Args[] = {&(DefltStrmPT::DevA), &(DefltStrmPT::N)};
  // launch Kernel
  if (NullStrm) {
    HIP_CHECK(hipLaunchKernel((const void*)DefltStrmPT_Square, dim3(Blocks, 1, 1),
                              dim3(ThrdsPerBlk, 1, 1), Args, 0, 0));
    HIP_CHECK(hipStreamSynchronize(0));
  } else {
    HIP_CHECK(hipLaunchKernel((const void*)DefltStrmPT_Square, dim3(Blocks, 1, 1),
                              dim3(ThrdsPerBlk, 1, 1), Args, 0, DefltStrmPT::Strm));
    HIP_CHECK(hipStreamSynchronize(DefltStrmPT::Strm));
  }
  HIP_CHECK(
      hipMemcpy(DefltStrmPT::HstA, DefltStrmPT::DevA, DefltStrmPT::Sz, hipMemcpyDeviceToHost));
  for (int64_t i = 0; i < DefltStrmPT::N; ++i) {
    if (DefltStrmPT::HstA[i] != (DefltStrmPT::CONST * DefltStrmPT::CONST)) {
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipStreamDestroy(DefltStrmPT::Strm));
  HIP_CHECK(hipFree(DefltStrmPT::DevA));
  free(DefltStrmPT::HstA);
}


void DefaultPT2_LaunchCooperativeKernel(int NullStrm) {
  bool IfTestPassed = true;
  uint32_t* dA;
  int64_t *dB, *dC;
  uint32_t BufferSizeInDwords = 448 * 1024 * 1024;
  uint32_t* init = new uint32_t[BufferSizeInDwords];
  for (uint32_t i = 0; i < BufferSizeInDwords; ++i) {
    init[i] = i;
  }
  size_t SIZE = BufferSizeInDwords * sizeof(uint);
  HIP_CHECK(hipStreamCreate(&(DefltStrmPT::Strm)));
  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&dA), SIZE));
  HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&dC), sizeof(int64_t)));
  HIPCHECK(hipMemcpy(dA, init, SIZE, hipMemcpyHostToDevice));

  dim3 dimBlock = dim3(1);
  dim3 dimGrid = dim3(1);
  int numBlocks = 0;
  dimBlock.x = 32;
  // Calculate the device occupancy to know how many blocks can be run
  //  concurrently
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, DefltStrmPT_Test_gws,
                                                         dimBlock.x * dimBlock.y * dimBlock.z,
                                                         dimBlock.x * sizeof(int64_t)));
  dimGrid.x = deviceProp.multiProcessorCount * std::min(numBlocks, 32);
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&dB), dimGrid.x * sizeof(int64_t)));

  void* params[4];
  params[0] = reinterpret_cast<void*>(&dA);
  params[1] = reinterpret_cast<void*>(&BufferSizeInDwords);
  params[2] = reinterpret_cast<void*>(&dB);
  params[3] = reinterpret_cast<void*>(&dC);
  if (NullStrm) {
    HIPCHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(DefltStrmPT_Test_gws), dimGrid,
                                        dimBlock, params, dimBlock.x * sizeof(int64_t), 0));
    HIP_CHECK(hipStreamSynchronize(0));
  } else {
    HIPCHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(DefltStrmPT_Test_gws), dimGrid,
                                        dimBlock, params, dimBlock.x * sizeof(int64_t),
                                        DefltStrmPT::Strm));
    HIP_CHECK(hipStreamSynchronize(DefltStrmPT::Strm));
  }
  HIPCHECK(hipMemcpy(init, dC, sizeof(int64_t), hipMemcpyDeviceToHost));

  if (*dC != (((int64_t)(BufferSizeInDwords) * (BufferSizeInDwords - 1)) / 2)) {
    std::cout << "Data validation failed for grid size = " << dimGrid.x
              << " and block size = " << dimBlock.x << "\n";
    std::cout << "Test failed! \n";
    IfTestPassed = false;
  }
  HIPCHECK(hipStreamDestroy(DefltStrmPT::Strm));
  HIPCHECK(hipHostFree(dC));
  HIPCHECK(hipFree(dB));
  HIPCHECK(hipFree(dA));
  delete[] init;
  REQUIRE(IfTestPassed);
}


void DefaultPT2_StrmGetFlag() {
  HIP_CHECK(hipStreamCreateWithFlags(&(DefltStrmPT::Strm), hipStreamDefault));
  unsigned int flag = 9999;
  HIP_CHECK(hipStreamGetFlags(DefltStrmPT::Strm, &flag));
  if (flag != 0) {
    INFO("Expected flag value: 0, Received flag value: %u\n" << flag);
    REQUIRE(false);
  }
  HIP_CHECK(hipStreamDestroy(DefltStrmPT::Strm));
  HIP_CHECK(hipStreamCreateWithFlags(&(DefltStrmPT::Strm), hipStreamNonBlocking));
  flag = 9999;
  HIP_CHECK(hipStreamGetFlags(DefltStrmPT::Strm, &flag));
  if (flag != 1) {
    INFO("Expected flag value: 1, Received flag value: %u\n" << flag);
    REQUIRE(false);
  }
  HIP_CHECK(hipStreamDestroy(DefltStrmPT::Strm));
}

void DefaultPT2_StrmGetPriority() {
  int low, high, ObsrvdPriority;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&low, &high));
  INFO("Lowest possible priority: %d\n" << low);
  INFO("Highest possible priority: %d\n" << high);
  INFO("Creating streams with flag hipStreamNonBlocking\n");
  // hipStrmFlg = 0 = hipStreamDefault
  // hipStrmFlg = 1 = hipStreamNonBlocking
  for (int hipStrmFlg = 0; hipStrmFlg < 2; ++hipStrmFlg) {
    for (int Priority = low; Priority <= high; ++Priority) {
      if (hipStrmFlg == 0) {
        HIP_CHECK(hipStreamCreateWithPriority(&(DefltStrmPT::Strm), hipStreamDefault, Priority));
      } else {
        HIP_CHECK(
            hipStreamCreateWithPriority(&(DefltStrmPT::Strm), hipStreamNonBlocking, Priority));
      }
      HIP_CHECK(hipStreamGetPriority(DefltStrmPT::Strm, &ObsrvdPriority));
      if (ObsrvdPriority != Priority) {
        INFO("Expected priority: %d" << Priority << " Observed Priority: %d\n" << ObsrvdPriority);
        INFO("Test Failed!\n\n");
        REQUIRE(false);
      }
      HIP_CHECK(hipStreamDestroy(DefltStrmPT::Strm));
    }
  }
  INFO("Checking priority on null stream!!\n");
  HIP_CHECK(hipStreamGetPriority(0, &ObsrvdPriority));
  if (ObsrvdPriority != 0) {
    INFO("Expected priority: 0, Observed Priority: %d\n" << ObsrvdPriority);
    INFO("Test Failed!\n\n");
    REQUIRE(false);
  }
}


void DefaultPT2_hipMemcpyFromSymbol() {
  int64_t* Hst = nullptr;
  HIP_CHECK(hipHostMalloc(&(Hst), DefltStrmPT::OneMBSz));
  HIP_CHECK(hipStreamCreate(&(DefltStrmPT::Strm)));
  for (int i = 0; i < DefltStrmPT::OneMB; ++i) {
    Hst[i] = DefltStrmPT::CONST;
  }
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(globalInDStrmPT), Hst, DefltStrmPT::OneMBSz, 0,
                              hipMemcpyHostToDevice));
  for (int i = 0; i < DefltStrmPT::OneMB; ++i) {
    Hst[i] = 0;
  }
  HIP_CHECK(hipMemcpyFromSymbol(Hst, HIP_SYMBOL(globalInDStrmPT), DefltStrmPT::OneMBSz, 0,
                                hipMemcpyDeviceToHost));
  for (int i = 0; i < DefltStrmPT::OneMB; ++i) {
    if (Hst[i] != DefltStrmPT::CONST) {
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipHostFree(Hst));
  HIP_CHECK(hipStreamDestroy(DefltStrmPT::Strm));
}

void DefaultPT2_hipMemcpy2D(int Async) {
  DefltStrmPT::numH = 1024;
  DefltStrmPT::numW = 1024;
  DefltStrmPT::width = DefltStrmPT::numW * sizeof(int64_t);
  HIP_CHECK(hipHostMalloc(&(DefltStrmPT::HstA),
                          (DefltStrmPT::numH * DefltStrmPT::numW * sizeof(int64_t))));
  HIP_CHECK(hipHostMalloc(&(DefltStrmPT::HstRes),
                          (DefltStrmPT::numH * DefltStrmPT::numW * sizeof(int64_t))));
  DefltStrmPT::width = DefltStrmPT::numW * sizeof(int64_t);
  for (size_t row = 0; row < DefltStrmPT::numH; ++row) {
    for (size_t column = 0; column < DefltStrmPT::numW; ++column) {
      DefltStrmPT::HstA[(row * DefltStrmPT::numW) + column] = DefltStrmPT::CONST;
      DefltStrmPT::HstRes[(row * DefltStrmPT::numW) + column] = 0;
    }
  }
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&(DefltStrmPT::DevA)), &(DefltStrmPT::pitch_A),
                           DefltStrmPT::width, DefltStrmPT::numH));
  if (Async) {
    HIP_CHECK(hipStreamCreate(&(DefltStrmPT::Strm)));
    HIP_CHECK(hipMemcpy2DAsync(DefltStrmPT::DevA, DefltStrmPT::pitch_A, DefltStrmPT::HstA,
                               DefltStrmPT::numW * sizeof(int64_t),
                               DefltStrmPT::numW * sizeof(int64_t), DefltStrmPT::numH,
                               hipMemcpyHostToDevice, DefltStrmPT::Strm));
    HIP_CHECK(hipMemcpy2DAsync(DefltStrmPT::HstRes, DefltStrmPT::numW * sizeof(int64_t),
                               DefltStrmPT::DevA, DefltStrmPT::pitch_A,
                               DefltStrmPT::numW * sizeof(int64_t), DefltStrmPT::numH,
                               hipMemcpyDeviceToHost, DefltStrmPT::Strm));
    HIP_CHECK(hipStreamSynchronize(DefltStrmPT::Strm));
    HIP_CHECK(hipStreamDestroy(DefltStrmPT::Strm));
  } else {
    HIP_CHECK(hipMemcpy2D(DefltStrmPT::DevA, DefltStrmPT::pitch_A, DefltStrmPT::HstA,
                          DefltStrmPT::numW * sizeof(int64_t), DefltStrmPT::numW * sizeof(int64_t),
                          DefltStrmPT::numH, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy2D(DefltStrmPT::HstRes, DefltStrmPT::numW * sizeof(int64_t),
                          DefltStrmPT::DevA, DefltStrmPT::pitch_A,
                          DefltStrmPT::numW * sizeof(int64_t), DefltStrmPT::numH,
                          hipMemcpyDeviceToHost));
  }
  for (size_t row = 0; row < DefltStrmPT::numH; ++row) {
    for (size_t column = 0; column < DefltStrmPT::numW; ++column) {
      if (DefltStrmPT::HstRes[(row * DefltStrmPT::numW) + column] != DefltStrmPT::CONST) {
        REQUIRE(false);
      }
    }
  }
  HIP_CHECK(hipFree(DefltStrmPT::DevA));
  HIP_CHECK(hipHostFree(DefltStrmPT::HstA));
}


void DefaultPT2_hipMemcpy2DToArray() {
  hipArray_t Dptr = nullptr;
  float *Hptr = nullptr, *HRes = nullptr;
  DefltStrmPT::numH = 1024;
  DefltStrmPT::numW = 1024;
  DefltStrmPT::width = DefltStrmPT::numW * sizeof(float);
  Hptr = new float[DefltStrmPT::width * DefltStrmPT::numH];
  HRes = new float[DefltStrmPT::width * DefltStrmPT::numH];
  for (size_t i = 0; i < DefltStrmPT::width * DefltStrmPT::numH; ++i) {
    Hptr[i] = DefltStrmPT::CONST;
  }
  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
  HIP_CHECK(hipMallocArray(&(Dptr), &desc, DefltStrmPT::numW, DefltStrmPT::numH, hipArrayDefault));
  HIP_CHECK(hipMemcpy2DToArray(Dptr, 0, 0, Hptr, DefltStrmPT::width, DefltStrmPT::width,
                               DefltStrmPT::numH, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy2DFromArray(HRes, DefltStrmPT::width, Dptr, 0, 0, DefltStrmPT::width,
                                 DefltStrmPT::numH, hipMemcpyDeviceToHost));
  // verifying the result
  for (size_t i = 0; i < DefltStrmPT::numW * DefltStrmPT::numH; ++i) {
    if (HRes[i] != DefltStrmPT::CONST) {
      REQUIRE(false);
    }
  }
  delete[] Hptr;
  delete[] HRes;
  HIP_CHECK(hipFreeArray(Dptr));
}


float DefaultPT2_hipMemcpy2DFromArray() {
  hipArray_t Dptr = nullptr;
  float *Hptr_A = nullptr, *Hptr_B = nullptr;
  DefltStrmPT::numH = 1024;
  DefltStrmPT::numW = 1024;
  DefltStrmPT::width = DefltStrmPT::numW * sizeof(float);
  HIP_CHECK(hipDeviceGetAttribute(&(DefltStrmPT::clockrate), hipDeviceAttributeMemoryClockRate, 0));
  HIP_CHECK(hipHostMalloc(&(Hptr_A), (DefltStrmPT::width * DefltStrmPT::numH * sizeof(float))));
  HIP_CHECK(hipHostMalloc(&(Hptr_B), (DefltStrmPT::width * DefltStrmPT::numH * sizeof(float))));
  for (size_t i = 0; i < (DefltStrmPT::width * DefltStrmPT::numH); ++i) {
    Hptr_A[i] = DefltStrmPT::CONST;
  }
  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
  HIP_CHECK(hipMallocArray(&(Dptr), &desc, DefltStrmPT::numW, DefltStrmPT::numH, hipArrayDefault));
  HIP_CHECK(hipStreamCreate(&(DefltStrmPT::Strm)));
  HIP_CHECK(hipMemcpy2DToArray(Dptr, 0, 0, Hptr_A, DefltStrmPT::width, DefltStrmPT::width,
                               DefltStrmPT::numH, hipMemcpyHostToDevice));
  Wait_Kernel3<<<1, 1, 0, DefltStrmPT::Strm>>>(DefltStrmPT::clockrate, 1);
  HIP_CHECK(hipMemcpy2DFromArray(Hptr_B, DefltStrmPT::width, Dptr, 0, 0, DefltStrmPT::width,
                                 DefltStrmPT::numH, hipMemcpyDeviceToHost));
  HIP_CHECK(hipStreamDestroy(DefltStrmPT::Strm));
  HIP_CHECK(hipFreeArray(Dptr));
  HIP_CHECK(hipHostFree(Hptr_A));
  HIP_CHECK(hipHostFree(Hptr_B));
  return true;
}

void DefaultPT2_hipMemcpy3D() {
  int width = 8, height = 8, depth = 8;
  unsigned int size = width * height * depth * sizeof(float);
  float* Hptr = reinterpret_cast<float*>(malloc(size));
  float* HRes = reinterpret_cast<float*>(malloc(size));
  memset(Hptr, 0, size);
  memset(HRes, 0, size);
  hipExtent extent = make_hipExtent(width, height, depth);

  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        Hptr[i * width * height + j * width + k] = i * width * height + j * width + k;
      }
    }
  }
  hipChannelFormatDesc channelDesc =
      hipCreateChannelDesc(sizeof(float) * 8, 0, 0, 0, hipChannelFormatKindFloat);
  hipArray_t arr;

  HIP_CHECK(
      hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width, height, depth), hipArrayDefault));
  hipMemcpy3DParms myparms{0,         {0, 0, 0},    {0, 0, 0, 0}, 0,
                           {0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0},    hipMemcpyDefault};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = make_hipPitchedPtr(Hptr, width * sizeof(float), width, height);
  myparms.dstArray = arr;
  myparms.extent = extent;

#ifdef __HIP_PLATFORM_NVIDIA__
  myparms.kind = cudaMemcpyHostToDevice;
#else
  myparms.kind = hipMemcpyHostToDevice;
#endif
  // Host to Device copy
  HIP_CHECK(hipMemcpy3D(&myparms));

  // Device to Host copy
  memset(&myparms, 0x0, sizeof(hipMemcpy3DParms));
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(HRes, width * sizeof(float), width, height);
  myparms.srcArray = arr;
  myparms.extent = extent;
#ifdef __HIP_PLATFORM_NVIDIA__
  myparms.kind = cudaMemcpyDeviceToHost;
#else
  myparms.kind = hipMemcpyDeviceToHost;
#endif
  HIP_CHECK(hipMemcpy3D(&myparms));

  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        if (HRes[i * width * height + j * width + k] != i * width * height + j * width + k) {
          REQUIRE(false);
        }
      }
    }
  }
  HIP_CHECK(hipFreeArray(arr));
  free(Hptr);
  free(HRes);
}


TEST_CASE("Unit_hipStrmPerThrdDefault") {
  CHECK_IMAGE_SUPPORT

  SECTION("Testing hipMemset/Memcpy() and their async version") {
    REQUIRE(DefaultPT2_Memcpy_MemSet(1, 0));
    REQUIRE(DefaultPT2_Memcpy_MemSet(1, 1));
    REQUIRE(DefaultPT2_Memcpy_MemSet(0, 1));
    REQUIRE(DefaultPT2_Memcpy_MemSet(0, 0));
  }
  SECTION("Testing hipMemset2D() and its async version") {
    DefaultPT2_Memset2D(0);
    DefaultPT2_Memset2D(1);
  }

  SECTION("Testing_hipMemset3D() and its async version") {
    PerThrdDefltStrm_Memset3D(0);
    PerThrdDefltStrm_Memset3D(1);
  }

  SECTION("Testing_hipStreamQuery()") { DefaultPT2_StrmQuery(); }

  SECTION("Testing_hipStreamSynchronize()") { DefaultPT2_StreamSync(); }

  SECTION("Testing_hipLaunchKernel()") {
    // launch with null stream
    Default_LaunchKernel(1);
    // launch with user created stream
    Default_LaunchKernel(0);
  }

  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));
  if (deviceProp.cooperativeLaunch) {
    SECTION("Testing_hipLaunchCooperativeKernel()") {
      // launching hipLaunchCooperativeKernel() with Null stream
      DefaultPT2_LaunchCooperativeKernel(1);
      // launching hipLaunchCooperativeKernel() with user created stream
      DefaultPT2_LaunchCooperativeKernel(0);
    }
  } else {
    INFO("Cooperative Launch feature is not supported, therefore skipping");
    INFO(" the test Testing_hipLaunchCooperativeKernel()");
  }

  SECTION("Testing_StrmWaitEvent()") { DefaultPT2_StrmWaitEvent(); }

  SECTION("Testing_hipStreamGetFlag()") { DefaultPT2_StrmGetFlag(); }

  SECTION("Testing_hipStreamGetPriority()") { DefaultPT2_StrmGetPriority(); }

  SECTION("Testing_hipMemcpyFrom/To/Symbol()") { DefaultPT2_hipMemcpyFromSymbol(); }

  SECTION("Testing_hipMemcpy2D() & its Async version") {
    DefaultPT2_hipMemcpy2D(0);
    DefaultPT2_hipMemcpy2D(1);
  }

  SECTION("Testing_hipMemcpy2DToArray()") { DefaultPT2_hipMemcpy2DToArray(); }

  SECTION("Testing_hipMemcpy3D()") { DefaultPT2_hipMemcpy3D(); }

  SECTION("Testing_hipEventQuery()") { DefaultPT2_EvtQuery(); }
}
