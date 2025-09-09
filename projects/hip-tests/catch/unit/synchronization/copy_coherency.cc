/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>

unsigned threadsPerBlock = 256;
unsigned blocksPerCU = 6;

class MemcpyFunction {
 public:
  MemcpyFunction(const char* fileName, const char* functionName) { load(fileName, functionName); }
  void load(const char* fileName, const char* functionName);
  void launch(int* dst, const int* src, size_t numElements, hipStream_t s);

 private:
  hipFunction_t _function;
  hipModule_t _module;
};


void MemcpyFunction::load(const char* fileName, const char* functionName) {
  HIP_CHECK(hipModuleLoad(&_module, fileName));
  HIP_CHECK(hipModuleGetFunction(&_function, _module, functionName));
}

void MemcpyFunction::launch(int* dst, const int* src, size_t numElements,
                            hipStream_t s) {  // NOLINT
  struct {
    int* _dst;
    const int* _src;
    size_t _numElements;
  } args;

  args._dst = dst;
  args._src = src;
  args._numElements = numElements;

  size_t size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);
  HIP_CHECK(hipModuleLaunchKernel(_function, blocks, 1, 1, threadsPerBlock, 1, 1, 0, s, NULL,
                                  reinterpret_cast<void**>(&config)));
}

bool g_warnOnFail = true;
int g_elementSizes[] = {128 * 1000, 256 * 1000, 16 * 1000 * 1000};

// Set value of array to specified 32-bit integer:
__global__ void memsetIntKernel(int* ptr, const int val, size_t numElements) {
  int gid = (blockIdx.x * blockDim.x + threadIdx.x);
  int stride = blockDim.x * gridDim.x;
  for (size_t i = gid; i < numElements; i += stride) {
    ptr[i] = val;
  }
}

__global__ void memcpyIntKernel(int* dst, const int* src, size_t numElements) {
  int gid = (blockIdx.x * blockDim.x + threadIdx.x);
  int stride = blockDim.x * gridDim.x;
  for (size_t i = gid; i < numElements; i += stride) {
    dst[i] = src[i];
  }
}

// Check arrays in reverse order, to more easily detect cases where
// the copy is "partially" done.
void checkReverse(const int* ptr, int numElements, int expected) {
  int mismatchCnt = 0;
  for (int i = numElements - 1; i >= 0; i--) {
    if (!g_warnOnFail) {
      REQUIRE(ptr[i] == expected);
    }
    if (++mismatchCnt >= 10) {
      break;
    }
  }
}

#define ENUM_CASE_STR(x)                                                                           \
  case x:                                                                                          \
    return #x

enum CmdType { COPY, KERNEL, MODULE_KERNEL, MAX_CmdType };

const char* CmdTypeStr(CmdType c) {
  switch (c) {
    ENUM_CASE_STR(COPY);
    ENUM_CASE_STR(KERNEL);
    ENUM_CASE_STR(MODULE_KERNEL);
    default:
      return "UNKNOWN";
  }
}

enum SyncType {
  NONE,
  EVENT_QUERY,
  EVENT_SYNC,
  STREAM_WAIT_EVENT,
  STREAM_QUERY,
  STREAM_SYNC,
  DEVICE_SYNC,
  MAX_SyncType
};

const char* SyncTypeStr(SyncType s) {
  switch (s) {
    ENUM_CASE_STR(NONE);
    ENUM_CASE_STR(EVENT_QUERY);
    ENUM_CASE_STR(EVENT_SYNC);
    ENUM_CASE_STR(STREAM_WAIT_EVENT);
    ENUM_CASE_STR(STREAM_QUERY);
    ENUM_CASE_STR(STREAM_SYNC);
    ENUM_CASE_STR(DEVICE_SYNC);
    default:
      return "UNKNOWN";
  }
}

void runCmd(CmdType cmd, int* dst, const int* src, hipStream_t s, size_t numElements) {
  switch (cmd) {
    case COPY:
      HIP_CHECK(hipMemcpyAsync(dst, src, numElements * sizeof(int), hipMemcpyDeviceToDevice, s));
      break;
    case KERNEL: {
      unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);
      hipLaunchKernelGGL(memcpyIntKernel, dim3(blocks), dim3(threadsPerBlock), 0, s, dst, src,
                         numElements);
    } break;
    case MODULE_KERNEL: {
      MemcpyFunction g_moduleMemcpy("memcpyInt.hsaco", "memcpyIntKernel");
      g_moduleMemcpy.launch(dst, src, numElements, s);
    } break;
    default:
      printf("Info:unknown cmd=%d type", cmd);
  }
}

void resetInputs(int* Ad, int* Bd, int* Ch, size_t numElements, int expected) {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, numElements);
  hipLaunchKernelGGL(memsetIntKernel, dim3(blocks), dim3(threadsPerBlock), 0, hipStream_t(0), Ad,
                     expected, numElements);
  // poison with bad value to ensure is overwritten correctly
  hipLaunchKernelGGL(memsetIntKernel, dim3(blocks), dim3(threadsPerBlock), 0, hipStream_t(0), Bd,
                     0xDEADBEEF, numElements);
  hipLaunchKernelGGL(memsetIntKernel, dim3(blocks), dim3(threadsPerBlock), 0, hipStream_t(0), Bd,
                     0xF000BA55, numElements);
  memset(Ch, 13, numElements * sizeof(int));
  HIP_CHECK(hipDeviceSynchronize());
}

// Intended to test proper synchronization and cache flushing
// between CMDA and CMDB. CMD are of type CmdType. All command copy memory,
// using either hipMemcpyAsync or kernel implementations.
// Some form of synchronization is applied. Then cmdB copies from Bd to Cd.
// CmdA copies from Ad to Bd, Cd is then copied to host Ch using a memory copy.
// Correct result at the end is that Ch contains the
// contents originally in Ad (integer 0x42)

void runTestImpl(CmdType cmdAType, SyncType syncType, CmdType cmdBType, hipStream_t stream1,
                 hipStream_t stream2, int numElements, int* Ad, int* Bd, int* Cd, int* Ch,
                 int expected) {
  hipEvent_t e;
  HIP_CHECK(hipEventCreateWithFlags(&e, 0));

  resetInputs(Ad, Bd, Ch, numElements, expected);

  const size_t sizeElements = numElements * sizeof(int);
  fprintf(stderr, "test: runTest with %zu bytes (%6.2f MB) cmdA=%s; sync=%s; cmdB=%s\n",  // NOLINT
          sizeElements, static_cast<double>(sizeElements / 1024.0), CmdTypeStr(cmdAType),
          SyncTypeStr(syncType), CmdTypeStr(cmdBType));

  /*if (SKIP_MODULE_KERNEL && ((cmdAType == MODULE_KERNEL) || (cmdBType == MODULE_KERNEL))) { //
  NOLINT fprintf(stderr, "warn: skipping since test infra does not yet support modules\n"); //
  NOLINT return;
  }*/

  // Step A:
  runCmd(cmdAType, Bd, Ad, stream1, numElements);

  // Sync in-between?
  switch (syncType) {
    case NONE:
      break;
    case EVENT_QUERY: {
      hipError_t st = hipErrorNotReady;
      HIP_CHECK(hipEventRecord(e, stream1));
      do {
        st = hipEventQuery(e);
      } while (st == hipErrorNotReady);
      HIP_CHECK(st);
    } break;
    case EVENT_SYNC:
      HIP_CHECK(hipEventRecord(e, stream1));
      HIP_CHECK(hipEventSynchronize(e));
      break;
    case STREAM_WAIT_EVENT:
      HIP_CHECK(hipEventRecord(e, stream1));
      HIP_CHECK(hipStreamWaitEvent(stream2, e, 0));
      break;
    case STREAM_QUERY: {
      hipError_t st = hipErrorNotReady;
      do {
        st = hipStreamQuery(stream1);
      } while (st == hipErrorNotReady);
      HIP_CHECK(st);
    } break;
    case STREAM_SYNC:
      HIP_CHECK(hipStreamSynchronize(stream1));
      break;
    case DEVICE_SYNC:
      HIP_CHECK(hipDeviceSynchronize());
      break;
    default:
      fprintf(stderr, "warning: unknown sync type=%s", SyncTypeStr(syncType));
      return;
  }
  runCmd(cmdBType, Cd, Bd, stream2, numElements);

  // Copy back to host, use async copy to avoid any extra synchronization
  //  that might mask issues.
  HIP_CHECK(hipMemcpyAsync(Ch, Cd, sizeElements, hipMemcpyDeviceToHost, stream2));
  HIP_CHECK(hipStreamSynchronize(stream2));

  checkReverse(Ch, numElements, expected);

  HIP_CHECK(hipEventDestroy(e));
}

void testWrapper(size_t numElements) {
  const size_t sizeElements = numElements * sizeof(int);
  const int expected = 0x42;
  int *Ad, *Bd, *Cd, *Ch;

  HIP_CHECK(hipMalloc(&Ad, sizeElements));
  HIP_CHECK(hipMalloc(&Bd, sizeElements));
  HIP_CHECK(hipMalloc(&Cd, sizeElements));
  HIP_CHECK(hipHostMalloc(&Ch, sizeElements));

  hipStream_t stream1, stream2;

  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipDeviceSynchronize());

  runTestImpl(COPY, EVENT_SYNC, KERNEL, stream1, stream2, numElements, Ad, Bd, Cd, Ch, expected);

  for (int cmdA = 0; cmdA < MAX_CmdType; cmdA++) {
    for (int cmdB = 0; cmdB < MAX_CmdType; cmdB++) {
      for (int syncMode = 0; syncMode < MAX_SyncType; syncMode++) {
        switch (syncMode) {
          // case NONE::
          case EVENT_QUERY:
          case EVENT_SYNC:
          case STREAM_WAIT_EVENT:
          // case STREAM_QUERY:
          case STREAM_SYNC:
          case DEVICE_SYNC:
            runTestImpl(CmdType(cmdA), SyncType(syncMode), CmdType(cmdB), stream1, stream2,
                        numElements, Ad, Bd, Cd, Ch, expected);
            break;
          default:
            break;
        }
      }
    }
  }

#if 0
  runTestImpl(COPY, STREAM_SYNC, MODULE_KERNEL, stream1, stream2,
              numElements, Ad, Bd, Cd, Ch, expected);
  runTestImpl(COPY, STREAM_SYNC, KERNEL, stream1, stream2, numElements,
              Ad, Bd, Cd, Ch, expected);
  runTestImpl(COPY, STREAM_WAIT_EVENT, MODULE_KERNEL, stream1, stream2,
               numElements, Ad, Bd, Cd, Ch, expected);
  runTestImpl(COPY, STREAM_WAIT_EVENT, KERNEL, stream1, stream2, numElements,
              Ad, Bd, Cd, Ch, expected);
#endif

  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Cd));
  HIP_CHECK(hipHostFree(Ch));

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
}

/**
 * Test Description
 * ------------------------
 *    - Test cache management (fences) and synchronization between
 * kernel and copy commands. Exhaustively tests 3 command types
 * (copy, kernel, module kernel), many sync types (see SyncType), followed by
 *  another command, across a sweep of data sizes designed to stress
 * various levels of the memory hierarchy.

 * Test source
 * ------------------------
 *    - catch/unit/synchronization/copy_coherency.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */

TEST_CASE("Unit_Copy_Coherency") {
  for (int index = 0; index < sizeof(g_elementSizes) / sizeof(int); index++) {
    size_t numElements = g_elementSizes[index];
    testWrapper(numElements);
  }
}
