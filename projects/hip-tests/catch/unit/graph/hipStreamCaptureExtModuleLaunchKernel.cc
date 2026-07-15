/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include "hip/hip_ext.h"
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <sys/stat.h>
#if !defined(S_IFREG) && defined(_S_IFREG)
#define S_IFREG _S_IFREG
#endif

struct gridblockDim {
  unsigned int gridX;
  unsigned int gridY;
  unsigned int gridZ;
  unsigned int blockX;
  unsigned int blockY;
  unsigned int blockZ;
};
class GraphModuleLaunchKernel {
  int N = 64;
  int SIZE = N * N;
  int *A, *B, *C;
  hipDeviceptr_t *Ad, *Bd;
  hipStream_t stream1, stream2;
  hipModule_t module;
  hipFunction_t multKernel;
  struct {
    void* _Ad;
    void* _Bd;
    void* _Cd;
    int _n;
  } args1, args2;
  size_t size1, size2;

  static constexpr char matmulK[] = "matmulK";

 public:
  GraphModuleLaunchKernel() {
    allocateMemory();
    moduleLoad();
  }

  ~GraphModuleLaunchKernel() { deAllocateMemory(); }

  void allocateMemory();
  void deAllocateMemory();
  void moduleLoad();
  bool extModuleKernelExecutionMatmul();
  bool extModuleKernelExecutionMatmulwithStreamCapture(bool LaunchByDifferentStream = false);
  bool extModuleKernelExecutionMatmulwithStreamCaptureN(int n, unsigned int blockX,
                                                        unsigned int blockY);
  static constexpr char fileName[] = "hipMatMul.code";
};

void GraphModuleLaunchKernel::allocateMemory() {
  A = new int[N * N * sizeof(int)];
  B = new int[N * N * sizeof(int)];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = 1;
      B[i * N + j] = 1;
    }
  }
  HIPCHECK(hipStreamCreate(&stream1));
  HIPCHECK(hipStreamCreate(&stream2));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE * sizeof(int)));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE * sizeof(int)));
  HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&C), SIZE * sizeof(int)));
  HIPCHECK(hipMemcpy(Ad, A, SIZE * sizeof(int), hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(Bd, B, SIZE * sizeof(int), hipMemcpyHostToDevice));
  args1._Ad = Ad;
  args1._Bd = Bd;
  args1._Cd = C;
  args1._n = N;
  args2._Ad = NULL;
  args2._Bd = NULL;
  args2._Cd = NULL;
  args2._n = 0;
  size1 = sizeof(args1);
  size2 = sizeof(args2);
}

void GraphModuleLaunchKernel::moduleLoad() {
  HIPCHECK(hipModuleLoad(&module, fileName));
  HIPCHECK(hipModuleGetFunction(&multKernel, module, matmulK));
}

void GraphModuleLaunchKernel::deAllocateMemory() {
  HIPCHECK(hipStreamDestroy(stream1));
  HIPCHECK(hipStreamDestroy(stream2));
  delete[] A;
  delete[] B;
  HIPCHECK(hipFree(Ad));
  HIPCHECK(hipFree(Bd));
  HIPCHECK(hipHostFree(C));
  HIPCHECK(hipModuleUnload(module));
}

bool GraphModuleLaunchKernel::extModuleKernelExecutionMatmul() {
  bool testStatus = true;
  int mismatch = 0;
  void* config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                     HIP_LAUNCH_PARAM_END};
  HIPCHECK(hipExtModuleLaunchKernel(multKernel, N, N, 1, 32, 32, 1, 0, stream1, NULL,
                                    reinterpret_cast<void**>(&config1), NULL, NULL, 0));
  HIPCHECK(hipStreamSynchronize(stream1));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (C[i * N + j] != N) mismatch++;
    }
  }
  if (mismatch) {
    printf("Test failed: the result of matrix multiplications incorrect.\n");
    testStatus = false;
  }
  return testStatus;
}

bool GraphModuleLaunchKernel::extModuleKernelExecutionMatmulwithStreamCapture(
    bool LaunchByDifferentStream) {
  bool testStatus = true;
  int mismatch = 0;

  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};

  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));

  void* config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                     HIP_LAUNCH_PARAM_END};

  HIPCHECK(hipExtModuleLaunchKernel(multKernel, N, N, 1, 32, 32, 1, 0, stream1, NULL,
                                    reinterpret_cast<void**>(&config1), NULL, NULL, 0));

  HIP_CHECK(hipStreamEndCapture(stream1, &graph));

  // Validate end capture is successful
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Replay the recorded sequence
  HIP_CHECK(hipGraphLaunch(graphExec, LaunchByDifferentStream ? stream2 : stream1));

  HIP_CHECK(hipStreamSynchronize(LaunchByDifferentStream ? stream2 : stream1));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (C[i * N + j] != N) mismatch++;
    }
  }
  if (mismatch) {
    printf("Test failed: the result of matrix multiplications incorrect.\n");
    testStatus = false;
  }
  return testStatus;
}

// Helper to test kernel launch with and without clean global/local sizes.
bool GraphModuleLaunchKernel::extModuleKernelExecutionMatmulwithStreamCaptureN(
    int n, unsigned int blockX, unsigned int blockY) {
  // Allocate fresh host/device buffers sized for n×n.
  int* hA = new int[n * n];
  int* hB = new int[n * n];
  int* hC = nullptr;
  HIPCHECK(hipHostMalloc(reinterpret_cast<void**>(&hC), n * n * sizeof(int)));

  for (int i = 0; i < n * n; i++) {
    hA[i] = 1;
    hB[i] = 1;
    hC[i] = 0;
  }

  int* dA = nullptr;
  int* dB = nullptr;
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&dA), n * n * sizeof(int)));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&dB), n * n * sizeof(int)));
  HIPCHECK(hipMemcpy(dA, hA, n * n * sizeof(int), hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(dB, hB, n * n * sizeof(int), hipMemcpyHostToDevice));

  struct {
    void* _Ad;
    void* _Bd;
    void* _Cd;
    int _n;
  } localArgs;
  localArgs._Ad = dA;
  localArgs._Bd = dB;
  localArgs._Cd = hC;
  localArgs._n = n;
  size_t localArgsSize = sizeof(localArgs);

  // global dims: ceil(n/blockX)*blockX is NOT used — we pass n directly so
  // the launch covers exactly the matrix; the kernel guard handles remainder
  // threads in the last block.
  unsigned int globalX = static_cast<unsigned int>(n);
  unsigned int globalY = static_cast<unsigned int>(n);

  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};

  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &localArgs, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &localArgsSize, HIP_LAUNCH_PARAM_END};

  HIPCHECK(hipExtModuleLaunchKernel(multKernel, globalX, globalY, 1, blockX, blockY, 1, 0, stream1,
                                    NULL, reinterpret_cast<void**>(&config), NULL, NULL, 0));

  HIP_CHECK(hipStreamEndCapture(stream1, &graph));
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  HIP_CHECK(hipGraphLaunch(graphExec, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));

  bool testStatus = true;
  // Validate every element of the n×n result; each C[i][j] = n (sum of n ones).
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (hC[i * n + j] != n) {
        testStatus = false;
        break;
      }
    }
    if (!testStatus) break;
  }

  HIPCHECK(hipFree(dA));
  HIPCHECK(hipFree(dB));
  HIPCHECK(hipHostFree(hC));
  delete[] hA;
  delete[] hB;

  return testStatus;
}

HIP_TEST_CASE(Unit_hipStreamCapture_ExtModuleLaunchKernel) {
  struct stat fileStat;
  if (stat(GraphModuleLaunchKernel::fileName, &fileStat) || !(fileStat.st_mode & S_IFREG)) {
    FAIL("module file " << GraphModuleLaunchKernel::fileName << " doesn't exist! aborted! \n"
                        << "To generate the file, type\n"
                        << "/opt/rocm/hip/bin/hipcc --cuda-device-only hipMatMul.cc -o hipMatMul.code");
    return;
  }
  HIPCHECK(hipSetDevice(0));
  GraphModuleLaunchKernel kernelLaunch;

  SECTION("extModuleKernelExecutionMatmul") {
    REQUIRE(kernelLaunch.extModuleKernelExecutionMatmul());
  }

  SECTION("extModuleKernelExecutionMatmul_withStreamCapture") {
    REQUIRE(kernelLaunch.extModuleKernelExecutionMatmulwithStreamCapture());
  }

  SECTION("extModuleKernelExecutionMatmul_withStreamCapture_launchByDifferentStream") {
    REQUIRE(kernelLaunch.extModuleKernelExecutionMatmulwithStreamCapture(true));
  }

  // Tests simple global % local == 0 and remainder global % local != 0 cases
  SECTION("extModuleKernelExecutionMatmul_withStreamCapture_remainder") {
    const auto cfg =
        GENERATE(gridblockDim{64, 1, 1, 32, 32, 1}, gridblockDim{100, 1, 1, 32, 32, 1});
    REQUIRE(kernelLaunch.extModuleKernelExecutionMatmulwithStreamCaptureN(
        static_cast<int>(cfg.gridX), cfg.blockX, cfg.blockY));
  }
}
