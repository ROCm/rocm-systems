/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */


/*
 Test Scenarios :
 1) Calling hipMemcpyTo/FromSymbolAsync() using user declared stream obj and hipStreamPerThread.
 2) Validate get symbol address/size for global const array.
 3) Validate get symbol address/size for static const variable.
*/

#include <hip_test_common.hh>

constexpr size_t NUM = 1024;
constexpr size_t SIZE = 1024 * 4;

__device__ int globalIn[NUM];
__device__ int globalOut[NUM];

__global__ static void Assign(int* Out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  Out[tid] = globalIn[tid];
  globalOut[tid] = globalIn[tid];
}

__device__ __constant__ int globalConst[NUM];
__device__ static __constant__ float statConstVar[NUM];

__global__ void checkAddress(int* addr, bool* out) { *out = (globalConst == addr); }
__global__ void checkStaticConstVarAddress(float* addr, bool* out) {
  *out = (statConstVar == addr);
}

HIP_TEST_CASE(Unit_hipMemcpyToSymbolAsync_ToNFrom) {
  int *A{nullptr}, *Am{nullptr}, *B{nullptr}, *Ad{nullptr}, *C{nullptr}, *Cm{nullptr};
  A = new int[NUM];
  B = new int[NUM];
  C = new int[NUM];

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&Am), SIZE));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&Cm), SIZE));

  for (size_t i = 0; i < NUM; i++) {
    A[i] = -1 * static_cast<int>(i);
    B[i] = 0;
    C[i] = 0;
    Am[i] = -1 * static_cast<int>(i);
    Cm[i] = 0;
  }


  SECTION("Calling hipMemcpyTo/FromSymbol using stream") {
    hipStream_t stream{};
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(
        hipMemcpyToSymbolAsync(HIP_SYMBOL(globalIn), Am, SIZE, 0, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpyFromSymbolAsync(Cm, HIP_SYMBOL(globalOut), SIZE, 0, hipMemcpyDeviceToHost,
                                       stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
    for (size_t i = 0; i < NUM; i++) {
      REQUIRE(Am[i] == B[i]);
      REQUIRE(Am[i] == Cm[i]);
    }
  }

  SECTION("Calling hipMemcpyTo/FromSymbol - validate value in host memory") {
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(globalIn), A, SIZE, 0, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpyFromSymbol(C, HIP_SYMBOL(globalOut), SIZE, 0, hipMemcpyDeviceToHost));

    for (size_t i = 0; i < NUM; i++) {
      REQUIRE(A[i] == B[i]);
      REQUIRE(A[i] == C[i]);
    }
  }

  SECTION("Calling hipMemcpyTo/FromSymbol using user declared stream obj") {
    hipStream_t stream{};
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(
        hipMemcpyToSymbolAsync(HIP_SYMBOL(globalIn), A, SIZE, 0, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpyFromSymbolAsync(C, HIP_SYMBOL(globalOut), SIZE, 0, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));

    for (size_t i = 0; i < NUM; i++) {
      REQUIRE(A[i] == B[i]);
      REQUIRE(A[i] == C[i]);
    }
  }

  SECTION("Calling hipMemcpyTo/FromSymbol using hipStreamPerThread") {
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(globalIn), A, SIZE, 0, hipMemcpyHostToDevice,
                                     hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpyFromSymbolAsync(C, HIP_SYMBOL(globalOut), SIZE, 0, hipMemcpyDeviceToHost,
                                       hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    for (size_t i = 0; i < NUM; i++) {
      REQUIRE(A[i] == B[i]);
      REQUIRE(A[i] == C[i]);
    }
  }

  // Check for address on GPU and CPU side and compare it
  // If address mismatch report error
  // Validate size of symbol as well, compare it with output of hipGetSymbolSize
  SECTION("Validate address on GPU") {
    bool* checkOkD{nullptr};
    bool checkOk = false;
    size_t symbolSize = 0;
    int* symbolAddress{nullptr};
    HIP_CHECK(hipGetSymbolSize(&symbolSize, HIP_SYMBOL(globalConst)));
    HIP_CHECK(
        hipGetSymbolAddress(reinterpret_cast<void**>(&symbolAddress), HIP_SYMBOL(globalConst)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&checkOkD), sizeof(bool)));
    hipLaunchKernelGGL(checkAddress, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, symbolAddress, checkOkD);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(&checkOk, checkOkD, sizeof(bool), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(checkOkD));
    REQUIRE(checkOk);
    REQUIRE((symbolSize == SIZE));
  }

  HIP_CHECK(hipHostFree(Am));
  HIP_CHECK(hipHostFree(Cm));
  HIP_CHECK(hipFree(Ad));
  delete[] A;
  delete[] B;
  delete[] C;
}
/*
 1) Validate get symbol address/size for static const variable.
*/
HIP_TEST_CASE(Unit_hipGetSymbolAddressAndSize_Validation) {
  bool* checkOkD{nullptr};
  bool checkOk = false;
  size_t symbolSize{};
  float* symbolVarAddress{};

  SECTION("Validate symbol size/address of static const variable") {
    HIP_CHECK(hipGetSymbolSize(&symbolSize, HIP_SYMBOL(statConstVar)));
    HIP_CHECK(
        hipGetSymbolAddress(reinterpret_cast<void**>(&symbolVarAddress), HIP_SYMBOL(statConstVar)));
    HIP_CHECK(hipMalloc(&checkOkD, sizeof(bool)));
    hipLaunchKernelGGL(checkStaticConstVarAddress, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                       symbolVarAddress, checkOkD);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(&checkOk, checkOkD, sizeof(bool), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(checkOkD));
    REQUIRE(checkOk);
    REQUIRE(symbolSize == SIZE);
  }
}

HIP_TEST_CASE(Unit_hipGetSymbolAddress_Negative) {
  SECTION("Invalid symbol") {
    int notADeviceSymbol{0};
    int* addr{nullptr};
    HIP_CHECK_ERROR(
        hipGetSymbolAddress(reinterpret_cast<void**>(&addr), HIP_SYMBOL(notADeviceSymbol)),
        hipErrorInvalidSymbol);
  }

  SECTION("Nullptr symbol") {
    int* addr{nullptr};
    HIP_CHECK_ERROR(hipGetSymbolAddress(reinterpret_cast<void**>(&addr), nullptr),
                    hipErrorInvalidSymbol);
  }
}

HIP_TEST_CASE(Unit_hipGetSymbolSize_Negative) {
  SECTION("Invalid symbol") {
    int notADeviceSymbol{0};
    size_t dsize{0};
    HIP_CHECK_ERROR(hipGetSymbolSize(&dsize, HIP_SYMBOL(notADeviceSymbol)), hipErrorInvalidSymbol);
  }

  SECTION("Nullptr symbol") {
    size_t size{0};
    HIP_CHECK_ERROR(hipGetSymbolSize(&size, nullptr), hipErrorInvalidSymbol);
  }
}

static __device__ int d_symbol{};
static __global__ void simple_kernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] += 1.0f;
  }
}

HIP_TEST_CASE(Unit_MemcpyToSymbolInParallelWithStreamLaunch) {
  std::atomic_bool thread_errored{false};
  std::atomic<bool> g_running{true};

  // Simple thread function where we run MemcpyToSymbol in parallel
  // While we create stream, enqueue work and destroy it
  // This pattern is seen in rocDecode, where we found this issue.
  // Creating this test so that we do not break this again.
  auto thread_func = [&](int thread_id) {
    int val = thread_id;
    // We run it until we notify it or the previous threads have errored
    while (g_running.load(std::memory_order_relaxed) &&
           !thread_errored.load(std::memory_order_relaxed)) {
      hipError_t err =
          hipMemcpyToSymbol(HIP_SYMBOL(d_symbol), &val, sizeof(int), 0, hipMemcpyHostToDevice);
      if (err != hipSuccess) {
        thread_errored = true;
      }
    }
  };

  constexpr int N = 256;
  constexpr int kNumWorkers = 4;
  constexpr int kIterations = 50000;

  float* d_data{nullptr};
  HIP_CHECK(hipMalloc(&d_data, N * sizeof(float)));

  std::vector<std::thread> workers;
  workers.reserve(kNumWorkers);
  for (int i = 0; i < kNumWorkers; ++i) {
    workers.emplace_back(thread_func, i);
  }

  // Main thread: rapid stream create -> kernel launch -> destroy cycle.
  for (int i = 0; i < kIterations; ++i) {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamDefault));
    simple_kernel<<<1, N, 0, stream>>>(d_data, N);
    // Intentionally skip hipStreamSynchronize
    HIP_CHECK(hipStreamDestroy(stream));
  }

  g_running.store(false, std::memory_order_relaxed);

  for (auto& w : workers) {
    w.join();
  }

  HIP_CHECK(hipFree(d_data));

  INFO("Checking if threads have errored.");
  REQUIRE(!thread_errored);
}
