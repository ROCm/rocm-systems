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
#define HIP_ENABLE_WARP_SYNC_BUILTINS
#define HIP_ENABLE_EXTRA_WARP_SYNC_TYPES

#include "warp_common.hh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip_test_common.hh>
#include <performance_common.hh>
#include <hip/amd_detail/amd_hip_atomic.h>
#include <resource_guards.hh>
#include <cmd_options.hh>
#include <algorithm>
#include <type_traits>
#include <limits>
#include <map>

/**
 * @addtogroup __reduce_op_sync __reduce_op_sync
 * @{
 * @ingroup WarpSyncPerformance
 * __reduce_op_sync(MaskT mask, T val)
 * Reduces the val as per the lanes described in mask and calculates the
 * aggregated result
 */

static constexpr int kBlockDim = 1024;

template <class T> struct AtomicAddOp {
  __device__ T operator()(T* lhs, const T& rhs) { return atomicAdd(lhs, rhs); }
};

template <class T> struct AtomicMinOp {
  __device__ T operator()(T* lhs, const T& rhs) { return atomicMin(lhs, rhs); }
};

template <class T> struct AtomicMaxOp {
  __device__ T operator()(T* lhs, const T& rhs) { return atomicMax(lhs, rhs); }
};

template <class T> struct AtomicAndOp {
  __device__ T operator()(T* lhs, const T& rhs) { return atomicAnd(lhs, rhs); }
};

template <class T> struct AtomicOrOp {
  __device__ T operator()(T* lhs, const T& rhs) { return atomicOr(lhs, rhs); }
};

template <class T> struct AtomicXorOp {
  __device__ T operator()(T* lhs, const T& rhs) { return atomicXor(lhs, rhs); }
};

// uses atomics to reduce the whole warp; depending on the mask our reduce should be faster
// @output   to store the result, one per warp
// @numItems must be a multiple of warpSize
template <class T, template <typename> class Op>
__global__ void reduceAllAtomics(T* __restrict__ output, const T* __restrict__ input,
                                 unsigned long long mask) {
  int idx = threadIdx.x + blockIdx.x * kBlockDim;
  extern __shared__ uint8_t shared_mem[];
  T* result = reinterpret_cast<T*>(shared_mem);  // one per warp
  Op<T> op;
  int numWarp = threadIdx.x / warpSize;

  // initialize result[numWarp] to the "identity" element for Op
  if constexpr (std::is_same<Op<T>, AtomicMinOp<T>>::value)
    result[numWarp] = std::numeric_limits<T>::max();
  else if constexpr (std::is_same<Op<T>, AtomicMaxOp<T>>::value)
    result[numWarp] = std::numeric_limits<T>::lowest();
  else if constexpr (std::is_same<Op<T>, AtomicAndOp<T>>::value)
    result[numWarp] = 1;
  else
    result[numWarp] = 0;

  __syncthreads();

  if (mask & (1ul << __ockl_lane_u32())) op(&result[numWarp], input[idx]);

  __syncthreads();

  if (__ockl_lane_u32() == 0) output[idx / warpSize] = result[numWarp];
}

template <class T, template <typename> class Op>
__global__ void reduceOpSync(T* __restrict__ output, const T* __restrict__ input,
                             unsigned long long mask) {
  int idx = threadIdx.x + blockIdx.x * kBlockDim;
  T result;

  if (mask & (1ul << __ockl_lane_u32())) {
    if constexpr (std::is_same<Op<T>, std::plus<T>>::value)
      result = __reduce_add_sync(mask, input[idx]);
    else if constexpr (std::is_same<Op<T>, MinOp<T>>::value)
      result = __reduce_min_sync(mask, input[idx]);
    else if constexpr (std::is_same<Op<T>, MaxOp<T>>::value)
      result = __reduce_max_sync(mask, input[idx]);
    else if constexpr (std::is_same<Op<T>, std::logical_and<T>>::value)
      result = __reduce_and_sync(mask, input[idx]);
    else if constexpr (std::is_same<Op<T>, std::logical_or<T>>::value)
      result = __reduce_or_sync(mask, input[idx]);
    else if constexpr (std::is_same<Op<T>, XorOp<T>>::value)
      result = __reduce_xor_sync(mask, input[idx]);
    else
      static_assert(std::is_void<T>::value, "Unsupported operator");

    if (__ockl_activelane_u32() == 0) output[idx / warpSize] = result;
  }
}

template <class T, template <typename> class Op> class AtomicBenchmark
    : public Benchmark<AtomicBenchmark<T, Op>> {
 public:
  void operator()(T* output, const T* input, int numItems, unsigned long long mask) {
    dim3 blockDim = {kBlockDim};
    dim3 gridDim = {static_cast<uint32_t>(std::ceil(numItems / static_cast<float>(blockDim.x)))};

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    int warpSize = props.warpSize;
    int numWarpsPerBlock = kBlockDim / warpSize;
    size_t sharedSize = numWarpsPerBlock * sizeof(T);
    TIMED_SECTION(kTimerTypeEvent) {
      if constexpr (std::is_same<Op<T>, std::plus<T>>::value)
        reduceAllAtomics<T, AtomicAddOp><<<gridDim, blockDim, sharedSize>>>(output, input, mask);
      else if constexpr (std::is_same<Op<T>, MinOp<T>>::value)
        reduceAllAtomics<T, AtomicMinOp><<<gridDim, blockDim, sharedSize>>>(output, input, mask);
      else if constexpr (std::is_same<Op<T>, MaxOp<T>>::value)
        reduceAllAtomics<T, AtomicMaxOp><<<gridDim, blockDim, sharedSize>>>(output, input, mask);
      else if constexpr (std::is_same<Op<T>, std::logical_and<T>>::value)
        reduceAllAtomics<T, AtomicAndOp><<<gridDim, blockDim, sharedSize>>>(output, input, mask);
      else if constexpr (std::is_same<Op<T>, std::logical_or<T>>::value)
        reduceAllAtomics<T, AtomicOrOp><<<gridDim, blockDim, sharedSize>>>(output, input, mask);
      else if constexpr (std::is_same<Op<T>, XorOp<T>>::value)
        reduceAllAtomics<T, AtomicXorOp><<<gridDim, blockDim, sharedSize>>>(output, input, mask);
      else
        static_assert(std::is_void<T>::value, "Unsupported operator");

      HIP_CHECK(hipDeviceSynchronize());
    }
  }
};

template <class T, template <typename> class Op> class ReduceSyncBenchmark
    : public Benchmark<ReduceSyncBenchmark<T, Op>> {
 public:
  void operator()(T* output, T* input, int numItems, unsigned long long mask) {
    dim3 blockDim = {kBlockDim};
    dim3 gridDim = {static_cast<uint32_t>(std::ceil(numItems / static_cast<float>(blockDim.x)))};


    TIMED_SECTION(kTimerTypeEvent) {
      reduceOpSync<T, Op><<<gridDim, blockDim>>>(output, input, mask);
      HIP_CHECK(hipDeviceSynchronize());
    }
  }
};

template <class T, template <typename> class Op>
void checkResults(T* d_atomicsResult, T* d_reduceResult, size_t numBytes, unsigned long long mask) {
  using namespace Catch::Matchers;
  LinearAllocGuard<T> outputAtomic(LinearAllocs::malloc, numBytes);
  LinearAllocGuard<T> outputReduce(LinearAllocs::malloc, numBytes);
  bool memcmpResult = std::memcmp(outputAtomic.ptr(), outputReduce.ptr(), numBytes);

  assert(numBytes % sizeof(T) == 0 && "numBytes needs to be a multiple of sizeof(T)");
  HIP_CHECK(hipMemcpy(outputAtomic.ptr(), d_atomicsResult, numBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(outputReduce.ptr(), d_reduceResult, numBytes, hipMemcpyDeviceToHost));

  if (memcmpResult) {
    for (int i = 0; i < numBytes / sizeof(T); i++) {
      auto& atomicResult = outputAtomic.ptr()[i];
      auto& reduceResult = outputReduce.ptr()[i];

      if constexpr (std::is_integral<T>::value || std::is_same<Op<T>, MinOp<T>>::value ||
                    std::is_same<Op<T>, MaxOp<T>>::value)
        // for integral types or min/max the result should match exactly
        REQUIRE(atomicResult == reduceResult);
      else
        // floating point types or operations which are lossy in terms of precision
        REQUIRE_THAT(reduceResult, WithinRel(atomicResult));
    }
  }
}

template <class T, template <typename> class Op> struct IsLogicalOp {
  static constexpr bool value = false;
};

template <class T> struct IsLogicalOp<T, std::logical_and> {
  static constexpr bool value = true;
};

template <class T> struct IsLogicalOp<T, std::logical_or> {
  static constexpr bool value = true;
};

template <class T> struct IsLogicalOp<T, XorOp> {
  static constexpr bool value = true;
};

// Neither long long or fp16 have atomic operations. In those cases
// we only benchmark reduce sync operations, we cannot compare with native atomics
template <class T> struct HasAtomicOps {
  static constexpr bool value = true;
};

template <> struct HasAtomicOps<half> {
  static constexpr bool value = false;
};

template <> struct HasAtomicOps<long long> {
  static constexpr bool value = false;
};

template <class T, template <typename> class Op> struct ReduceBenchmark {
  void Run() {
    static constexpr int numMasks = 6;
    using distribution = typename DistributionType<T>::type;
    ReduceSyncBenchmark<T, Op> benchmarkReduce;
    uint64_t inputSize = cmd_options.reduce_input_size * 1_MB;
    int numItems = inputSize / sizeof(T);
    int wavefrontSize = getWarpSize();
    int outputNumBytes = inputSize / wavefrontSize;
    LinearAllocGuard<T> input(LinearAllocs::malloc, inputSize);
    LinearAllocGuard<T> d_input(LinearAllocs::hipMalloc, inputSize);
    LinearAllocGuard<T> d_outputsAtomic[numMasks];
    LinearAllocGuard<T> d_outputsReduce[numMasks];
    LinearAllocGuard<T>* d_outputAtomic = &d_outputsAtomic[0];
    LinearAllocGuard<T>* d_outputReduce = &d_outputsReduce[0];
    std::mt19937_64 gen(123);
    distribution dist;
    int halfWaveSize = wavefrontSize / 2;
    unsigned long long halfBitsOn = (1ul << (wavefrontSize / 2)) - 1;
    unsigned long long fullMask = -1ul, halfHighBitsOn = halfBitsOn << halfWaveSize,
                       high16BitsOn = halfBitsOn << (wavefrontSize - 16),
                       high8BitsOn = halfBitsOn << (wavefrontSize - 8),
                       high4BitsOn = halfBitsOn << (wavefrontSize - 4), allButOne = -1 & ~1;
    const char* typeStr = typeToString<T>();
    const char* opStr = opToString<T, Op>();
    std::map<std::string, unsigned long long> masks;
    std::pair<std::string, unsigned long long> masksPairs[] = {
        {"full mask", fullMask},
        {"high order 32 bits on", halfHighBitsOn},
        {"high order 16 bits on", high16BitsOn},
        {"high order 8 bits on", high8BitsOn},
        {"high order 4 bits on", high4BitsOn},
        {"all but one", allButOne}};
    int pos = 0, numMask = 0;

    for (auto& mask : masksPairs) {
      // don't use 'halfHighBitsOn' on warp size 32; it's the same as high16BitsOn
      if (wavefrontSize != 32 || mask.second != halfHighBitsOn) {
        masks.emplace(std::to_string(numMask) + " - " + mask.first,
                      wavefrontSize == 64 ? mask.second : mask.second & 0xFFFFFFFF);
        numMask++;
      }
    }

    // avoid generating values different than 1 or 0 for logical operators;
    // otherwise the atomic version of the kernels would produce different results as
    // atomicAnd/Or() are bitwise operations, not logical
    if constexpr (IsLogicalOp<T, Op>::value) dist = distribution(0, 1);

    for (int i = 0; i < numItems; i++) {
      input.ptr()[i] = dist(gen);
    }

    for (auto& buffer : d_outputsAtomic) {
      buffer = LinearAllocGuard<T>(LinearAllocs::hipMalloc, outputNumBytes);
    }

    for (auto& buffer : d_outputsReduce) {
      buffer = LinearAllocGuard<T>(LinearAllocs::hipMalloc, outputNumBytes);
    }

    HIP_CHECK(hipMemcpy(d_input.ptr(), input.ptr(), inputSize, hipMemcpyHostToDevice));

    if constexpr (HasAtomicOps<T>::value) {
      AtomicBenchmark<T, Op> benchmarkAtomics;

      printf("\n--- atomics %s %s---\n", opStr, typeStr);

      for (auto& mask : masks) {
        printf("%s %llx\n", mask.first.c_str(), mask.second);
        benchmarkAtomics.Run((d_outputAtomic++)->ptr(), d_input.ptr(), numItems, mask.second);
      }
    }

    printf("\n--- reduce %s %s--- \n", opStr, typeStr);

    for (const auto& mask : masks) {
      printf("%s %llx\n", mask.first.c_str(), mask.second);
      benchmarkReduce.Run((d_outputReduce++)->ptr(), d_input.ptr(), numItems, mask.second);
    }

    printf("\n");

    if constexpr (HasAtomicOps<T>::value) {
      printf("Checking results...\n");

      for (const auto& mask : masks) {
        checkResults<T, Op>(d_outputsAtomic[pos].ptr(), d_outputsReduce[pos].ptr(), outputNumBytes,
                            mask.second);
        pos++;
      }
    }
  }
};

TEMPLATE_TEST_CASE("Performance_Reduce_Sync_Add", "", int, unsigned int, unsigned long long,
                   long long, float, half, double) {
  ReduceBenchmark<TestType, std::plus> benchmark;

  benchmark.Run();
}

TEMPLATE_TEST_CASE("Performance_Reduce_Sync_Min", "", int, unsigned int, unsigned long long,
                   long long, float, half, double) {
  ReduceBenchmark<TestType, MinOp> benchmark;

  benchmark.Run();
}

TEMPLATE_TEST_CASE("Performance_Reduce_Sync_Max", "", int, unsigned int, unsigned long long,
                   long long, float, half, double) {
  ReduceBenchmark<TestType, MaxOp> benchmark;

  benchmark.Run();
}

TEMPLATE_TEST_CASE("Performance_Reduce_Sync_And", "", int, unsigned int, unsigned long long,
                   long long) {
  ReduceBenchmark<TestType, std::logical_and> benchmark;

  benchmark.Run();
}

TEMPLATE_TEST_CASE("Performance_Reduce_Sync_Or", "", int, unsigned int, unsigned long long,
                   long long) {
  ReduceBenchmark<TestType, std::logical_or> benchmark;

  benchmark.Run();
}

TEMPLATE_TEST_CASE("Performance_Reduce_Sync_Xor", "", int, unsigned int, unsigned long long,
                   long long) {
  ReduceBenchmark<TestType, XorOp> benchmark;

  benchmark.Run();
}
