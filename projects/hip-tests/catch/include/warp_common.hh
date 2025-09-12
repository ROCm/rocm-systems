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

#pragma once

#define HIP_ENABLE_WARP_SYNC_BUILTINS
#define HIP_ENABLE_EXTRA_WARP_SYNC_TYPES

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_fp16.h>
#include <limits>
#include <cmath>
#include <iostream>
#include <ios>

#define MASK_SHIFT(x, n) \
  (x & (static_cast<uint64_t>(1) << n)) >> n

const unsigned long long Every5thBit = 0x1084210842108421;
const unsigned long long Every9thBit = 0x8040201008040201;
const unsigned long long Every5thBut9th = Every5thBit & ~Every9thBit;
const unsigned long long AllThreads = ~0;
static constexpr int kNumReduces = 5000;

inline __device__ bool deactivate_thread(const uint64_t* const active_masks) {
  const auto warp =
      cooperative_groups::tiled_partition(cooperative_groups::this_thread_block(), warpSize);
  const auto block = cooperative_groups::this_thread_block();
  const auto warps_per_block = (block.size() + warpSize - 1) / warpSize;
  const auto block_rank = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto idx = block_rank * warps_per_block + block.thread_rank() / warpSize;
  return !(active_masks[idx] & (static_cast<uint64_t>(1) << warp.thread_rank()));
}

inline std::mt19937& GetRandomGenerator() {
  static std::mt19937 mt(std::random_device{}());
  return mt;
}

template <typename T> inline T GenerateRandomInteger(const T min, const T max) {
  std::uniform_int_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

template <typename T> inline T GenerateRandomReal(const T min, const T max) {
  std::uniform_real_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

inline int generate_width(int warp_size) {
  int exponent = 0;
  while (warp_size >>= 1) {
    ++exponent;
  }

  return GENERATE_COPY(map([](int e) { return 1 << e; }, range(1, exponent + 1)));
}

inline uint64_t get_active_mask(unsigned int warp_id, unsigned int warp_size) {
  uint64_t active_mask = 0;
  switch (warp_id % 5) {
    case 0:  // even threads in the warp
      active_mask = 0xAAAAAAAAAAAAAAAA;
      break;
    case 1:  // odd threads in the warp
      active_mask = 0x5555555555555555;
      break;
    case 2:  // first half of the warp
      for (int i = 0; i < warp_size / 2; i++) {
        active_mask = active_mask | (static_cast<uint64_t>(1) << i);
      }
      break;
    case 3:  // second half of the warp
      for (int i = warp_size / 2; i < warp_size; i++) {
        active_mask = active_mask | (static_cast<uint64_t>(1) << i);
      }
      break;
    case 4:  // all threads
      active_mask = 0xFFFFFFFFFFFFFFFF;
      break;
  }
  return active_mask;
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline T expandPrecision(int X) { return X; }

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
inline T expandPrecision(int X) {
  return X * 3.141592653589793115997963468544185161590576171875;
}

template <typename T, std::enable_if_t<std::is_same<T, __half>::value, bool> = true>
inline __half expandPrecision(int X) {
  return (__half)expandPrecision<float>(X);
}

template <typename T, std::enable_if_t<std::is_same<T, __half2>::value, bool> = true>
inline __half2 expandPrecision(int X) {
  __half H = expandPrecision<float>(X);
  return {H, H};
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline void expandPrecision(T* Array, int size) {
  (void)Array;
  (void)size;
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
inline void expandPrecision(T *Array, int size) {
  for (int i = 0; i != size; ++i) {
    Array[i] *= 3.141592653589793115997963468544185161590576171875;
  }
}

template <typename T>
inline void initializeInput(T *Input, int size) {
  int Values[] = {0, -1, 2, 3, 4, 5, -6, 7,
                  8, -9, 10, 11, 12, 13, -14, 15,
                  16, 17, -18, 19, 20, -21, 22, 23,
                  24, 25, 26, -27, 28, 29, 30, 31,
                  -32, 33, 34, 35, -36, 37, 38, -39,
                  40, 41, 42, 43, -44, -45, 46, 47,
                  48, 49, 50, -51, 52, 53, -54, 55,
                  56, 57, -58, 59, 60, 61, 62, -63};

  for (int i = 0; i != size; ++i) {
    Input[i] = expandPrecision<T>(Values[i]);
  }
}

template <typename T>
inline void initializeExpected(T *Expected, int *Values, int size) {
  for (int i = 0; i != size; ++i) {
    Expected[i] = expandPrecision<T>(Values[i]);
  }
}

template <typename T>
inline bool compareEqual(T X, T Y) { return X == Y; }

template <>
inline bool compareEqual(__half X, __half Y) {
  return __half2float(X) == __half2float(Y);
}

template <>
inline bool compareEqual(__half2 X, __half2 Y) {
  return compareEqual(X.x, Y.x) && compareEqual(X.y, Y.y);
}

inline bool compareMaskEqual(unsigned long long *Actual, unsigned long long *Expected,
                       int i, int warpSize) {
  if (warpSize == 32)
    return (unsigned)Actual[i] == (unsigned)Expected[i];
  return Actual[i] == Expected[i];
}

template <typename T>
inline T alignUp(T num, size_t n) {
  if (num % n == 0) {
    return num;
  }

  return ((num + n - 1) / n) * n;
}

template <class T>
struct DistributionType {
  using type = std::uniform_int_distribution<T>;
};

// there is no std::uniform_real_distribution for 'half' type, so we cast from
// unsigned short, avoiding Nan and Infinity
template <>
struct DistributionType<__half> {
  using type = std::uniform_int_distribution<unsigned short>;
};

template <>
struct DistributionType<float> {
  using type = std::uniform_real_distribution<float>;
};

template <>
struct DistributionType<double> {
  using type = std::uniform_real_distribution<double>;
};


template <class T>
struct MinOp {
  T operator()(const T& lhs, const T& rhs) const
  {
    return std::min(lhs, rhs);
  }
};

template <class T>
struct MaxOp {
  T operator()(const T& lhs, const T& rhs) const
  {
    return std::max(lhs, rhs);
  }
};

template <class T>
struct XorOp {
  __host__ __device__ T operator()(const T& lhs, const T& rhs)
  {
    return (!lhs) != (!rhs) == 1;
  }
};

// typeid(T).name() does seem to return a very descriptive name for primitive types,
// at least on clang, so we roll out an equivalent
template<class T>
const char* typeToString()
{
  if (std::is_same<T, int>::value)
    return "int";
  if (std::is_same<T, unsigned int>::value)
    return "unsigned int";
  if (std::is_same<T, long long>::value)
    return "long long";
  if (std::is_same<T, unsigned long long>::value)
    return "unsigned long long";
  if (std::is_same<T, half>::value)
    return "half";
  if (std::is_same<T, float>::value)
    return "float";
  if (std::is_same<T, double>::value)
    return "double";

  assert(false && "Missing conversion to string for type");
  return "";
}

template<class T, template <typename> class Op>
const char* opToString()
{
  if constexpr (std::is_same<Op<T>, std::plus<T>>::value)
    return "add";
  else if constexpr (std::is_same<Op<T>, MinOp<T>>::value)
    return "min";
  else if constexpr (std::is_same<Op<T>, MaxOp<T>>::value)
    return "max";
  else if constexpr (std::is_same<Op<T>, std::logical_and<T>>::value)
    return "logical_and";
  else if constexpr (std::is_same<Op<T>, std::logical_or<T>>::value)
    return "logical_or";
  else if constexpr (std::is_same<Op<T>, XorOp<T>>::value)
    return "logical_xor";
  else {
    static_assert(std::is_void<T>::value, "Unsupported operator");
    return "";
  }
}

template <class T, class Gen>
void genRandomMasks(LinearAllocGuard<T>& d_buf,
                    LinearAllocGuard<T>& buf,
                    Gen& gen,
                    int numItems)
{
  // masks must be != 0, hence passing 1 as the 'a' distribution parameter
  std::uniform_int_distribution<unsigned long long> dist(1);
  int numBytes = numItems * sizeof(T);
  LinearAllocGuard<T> tmp(LinearAllocs::malloc, numBytes);
  LinearAllocGuard<T> d_tmp(LinearAllocs::hipMalloc, numBytes);

  buf = std::move(tmp);
  d_buf = std::move(d_tmp);

  for (int i = 0; i < numItems; i++) {
    T mask = dist(gen);

    if (getWarpSize() == 32)
      mask &= 0xFFFFFFFF;

    buf.ptr()[i] = mask;
  }

  HIP_CHECK(hipMemcpy(d_buf.ptr(), buf.ptr(), numBytes, hipMemcpyHostToDevice));
}

// generates a random __half (instead of using uniform_real_distribution<float> casting to __half
// which is problematic)
// @expDist needs to be between [0-2^5-2]
template <class Gen>
__half genRandomHalf(std::uniform_int_distribution<unsigned short>& dist,
                     Gen& gen)
{
  __half_raw tmp;

  tmp.x = dist(gen);
  // rewrite the exponent to force the number to be (-8<x<8) and at the same time avoid NaN or
  // infinity
  tmp.x &= 0xBBFF;
  return tmp;
}

// generates a random buffer in buf, copies it to device memory in d_buf
template <class T, class Dist, class Gen>
void genRandomBuffers(LinearAllocGuard<T>& d_buf,
                      LinearAllocGuard<T>& buf,
                      Dist& dist,
                      Gen& gen,
                      int numItems)
{
  int numBytes = numItems * sizeof(T);
  LinearAllocGuard<T> tmp(LinearAllocs::malloc, numBytes);
  LinearAllocGuard<T> d_tmp(LinearAllocs::hipMalloc, numBytes);

  buf = std::move(tmp);
  d_buf = std::move(d_tmp);

  for (int i = 0; i < numItems; i++)
    if constexpr (std::is_same<T, __half>::value)
      buf.ptr()[i] = genRandomHalf(dist, gen);
    else
      buf.ptr()[i] = dist(gen);

  HIP_CHECK(hipMemcpy(d_buf.ptr(), buf.ptr(), numBytes, hipMemcpyHostToDevice));
}

// given an operation produces the expected result of the reduction
// @mask indicates the lanes that will participate in the computation
template <class T, class Op>
T calculateExpected(const T* input, Op op, unsigned long long mask)
{
  T result;
  int wavefrontSize = getWarpSize();

  if (std::is_same<Op, std::plus<T>>::value) {
    T tmp[64] = { 0 };

    for (int i = 0; i < wavefrontSize; i++) {
       if (mask & (1ul << i)) {
         tmp[i] = input[i];
       }
    }

    for (int modulo = 2; modulo <= wavefrontSize; modulo *= 2) {
      for (int i = 0; i < wavefrontSize; i += modulo) {
        int j = i + modulo / 2;

        if (j < wavefrontSize)
          tmp[i] += tmp[j];
      }
    }
    result = tmp[0];
  } else {
    bool initialized = false;

    for (int i = 0; i < wavefrontSize; i++) {
      if (mask & (1ul << i)) {
        if (initialized)
          result = op(input[i], result);
        else {
          result = input[i];
          initialized = true;
        }
      }
    }
  }
  return result;
}

template <class T>
void printMismatch(const T& result, const T& expected, const T* input, unsigned long long mask)
{
  std::ios init(NULL);

  init.copyfmt(std::cout);
  std::cout << "\nMismatch\n";
  std::cout << "Mask: 0x" << std::hex << std::setfill('0') << std::setw(16) << mask << "\n";
  std::cout.copyfmt(init);

  for (int i = 0; i < getWarpSize(); i++) {
    if ((1ul << i) & mask) {
      if constexpr (std::is_same<T, __half>::value)
                     std::cout << "Lane " << i << ": " << __half2float(input[i]) << "\n";
      else
        std::cout << "Lane " << i << ": " << input[i] << "\n";
    }
  }

  if constexpr (std::is_same<T, __half>::value) {
    std::cout << "Result: " << __half2float(result) << "\n";
    std::cout << "Expected: " << __half2float(expected) << "\n";
  } else {
    std::cout << "Result: " << result << "\n";
    std::cout << "Expected: " << expected << "\n";
  }
}

template <class T>
void compareFloatingPoint(const T& result, const T& expected, unsigned long long mask, const T* input)
{
  using namespace Catch::Matchers;
  if constexpr (std::is_same<T, __half>::value) {
    float resultFloat = __half2float(result);
    float expectedFloat = __half2float(expected);
    float absDifference = fabs(resultFloat - expectedFloat);
    float relativeEpsilon = 0.1 * fmax(resultFloat, expectedFloat);
    float eps = 0.01f;

    REQUIRE(!__hisnan(result));
    REQUIRE(!__hisinf(result));

    if (relativeEpsilon > eps) {
      if (absDifference > 0.0001) {
        if (absDifference >= eps * fabs(fmax(resultFloat, expectedFloat))) {
          printMismatch(result, expected, input, mask);
          std::cout << "Relative epsilon: " << relativeEpsilon << "\n";
          std::cout << "Difference: " << absDifference << "\n";
        }
       }

      REQUIRE_THAT(__half2float(resultFloat), WithinRel(expectedFloat, eps));
    }
  } else {
    // for float or double, also lossy in terms of precision
    T absDifference = fabs(result - expected);
    T relativeEpsilon = 0.1 * fmax(result, expected);
    T eps = 0.01;

    if (relativeEpsilon > eps) {
      if (absDifference > 0.0001) {
        if (absDifference >= eps * fabs(fmax(result, expected))) {
          printMismatch(result, expected, input, mask);
          std::cout << "Relative epsilon: " << relativeEpsilon << "\n";
          std::cout << "Difference: " << absDifference << "\n";
        }

        REQUIRE_THAT(result, WithinRel(expected, eps));
      }
    }
  }
}

// @tparam Reduce a functor; abstracts away kernel dispatching
//         (via hiprtc or normal execution)
template <class T, class Reduce, template <typename> class Op>
void runTestReduce(int iteration, Reduce reduce)
{
  using namespace Catch::Matchers;
  using distribution = typename DistributionType<T>::type;
  unsigned int wavefrontSize = getWarpSize();
  // one result per reduce per thread to be checked
  LinearAllocGuard<T> d_output(LinearAllocs::hipMalloc, kNumReduces * wavefrontSize * sizeof(T));
  LinearAllocGuard<T> output(LinearAllocs::malloc, kNumReduces * wavefrontSize * sizeof(T));
  std::mt19937_64 gen(iteration);
  // for float16, we generate any random unsigned short, but cap the exponent later on
  // to keep it in the range (-8.0..8.0) (just to avoid overflows)
  // On the rest of the types, just use a bigger reduced range of numbers to avoid overflows too
  T a = std::is_same<T, half>::value? std::numeric_limits<unsigned short>::lowest() : -1023;
  T b = std::is_same<T, half>::value? std::numeric_limits<unsigned short>::max() : 1023;
  distribution dist(a, b);
  LinearAllocGuard<T> input, d_input;
  LinearAllocGuard<unsigned long long> masks, d_masks;
  Op<T> op;
  int numReduce = 0;

  genRandomBuffers(d_input, input, dist, gen, kNumReduces * wavefrontSize);
  genRandomMasks(d_masks, masks, gen, kNumReduces);
  reduce(d_output.ptr(), d_input.ptr(), d_masks.ptr(), kNumReduces, op);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(output.ptr(), d_output.ptr(), d_output.size_bytes(), hipMemcpyDeviceToHost));

  while (numReduce < kNumReduces) {
    T expected = calculateExpected<T>(input.ptr(), op, masks.ptr()[numReduce]);
    int lane = 0;

    while (lane < wavefrontSize) {
      auto result = output.ptr()[numReduce * wavefrontSize + lane];
      unsigned long long mask = masks.ptr()[numReduce];

      if ((1ul << lane) & mask) {
        if constexpr (std::is_integral<T>::value || std::is_same<Op<T>, MinOp<T>>::value ||
                      std::is_same<Op<T>, MaxOp<T>>::value) {
          // for integral types or min/max the result should match exactly
          if constexpr (std::is_same<T, __half>::value)
            REQUIRE(__half2float(result) == __half2float(expected));
          else {
            if (result != expected) {
              printMismatch(result, expected, input.ptr(), mask);
              REQUIRE(result == expected);
            }
          }
        } else
          compareFloatingPoint(result, expected, mask, input.ptr());

      }
      lane++;
    }
    numReduce++;
  }
}
