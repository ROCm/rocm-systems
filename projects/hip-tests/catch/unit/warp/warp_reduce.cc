/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#define HIP_ENABLE_WARP_SYNC_BUILTINS
#define HIP_ENABLE_EXTRA_WARP_SYNC_TYPES

#include <hip_test_common.hh>
#include "warp_common.hh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <resource_guards.hh>
#include <memory>
#include <vector>
#include <functional>
#include <algorithm>
#include <cstdlib>
#include <cmd_options.hh>
#include <tuple>

#define NELEMS(array) (sizeof(array) / sizeof(array[0]))

template <class T>
// @input an array containing one value per lane to be used as input for the reduction
// @masks a list of masks, none of them sharing bits
__global__ void multipleMasksKernel(T* output, const T* input, const unsigned long long* masks,
                                    int numMasks) {
  bool isInAnyOfTheMasks = false;
  int numMask = 0;
  unsigned long long mask;

  while (numMask < numMasks && !isInAnyOfTheMasks) {
    mask = masks[numMask];
    if ((1ul << threadIdx.x) & mask) isInAnyOfTheMasks = true;

    numMask++;
  }

  if (!isInAnyOfTheMasks) return;

  output[threadIdx.x] = __reduce_add_sync<decltype(mask)>(mask, input[threadIdx.x]);
}

template <class T, class Op, class MaskType>
__global__ void reduceOp(T* output, const T* input, const MaskType* masks, int numReduces, Op) {
  int tid = threadIdx.x;

  for (int i = 0; i < numReduces; i++) {
    if (masks[i] & (1ul << tid)) {
      // call the operator only if the lane is mentioned in the mask
      T& result = output[warpSize * i + tid];

      if constexpr (std::is_same<Op, std::plus<T>>::value)
        result = __reduce_add_sync(masks[i], input[tid]);
      else if constexpr (std::is_same<Op, MinOp<T>>::value)
        result = __reduce_min_sync(masks[i], input[tid]);
      else if constexpr (std::is_same<Op, MaxOp<T>>::value)
        result = __reduce_max_sync(masks[i], input[tid]);
      else if constexpr (std::is_same<Op, AndOp<T>>::value)
        result = __reduce_and_sync(masks[i], input[tid]);
      else if (std::is_same<Op, OrOp<T>>::value)
        result = __reduce_or_sync(masks[i], input[tid]);
      else if (std::is_same<Op, XorOp<T>>::value)
        result = __reduce_xor_sync(masks[i], input[tid]);
      else
        assert(false && "Unsupported operator");
    }
  }
}

template <class T> void runTestMultipleMasks(unsigned long long masks[], int numMasks) {
  using namespace Catch::Matchers;
  using distribution = typename DistributionType<T>::type;
  unsigned int wavefrontSize = getWarpSize();
  LinearAllocGuard<unsigned long long> d_masks(LinearAllocs::hipMalloc,
                                               numMasks * sizeof(decltype(masks[0])));
  LinearAllocGuard<T> d_input, input;
  LinearAllocGuard<T> output(LinearAllocs::malloc, wavefrontSize * sizeof(T));
  LinearAllocGuard<T> d_output(LinearAllocs::hipMalloc, wavefrontSize * sizeof(T));
  std::plus<T> op;
  std::mt19937_64 gen(123);
  typename distribution::result_type a = std::is_same<T, half>::value? std::numeric_limits<unsigned short>::lowest() :
                                         (std::is_signed<T>::value? -1023 : 0);
  typename distribution::result_type b = std::is_same<T, half>::value? std::numeric_limits<unsigned short>::max() :
                                         1023;
  distribution distInput(a, b);
  dim3 blkDim{wavefrontSize};
  dim3 grdDim{1u};

  HIP_CHECK(hipMemcpy(d_masks.ptr(), &masks[0], d_masks.size_bytes(), hipMemcpyHostToDevice));
  genRandomBuffers(d_input, input, distInput, gen, wavefrontSize);
  multipleMasksKernel<T>
      <<<grdDim, blkDim>>>(d_output.ptr(), d_input.ptr(), d_masks.ptr(), numMasks);
  HIP_CHECK(hipMemcpy(output.ptr(), d_output.ptr(), d_output.size_bytes(), hipMemcpyDeviceToHost));

  for (int numMask = 0; numMask < numMasks; numMask++) {
    unsigned long long mask = masks[numMask];
    T expected = calculateExpected<T>(input.ptr(), op, mask);
    int lane = 0;

    while (lane < wavefrontSize) {
      if ((1ul << lane) & mask) {
        T result = output.ptr()[lane];

        if constexpr (std::is_integral<T>::value) {
          // for integral types the result should match exactly
          if (result != expected) {
            printMismatch(result, expected, input.ptr(), mask);
            REQUIRE(result == expected);
          }
        } else
          compareFloatingPoint(result, expected, mask, input.ptr());
      }

      lane++;
    }
  }
}

HIP_TEMPLATE_TEST_CASE(Unit_hipReduceSingleMasks, int, unsigned int, long long,
                   unsigned long long, float, half, double) {
  unsigned long long fullMask = getWarpSize() == 64 ? ~0ul : 0xFFFFFFFF;
  unsigned long long oneBitMasks[] = {0b1 & fullMask};
  unsigned long long everyFifthMasks[] = {Every5thBit & fullMask};
  unsigned long long everyNinethMasks[] = {Every9thBit & fullMask};
  unsigned long long everyFifthButNinethMasks[] = {Every5thBut9th & fullMask};

  runTestMultipleMasks<TestType>(oneBitMasks, NELEMS(oneBitMasks));
  runTestMultipleMasks<TestType>(everyFifthMasks, NELEMS(everyFifthMasks));
  runTestMultipleMasks<TestType>(everyNinethMasks, NELEMS(everyNinethMasks));
  runTestMultipleMasks<TestType>(everyFifthButNinethMasks, NELEMS(everyFifthButNinethMasks));
}

HIP_TEMPLATE_TEST_CASE(Unit_hipReduceMultipleMasks, int, unsigned int, long long,
                   unsigned long long, float, half, double) {
  if (getWarpSize() == 64) {
    unsigned long long masks[] = {0b0110011, 0x0F0F0F0F00000000, 0xF0F0F0F000000000,
                                  0x000000000F0F0F00, 0b0000100};
    // these divergent masks, when combined, occupy the whole set of lanes
    unsigned long long fullMasks[] = {0xFFFF000000000000, 0x0000FFFFFFFF0000, 0x000000000000FFFF};
    unsigned long long fullMasksEvenOdd[] = {0x5555555555555555,   // even lanes
                                             0xAAAAAAAAAAAAAAAA};  // odd lanes

    runTestMultipleMasks<TestType>(masks, NELEMS(masks));
    runTestMultipleMasks<TestType>(fullMasks, NELEMS(fullMasks));
    runTestMultipleMasks<TestType>(fullMasksEvenOdd, NELEMS(fullMasksEvenOdd));
  } else {
    unsigned long long masks1[] = {0x0F0F0F0F, 0xF0F0F0F0};
    unsigned long long masks2[] = {0b0110011, 0x0F0F0F00, 0b0000100};
    runTestMultipleMasks<TestType>(masks1, NELEMS(masks1));
    runTestMultipleMasks<TestType>(masks2, NELEMS(masks2));
  }
}

template <template <typename> class Op, class Type = void>
void runTestReduceForTypes(const std::tuple<>) {}

template <template <typename> class Op, class T, typename... Types>
void runTestReduceForTypes(const std::tuple<T, Types...>) {
  unsigned int wavefrontSize = getWarpSize();
  dim3 blkDim{wavefrontSize};
  dim3 grdDim{1u};
  std::tuple<Types...> remainingTypes;
  int iteration = 0;
  auto reduceFunc = [&](T* d_output, const T* d_input, const unsigned long long* d_masks,
                        int numReduces, Op<T> op) {
    reduceOp<T><<<grdDim, blkDim>>>(d_output, d_input, d_masks, numReduces, op);
  };
  bool customNumIterations = cmd_options.reduce_iterations != 1;

  if (customNumIterations)
    std::cout << "\n" << opToString<T, Op>() << " - " << typeToString<T>() << "\n";

  while (iteration < cmd_options.reduce_iterations) {
    runTestReduce<T, decltype(reduceFunc), Op>(iteration, reduceFunc);
    iteration++;

    if (customNumIterations) {
      std::cout << "\rIteration: " << iteration;
      std::flush(std::cout);
    }
  }

  runTestReduceForTypes<Op>(remainingTypes);
}

HIP_TEST_CASE(Unit_hipReduceRandom) {
  const std::tuple<int, unsigned int, long long, unsigned long long, float, half, double> allTypes;
  const std::tuple<int, unsigned int, long long, unsigned long long> integralTypes;

  SECTION("add") { runTestReduceForTypes<std::plus>(allTypes); }

  SECTION("min") { runTestReduceForTypes<MinOp>(allTypes); }

  SECTION("max") { runTestReduceForTypes<MaxOp>(allTypes); }

  SECTION("and") { runTestReduceForTypes<AndOp>(integralTypes); }

  SECTION("or") { runTestReduceForTypes<OrOp>(integralTypes); }

  SECTION("xor") { runTestReduceForTypes<XorOp>(integralTypes); }
}
