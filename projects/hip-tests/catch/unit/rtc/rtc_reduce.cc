/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#define HIP_ENABLE_WARP_SYNC_BUILTINS
#define HIP_ENABLE_EXTRA_WARP_SYNC_TYPES

#include "warp_common.hh"
#include <hip/hip_runtime.h>
#include <tuple>
#include <cmd_options.hh>
#include <functional>

#define NELEMS(array) (sizeof(array) / sizeof(array[0]))

// compiles the program, reusing the same compiling session for all the types
// (as opposed as calling the rtc compiler for each of the types)
template <template <typename> class Op, class T, typename... Types>
void compileProgram(hiprtcProgram& prog, const std::tuple<T, Types...>&) {
  std::string scalarName, intrinsicName, expression;
  std::tuple<Types...> remainingTypes;

  expression = std::string("reduceRtcKernel<") + typeToString<T>() + ", unsigned long long>";
  HIPRTC_CHECK(hiprtcAddNameExpression(prog, expression.c_str()));
  compileProgram<Op>(prog, remainingTypes);
}

template <class T, class MaskType, template <typename> class Op>
void runRtcReduceOp(hiprtcProgram& prog, T* output, const T* input, const MaskType* masks,
                    int numReduces, Op<T>) {
  unsigned int wavefrontSize = getWarpSize();
  const char* loweredName;
  hipFunction_t kernel;
  hipModule_t module;
  LinearAllocGuard<int> d_numReduces(LinearAllocs::hipMalloc, sizeof(int));

  HIP_CHECK(hipMemcpy(d_numReduces.ptr(), &numReduces, sizeof(int), hipMemcpyHostToDevice));
  std::vector<const void*> args = {output, input, masks, d_numReduces.ptr()};
  std::size_t sizeBytes = args.size() * sizeof(void*);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &sizeBytes, HIP_LAUNCH_PARAM_END};
  std::vector<char> code;
  size_t codeSize;
  std::string expression =
      std::string("reduceRtcKernel<") + typeToString<T>() + ", unsigned long long>";
  dim3 grdDim{1u};
  dim3 blkDim{wavefrontSize};

  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
  code.resize(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  HIPRTC_CHECK(hiprtcGetLoweredName(prog, expression.c_str(), &loweredName));
  HIP_CHECK(hipModuleGetFunction(&kernel, module, loweredName));
  HIP_CHECK(hipModuleLaunchKernel(kernel, grdDim.x, grdDim.y, grdDim.z, blkDim.x, blkDim.y,
                                  blkDim.z, 0, 0, nullptr, config));
  HIP_CHECK(hipModuleUnload(module));
}

template <template <typename> class Op, class Type = void>
void runTestReduceForTypes(hiprtcProgram&, const std::tuple<>) {}

template <template <typename> class Op, class T, typename... Types>
void runTestReduceForTypes(hiprtcProgram& prog, const std::tuple<T, Types...>) {
  std::tuple<Types...> remainingTypes;
  int iteration = 0;

  auto reduceFunc = [&prog](T* d_output, const T* d_input, const unsigned long long* d_masks,
                            int numReduces, Op<T> op) {
    runRtcReduceOp(prog, d_output, d_input, d_masks, numReduces, op);
  };

  std::cout << typeToString<T>() << std::endl;

  while (iteration < cmd_options.reduce_iterations) {
    runTestReduce<T, decltype(reduceFunc), Op>(iteration, reduceFunc);
    iteration++;

    if (cmd_options.reduce_iterations != 1) {
      std::cout << "\rIteration: " << iteration;
      std::flush(std::cout);
    }
  }

  runTestReduceForTypes<Op>(prog, remainingTypes);
}

template <class T, template <typename> class Op>
void reduceOpToString(std::string& scalarName, std::string& intrinsicName) {
  if constexpr (std::is_same<Op<T>, std::plus<T>>::value) {
    scalarName = "std::plus";
    intrinsicName = "__reduce_add_sync";
  } else if constexpr (std::is_same<Op<T>, MinOp<T>>::value) {
    scalarName = "MinOp";
    intrinsicName = "__reduce_min_sync";
  } else if constexpr (std::is_same<Op<T>, MaxOp<T>>::value) {
    scalarName = "MaxOp";
    intrinsicName = "__reduce_max_sync";
  } else if constexpr (std::is_same<Op<T>, AndOp<T>>::value) {
    scalarName = "std::bit_and";
    intrinsicName = "__reduce_and_sync";
  } else if constexpr (std::is_same<Op<T>, OrOp<T>>::value) {
    scalarName = "std::bit_or";
    intrinsicName = "__reduce_or_sync";
  } else if constexpr (std::is_same<Op<T>, XorOp<T>>::value) {
    scalarName = "std::bit_xor";
    intrinsicName = "__reduce_xor_sync";
  } else
    static_assert(std::is_void<T>::value, "Unexpected operator");
}

template <template <typename> class Op, class T = void>
void compileProgram(hiprtcProgram& prog, const std::tuple<>&) {
  size_t logSize;
  std::string scalarName, intrinsicName;
  hiprtcResult compileResult;
  const char* options[] = {"-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-DHIP_ENABLE_EXTRA_WARP_SYNC_TYPES"};

  reduceOpToString<int, Op>(scalarName, intrinsicName);
  compileResult = hiprtcResult{hiprtcCompileProgram(prog, NELEMS(options), options)};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));

  if (compileResult != HIPRTC_SUCCESS || logSize > 0) {
    std::string log(logSize, '\0');

    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    std::cerr << "Runtime compilation failed or contained warnings for operator: " << scalarName
              << " associated reduce function: " << intrinsicName << "\n";
    std::cerr << log << '\n';
    REQUIRE(false);
  }
}

template <template <typename> class Op, typename... Types>
void runAndCompileTest(const std::tuple<Types...> types) {
  std::string scalarName, intrinsicName, kernelStr;
  hiprtcProgram prog;

  reduceOpToString<int, Op>(scalarName, intrinsicName);
  kernelStr = R"(
    template <class T, class MaskType>
    __global__ void reduceRtcKernel(T* output, const T* input, const MaskType* masks, int* numReduces)
    {
      int tid = threadIdx.x;
      int laneId = tid % warpSize;

      for (int i = 0; i < *numReduces; i++) {
        int idx = warpSize * i + laneId;
        if (masks[i] & (1ull << laneId)) {
          // call the operator only if the lane is mentioned in the mask
          T& result = output[idx];
          result = )" +
              intrinsicName + R"((masks[i], input[idx]);
        }
      }
   })";

  HIPRTC_CHECK(
      hiprtcCreateProgram(&prog, kernelStr.c_str(), "warp_reduce.hip", 0, nullptr, nullptr));
  compileProgram<Op>(prog, types);
  runTestReduceForTypes<Op>(prog, types);
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
}

HIP_TEST_CASE(Unit_Rtc_ReduceRandom) {
  const std::tuple<float, half> allTypes;

  SECTION("add") { runAndCompileTest<std::plus>(allTypes); }

  SECTION("min") { runAndCompileTest<MinOp>(allTypes); }

  SECTION("max") { runAndCompileTest<MaxOp>(allTypes); }

}

HIP_TEMPLATE_TEST_CASE(Unit_Rtc_Reduce_Simple, float, __half) {
  std::string scalarName, intrinsicName, kernelStr;
  hiprtcProgram prog;
  unsigned int wavefrontSize = getWarpSize();
  int numReduces = std::pow(2, getWarpSize() / 8 );
  LinearAllocGuard<unsigned long long> d_masks(LinearAllocs::hipMalloc, numReduces * sizeof(unsigned long long));
  LinearAllocGuard<unsigned long long> masks(LinearAllocs::malloc, d_masks.size_bytes());
  int numReduce = 0;
  LinearAllocGuard<TestType> d_output(LinearAllocs::hipMalloc, numReduces * wavefrontSize * sizeof(TestType));
  LinearAllocGuard<TestType> output(LinearAllocs::malloc, numReduces * wavefrontSize * sizeof(TestType));
  std::plus<TestType> op;
  LinearAllocGuard<TestType> d_input(LinearAllocs::hipMalloc, wavefrontSize * sizeof(TestType));
  LinearAllocGuard<TestType> input(LinearAllocs::malloc, d_input.size_bytes());
  std::string opName = opToString<TestType, std::plus<TestType>>();
  std::tuple<TestType> types;

  for (int i = 0; i < wavefrontSize; i++) {
    input.host_ptr()[i] = i;
  }

  HIP_CHECK(hipMemcpy(d_input.ptr(), input.host_ptr(), d_input.size_bytes(), hipMemcpyHostToDevice));
  reduceOpToString<int, std::plus>(scalarName, intrinsicName);
  kernelStr = R"(
    template <class T, class MaskType>
    __global__ void reduceRtcKernel(T* output, const T* input, const MaskType* masks, int* numReduces)
    {
      int tid = threadIdx.x;
      int laneId = tid % warpSize;

      for (int i = 0; i < *numReduces; i++) {
        int idx = warpSize * i + laneId;
        if (masks[i] & (1ull << laneId)) {
          // call the operator only if the lane is mentioned in the mask
          T& result = output[idx];
          result = )" +
              intrinsicName + R"((masks[i], input[laneId]);
        }
      }
   })";

  HIPRTC_CHECK(
      hiprtcCreateProgram(&prog, kernelStr.c_str(), "warp_reduce.hip", 0, nullptr, nullptr));
  compileProgram<std::plus>(prog, types);


  for (int i = 0; i < numReduces; i++) {
    unsigned long long mask = 0ull;
    unsigned char* maskAsUChar = reinterpret_cast<unsigned char*>(&mask);

    // map each bit of maskCounter to a byte of mask
    // e.g. 101 --> 0xFF00FF
    for (int i = 0; i < sizeof(maskCounter) * 8; i++) {
      if (maskCounter & (1 << i)) {
        maskAsUChar[i] = 0xFF;
      }
    }

    maskCounter++;
    maskCounter = (maskCounter && (maskCounter < std::pow(2, getWarpSize() / 8 )))? maskCounter : 1;
    masks.host_ptr()[i] = mask;
  }

  HIP_CHECK(hipMemcpy(d_masks.ptr(), masks.ptr(), masks.size_bytes(), hipMemcpyHostToDevice));
  runRtcReduceOp(prog, d_output.ptr(), d_input.ptr(), d_masks.ptr(), numReduces, op);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(output.ptr(), d_output.ptr(), d_output.size_bytes(), hipMemcpyDeviceToHost));

  while (numReduce < numReduces) {
    TestType expectedByLane[64];
    TestType expected = calculateExpected<TestType>(expectedByLane,
                                      input.ptr(),
                                      op,
                                      masks.ptr()[numReduce],
                                      AggregationType::Reduce);
    int lane = 0;

    while (lane < wavefrontSize) {
      auto result = output.ptr()[numReduce * wavefrontSize + lane];
      unsigned long long mask = masks.ptr()[numReduce];

      if ((1ull << lane) & mask) {
        if constexpr (std::is_integral<TestType>::value || std::is_same<std::plus<TestType>, MinOp<TestType>>::value ||
                      std::is_same<std::plus<TestType>, MaxOp<TestType>>::value) {
          // for integral types or min/max the result should match exactly
          if constexpr (std::is_same<TestType, __half>::value)
            REQUIRE(__half2float(result) == __half2float(expected));
          else {
            if (result != expected) {
              printMismatch(result, expected, input.ptr(), mask, lane);
              INFO("Operator: " << opName << " mask: 0x" << std::hex << mask);
              REQUIRE(result == expected);
            }
          }
        } else
          compareFloatingPoint<std::plus<TestType>>(result, expected, mask, input.ptr(), lane);

      }
      lane++;
    }
    numReduce++;
  }
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
}
