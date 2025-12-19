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
#include <tuple>
#include <cmd_options.hh>
#include <functional>
#include <algorithm>

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
  struct {
    const T* d_output;
    const T* d_input;
    const MaskType* d_masks;
    int numReduces;
  } args{output, input, masks, numReduces};
  int size = 4;
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
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
void opToString(std::string& scalarName, std::string& intrinsicName) {
  if constexpr (std::is_same<Op<T>, std::plus<T>>::value) {
    scalarName = "std::plus";
    intrinsicName = "__reduce_add_sync";
  } else if constexpr (std::is_same<Op<T>, MinOp<T>>::value) {
    scalarName = "MinOp";
    intrinsicName = "__reduce_min_sync";
  } else if constexpr (std::is_same<Op<T>, MaxOp<T>>::value) {
    scalarName = "MaxOp";
    intrinsicName = "__reduce_max_sync";
  } else if constexpr (std::is_same<Op<T>, std::logical_and<T>>::value) {
    scalarName = "std::logical_and";
    intrinsicName = "__reduce_and_sync";
  } else if constexpr (std::is_same<Op<T>, std::logical_or<T>>::value) {
    scalarName = "std::logical_or";
    intrinsicName = "__reduce_or_sync";
  } else if constexpr (std::is_same<Op<T>, XorOp<T>>::value) {
    scalarName = "LogicalXor";
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

  opToString<int, Op>(scalarName, intrinsicName);
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

  opToString<int, Op>(scalarName, intrinsicName);
  kernelStr = R"(
    template <class T, class MaskType>
    __global__ void reduceRtcKernel(T* output, const T* input, const MaskType* masks, int numReduces)
    {
      int tid = threadIdx.x;

      for (int i = 0; i < numReduces; i++) {
        if (masks[i] & (1ul << tid)) {
          // call the operator only if the lane is mentioned in the mask
          T& result = output[warpSize * i + tid];
          result = )" +
              intrinsicName + R"((masks[i], input[tid]);
        }
      }
   })";

  HIPRTC_CHECK(
      hiprtcCreateProgram(&prog, kernelStr.c_str(), "warp_reduce.hip", 0, nullptr, nullptr));
  compileProgram<Op>(prog, types);
  runTestReduceForTypes<Op>(prog, types);
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
}

TEST_CASE("Unit_Rtc_ReduceRandom") {
  const std::tuple<int, unsigned int, long long, unsigned long long, float, half, double> allTypes;
  const std::tuple<int, unsigned int, long long, unsigned long long> integralTypes;

  SECTION("add") { runAndCompileTest<std::plus>(allTypes); }

  SECTION("min") { runAndCompileTest<MinOp>(allTypes); }

  SECTION("max") { runAndCompileTest<MaxOp>(allTypes); }

  SECTION("and") { runAndCompileTest<std::logical_and>(integralTypes); }

  SECTION("or") { runAndCompileTest<std::logical_or>(integralTypes); }

  SECTION("xor") { runAndCompileTest<XorOp>(integralTypes); }
}
