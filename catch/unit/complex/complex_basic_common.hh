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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <hip/hip_complex.h>

template <typename T>
__host__ __device__ T MakeComplexType(decltype(T().x) input_val1, decltype(T().x) input_val2) {
  if constexpr (std::is_same_v<T, hipFloatComplex>) {
    return make_hipFloatComplex(input_val1, input_val2);
  } else {
    return make_hipDoubleComplex(input_val1, input_val2);
  }
}

template <typename T>
__global__ void MakeComplexTypeKernel(T* const output_val, decltype(T().x) const input_val1,
                                      decltype(T().x) const input_val2) {
  *output_val = MakeComplexType<T>(input_val1, input_val2);
}
#if HT_AMD  // EXSWHTEC-321
__global__ void MakeHipComplexTypeKernel(hipComplex* const output_val, float const input_val1,
                                         float const input_val2) {
  *output_val = make_hipComplex(input_val1, input_val2);
}
#endif
template <typename T> struct CastType {};

template <> struct CastType<hipFloatComplex> {
  using type = hipDoubleComplex;
};

template <> struct CastType<hipDoubleComplex> {
  using type = hipFloatComplex;
};

template <typename T> using CastType_t = typename CastType<T>::type;

template <typename T1, typename T2> __device__ __host__ T1 CastComplexType(T2 const input_val) {
  if constexpr (std::is_same_v<hipDoubleComplex, T2>) {
    return hipComplexDoubleToFloat(input_val);
  } else if constexpr (std::is_same_v<hipFloatComplex, T2>) {
    return hipComplexFloatToDouble(input_val);
  }
}

template <typename T1, typename T2>
__global__ void CastComplexTypeKernel(T1* const output_val, T2 const input_val) {
  *output_val = CastComplexType<T1, T2>(input_val);
}

template <typename T> void CompareValues(T actual_val, T ref_val, double margin) {
  if (!std::isnan(ref_val)) {
    REQUIRE_THAT(actual_val, Catch::WithinAbs(ref_val, margin));
  }
}

template <int expected_errors_num> void ComplexTypeRTCWrapper(const char* program_source) {
  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, program_source, "complex_type_kernels.cc", 0, nullptr,
                                   nullptr));

#if HT_AMD
  std::string args = std::string("-ferror-limit=100");
  const char* options[] = {args.c_str()};
  hiprtcResult result{hiprtcCompileProgram(program, 1, options)};
#else
  // read from CMakeLists.txt file add_definitions
  std::string hip_path = HIP_PATH;
  // nvcc errors if headers with relative paths are included
  std::string opts_pre = "--pre-include="+ hip_path + "/include/hip/nvidia_detail/nvidia_hip_complex.h";
  std::string opts_pos = "--include-path=/usr/local/cuda/include";
  const char* options[] = {"-default-device" , opts_pre.c_str(), opts_pos.c_str()};
  hiprtcResult result{hiprtcCompileProgram(program, 3, options)};
#endif

  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};

  std::string error_message{"error:"};

  size_t npos_e = log.find(error_message, 0);
  while (npos_e != std::string::npos) {
    ++error_count;
    npos_e = log.find(error_message, npos_e + 1);
  }

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  HIPRTC_CHECK_ERROR(result, HIPRTC_ERROR_COMPILATION);
  REQUIRE(error_count == expected_errors_num);
}
