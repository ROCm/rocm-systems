# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


###############################################################################
# HIPRTC.cmake
###############################################################################

# This file includes macros required to generate the hiprtc builtins library.

function(get_hiprtc_macros HIPRTC_DEFINES)
  set(${HIPRTC_DEFINES}
"#define __device__ __attribute__((device))\n\
#define __host__ __attribute__((host))\n\
#define __global__ __attribute__((global))\n\
#define __constant__ __attribute__((constant))\n\
#define __shared__ __attribute__((shared))\n\
#define __align__(x) __attribute__((aligned(x)))\n\
#if !defined(__has_feature) || !__has_feature(cuda_noinline_keyword)\n\
#define __noinline__ __attribute__((noinline))\n\
#endif\n\
#define __forceinline__ inline __attribute__((always_inline))\n\
#if __HIP_NO_IMAGE_SUPPORT\n\
#define __hip_img_chk__ __attribute__((unavailable(\"The image/texture API not supported on the device\")))\n\
#else\n\
#define __hip_img_chk__\n\
#endif\n\
#define launch_bounds_impl0(requiredMaxThreadsPerBlock)                                       \\\n\
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock)))\n\
#define launch_bounds_impl1(requiredMaxThreadsPerBlock, minBlocksPerMultiprocessor)           \\\n\
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock),                \\\n\
                   amdgpu_waves_per_eu(minBlocksPerMultiprocessor)))\n\
#define select_impl_(_1, _2, impl_, ...) impl_\n\
#define __launch_bounds__(...)                                                                \\\n\
    select_impl_(__VA_ARGS__, launch_bounds_impl1, launch_bounds_impl0)(__VA_ARGS__)           \n\
#define HIP_INCLUDE_HIP_HIP_RUNTIME_H\n\
#define _HIP_BFLOAT16_H_\n\
#define HIP_INCLUDE_HIP_MATH_FUNCTIONS_H\n\
#define HIP_INCLUDE_HIP_HIP_VECTOR_TYPES_H\n\
#if !__HIP_NO_STD_DEFS__\n\
#if defined(__HIPRTC_PTRDIFF_T_IS_LONG_LONG__) && __HIPRTC_PTRDIFF_T_IS_LONG_LONG__==1\n\
typedef long long ptrdiff_t;\n\
#else\n\
typedef __PTRDIFF_TYPE__ ptrdiff_t;\n\
#endif\n\
typedef long clock_t;\n\
namespace std {\n\
using ::ptrdiff_t;\n\
using ::clock_t;\n\
}\n\
#endif // __HIP_NO_STD_DEFS__\n"
  PARENT_SCOPE)
endfunction(get_hiprtc_macros)

# To allow concatenating above macros during build time, call this file in script mode.
if(HIPRTC_ADD_MACROS)
# Read the existing content of the preprocessed file into a temporary variable
  FILE(READ "${HIPRTC_PREPROCESSED_FILE}" ORIGINAL_PREPROCESSED_FILE)
# Prepend the push and ignore pragmas to the original preprocessed file
  set(PRAGMA_PUSH "#pragma clang diagnostic push")
  set(PRAGMA_EVERYTHING "#pragma clang diagnostic ignored \"-Weverything\"")
  set(MODIFIED_PREPROCESSED_FILE "${PRAGMA_PUSH}\n${PRAGMA_EVERYTHING}
      \n${ORIGINAL_PREPROCESSED_FILE}")
# Write the modified preprocessed content back to the original file
  FILE(WRITE ${HIPRTC_PREPROCESSED_FILE} "${MODIFIED_PREPROCESSED_FILE}")

  message(STATUS "Appending hiprtc macros to ${HIPRTC_PREPROCESSED_FILE}.")
  get_hiprtc_macros(HIPRTC_DEFINES)
  FILE(APPEND ${HIPRTC_PREPROCESSED_FILE} "${HIPRTC_DEFINES}")
  set(HIPRTC_HEADER_LIST ${HIPRTC_HEADERS})
  separate_arguments(HIPRTC_HEADER_LIST)
# Appends all the headers from the list to the hiprtc preprocessed file
  foreach(header ${HIPRTC_HEADER_LIST})
    FILE(READ "${header}" HEADER_FILE)
    FILE(APPEND ${HIPRTC_PREPROCESSED_FILE} "${HEADER_FILE}")
  endforeach()
# Append the pop pragma to the preprocessed file
  set(PRAGMA_POP "#pragma clang diagnostic pop\n")
  FILE(APPEND ${HIPRTC_PREPROCESSED_FILE} "${PRAGMA_POP}")
endif()

macro(generate_hiprtc_header HiprtcHeader)
  FILE(WRITE ${HiprtcHeader}
"#pragma push_macro(\"CHAR_BIT\")\n\
#pragma push_macro(\"INT_MAX\")\n\
#define CHAR_BIT __CHAR_BIT__\n\
#define INT_MAX __INTMAX_MAX__\n\
#include \"hip/hip_runtime.h\"\n\
#include \"hip/hip_bfloat16.h\"\n\
#pragma pop_macro(\"CHAR_BIT\")\n\
#pragma pop_macro(\"INT_MAX\")")
endmacro(generate_hiprtc_header)

macro(generate_hiprtc_mcin HiprtcMcin HiprtcPreprocessedInput)
  if(WIN32)
    set(HIPRTC_TYPE_LINUX_ONLY "")
  else()
    set(HIPRTC_TYPE_LINUX_ONLY
      "  .section .note.GNU-stack,\"\",@progbits\n"
      "  .type __hipRTC_header,@object\n"
      "  .type __hipRTC_header_size,@object")
  endif()
  FILE(WRITE ${HiprtcMcin}
"// Automatically generated script for HIPRTC.\n\
${HIPRTC_TYPE_LINUX_ONLY}\n\
  .section .hipRTC_header,\"a\"\n\
  .globl __hipRTC_header\n\
  .globl __hipRTC_header_size\n\
  .p2align 3\n\
__hipRTC_header:\n\
  .incbin \"${HiprtcPreprocessedInput}\"\n\
__hipRTC_header_size:\n\
  .long __hipRTC_header_size - __hipRTC_header\n")
endmacro(generate_hiprtc_mcin)

