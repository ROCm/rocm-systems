# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
  trig_funcs.cc
  misc_funcs.cc
  remainder_and_rounding_funcs.cc
  single_precision_intrinsics.cc
  double_precision_intrinsics.cc
  integer_intrinsics.cc
  root_funcs.cc
  log_funcs.cc
  special_funcs.cc
  casting_double_funcs.cc
  casting_float_funcs.cc
  casting_int_funcs.cc
  casting_half2int_funcs.cc
  casting_int2half_funcs.cc
  casting_half_float_funcs.cc
)

if(HIP_PLATFORM MATCHES "nvidia")
  set(LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC}
    pow_funcs.cc
    casting_half2_funcs.cc
    half_precision_math.cc
    half_precision_arithmetic.cc
    half_precision_comparison.cc
  )
  set(LINKER_LIBS hiprtc::hiprtc)
endif()

hip_add_exe_to_target(NAME MathsTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMMON_SHARED_SRC ${COMMON_SHARED_SRC}
                      LINKER_LIBS ${LINKER_LIBS})


if(UNIX)
  file(GLOB NEGATIVE_TEST_SRC
    "trig_single_precision_negative_kernels.cc"
    "trig_double_precision_negative_kernels.cc"
    "misc_negative_kernels.cc"
    "math_remainder_negative_kernels.cc"
    "math_rounding_negative_kernels.cc"
    "single_precision_intrinsics_negative_kernels.cc"
    "double_precision_intrinsics_negative_kernels.cc"
    "integer_intrinsics_negative_kernels.cc"
    "math_root_negative_kernels_1Dand2D.cc"
    "math_root_negative_kernels_3Dand4D.cc"
    "math_pow_negative_kernels.cc"
    "math_log_negative_kernels.cc"
    "math_special_func_kernels.cc"
    "casting_double_negative_kernels.cc"
    "casting_float_negative_kernels.cc"
    "casting_int_negative_kernels.cc"
    "casting_half2_negative_kernels.cc"
    "half_precision_math_negative_kernels.cc"
    "half_precision_arithmetic_negative_kernels.cc"
    "half_precision_comparison_negative_kernels.cc"
    "casting_half2int_negative_kernels.cc"
    "casting_int2half_negative_kernels.cc"
    "casting_half_float_negative_kernels.cc"
      )

  file(COPY ${NEGATIVE_TEST_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_SRC_FILES ${NEGATIVE_TEST_SRC})

  #if(HIP_PLATFORM MATCHES "nvidia")
  # set(ERRORS_1 81)
  # set(ERRORS_2 100)
  #elseif(HIP_PLATFORM MATCHES "amd")
  # set(ERRORS_1 76)
  # set(ERRORS_2 168)
  #endif()
  #
  #add_test(NAME Unit_Device_Single_Precision_Trig_Functions_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         trig_single_precision_negative_kernels.cc 66)
  #add_test(NAME Unit_Device_Double_Precision_Trig_Functions_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         trig_double_precision_negative_kernels.cc 66)
  #add_test(NAME Unit_Device_Misc_Functions_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         misc_negative_kernels.cc ${ERRORS_1})
  #add_test(NAME Unit_Device_remainder_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         math_remainder_negative_kernels.cc 67)
  #add_test(NAME Unit_Device_rounding_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         math_rounding_negative_kernels.cc 40)
  #add_test(NAME Unit_Single_Precision_Intrinsics_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         single_precision_intrinsics_negative_kernels.cc 42)
  #add_test(NAME Unit_Double_Precision_Intrinsics_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         double_precision_intrinsics_negative_kernels.cc 18)
  #add_test(NAME Unit_Integer_Intrinsics_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         integer_intrinsics_negative_kernels.cc 20)
  #add_test(NAME Unit_Device_root_1Dand2D_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         math_root_negative_kernels_1Dand2D.cc 68)
  #add_test(NAME Unit_Device_root_3Dand4D_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         math_root_negative_kernels_3Dand4D.cc 56)
  #add_test(NAME Unit_Device_pow_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         math_pow_negative_kernels.cc 76)
  #add_test(NAME Unit_Device_log_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         math_log_negative_kernels.cc 24)
  #add_test(NAME Unit_Device_special_funcs_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         math_special_func_kernels.cc 76)
  #add_test(NAME Unit_Device_casting_double_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         casting_double_negative_kernels.cc 69)
  #add_test(NAME Unit_Device_casting_float_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         casting_float_negative_kernels.cc 54)
  #add_test(NAME Unit_Device_casting_int_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         casting_int_negative_kernels.cc 92)
  #add_test(NAME Unit_Device_casting_half2_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         casting_half2_negative_kernels.cc 53)
  #add_test(NAME Unit_Half_Precision_Math_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         half_precision_math_negative_kernels.cc 60)
  #add_test(NAME Unit_Half_Precision_Arithmetic_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         half_precision_arithmetic_negative_kernels.cc 88)
  #add_test(NAME Unit_Half_Precision_Comparison_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         half_precision_comparison_negative_kernels.cc ${ERRORS_2})
  #add_test(NAME Unit_Device_casting_half2int_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         casting_half2int_negative_kernels.cc 78)
  #add_test(NAME Unit_Device_casting_int2half_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         casting_int2half_negative_kernels.cc 78)
  #add_test(NAME Unit_Device_casting_half_float_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         casting_half_float_negative_kernels.cc 18)
endif()
