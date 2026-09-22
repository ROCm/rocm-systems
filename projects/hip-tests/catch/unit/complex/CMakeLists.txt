# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Common Tests - Test independent of all platforms

set(TEST_SRC
    complex.cc
)

if(HIP_PLATFORM MATCHES "nvidia")
  set(LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
  set(LINKER_LIBS hiprtc::hiprtc)
endif()

hip_add_exe_to_target(NAME ComplexTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS ${LINKER_LIBS})
if(UNIX)
  file(GLOB NEGATIVE_TEST_SRC
      "complex_make_negative_kernels.cc"
      "complex_cast_negative_kernels.cc"
      "complex_negative_kernels_1Arg_float.cc"
      "complex_negative_kernels_1Arg_double.cc"
      "complex_negative_kernels_2Arg_float.cc"
      "complex_negative_kernels_2Arg_double.cc"
      "complex_negative_kernels_3Arg.cc"
       )

  file(COPY ${NEGATIVE_TEST_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_SRC_FILES ${NEGATIVE_TEST_SRC})

  #add_test(NAME Unit_Device_Complex_make_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         complex_make_negative_kernels.cc 54)
  #
  #add_test(NAME Unit_Device_Complex_Cast_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         complex_cast_negative_kernels.cc 28)
  #
  #add_test(NAME Unit_Device_Complex_Unary_float_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         complex_negative_kernels_1Arg_float.cc 62)
  #
  #add_test(NAME Unit_Device_Complex_Unary_double_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         complex_negative_kernels_1Arg_double.cc 62)
  #
  #add_test(NAME Unit_Device_Complex_Binary_float_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         complex_negative_kernels_2Arg_float.cc 88)
  #
  #add_test(NAME Unit_Device_Complex_Binary_double_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         complex_negative_kernels_2Arg_double.cc 88)
  #
  #add_test(NAME Unit_Device_Complex_hipCfma_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         complex_negative_kernels_3Arg.cc 60)
endif()
