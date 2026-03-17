# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Common Tests - Test independent of all platforms

set(TEST_SRC
    memcpy.cc
    memset.cc
)

if(HIP_PLATFORM MATCHES "nvidia")
  set(LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
  set(LINKER_LIBS hiprtc::hiprtc)
endif()

hip_add_exe_to_target(NAME DeviceMemoryTest
                    TEST_SRC ${TEST_SRC}
                    TEST_TARGET_NAME build_tests
                    LINKER_LIBS ${LINKER_LIBS}
                    PROPERTY CXX_STANDARD 17)

if(UNIX)
  file(GLOB NEGATIVE_TEST_SRC
     "memcpy_negative_kernels.cc"
     "memset_negative_kernels.cc"
    )

  file(COPY ${NEGATIVE_TEST_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_SRC_FILES ${NEGATIVE_TEST_SRC})

  #add_test(NAME Unit_Device_memcpy_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         memcpy_negative_kernels.cc 4)
  #
  #add_test(NAME Unit_Device_memset_Negative
  #         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
  #         ./src ${HIP_PLATFORM} ${HIP_PATH}
  #         memset_negative_kernels.cc 4)
endif()
