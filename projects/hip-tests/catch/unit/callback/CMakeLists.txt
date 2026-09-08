# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Common Tests - Test independent of all platforms

set(TEST_SRC
    hipApiName.cc
    hipGetStreamDeviceId.cc
    hipKernelNameRef.cc
    hipKernelNameRefByPtr.cc
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/SimpleKernel.code
                  COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                  -x hip ${CMAKE_CURRENT_SOURCE_DIR}/SimpleKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/SimpleKernel.code
                  -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/SimpleKernel.cc
                  COMMENT "Compiling SimpleKernel.code")
add_custom_target(SimpleKernel_code DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/SimpleKernel.code)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/SimpleKernel.code)


hip_add_exe_to_target(NAME CallbackTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

add_dependencies(CallbackTest SimpleKernel_code)
