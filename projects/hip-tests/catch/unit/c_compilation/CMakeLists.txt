# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    hipGetDeviceProp.cc
    hipBf16Compile.cc
)


if(HIP_PLATFORM MATCHES "nvidia")
  set(PLATFORM_DEFINE __HIP_PLATFORM_NVIDIA__)
  add_compile_definitions(__HIP_PLATFORM_NVIDIA__)
else()
  set(PLATFORM_DEFINE __HIP_PLATFORM_AMD__)
  add_compile_definitions(__HIP_PLATFORM_AMD__)
endif()

# Creating Custom object file
if (HIP_PLATFORM MATCHES "amd")
  add_custom_target(devprop_c_custom
    COMMAND ${CMAKE_C_COMPILER}
            -c -Wno-deprecated-declarations -x c ${CMAKE_CURRENT_SOURCE_DIR}/hipGetDeviceProp.c
            -I${HIP_INCLUDE_DIR}
            -D${PLATFORM_DEFINE}
            -o hipGetDeviceProp.o
    BYPRODUCTS hipGetDeviceProp.o
)
  # Regression guard: hip/hip_bf16.h must compile as plain C.
  add_custom_target(bf16_c_custom
    COMMAND ${CMAKE_C_COMPILER}
            -c -Wno-deprecated-declarations -x c ${CMAKE_CURRENT_SOURCE_DIR}/hipBf16Compile.c
            -I${HIP_INCLUDE_DIR}
            -D${PLATFORM_DEFINE}
            -o hipBf16Compile.o
    BYPRODUCTS hipBf16Compile.o
)
else()
  add_custom_target(devprop_c_custom
    COMMAND ${CMAKE_C_COMPILER}
            -c -Wno-deprecated-declarations -x c ${CMAKE_CURRENT_SOURCE_DIR}/hipGetDeviceProp.c
            -D${PLATFORM_DEFINE}
            -I${HIP_INCLUDE_DIR}
            -I${CUDA_INCLUDE_DIR}
            -o hipGetDeviceProp.o
    BYPRODUCTS hipGetDeviceProp.o
)
  add_custom_target(bf16_c_custom
    COMMAND ${CMAKE_C_COMPILER}
            -c -Wno-deprecated-declarations -x c ${CMAKE_CURRENT_SOURCE_DIR}/hipBf16Compile.c
            -D${PLATFORM_DEFINE}
            -I${HIP_INCLUDE_DIR}
            -I${CUDA_INCLUDE_DIR}
            -o hipBf16Compile.o
    BYPRODUCTS hipBf16Compile.o
)

endif()

add_library(devprop_c OBJECT IMPORTED)
set_property(TARGET devprop_c PROPERTY IMPORTED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/hipGetDeviceProp.o")

add_library(bf16_c OBJECT IMPORTED)
set_property(TARGET bf16_c PROPERTY IMPORTED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/hipBf16Compile.o")

hip_add_exe_to_target(NAME CTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS devprop_c bf16_c)
set_source_files_properties(${TEST_SRC} PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})

add_dependencies(CTests devprop_c_custom bf16_c_custom)
