# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    hipGetDeviceProp.cc
    hipEmulationTypes.cc
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
  add_custom_target(emulation_types_c_custom
    COMMAND ${CMAKE_C_COMPILER}
            -c -std=c11 -x c ${CMAKE_CURRENT_SOURCE_DIR}/hipEmulationTypes.c
            -I${HIP_INCLUDE_DIR}
            -D${PLATFORM_DEFINE}
            -o hipEmulationTypes.o
    BYPRODUCTS hipEmulationTypes.o
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
  add_custom_target(emulation_types_c_custom
    COMMAND ${CMAKE_C_COMPILER}
            -c -std=c11 -x c ${CMAKE_CURRENT_SOURCE_DIR}/hipEmulationTypes.c
            -D${PLATFORM_DEFINE}
            -I${HIP_INCLUDE_DIR}
            -I${CUDA_INCLUDE_DIR}
            -o hipEmulationTypes.o
    BYPRODUCTS hipEmulationTypes.o
)

endif()

add_library(devprop_c OBJECT IMPORTED)
set_property(TARGET devprop_c PROPERTY IMPORTED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/hipGetDeviceProp.o")

add_library(emulation_types_c OBJECT IMPORTED)
set_property(TARGET emulation_types_c PROPERTY IMPORTED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/hipEmulationTypes.o")

hip_add_exe_to_target(NAME CTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS devprop_c emulation_types_c)
set_source_files_properties(${TEST_SRC} PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})

add_dependencies(CTests devprop_c_custom emulation_types_c_custom)
