###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# Builds HSACOs for the device_bitcode functional test (kernel + rocshmem
# device bitcode). Included from tests/functional_tests/CMakeLists.txt.
#
# Builds one HSACO per architecture in BITCODE_GPU_ARCHS (derived from the
# required -DGPU_TARGETS option). Tests may be dispatched to GPU nodes that
# differ from the build machine, so the presence of a local GPU is not a
# reliable guide — GPU_TARGETS is the authoritative statement of intent.
#
# At runtime, device_bitcode_tester.cpp selects the HSACO matching the local GPU
# and skips gracefully if none is found.

# LLVM_CLANG and LLVM_LINK are already cached by DeviceBitcode.cmake.
# Only search for the additional tools needed for the final HSACO steps.
find_program(LLVM_LLC  llc    PATHS ${ROCM_PATH}/llvm/bin NO_DEFAULT_PATH)
find_program(LLVM_LLD  ld.lld PATHS ${ROCM_PATH}/llvm/bin NO_DEFAULT_PATH)

if(NOT LLVM_CLANG OR NOT LLVM_LINK OR NOT LLVM_LLC OR NOT LLVM_LLD)
  message(WARNING
    "LLVM tools not found (ROCM_PATH=${ROCM_PATH}). "
    "device_bitcode tester HSACOs will not be built; test will skip at runtime.")
  return()
endif()

if(NOT BITCODE_GPU_ARCHS)
  message(WARNING
    "device_bitcode_tester: BITCODE_GPU_ARCHS is empty (GPU_TARGETS not set?). "
    "HSACOs will not be built; test will skip at runtime.")
  return()
endif()

set(_TESTER_ARCHS "${BITCODE_GPU_ARCHS}")

# --- HSACO build steps --------------------------------------------------------

set(BITCODE_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/device_bitcode_tester_kernel.hip)
set(BITCODE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(_TESTER_HSACOS "")

foreach(GPU_ARCH ${_TESTER_ARCHS})
  set(KERNEL_BC  ${BITCODE_OUTPUT_DIR}/device_bitcode_tester_kernel_${GPU_ARCH}.bc)
  set(LINKED_BC  ${BITCODE_OUTPUT_DIR}/device_bitcode_tester_kernel_${GPU_ARCH}_linked.bc)
  set(OBJ_FILE   ${BITCODE_OUTPUT_DIR}/device_bitcode_tester_kernel_${GPU_ARCH}.o)
  set(HSACO_FILE ${BITCODE_OUTPUT_DIR}/device_bitcode_tester_kernel_${GPU_ARCH}.hsaco)
  set(DEVICE_LIB ${CMAKE_BINARY_DIR}/librocshmem_device_${GPU_ARCH}.bc)

  add_custom_command(
    OUTPUT ${KERNEL_BC}
    COMMAND ${LLVM_CLANG}
      -x hip --cuda-device-only -std=c++20 -emit-llvm
      --offload-arch=${GPU_ARCH}
      -fvisibility=default
      -c ${BITCODE_KERNEL_SRC}
      -o ${KERNEL_BC}
    DEPENDS ${BITCODE_KERNEL_SRC}
    COMMENT "device_bitcode_tester: compiling kernel for ${GPU_ARCH}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${LINKED_BC}
    COMMAND ${LLVM_LINK}
      ${KERNEL_BC}
      --override=${DEVICE_LIB}
      -o ${LINKED_BC}
    DEPENDS ${KERNEL_BC} ${DEVICE_LIB} rocshmem_device_bitcode
    COMMENT "device_bitcode_tester: linking with device bitcode for ${GPU_ARCH}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${LLVM_LLC}
      -mtriple=amdgcn-amd-amdhsa
      -mcpu=${GPU_ARCH}
      --amdgpu-internalize-symbols=false
      -filetype=obj
      ${LINKED_BC}
      -o ${OBJ_FILE}
    DEPENDS ${LINKED_BC}
    COMMENT "device_bitcode_tester: compiling to object for ${GPU_ARCH}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${HSACO_FILE}
    COMMAND ${LLVM_LLD} -shared ${OBJ_FILE} -o ${HSACO_FILE}
    DEPENDS ${OBJ_FILE}
    COMMENT "device_bitcode_tester: linking HSACO for ${GPU_ARCH}"
    VERBATIM
  )

  list(APPEND _TESTER_HSACOS ${HSACO_FILE})

  rocm_install(FILES ${HSACO_FILE} COMPONENT tests
    DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem)

endforeach()

add_custom_target(device_bitcode_tester_hsacos ALL
  DEPENDS ${_TESTER_HSACOS}
)

add_dependencies(${PROJECT_NAME} device_bitcode_tester_hsacos)

message(STATUS "Device bitcode test (in rocshmem_functional_tests) enabled for: ${_TESTER_ARCHS}")
