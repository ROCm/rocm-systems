###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# Builds HSACOs for the device_bitcode functional test (kernel + rocshmem device
# bitcode). Included from tests/functional_tests/CMakeLists.txt. Depends on
# BITCODE_GPU_ARCHS and librocshmem device bitcode from cmake/DeviceBitcode.cmake.
#
# HSACOs are built only for the local GPU architecture. If the HSACO cannot be
# built (LLVM tools missing or no matching bitcode), the test binary detects the
# missing HSACO at runtime and skips gracefully.

# Prefer ROCm LLVM so we match the bitcode toolchain instead of system clang;
find_program(LLVM_CLANG clang++ PATHS ${ROCM_PATH}/llvm/bin NO_DEFAULT_PATH)
find_program(LLVM_LINK llvm-link PATHS ${ROCM_PATH}/llvm/bin NO_DEFAULT_PATH)
find_program(LLVM_LLC llc PATHS ${ROCM_PATH}/llvm/bin NO_DEFAULT_PATH)
find_program(LLVM_LLD ld.lld PATHS ${ROCM_PATH}/llvm/bin NO_DEFAULT_PATH)

if(NOT LLVM_CLANG OR NOT LLVM_LINK OR NOT LLVM_LLC OR NOT LLVM_LLD)
  message(WARNING "LLVM tools not found (ROCM_PATH=${ROCM_PATH}). "
                  "device_bitcode tester HSACOs will not be built; test will skip at runtime.")
  return()
endif()

set(BITCODE_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/device_bitcode_tester_kernel.hip)
set(BITCODE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Detect the local GPU architecture (e.g. gfx942) and check it has matching
# device bitcode. All GPUs on a test machine are the same arch.
set(_LOCAL_GPU "NOTFOUND")
if(COMMAND rocm_local_targets)
  rocm_local_targets(_LOCAL_GPU)
endif()

if(_LOCAL_GPU STREQUAL "NOTFOUND" OR NOT _LOCAL_GPU)
  message(WARNING "device_bitcode_tester: could not detect local GPU. "
                  "HSACOs not built; test will skip at runtime.")
  return()
endif()

# Strip target features (e.g. gfx942:sramecc+:xnack- -> gfx942).
list(GET _LOCAL_GPU 0 _LOCAL_GPU)
string(REGEX REPLACE ":.*" "" TESTER_GPU_ARCH "${_LOCAL_GPU}")

# Verify the local arch has a matching device bitcode library.
if(NOT TESTER_GPU_ARCH IN_LIST BITCODE_GPU_ARCHS)
  message(WARNING "device_bitcode_tester: local GPU (${TESTER_GPU_ARCH}) has no matching "
                  "device bitcode (built for ${BITCODE_GPU_ARCHS}). HSACO not built; test will skip at runtime.")
  return()
endif()

set(KERNEL_BC    ${BITCODE_OUTPUT_DIR}/device_bitcode_tester_kernel_${TESTER_GPU_ARCH}.bc)
set(LINKED_BC    ${BITCODE_OUTPUT_DIR}/device_bitcode_tester_kernel_${TESTER_GPU_ARCH}_linked.bc)
set(OBJ_FILE     ${BITCODE_OUTPUT_DIR}/device_bitcode_tester_kernel_${TESTER_GPU_ARCH}.o)
set(HSACO_FILE   ${BITCODE_OUTPUT_DIR}/device_bitcode_tester_kernel_${TESTER_GPU_ARCH}.hsaco)
set(DEVICE_LIB   ${CMAKE_BINARY_DIR}/librocshmem_device_${TESTER_GPU_ARCH}.bc)

add_custom_command(
  OUTPUT ${KERNEL_BC}
  COMMAND ${LLVM_CLANG}
    -x hip --cuda-device-only -std=c++20 -emit-llvm
    --offload-arch=${TESTER_GPU_ARCH}
    -fvisibility=default
    -c ${BITCODE_KERNEL_SRC}
    -o ${KERNEL_BC}
  DEPENDS ${BITCODE_KERNEL_SRC}
  COMMENT "device_bitcode_tester: compiling kernel for ${TESTER_GPU_ARCH}"
  VERBATIM
)

add_custom_command(
  OUTPUT ${LINKED_BC}
  COMMAND ${LLVM_LINK}
    ${KERNEL_BC}
    --override=${DEVICE_LIB}
    -o ${LINKED_BC}
  DEPENDS ${KERNEL_BC} ${DEVICE_LIB} rocshmem_device_bitcode
  COMMENT "device_bitcode_tester: linking with device bitcode for ${TESTER_GPU_ARCH}"
  VERBATIM
)

add_custom_command(
  OUTPUT ${OBJ_FILE}
  COMMAND ${LLVM_LLC}
    -mtriple=amdgcn-amd-amdhsa
    -mcpu=${TESTER_GPU_ARCH}
    --amdhsa-code-object-version=4
    --amdgpu-internalize-symbols=false
    -filetype=obj
    ${LINKED_BC}
    -o ${OBJ_FILE}
  DEPENDS ${LINKED_BC}
  COMMENT "device_bitcode_tester: compiling to object for ${TESTER_GPU_ARCH}"
  VERBATIM
)

add_custom_command(
  OUTPUT ${HSACO_FILE}
  COMMAND ${LLVM_LLD} -shared ${OBJ_FILE} -o ${HSACO_FILE}
  DEPENDS ${OBJ_FILE}
  COMMENT "device_bitcode_tester: linking HSACO for ${TESTER_GPU_ARCH}"
  VERBATIM
)

add_custom_target(device_bitcode_tester_hsacos ALL
  DEPENDS ${HSACO_FILE}
)

add_dependencies(${PROJECT_NAME} device_bitcode_tester_hsacos)

rocm_install(FILES ${HSACO_FILE} COMPONENT tests
  DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem)

message(STATUS "Device bitcode test (in rocshmem_functional_tests) enabled for: ${TESTER_GPU_ARCH}")
