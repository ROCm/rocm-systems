###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# Builds HSACOs for the device_bitcode functional test (kernel + rocshmem
# device bitcode). Included from tests/functional_tests/CMakeLists.txt.
#
# Builds one HSACO per architecture in BITCODE_GPU_ARCHS (set by DeviceBitcode.cmake).
# At runtime, device_bitcode_tester.cpp selects the HSACO matching the local GPU
# and skips gracefully if none is found.

# Verify rocshmem_device_bitcode target exists (created by DeviceBitcode.cmake)
if(NOT TARGET rocshmem_device_bitcode)
  message(WARNING "device_bitcode_tester: rocshmem_device_bitcode target not found. "
    "HSACOs will not be built; test will skip at runtime.")
  return()
endif()

if(NOT BITCODE_GPU_ARCHS)
  message(WARNING "device_bitcode_tester: BITCODE_GPU_ARCHS is empty. "
    "HSACOs will not be built; test will skip at runtime.")
  return()
endif()

# LLVM_CLANG and LLVM_LINK are already set by DeviceBitcode.cmake.
# Search for additional tools needed for HSACO generation.
find_program(LLVM_LLD ld.lld PATHS ${ROCM_PATH}/llvm/bin ${THEROCK_TOOLCHAIN_ROOT}/lib/llvm/bin NO_DEFAULT_PATH QUIET)
find_program(LLVM_OPT opt PATHS ${ROCM_PATH}/llvm/bin ${THEROCK_TOOLCHAIN_ROOT}/lib/llvm/bin NO_DEFAULT_PATH QUIET)

if(NOT LLVM_LLD OR NOT LLVM_OPT)
  message(WARNING "device_bitcode_tester: ld.lld/opt not found (ROCM_PATH=${ROCM_PATH}). "
    "HSACOs will not be built; test will skip at runtime.")
  return()
endif()

# --- HSACO build steps --------------------------------------------------------

set(ALL_TESTER_HSACOS "")

foreach(GPU_ARCH ${BITCODE_GPU_ARCHS})
  set(KERNEL_BC  ${CMAKE_CURRENT_BINARY_DIR}/device_bitcode_tester_kernel_${GPU_ARCH}.bc)
  set(LINKED_BC  ${CMAKE_CURRENT_BINARY_DIR}/device_bitcode_tester_kernel_${GPU_ARCH}_linked.bc)
  set(OBJ_FILE   ${CMAKE_CURRENT_BINARY_DIR}/device_bitcode_tester_kernel_${GPU_ARCH}.o)
  set(HSACO_FILE ${CMAKE_CURRENT_BINARY_DIR}/device_bitcode_tester_kernel_${GPU_ARCH}.hsaco)

  # Find the full arch string (with feature suffixes) for this base arch and
  # pass it directly via --offload-arch= to the kernel/device-source compile
  # steps below, so the frontend embeds the correct amdhsa.target metadata
  # from the start. Without features the amdhsa.target metadata in the HSACO
  # omits the suffix, causing hipModuleLoadData error 209 on devices that
  # report e.g. gfx950:sramecc+:xnack-.
  set(_FULL_ARCH "${GPU_ARCH}")
  foreach(_candidate ${BITCODE_GPU_ARCHS_FULL})
    string(REGEX REPLACE ":.*" "" _candidate_base "${_candidate}")
    if("${_candidate_base}" STREQUAL "${GPU_ARCH}")
      set(_FULL_ARCH "${_candidate}")
      break()
    endif()
  endforeach()

  # The kernel is compiled fresh (it stands in for an end user's own code),
  # then linked directly against BITCODE_OUTPUT_${GPU_ARCH} -- the exact
  # librocshmem_device_${GPU_ARCH}.bc that DeviceBitcode.cmake already built
  # and ships to real users -- instead of independently recompiling
  # BITCODE_SOURCES a second time.
  add_custom_command(
    OUTPUT ${KERNEL_BC}
    COMMAND ${LLVM_CLANG}
      ${BITCODE_COMPILE_FLAGS_BASE}
      --offload-arch=${_FULL_ARCH}
      -c ${CMAKE_CURRENT_SOURCE_DIR}/device_bitcode_tester_kernel.hip
      -o ${KERNEL_BC}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/device_bitcode_tester_kernel.hip
    COMMENT "device_bitcode_tester: compiling kernel for ${GPU_ARCH}"
    VERBATIM
  )

  set(_UNOPT_BC ${CMAKE_CURRENT_BINARY_DIR}/device_bitcode_tester_kernel_${GPU_ARCH}_unopt.bc)

  # rocshmem_device_bitcode is listed explicitly (in addition to the file
  # itself) because the Unix Makefiles generator only resolves custom-command
  # OUTPUT dependencies within the directory that defines them; a bare file
  # dependency on BITCODE_OUTPUT_${GPU_ARCH} (produced in the top-level
  # directory scope by DeviceBitcode.cmake) has no rule when recursed into
  # from this directory's own generated Makefile on a clean build. Depending
  # on the target instead gives a global build-order edge that works
  # regardless of generator.
  add_custom_command(
    OUTPUT ${_UNOPT_BC}
    COMMAND ${LLVM_LINK}
      ${KERNEL_BC}
      ${BITCODE_OUTPUT_${GPU_ARCH}}
      -o ${_UNOPT_BC}
    DEPENDS ${KERNEL_BC} ${BITCODE_OUTPUT_${GPU_ARCH}} rocshmem_device_bitcode
    COMMENT "device_bitcode_tester: linking kernel against rocshmem device bitcode for ${GPU_ARCH}"
    VERBATIM
  )

  # Optimize the merged BC at -O3 so the final HSACO has efficient code.
  add_custom_command(
    OUTPUT ${LINKED_BC}
    COMMAND ${LLVM_OPT}
      -O3
      -mtriple=amdgcn-amd-amdhsa
      -mcpu=${GPU_ARCH}
      ${_UNOPT_BC}
      -o ${LINKED_BC}
    DEPENDS ${_UNOPT_BC}
    COMMENT "device_bitcode_tester: optimizing merged BC for ${GPU_ARCH}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${LLVM_CLANG}
      -target amdgcn-amd-amdhsa
      -mcpu=${GPU_ARCH}
      -mllvm -amdgpu-internalize-symbols=false
      -x ir
      -c ${LINKED_BC}
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

  list(APPEND ALL_TESTER_HSACOS ${HSACO_FILE})

  rocm_install(FILES ${HSACO_FILE} COMPONENT tests
    DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem)

endforeach()

add_custom_target(device_bitcode_tester_hsacos ALL
  DEPENDS ${ALL_TESTER_HSACOS}
)

add_dependencies(${PROJECT_NAME} device_bitcode_tester_hsacos)

message(STATUS "Device bitcode test (in rocshmem_functional_tests) enabled for: ${BITCODE_GPU_ARCHS}")
