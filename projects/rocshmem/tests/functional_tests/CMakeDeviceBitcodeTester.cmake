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

  # Find the full arch string (with feature suffixes) for this base arch so we can
  # pass target-feature flags. Without features the amdhsa.target metadata in the
  # HSACO omits the suffix, causing hipModuleLoadData error 209 on devices that
  # report e.g. gfx950:sramecc+:xnack-.
  set(_FULL_ARCH "${GPU_ARCH}")
  foreach(_candidate ${BITCODE_GPU_ARCHS_FULL})
    string(REGEX REPLACE ":.*" "" _candidate_base "${_candidate}")
    if("${_candidate_base}" STREQUAL "${GPU_ARCH}")
      set(_FULL_ARCH "${_candidate}")
      break()
    endif()
  endforeach()
  arch_features_to_target_feature_flags("${_FULL_ARCH}" _CLANG_MATTR_FLAGS)

  # The device API functions (rocshmem_my_pe, rocshmem_putmem, etc.) are plain
  # __device__ functions. When compiled at -O3 independently, LLVM DCEs them
  # because no amdgpu_kernel in the same TU calls them. Compiling at -O0
  # avoids DCE but produces unoptimized IR patterns that trigger an AMDGPU
  # backend register-class bug (V_CMP_NE_U32 on $src_private_base).
  #
  # Solution: compile with -Xclang -disable-llvm-passes, which runs the
  # frontend at -O3 (generating valid, structured IR) but skips the LLVM
  # optimization and DCE passes. The resulting BC retains all device function
  # bodies. After linking with the kernel (which provides callers), a final
  # opt -O3 pass over the merged BC optimizes everything together.

  # Device sources: suppress LLVM passes so DCE doesn't eliminate __device__
  # functions that have no callers within the same TU. The kernel BC is compiled
  # at plain -O3 (it has an amdgpu_kernel entry point, so nothing gets DCE'd).
  set(_TESTER_DEVICE_FLAGS ${BITCODE_COMPILE_FLAGS_BASE})
  list(APPEND _TESTER_DEVICE_FLAGS -Xclang -disable-llvm-passes)

  # Compile the kernel and each device source into tester-private BCs.
  set(_TESTER_BCS "")

  add_custom_command(
    OUTPUT ${KERNEL_BC}
    COMMAND ${LLVM_CLANG}
      ${BITCODE_COMPILE_FLAGS_BASE}
      --offload-arch=${GPU_ARCH}
      -c ${CMAKE_CURRENT_SOURCE_DIR}/device_bitcode_tester_kernel.hip
      -o ${KERNEL_BC}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/device_bitcode_tester_kernel.hip
    COMMENT "device_bitcode_tester: compiling kernel for ${GPU_ARCH}"
    VERBATIM
  )
  list(APPEND _TESTER_BCS ${KERNEL_BC})

  foreach(_src ${BITCODE_SOURCES})
    get_filename_component(_src_name "${_src}" NAME_WE)
    set(_src_bc ${CMAKE_CURRENT_BINARY_DIR}/tester_device_${GPU_ARCH}_${_src_name}.bc)
    add_custom_command(
      OUTPUT ${_src_bc}
      COMMAND ${LLVM_CLANG}
        ${_TESTER_DEVICE_FLAGS}
        --offload-arch=${GPU_ARCH}
        -c ${_src}
        -o ${_src_bc}
      DEPENDS ${_src}
      COMMENT "device_bitcode_tester: compiling ${_src_name} for ${GPU_ARCH}"
      VERBATIM
    )
    list(APPEND _TESTER_BCS ${_src_bc})
  endforeach()

  set(_UNOPT_BC ${CMAKE_CURRENT_BINARY_DIR}/device_bitcode_tester_kernel_${GPU_ARCH}_unopt.bc)

  add_custom_command(
    OUTPUT ${_UNOPT_BC}
    COMMAND ${LLVM_LINK}
      ${_TESTER_BCS}
      -o ${_UNOPT_BC}
    DEPENDS ${_TESTER_BCS}
    COMMENT "device_bitcode_tester: linking all device sources for ${GPU_ARCH}"
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
      ${_CLANG_MATTR_FLAGS}
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
