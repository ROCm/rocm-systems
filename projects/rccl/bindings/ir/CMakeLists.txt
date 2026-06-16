# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# Adapted from NVIDIA NCCL ir/CMakeLists.txt (v2.29.2-1).
#
# See LICENSE.txt for license information
#
# LLVM IR / bitcode generation for RCCL device APIs (HIP / AMDGPU).
#
# This file is meant to be pulled in from the top-level RCCL CMakeLists.txt
# via:
#     option(EMIT_LLVM_IR "Generate LLVM IR/bitcode for device APIs" OFF)
#     if(EMIT_LLVM_IR)
#       add_subdirectory(bindings/ir)
#       add_dependencies(llvm_ir copy_nccl_device_headers)
#       add_custom_target(rccl_with_ir ALL DEPENDS rccl llvm_ir)
#     endif()
#
# The output mirrors the standalone bindings/ir/Makefile:
#   ${CMAKE_BINARY_DIR}/lib/librccl_device.bc          (final clean bitcode)
#   ${CMAKE_BINARY_DIR}/obj/llvm_ir/*.bc.{un,}optimized
#   ${CMAKE_BINARY_DIR}/obj/llvm_ir/librccl_device.ll  (textual IR)
#   ${CMAKE_BINARY_DIR}/include/nccl_device_wrapper.h  (public C wrapper)

##### Source and output paths #####
set(LLVM_SRC      ${CMAKE_CURRENT_SOURCE_DIR}/nccl_device_wrapper__impl.h)
set(OBJDIR        ${CMAKE_BINARY_DIR}/obj/llvm_ir)
set(LIBDIR        ${CMAKE_BINARY_DIR}/lib)
set(INCLUDEDIR    ${CMAKE_BINARY_DIR}/include)

set(UNOPTIMIZED_BC ${OBJDIR}/librccl_device.bc.unoptimized)
set(OPTIMIZED_BC   ${OBJDIR}/librccl_device.bc.optimized)
set(LLVM_IR_FILE   ${OBJDIR}/librccl_device.ll)
set(FINAL_BC       ${LIBDIR}/librccl_device.bc)

set(WRAPPER_HEADER ${INCLUDEDIR}/nccl_device_wrapper.h)

##### Build configuration #####
# Default GPU architecture for the bitcode artifact. Override via:
#     cmake -DEMIT_LLVM_IR=ON -DBITCODE_LIB_ARCH=gfx942 ...
set(BITCODE_LIB_ARCH "gfx950" CACHE STRING
    "AMDGPU architecture for the LLVM bitcode (e.g. gfx942, gfx950)")
set(BITCODE_CXX_STD  "c++17"  CACHE STRING
    "C++ standard for the LLVM bitcode build")

# ROCM_PATH is normally set by the top-level CMakeLists.txt; provide a
# safe default so this file is also usable standalone (cmake -P style).
if(NOT DEFINED ROCM_PATH OR ROCM_PATH STREQUAL "")
  set(ROCM_PATH "/opt/rocm")
endif()
set(ROCM_LLVM_BIN "${ROCM_PATH}/llvm/bin")

##### Locate ROCm LLVM toolchain #####
# RCCL's cmake_minimum_required is 3.16, which predates
# `find_program(... REQUIRED)`, so check each one explicitly.
find_program(CLANG_EXECUTABLE    NAMES amdclang++ clang++ HINTS ${ROCM_LLVM_BIN})
find_program(OPT_EXECUTABLE      NAMES opt                HINTS ${ROCM_LLVM_BIN})
find_program(LLVM_DIS_EXECUTABLE NAMES llvm-dis           HINTS ${ROCM_LLVM_BIN})
find_program(LLVM_AS_EXECUTABLE  NAMES llvm-as            HINTS ${ROCM_LLVM_BIN})

foreach(_tool CLANG_EXECUTABLE OPT_EXECUTABLE LLVM_DIS_EXECUTABLE LLVM_AS_EXECUTABLE)
  if(NOT ${_tool})
    message(FATAL_ERROR
      "RCCL IR (EMIT_LLVM_IR=ON): could not locate ${_tool} under "
      "${ROCM_LLVM_BIN}. Install the ROCm LLVM toolchain or override "
      "${_tool} on the command line.")
  endif()
endforeach()

message(STATUS "RCCL IR: clang     = ${CLANG_EXECUTABLE}")
message(STATUS "RCCL IR: opt       = ${OPT_EXECUTABLE}")
message(STATUS "RCCL IR: llvm-dis  = ${LLVM_DIS_EXECUTABLE}")
message(STATUS "RCCL IR: llvm-as   = ${LLVM_AS_EXECUTABLE}")
message(STATUS "RCCL IR: target    = ${BITCODE_LIB_ARCH} (${BITCODE_CXX_STD})")

##### Include paths #####
# Point at the hipify staging directory so this build sees exactly the
# same headers librccl.so is compiled from. HIPIFY_DIR is defined in
# src/CMakeLists.txt without CACHE/PARENT_SCOPE so it isn't visible from
# this sibling subdirectory; we re-derive the same well-known path.
# The dependency on the hipify pass having actually run must be wired
# up by the parent (`add_dependencies(llvm_ir copy_nccl_device_headers)`).
set(IR_HIPIFY_DIR "${PROJECT_BINARY_DIR}/hipify")

set(RCCL_INCLUDES
  -I${CMAKE_BINARY_DIR}/include              # generated rccl.h / nccl.h
  -I${IR_HIPIFY_DIR}/src/include
  -I${IR_HIPIFY_DIR}/src/include/nccl_device
  -I${IR_HIPIFY_DIR}/src/device
)

set(ROCM_INCLUDES
  -I${ROCM_PATH}/include
)

##### Compiler flags #####
# HIP device-only compilation. --offload-device-only is the modern,
# target-agnostic spelling of NCCL's --cuda-device-only.
set(COMMON_CLANG_FLAGS
  -std=${BITCODE_CXX_STD}
  -x hip
  --rocm-path=${ROCM_PATH}
  --offload-device-only
  --offload-arch=${BITCODE_LIB_ARCH}
  ${RCCL_INCLUDES}
  ${ROCM_INCLUDES}
  -D__clang_llvm_bitcode_lib__
  -D__HIP_PLATFORM_AMD__=1
)

set(CLANG_FLAGS
  -c -emit-llvm -O1
  ${COMMON_CLANG_FLAGS}
)

##### Stage the public wrapper header into build/include #####
add_custom_command(
  OUTPUT ${WRAPPER_HEADER}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${INCLUDEDIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/nccl_device_wrapper.h ${WRAPPER_HEADER}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/nccl_device_wrapper.h
  COMMENT "Staging nccl_device_wrapper.h -> ${WRAPPER_HEADER}"
  VERBATIM
)

##### Step 1: clang -> unoptimized bitcode #####
add_custom_command(
  OUTPUT ${UNOPTIMIZED_BC}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OBJDIR}
  COMMAND ${CLANG_EXECUTABLE} ${CLANG_FLAGS} ${LLVM_SRC} -o ${UNOPTIMIZED_BC}
  # Track both the impl TU (clang's direct input) and the staged wrapper
  # header it transitively includes. Without the WRAPPER_HEADER dep,
  # editing nccl_device_wrapper.h would not retrigger this command.
  DEPENDS ${LLVM_SRC} ${WRAPPER_HEADER}
  COMMENT "Generating unoptimized LLVM bitcode for ${BITCODE_LIB_ARCH}..."
  VERBATIM
)

##### Step 2: opt -> optimized bitcode #####
# Internalize everything outside the public surface, then inline and drop
# dead globals. NCCL's recipe uses "-internalize-public-api-list=nccl*",
# but opt's `internalize` pass takes EXACT symbol names, not globs -- the
# literal token "nccl*" matches no symbol, so every wrapper thunk gets
# internalized and dropped to internal linkage. We list the public surface
# explicitly instead; keep this set in sync with the NCCL_IR_EXTERN_C
# entries in nccl_device_wrapper.h.
set(RCCL_DEVICE_BC_PUBLIC_API
    # [A] Always-on
    ncclGetPeerPointerTeam
    # [B] ncclCoopAny + LSA barrier session
    ncclCoopAnyInitThread
    ncclCoopAnyInitWarp
    ncclCoopAnyInitLanes
    ncclCoopAnyInitWarpSpan
    ncclCoopAnyInitCta
    ncclCoopThreadRank
    ncclCoopSize
    ncclCoopNumThreads
    ncclCoopSync
    ncclLsaBarrierSessionInit
    ncclLsaBarrierSessionArrive
    ncclLsaBarrierSessionWait
    ncclLsaBarrierSessionSync
)
string(REPLACE ";" "," RCCL_DEVICE_BC_PUBLIC_API_CSV "${RCCL_DEVICE_BC_PUBLIC_API}")

add_custom_command(
  OUTPUT ${OPTIMIZED_BC}
  COMMAND ${OPT_EXECUTABLE}
          --passes=internalize,inline,globaldce
          -internalize-public-api-list=${RCCL_DEVICE_BC_PUBLIC_API_CSV}
          ${UNOPTIMIZED_BC} -o ${OPTIMIZED_BC}
  DEPENDS ${UNOPTIMIZED_BC}
  COMMENT "Optimizing LLVM bitcode (internalize/inline/globaldce)..."
  VERBATIM
)

##### Step 3: llvm-dis -> textual IR (kept for inspection) #####
# AMDGPU has no equivalent of NCCL's "nvvm-reflect-ftz" metadata, so
# the awk/sed scrubbing pass that NCCL applies between dis and as is
# intentionally omitted.
add_custom_command(
  OUTPUT ${LLVM_IR_FILE}
  COMMAND ${LLVM_DIS_EXECUTABLE} ${OPTIMIZED_BC} -o ${LLVM_IR_FILE}
  DEPENDS ${OPTIMIZED_BC}
  COMMENT "Disassembling bitcode -> textual LLVM IR..."
  VERBATIM
)

##### Step 4: llvm-as -> final bitcode artifact in build/lib #####
add_custom_command(
  OUTPUT ${FINAL_BC}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBDIR}
  COMMAND ${LLVM_AS_EXECUTABLE} ${LLVM_IR_FILE} -o ${FINAL_BC}
  DEPENDS ${LLVM_IR_FILE}
  COMMENT "Assembling final bitcode -> librccl_device.bc..."
  VERBATIM
)

##### Public target #####
add_custom_target(llvm_ir
  DEPENDS ${FINAL_BC} ${OPTIMIZED_BC} ${WRAPPER_HEADER}
  COMMENT "RCCL LLVM IR / bitcode generation"
)

##### Post-build summary #####
add_custom_command(TARGET llvm_ir POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "RCCL LLVM IR and bitcode generated successfully:"
  COMMAND ${CMAKE_COMMAND} -E echo "  C++ Standard:      ${BITCODE_CXX_STD}"
  COMMAND ${CMAKE_COMMAND} -E echo "  GPU Architecture:  ${BITCODE_LIB_ARCH}"
  COMMAND ${CMAKE_COMMAND} -E echo "  Unoptimized BC:    ${UNOPTIMIZED_BC}"
  COMMAND ${CMAKE_COMMAND} -E echo "  Optimized BC:      ${OPTIMIZED_BC}"
  COMMAND ${CMAKE_COMMAND} -E echo "  LLVM IR (textual): ${LLVM_IR_FILE}"
  COMMAND ${CMAKE_COMMAND} -E echo "  Final BC:          ${FINAL_BC}"
  COMMAND ${CMAKE_COMMAND} -E echo "  Wrapper Header:    ${WRAPPER_HEADER}"
  VERBATIM
)
