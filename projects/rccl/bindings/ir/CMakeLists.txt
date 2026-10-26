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
set(LLVM_SRC      ${CMAKE_CURRENT_SOURCE_DIR}/nccl_device_ir.cu)
set(OBJDIR        ${CMAKE_BINARY_DIR}/obj/llvm_ir)
set(LIBDIR        ${CMAKE_BINARY_DIR}/lib)
set(INCLUDEDIR    ${CMAKE_BINARY_DIR}/include)

set(UNOPTIMIZED_BC ${OBJDIR}/librccl_device.bc.unoptimized)
set(OPTIMIZED_BC   ${OBJDIR}/librccl_device.bc.optimized)
set(LLVM_IR_FILE   ${OBJDIR}/librccl_device.ll)
set(FINAL_BC       ${LIBDIR}/librccl_device.bc)

set(WRAPPER_HEADER ${INCLUDEDIR}/nccl_device_wrapper.h)

##### LTO (deferred-inlining) bitcode artifact #####
# Opt-in: EMIT_LTO_BC=ON additionally emits librccl_device.lto.bc, a bitcode
# variant with no opt internalize/inline/globaldce pass applied, intended
# for downstream AMDGPU LTO consumption (-Xoffload-linker <bc>
# -Xoffload-linker -plugin-opt=-amdgpu-internalize-symbols=false), analogous
# to NCCL's nvcc -dlto --ltoir artifact.
option(EMIT_LTO_BC "Additionally emit a deferred-inlining LTO bitcode variant (librccl_device.lto.bc)" OFF)

set(LTO_BC_RAW   ${OBJDIR}/librccl_device.lto.bc.raw)
set(LTO_IR_FILE  ${OBJDIR}/librccl_device.lto.ll)
set(FINAL_LTO_BC ${LIBDIR}/librccl_device.lto.bc)

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

##### Device API defines #####
# Enable all ReduceCopy type and block-size combinations in generated IR.
set(NCCL_DEVICE_DEFINES
  -DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1
)

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
# Clang supports extended device lambdas but does not define nvcc's feature
# macro; the ReduceCopy device APIs require it.
set(BASE_CLANG_FLAGS
  -std=${BITCODE_CXX_STD}
  -x hip
  --rocm-path=${ROCM_PATH}
  --offload-device-only
  --offload-arch=${BITCODE_LIB_ARCH}
  ${RCCL_INCLUDES}
  ${ROCM_INCLUDES}
  -D__CUDACC_EXTENDED_LAMBDA__
  ${NCCL_DEVICE_DEFINES}
  -D__HIP_PLATFORM_AMD__=1
)

# Both artifacts are the bitcode library, so both define
# __clang_llvm_bitcode_lib__. Beyond inlining policy that macro also selects
# the caller-owned-LDS path in nccl_device/coop.h and extern "C" linkage for
# NCCL_IR_EXTERN_C in nccl_device/utility.h; an artifact that omits it gets a
# function-local addrspace(3) barrier array that the AMDGPU group-segment
# sizing pass cannot see through the vtable indirect call. See the macro
# commentary in nccl_device_wrapper.h.
set(BITCODE_LIB_DEFINE -D__clang_llvm_bitcode_lib__)

# Plain (collapsed) bitcode variant: thunks' callees are always_inline, and
# the opt internalize/inline/globaldce pass below collapses everything
# outside RCCL_DEVICE_BC_PUBLIC_API.
set(COMMON_CLANG_FLAGS
  ${BASE_CLANG_FLAGS}
  ${BITCODE_LIB_DEFINE}
)

# Deferred-inlining ("LTO") variant: same library, one modifier on top.
# __RCCL_DEVICE_LTO_BC_LIB__ drops always_inline from the thunks' callees
# (see NCCL_DEVICE_INLINE in nccl_device_wrapper.h) and NO opt pass runs, so
# a downstream AMDGPU LTO link (lld) can inline across the real call site
# with full visibility.
set(LTO_BC_CLANG_FLAGS
  ${BASE_CLANG_FLAGS}
  ${BITCODE_LIB_DEFINE}
  -D__RCCL_DEVICE_LTO_BC_LIB__
)

set(CLANG_FLAGS
  -c -emit-llvm -O1
  ${COMMON_CLANG_FLAGS}
)

set(LTO_BC_CLANG_COMPILE_FLAGS
  -c -emit-llvm -O1
  ${LTO_BC_CLANG_FLAGS}
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
  # header it transitively includes (plus util.h's type lists). Without
  # the WRAPPER_HEADER dep, editing nccl_device_wrapper.h would not
  # retrigger this command.
  DEPENDS ${LLVM_SRC} ${WRAPPER_HEADER} ${CMAKE_CURRENT_SOURCE_DIR}/util.h
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
    ncclLsaBarrierSession_C_size
    ncclGinBarrierSession_C_size
    ncclBarrierSession_C_size
    ncclDevComm_Rank
    ncclDevComm_NRanks
    ncclDevComm_LsaRank
    ncclDevComm_LsaSize
    ncclDevComm_LsaBarrier
    ncclDevComm_RailGinBarrier
    ncclDevComm_HybridLsaBarrier
    ncclDevComm_HybridRailGinBarrier
    ncclDevComm_WorldGinBarrier
    ncclDevComm_LsaMultimem
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
    ncclLsaBarrierSessionDestroy
    # [C] GIN + composite barrier session
    ncclGinBarrierSessionInit
    ncclGinBarrierSessionInitAllContexts
    ncclGinBarrierSessionSync
    ncclGinBarrierSessionDestroy
    ncclBarrierSessionInit
    ncclBarrierSessionSync
    ncclBarrierSessionDestroy
)
# [D] ReduceCopy: one thunk per op and type, matching NCCL_IR_TYPES /
# NCCL_IR_MULTIMEM_TYPES in util.h.
set(RCCL_IR_ALL_TYPES      I8 U8 I32 U32 I64 U64 F16 F32 F64 BF16)
set(RCCL_IR_MULTIMEM_TYPES I32 U32 I64 U64 F16 F32 F64 BF16)
foreach(_op ncclLsaReduceSum ncclLsaCopy ncclLsaReduceSumCopy ncclLocalReduceSumCopy)
  foreach(_ty ${RCCL_IR_ALL_TYPES})
    list(APPEND RCCL_DEVICE_BC_PUBLIC_API ${_op}_${_ty})
  endforeach()
endforeach()
foreach(_op ncclMultimemReduceSum ncclMultimemCopy ncclMultimemReduceSumCopy)
  foreach(_ty ${RCCL_IR_MULTIMEM_TYPES})
    list(APPEND RCCL_DEVICE_BC_PUBLIC_API ${_op}_${_ty})
  endforeach()
endforeach()
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

##### LTO (deferred-inlining) bitcode custom commands #####
if(EMIT_LTO_BC)
  ##### Step 1: clang -> raw (unoptimized, no opt pass ever runs) bitcode #####
  add_custom_command(
    OUTPUT ${LTO_BC_RAW}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OBJDIR}
    COMMAND ${CLANG_EXECUTABLE} ${LTO_BC_CLANG_COMPILE_FLAGS} ${LLVM_SRC} -o ${LTO_BC_RAW}
    DEPENDS ${LLVM_SRC} ${WRAPPER_HEADER} ${CMAKE_CURRENT_SOURCE_DIR}/util.h
    COMMENT "Generating deferred-inlining (LTO) LLVM bitcode for ${BITCODE_LIB_ARCH}..."
    VERBATIM
  )

  ##### Step 2: llvm-dis -> textual IR (kept for inspection; no opt step) #####
  add_custom_command(
    OUTPUT ${LTO_IR_FILE}
    COMMAND ${LLVM_DIS_EXECUTABLE} ${LTO_BC_RAW} -o ${LTO_IR_FILE}
    DEPENDS ${LTO_BC_RAW}
    COMMENT "Disassembling LTO bitcode -> textual LLVM IR..."
    VERBATIM
  )

  ##### Step 3: llvm-as -> final LTO bitcode artifact in build/lib #####
  # Deliberately does NOT depend on ${FINAL_BC}. The two artifacts are
  # independent outputs of independent inputs; the only thing they share is
  # ${LIBDIR}, and `cmake -E make_directory` tolerates the directory already
  # existing, so there is no race to serialize. An earlier revision listed
  # ${FINAL_BC} here claiming build-order-only semantics, but CMake's DEPENDS
  # has no order-only form -- it is a real file dependency, so it rebuilt
  # librccl_device.lto.bc every time the plain .bc changed, which is exactly
  # what that comment said it avoided. The standalone Makefile expresses the
  # intent correctly with `| $(FINAL_BC)`.
  add_custom_command(
    OUTPUT ${FINAL_LTO_BC}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBDIR}
    COMMAND ${LLVM_AS_EXECUTABLE} ${LTO_IR_FILE} -o ${FINAL_LTO_BC}
    DEPENDS ${LTO_IR_FILE}
    COMMENT "Assembling final LTO bitcode -> librccl_device.lto.bc..."
    VERBATIM
  )

  add_custom_target(lto_bc
    DEPENDS ${FINAL_LTO_BC}
    COMMENT "RCCL deferred-inlining (LTO) bitcode generation"
  )

  add_custom_command(TARGET lto_bc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "RCCL deferred-inlining (LTO) bitcode generated successfully:"
    COMMAND ${CMAKE_COMMAND} -E echo "  Raw LTO BC:        ${LTO_BC_RAW}"
    COMMAND ${CMAKE_COMMAND} -E echo "  LTO IR (textual):  ${LTO_IR_FILE}"
    COMMAND ${CMAKE_COMMAND} -E echo "  Final LTO BC:      ${FINAL_LTO_BC}"
    VERBATIM
  )
endif()

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
