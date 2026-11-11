# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# Adapted from NVIDIA NCCL ir/CMakeLists.txt (v2.29.2-1).
#
# See LICENSE.txt for license information
#
# LLVM IR / bitcode generation for RCCL device APIs (HIP / AMDGPU).
#
# This file is meant to be pulled in from the top-level RCCL CMakeLists.txt
# via:
#     option(EMIT_LLVM_IR     "Generate LLVM IR/bitcode for device APIs" OFF)
#     option(NCCL_EMIT_LTO_IR "Generate LTO IR for device APIs"          OFF)
#     if(EMIT_LLVM_IR OR NCCL_EMIT_LTO_IR)
#       add_subdirectory(bindings/ir)
#     endif()
#     if(EMIT_LLVM_IR)
#       add_dependencies(llvm_ir copy_nccl_device_headers)
#       add_custom_target(rccl_with_ir ALL DEPENDS rccl llvm_ir)
#     endif()
#     if(NCCL_EMIT_LTO_IR)
#       add_dependencies(ltoir copy_nccl_device_headers)
#       add_custom_target(rccl_with_ltoir ALL DEPENDS rccl ltoir)
#     endif()
#
# The output mirrors the standalone bindings/ir/Makefile:
#   ${CMAKE_BINARY_DIR}/lib/librccl_device.bc          (final clean bitcode)
#   ${CMAKE_BINARY_DIR}/obj/llvm_ir/*.bc.{un,}optimized
#   ${CMAKE_BINARY_DIR}/obj/llvm_ir/librccl_device.ll  (textual IR)
#   ${CMAKE_BINARY_DIR}/lib/librccl_device.ltoir       (LTO IR, deferred inlining)
#   ${CMAKE_BINARY_DIR}/include/nccl_device_wrapper.h  (public C wrapper)
#
# Two independent knobs, matching NCCL and the standalone Makefile:
#   EMIT_LLVM_IR      -> target llvm_ir -> librccl_device.bc
#   NCCL_EMIT_LTO_IR  -> target ltoir   -> librccl_device.ltoir
# Everything up to the "Compiler flags" section below is shared by both and
# must stay outside either guard.

##### Source and output paths #####
set(LLVM_SRC      ${CMAKE_CURRENT_SOURCE_DIR}/nccl_device_wrapper__impl.h)
set(OBJDIR        ${CMAKE_BINARY_DIR}/obj/llvm_ir)
set(LIBDIR        ${CMAKE_BINARY_DIR}/lib)
set(INCLUDEDIR    ${CMAKE_BINARY_DIR}/include)

set(UNOPTIMIZED_BC ${OBJDIR}/librccl_device.bc.unoptimized)
set(OPTIMIZED_BC   ${OBJDIR}/librccl_device.bc.optimized)
set(LLVM_IR_FILE   ${OBJDIR}/librccl_device.ll)
set(FINAL_BC       ${LIBDIR}/librccl_device.bc)

set(LTO_IR_RAW     ${OBJDIR}/librccl_device.ltoir.raw)
set(FINAL_LTO_IR   ${LIBDIR}/librccl_device.ltoir)

set(WRAPPER_HEADER ${INCLUDEDIR}/nccl_device_wrapper.h)

##### Build configuration #####
# Default GPU architecture for the bitcode artifact. Override via:
#     cmake -DEMIT_LLVM_IR=ON -DBITCODE_LIB_ARCH=gfx942 ...
set(BITCODE_LIB_ARCH "gfx950" CACHE STRING
    "AMDGPU architecture for the LLVM bitcode (e.g. gfx942, gfx950)")
set(BITCODE_CXX_STD  "c++17"  CACHE STRING
    "C++ standard for the LLVM bitcode build")

# LTO IR knobs, mirroring NCCL's LTO_IR_ARCH / LTO_IR_CXX_STD. Both default
# to the bitcode library's settings, so the two artifacts stay in step
# unless deliberately diverged.
set(LTO_IR_ARCH    "" CACHE STRING
    "AMDGPU architecture for the LTO IR artifact (empty = BITCODE_LIB_ARCH)")
set(LTO_IR_CXX_STD "" CACHE STRING
    "C++ standard for the LTO IR build (empty = BITCODE_CXX_STD)")
if(NOT LTO_IR_ARCH)
  set(LTO_IR_ARCH "${BITCODE_LIB_ARCH}")
endif()
if(NOT LTO_IR_CXX_STD)
  set(LTO_IR_CXX_STD "${BITCODE_CXX_STD}")
endif()

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

# llvm-dis/llvm-as are only used by the llvm_ir recipe's textual round-trip;
# the ltoir recipe needs clang and opt alone. Only require what is actually
# going to be invoked, so a toolchain missing llvm-dis can still build the
# LTO IR artifact.
set(_required_ir_tools CLANG_EXECUTABLE OPT_EXECUTABLE)
if(EMIT_LLVM_IR)
  list(APPEND _required_ir_tools LLVM_DIS_EXECUTABLE LLVM_AS_EXECUTABLE)
endif()

foreach(_tool ${_required_ir_tools})
  if(NOT ${_tool})
    message(FATAL_ERROR
      "RCCL IR (EMIT_LLVM_IR=${EMIT_LLVM_IR}, "
      "NCCL_EMIT_LTO_IR=${NCCL_EMIT_LTO_IR}): could not locate ${_tool} "
      "under ${ROCM_LLVM_BIN}. Install the ROCm LLVM toolchain or override "
      "${_tool} on the command line.")
  endif()
endforeach()

message(STATUS "RCCL IR: clang     = ${CLANG_EXECUTABLE}")
message(STATUS "RCCL IR: opt       = ${OPT_EXECUTABLE}")
if(EMIT_LLVM_IR)
  message(STATUS "RCCL IR: llvm-dis  = ${LLVM_DIS_EXECUTABLE}")
  message(STATUS "RCCL IR: llvm-as   = ${LLVM_AS_EXECUTABLE}")
  message(STATUS "RCCL IR: bitcode   = ${BITCODE_LIB_ARCH} (${BITCODE_CXX_STD})")
endif()
if(NCCL_EMIT_LTO_IR)
  message(STATUS "RCCL IR: lto ir    = ${LTO_IR_ARCH} (${LTO_IR_CXX_STD})")
endif()

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

##### Stage the public wrapper header into build/include #####
# Owned by one utility target so the copy runs once under -j, not once per dependent target.
set(WRAPPER_HEADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/nccl_device_wrapper.h)

add_custom_command(
  OUTPUT ${WRAPPER_HEADER}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${INCLUDEDIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${WRAPPER_HEADER_SRC} ${WRAPPER_HEADER}
  DEPENDS ${WRAPPER_HEADER_SRC}
  COMMENT "Staging nccl_device_wrapper.h -> ${WRAPPER_HEADER}"
  VERBATIM
)

add_custom_target(nccl_device_wrapper_header DEPENDS ${WRAPPER_HEADER})

# ===========================================================================
# Bitcode library (EMIT_LLVM_IR) -> librccl_device.bc
# ===========================================================================
if(EMIT_LLVM_IR)

##### Compiler flags #####
# HIP device-only compilation. --offload-device-only is the modern,
# target-agnostic spelling of NCCL's --cuda-device-only.
# Clang supports extended device lambdas but does not define nvcc's feature
# macro; the ReduceCopy device APIs require it.
set(COMMON_CLANG_FLAGS
  -std=${BITCODE_CXX_STD}
  -x hip
  --rocm-path=${ROCM_PATH}
  --offload-device-only
  --offload-arch=${BITCODE_LIB_ARCH}
  ${RCCL_INCLUDES}
  ${ROCM_INCLUDES}
  -D__clang_llvm_bitcode_lib__
  -D__CUDACC_EXTENDED_LAMBDA__
  ${NCCL_DEVICE_DEFINES}
  -D__HIP_PLATFORM_AMD__=1
)

set(CLANG_FLAGS
  -c -emit-llvm -O1
  ${COMMON_CLANG_FLAGS}
)

##### Step 1: clang -> unoptimized bitcode #####
add_custom_command(
  OUTPUT ${UNOPTIMIZED_BC}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OBJDIR}
  COMMAND ${CLANG_EXECUTABLE} ${CLANG_FLAGS} ${LLVM_SRC} -o ${UNOPTIMIZED_BC}
  # Track both the impl TU (clang's direct input) and the wrapper header it
  # transitively includes (plus util.h's type lists). Without the wrapper
  # header dep, editing nccl_device_wrapper.h would not retrigger this
  # command. Ordering against the staging copy comes from
  # add_dependencies(llvm_ir nccl_device_wrapper_header) below.
  DEPENDS ${LLVM_SRC} ${WRAPPER_HEADER_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/util.h
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
    # [C] GIN + composite barrier session
    ncclGinBarrierSessionInit
    ncclGinBarrierSessionSync
    ncclBarrierSessionInit
    ncclBarrierSessionSync
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
  DEPENDS ${FINAL_BC} ${OPTIMIZED_BC}
  COMMENT "RCCL LLVM IR / bitcode generation"
)
add_dependencies(llvm_ir nccl_device_wrapper_header)

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

endif() # EMIT_LLVM_IR

# ===========================================================================
# LTO IR (NCCL_EMIT_LTO_IR) -> librccl_device.ltoir
# ===========================================================================
# Same TU as the .bc; -O3 -Xclang -disable-llvm-passes defers inlining to the consumer's link
# (-O3, not -O1: -O1 stamps llvm.loop.unroll.disable; see bindings/ir/Makefile).
if(NCCL_EMIT_LTO_IR)

##### Compiler flags #####
set(LTO_IR_CLANG_FLAGS
  -c -emit-llvm -O3 -Xclang -disable-llvm-passes
  -std=${LTO_IR_CXX_STD}
  -x hip
  --rocm-path=${ROCM_PATH}
  --offload-device-only
  --offload-arch=${LTO_IR_ARCH}
  ${RCCL_INCLUDES}
  ${ROCM_INCLUDES}
  -D__clang_llvm_bitcode_lib__
  -D__NCCL_DEVICE_LTOIR_LIB__
  -D__CUDACC_EXTENDED_LAMBDA__
  ${NCCL_DEVICE_DEFINES}
  -D__HIP_PLATFORM_AMD__=1
)

# Track staged device headers explicitly; DEPFILE support for custom commands
# is not available with every generator supported by RCCL's CMake minimum.
file(GLOB_RECURSE RCCL_DEVICE_HEADER_DEPS CONFIGURE_DEPENDS
  "${CMAKE_BINARY_DIR}/include/*.h"
  "${CMAKE_BINARY_DIR}/include/*.hpp"
  "${CMAKE_BINARY_DIR}/include/*.cuh"
  "${IR_HIPIFY_DIR}/src/include/*.h"
  "${IR_HIPIFY_DIR}/src/include/*.hpp"
  "${IR_HIPIFY_DIR}/src/include/*.cuh"
  "${IR_HIPIFY_DIR}/src/device/*.h"
  "${IR_HIPIFY_DIR}/src/device/*.hpp"
  "${IR_HIPIFY_DIR}/src/device/*.cuh"
)

##### Step 1: clang -> raw LTO IR #####
add_custom_command(
  OUTPUT ${LTO_IR_RAW}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OBJDIR}
  COMMAND ${CLANG_EXECUTABLE} ${LTO_IR_CLANG_FLAGS} ${LLVM_SRC} -o ${LTO_IR_RAW}
  DEPENDS ${LLVM_SRC} ${WRAPPER_HEADER_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/util.h
          ${RCCL_DEVICE_HEADER_DEPS}
  COMMENT "Generating LTO IR (deferred inlining) for ${LTO_IR_ARCH}..."
  VERBATIM
)

##### Step 2: opt -> internalized LTO IR #####
# internalize ONLY. Adding `inline` or `globaldce` here would undo the whole
# point of the artifact and collapse it back into something equivalent to the
# .bc. No -internalize-public-api-list is needed: NCCL_IR_EXPORT carries
# `used, retain`, so @llvm.used already names the public surface and
# internalize honours it as its root set.
add_custom_command(
  OUTPUT ${FINAL_LTO_IR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBDIR}
  COMMAND ${OPT_EXECUTABLE} --passes=internalize ${LTO_IR_RAW} -o ${FINAL_LTO_IR}
  DEPENDS ${LTO_IR_RAW}
  COMMENT "Internalizing LTO IR -> librccl_device.ltoir..."
  VERBATIM
)

##### Public target #####
add_custom_target(ltoir
  DEPENDS ${FINAL_LTO_IR}
  COMMENT "RCCL LTO IR generation"
)
add_dependencies(ltoir nccl_device_wrapper_header)

# Asking for the variant builds the library it is a variant of, so both build
# systems answer `--target ltoir` the same way. This is target-level ordering
# (the CMake equivalent of make's order-only `|`), deliberately NOT a DEPENDS
# entry on the custom command above -- a DEPENDS would make .ltoir regenerate
# every time .bc changes, which it has no reason to do.
if(EMIT_LLVM_IR)
  add_dependencies(ltoir llvm_ir)
endif()

##### Post-build summary #####
add_custom_command(TARGET ltoir POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "RCCL LTO IR generated successfully:"
  COMMAND ${CMAKE_COMMAND} -E echo "  C++ Standard:      ${LTO_IR_CXX_STD}"
  COMMAND ${CMAKE_COMMAND} -E echo "  GPU Architecture:  ${LTO_IR_ARCH}"
  COMMAND ${CMAKE_COMMAND} -E echo "  Raw LTO IR:        ${LTO_IR_RAW}"
  COMMAND ${CMAKE_COMMAND} -E echo "  Final LTO IR:      ${FINAL_LTO_IR}"
  COMMAND ${CMAKE_COMMAND} -E echo "  Wrapper Header:    ${WRAPPER_HEADER}"
  VERBATIM
)

endif() # NCCL_EMIT_LTO_IR
