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
#   ${CMAKE_BINARY_DIR}/include/nccl_device_ir.h       (public C ABI)

##### Source and output paths #####
set(LLVM_SRC      ${CMAKE_CURRENT_SOURCE_DIR}/nccl_device_ir.cu)
set(OBJDIR        ${CMAKE_BINARY_DIR}/obj/llvm_ir)
set(LIBDIR        ${CMAKE_BINARY_DIR}/lib)
set(INCLUDEDIR    ${CMAKE_BINARY_DIR}/include)

set(UNOPTIMIZED_BC ${OBJDIR}/librccl_device.bc.unoptimized)
set(OPTIMIZED_BC   ${OBJDIR}/librccl_device.bc.optimized)
set(LLVM_IR_FILE   ${OBJDIR}/librccl_device.ll)
set(FINAL_BC       ${LIBDIR}/librccl_device.bc)

set(IR_HEADER ${INCLUDEDIR}/nccl_device_ir.h)

##### LTO (deferred-inlining) bitcode artifact #####
# Opt-in: EMIT_LTO_BC=ON additionally emits librccl_device.lto.bc, a bitcode
# variant compiled with no LLVM passes and internalized but never inlined,
# intended for downstream AMDGPU LTO consumption (-Xoffload-linker <bc>
# -Xoffload-linker -plugin-opt=-amdgpu-internalize-symbols=false), analogous
# to NCCL's nvcc -dlto --ltoir artifact.
#
# The deferral comes from -Xclang -disable-llvm-passes on the compile, not
# from the __RCCL_DEVICE_LTO_BC_LIB__ macro. Dropping always_inline from the
# callees is not sufficient on its own: LLVM's inliner gives internal,
# single-use callees a "last call to local" bonus and collapses them anyway,
# which is why this artifact was previously byte-equivalent to the plain .bc
# in everything but two out-of-line C++ functions. See
# LTO_BC_CLANG_COMPILE_FLAGS and LTO_OPT_PASSES below for the measurements.
option(EMIT_LTO_BC "Additionally emit a deferred-inlining LTO bitcode variant (librccl_device.lto.bc)" OFF)

set(LTO_BC_RAW   ${OBJDIR}/librccl_device.lto.bc.raw)
set(LTO_BC_INT   ${OBJDIR}/librccl_device.lto.int.bc)
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

# ${CMAKE_BINARY_DIR}/include must come LAST. It is wanted only for the
# generated rccl.h / nccl.h, but copy_nccl_device_headers also drops a raw,
# *un-hipified* copy of src/include/nccl_device there. hipify prepends
# `#include "hip/hip_runtime.h"` to every header it processes (that is where
# __forceinline__, __popc, __syncthreads come from) and rewrites __CUDACC__
# to __HIPCC__; the raw copy has neither. Searched first, it shadows the
# hipified tree and the device TU fails to compile.
set(RCCL_INCLUDES
  -I${IR_HIPIFY_DIR}/src/include
  -I${IR_HIPIFY_DIR}/src/include/nccl_device
  -I${IR_HIPIFY_DIR}/src/device
  -I${CMAKE_BINARY_DIR}/include              # generated rccl.h / nccl.h
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
# the caller-owned-LDS path in nccl_device/coop.h; an artifact that omits it
# gets a function-local addrspace(3) barrier array that the AMDGPU
# group-segment sizing pass cannot see through the vtable indirect call. See
# the macro commentary in nccl_device_ir.h.
#
# It no longer gates linkage: the public device API in src/include/nccl_device/
# is plain NCCL_DEVICE_INLINE and inlines into the thunks, and the thunks get
# their extern "C" from NCCL_IR_EXPORT plus the prototypes in
# bindings/ir/impl/.
set(BITCODE_LIB_DEFINE -D__clang_llvm_bitcode_lib__)

# Plain (collapsed) bitcode variant: thunks' callees are always_inline, and
# the opt internalize/inline/globaldce pass below collapses everything
# outside the @llvm.used export surface.
set(COMMON_CLANG_FLAGS
  ${BASE_CLANG_FLAGS}
  ${BITCODE_LIB_DEFINE}
)

# Deferred-inlining ("LTO") variant: same library, one modifier on top.
# __RCCL_DEVICE_LTO_BC_LIB__ keys the NCCL_DEVICE_INLINE branch in
# nccl_device_ir.h, mirroring NCCL's __NCCL_DEVICE_LTOIR_LIB__.
set(LTO_BC_CLANG_FLAGS
  ${BASE_CLANG_FLAGS}
  ${BITCODE_LIB_DEFINE}
  -D__RCCL_DEVICE_LTO_BC_LIB__
)

set(CLANG_FLAGS
  -c -emit-llvm -O1
  ${COMMON_CLANG_FLAGS}
)

# -Xclang -disable-llvm-passes is what actually makes this artifact deferred.
# It is the analogue of NCCL's `nvcc -dlto --ltoir`, which emits NVVM IR
# straight from the frontend and leaves every inlining decision to nvlink.
# Clang runs no LLVM pass at all, so the thunks' callees survive as separate
# functions for the downstream AMDGPU LTO link (lld) to re-decide with the
# real call site visible.
#
# -O1 is deliberate and must NOT be lowered to -O0. Measured on gfx950 /
# ROCm 7.2.4, on a function with the same shape as a real thunk:
#
#   -O0                              -> noinline optnone   (optnone even with
#   -O0 -Xclang -disable-llvm-passes -> noinline optnone    -disable-llvm-passes)
#   -O1                              -> fully optimized body, ~20 inferred
#                                       amdgpu-no-* attributes baked in
#   -O1 -Xclang -disable-llvm-passes -> convergent mustprogress nounwind,
#                                       pristine 53-line body
#
# optnone permanently forbids the link-time optimizer from touching the
# function, which is strictly worse than --ltoir: that artifact is
# unoptimized but still *optimizable*. -O1 sets CodeGenOpts.OptimizationLevel
# so the frontend stamps neither optnone nor noinline, while
# -disable-llvm-passes suppresses the pipeline. Only this pairing works.
set(LTO_BC_CLANG_COMPILE_FLAGS
  -c -emit-llvm -O1 -Xclang -disable-llvm-passes
  ${LTO_BC_CLANG_FLAGS}
)

##### Stage the public ABI header into build/include #####
add_custom_command(
  OUTPUT ${IR_HEADER}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${INCLUDEDIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/nccl_device_ir.h ${IR_HEADER}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/nccl_device_ir.h
  COMMENT "Staging nccl_device_ir.h -> ${IR_HEADER}"
  VERBATIM
)

##### Step 1: clang -> unoptimized bitcode #####
add_custom_command(
  OUTPUT ${UNOPTIMIZED_BC}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OBJDIR}
  COMMAND ${CLANG_EXECUTABLE} ${CLANG_FLAGS} ${LLVM_SRC} -o ${UNOPTIMIZED_BC}
  # Track both the impl TU (clang's direct input) and the staged ABI
  # header it transitively includes (plus util.h's type lists). Without
  # the IR_HEADER dep, editing nccl_device_ir.h would not
  # retrigger this command.
  DEPENDS ${LLVM_SRC} ${IR_HEADER} ${CMAKE_CURRENT_SOURCE_DIR}/util.h
  COMMENT "Generating unoptimized LLVM bitcode for ${BITCODE_LIB_ARCH}..."
  VERBATIM
)

##### Step 2: opt -> optimized bitcode #####
# Internalize everything outside the public surface, then inline and drop
# dead globals.
#
# No -internalize-public-api-list. The export surface is already declared in
# the IR: NCCL_IR_EXPORT carries `used, retain`, so every thunk lands in
# @llvm.used (`appending addrspace(1) global [N x ptr]`), and opt's
# internalize pass treats @llvm.used entries as roots. Verified empirically --
# dropping the flag leaves the artifact byte-for-byte identical in symbol
# content (same exported `T` set, same `define internal` count) to the build
# that passed all 99 names explicitly.
#
# This replaces a hand-maintained 99-name list that was duplicated here and in
# bindings/ir/Makefile, had to be edited for every new entry point, and
# silently dropped a thunk to internal linkage whenever someone forgot. NCCL's
# recipe spells it "-internalize-public-api-list=nccl*", which does not work at
# all: opt's internalize pass takes EXACT symbol names, never globs, so the
# literal token "nccl*" matches nothing.

add_custom_command(
  OUTPUT ${OPTIMIZED_BC}
  COMMAND ${OPT_EXECUTABLE}
          --passes=internalize,inline,globaldce
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
  DEPENDS ${FINAL_BC} ${OPTIMIZED_BC} ${IR_HEADER}
  COMMENT "RCCL LLVM IR / bitcode generation"
)

##### LTO (deferred-inlining) bitcode custom commands #####
if(EMIT_LTO_BC)
  ##### Step 1: clang -> raw pristine (no LLVM pass runs) bitcode #####
  add_custom_command(
    OUTPUT ${LTO_BC_RAW}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OBJDIR}
    COMMAND ${CLANG_EXECUTABLE} ${LTO_BC_CLANG_COMPILE_FLAGS} ${LLVM_SRC} -o ${LTO_BC_RAW}
    DEPENDS ${LLVM_SRC} ${IR_HEADER} ${CMAKE_CURRENT_SOURCE_DIR}/util.h
    COMMENT "Generating deferred-inlining (LTO) LLVM bitcode for ${BITCODE_LIB_ARCH}..."
    VERBATIM
  )

  ##### Step 2: opt -> internalize + globaldce, but NOT inline #####
  # This is the plain path's pass list minus exactly one pass: `inline`.
  # That one pass is the whole difference between the two artifacts.
  # internalize and globaldce only rewrite linkage and delete unreachable
  # globals; neither inlines anything, so neither costs any deferral.
  #
  # internalize is mandatory, not cosmetic. Without it the artifact leaks
  # C++-mangled symbols into its export surface; the plain .bc never showed
  # this because its opt pass internalized them as a side effect. Measured
  # on the pre-change artifact: 97 defined T with 1 mangled
  # (_ZN9ncclGin_CC1ERK11ncclDevComm...) before, 96 defined T with 0 mangled
  # after, all 96 ncclIr* preserved via @llvm.used. Pristine IR raises the
  # stakes: with no LLVM passes run at compile time, every non-static
  # function reachable from the headers would otherwise survive as an
  # external mangled symbol.
  #
  # globaldce is equally mandatory, for a subtler reason. coop.h's
  # ncclCoopBcast_{WarpSpan,Cta}_stash() hold function-local __shared__
  # arrays. __clang_llvm_bitcode_lib__ selects the caller-owned-LDS path so
  # nothing calls them, but "uncalled" is not "absent": without DCE they
  # survive as addrspace(3) globals that the AMDGPU group-segment sizing
  # pass cannot account for through the vtable indirect call (HSA exception
  # 0x1016). Measured: internalize alone leaves 4 addrspace(3) references,
  # internalize,globaldce leaves 0, and the define count only moves
  # 4737 -> 4567 -- it deletes dead code, not deferred callees.
  add_custom_command(
    OUTPUT ${LTO_BC_INT}
    COMMAND ${OPT_EXECUTABLE}
            --passes=internalize,globaldce
            ${LTO_BC_RAW} -o ${LTO_BC_INT}
    DEPENDS ${LTO_BC_RAW}
    COMMENT "Internalizing LTO bitcode (internalize/globaldce)..."
    VERBATIM
  )

  ##### Step 3: llvm-dis -> textual IR (kept for inspection) #####
  add_custom_command(
    OUTPUT ${LTO_IR_FILE}
    COMMAND ${LLVM_DIS_EXECUTABLE} ${LTO_BC_INT} -o ${LTO_IR_FILE}
    DEPENDS ${LTO_BC_INT}
    COMMENT "Disassembling LTO bitcode -> textual LLVM IR..."
    VERBATIM
  )

  ##### Step 4: llvm-as -> final LTO bitcode artifact in build/lib #####
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

  # Asking for the variant builds the library it is a variant of, matching the
  # standalone Makefile's `| $(FINAL_BC)`. Without this,
  # `cmake --build . --target lto_bc` on a clean tree yields only
  # librccl_device.lto.bc and no librccl_device.bc -- the two build systems
  # would answer the same request differently. (A normal build is unaffected
  # either way: the top-level rccl_with_ir ALL target already depends on both.)
  #
  # add_dependencies, not DEPENDS: this is target-level ordering, so it does
  # not re-introduce the spurious-rebuild problem described above -- the custom
  # command's DEPENDS list is untouched, and librccl_device.lto.bc is still not
  # regenerated when librccl_device.bc changes.
  add_dependencies(lto_bc llvm_ir)

  add_custom_command(TARGET lto_bc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "RCCL deferred-inlining (LTO) bitcode generated successfully:"
    COMMAND ${CMAKE_COMMAND} -E echo "  Raw LTO BC:        ${LTO_BC_RAW}"
    COMMAND ${CMAKE_COMMAND} -E echo "  Internalized BC:   ${LTO_BC_INT}"
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
  COMMAND ${CMAKE_COMMAND} -E echo "  IR Header:         ${IR_HEADER}"
  VERBATIM
)
