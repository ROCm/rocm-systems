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
# The output mirrors the standalone bindings/ir/Makefile, once per target arch:
#   ${CMAKE_BINARY_DIR}/lib/librccl_device_<arch>.bc   (final clean bitcode)
#   ${CMAKE_BINARY_DIR}/lib/librccl_device.bc          (copy of the first arch)
#   ${CMAKE_BINARY_DIR}/obj/llvm_ir/<arch>/*           (intermediates, textual IR)
#   ${CMAKE_BINARY_DIR}/include/nccl_device_wrapper.h  (public C wrapper)

##### Source and output paths #####
set(LLVM_SRC      ${CMAKE_CURRENT_SOURCE_DIR}/nccl_device_wrapper__impl.h)
set(OBJDIR        ${CMAKE_BINARY_DIR}/obj/llvm_ir)
set(LIBDIR        ${CMAKE_BINARY_DIR}/lib)
set(INCLUDEDIR    ${CMAKE_BINARY_DIR}/include)

set(FINAL_BC       ${LIBDIR}/librccl_device.bc)

set(WRAPPER_HEADER ${INCLUDEDIR}/nccl_device_wrapper.h)

##### Build configuration #####
# GPU architectures to emit librccl_device_<arch>.bc for. Accepts comma- or
# semicolon-separated targets, optionally with feature suffixes
# (gfx950:sramecc+:xnack-): the suffix is passed to --offload-arch so the
# embedded amdhsa.target matches the HSACO the consumer loads, and is stripped
# from the file name. The first entry also becomes the unsuffixed
# librccl_device.bc. When empty, only BITCODE_LIB_ARCH is built:
#     cmake -DEMIT_LLVM_IR=ON -DBITCODE_GPU_ARCHS="gfx942;gfx950" ...
set(BITCODE_GPU_ARCHS "" CACHE STRING
    "GPU architectures for librccl_device_<arch>.bc (empty: BITCODE_LIB_ARCH only)")
set(BITCODE_LIB_ARCH "gfx950" CACHE STRING
    "Single AMDGPU architecture for the LLVM bitcode when BITCODE_GPU_ARCHS is empty")
if(BITCODE_GPU_ARCHS)
  string(REPLACE "," ";" _ir_requested "${BITCODE_GPU_ARCHS}")
else()
  set(_ir_requested "${BITCODE_LIB_ARCH}")
endif()
set(RCCL_IR_ARCHS)        # full target strings, first occurrence of each base arch
set(RCCL_IR_ARCH_BASES)   # same, feature suffixes stripped
foreach(_arch ${_ir_requested})
  string(REGEX REPLACE ":.*" "" _base "${_arch}")
  if(NOT _base IN_LIST RCCL_IR_ARCH_BASES)
    list(APPEND RCCL_IR_ARCHS "${_arch}")
    list(APPEND RCCL_IR_ARCH_BASES "${_base}")
  endif()
endforeach()
list(GET RCCL_IR_ARCH_BASES 0 RCCL_IR_PRIMARY_ARCH)
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
message(STATUS "RCCL IR: targets   = ${RCCL_IR_ARCH_BASES} (${BITCODE_CXX_STD})")

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
set(COMMON_CLANG_FLAGS
  -std=${BITCODE_CXX_STD}
  -x hip
  --rocm-path=${ROCM_PATH}
  --offload-device-only
  ${RCCL_INCLUDES}
  ${ROCM_INCLUDES}
  -D__clang_llvm_bitcode_lib__
  -D__CUDACC_EXTENDED_LAMBDA__
  ${NCCL_DEVICE_DEFINES}
  -D__HIP_PLATFORM_AMD__=1
)

# The bitcode must be compiled with the same GIN backend selection as the host
# library: backend dispatch and the ncclDevComm/GIN context layouts depend on
# it, so a bitcode built without these defines is ABI-incompatible with an
# ENABLE_ROCSHMEM_GIN=ON librccl.so.
if(ENABLE_ROCSHMEM_GIN)
  list(APPEND COMMON_CLANG_FLAGS -DENABLE_ROCSHMEM_GIN)
  foreach(_dir ${ROCSHMEM_INCLUDE_DIR} ${ROCSHMEM_BUILD_DIR})
    list(APPEND COMMON_CLANG_FLAGS -I${_dir})
  endforeach()
  if(ROCSHMEM_SOURCE_DIR)
    list(APPEND COMMON_CLANG_FLAGS -I${ROCSHMEM_SOURCE_DIR}/src -I${ROCSHMEM_SOURCE_DIR}/include)
  endif()
endif()

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

##### Exported surface #####
# Every NCCL_IR_EXPORT thunk carries __attribute__((used, retain)), which puts
# it in @llvm.used; opt's `internalize` always preserves those, and everything
# else is internalized and then inlined or dropped. No explicit public-API list
# is needed, so adding a thunk needs no build-file change. verify_exports.cmake
# checks that every @llvm.used entry survives with external linkage and that
# these required thunks are among them.
set(RCCL_IR_REQUIRED_EXPORTS
    rccl_gin_put rccl_gin_put_signal rccl_gin_put_counter
    rccl_gin_wait_signal rccl_gin_wait_counter rccl_gin_flush)
string(REPLACE ";" "," RCCL_IR_REQUIRED_EXPORTS_CSV "${RCCL_IR_REQUIRED_EXPORTS}")

##### Per-architecture pipeline #####
# clang -> opt (internalize/inline/globaldce) -> llvm-dis -> llvm-as, then check
# that every public symbol survived with external linkage. Each arch gets its
# own intermediates under ${OBJDIR}/<arch>/ and a librccl_device_<arch>.bc.
# AMDGPU has no equivalent of NCCL's "nvvm-reflect-ftz" metadata, so the
# awk/sed scrubbing pass that NCCL applies between dis and as is intentionally
# omitted.
set(RCCL_IR_ARCH_BC_FILES)
function(rccl_ir_add_arch full_arch base_arch)
  set(_dir   ${OBJDIR}/${base_arch})
  set(_unopt ${_dir}/librccl_device.bc.unoptimized)
  set(_opt   ${_dir}/librccl_device.bc.optimized)
  set(_ll    ${_dir}/librccl_device.ll)
  set(_final ${LIBDIR}/librccl_device_${base_arch}.bc)

  add_custom_command(
    OUTPUT ${_unopt}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_dir}
    COMMAND ${CLANG_EXECUTABLE} ${CLANG_FLAGS} --offload-arch=${full_arch} ${LLVM_SRC} -o ${_unopt}
    # Track both the impl TU (clang's direct input) and the staged wrapper
    # header it transitively includes (plus util.h's type lists). Without
    # the WRAPPER_HEADER dep, editing nccl_device_wrapper.h would not
    # retrigger this command.
    DEPENDS ${LLVM_SRC} ${WRAPPER_HEADER} ${CMAKE_CURRENT_SOURCE_DIR}/util.h
    COMMENT "Generating unoptimized LLVM bitcode for ${full_arch}..."
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${_opt}
    COMMAND ${OPT_EXECUTABLE}
            --passes=internalize,inline,globaldce
            ${_unopt} -o ${_opt}
    DEPENDS ${_unopt}
    COMMENT "Optimizing LLVM bitcode for ${base_arch} (internalize/inline/globaldce)..."
    VERBATIM
  )

  # Textual IR is kept for inspection and is what the export check reads.
  add_custom_command(
    OUTPUT ${_ll}
    COMMAND ${LLVM_DIS_EXECUTABLE} ${_opt} -o ${_ll}
    COMMAND ${CMAKE_COMMAND} -DINPUT=${_ll} -DREQUIRED=${RCCL_IR_REQUIRED_EXPORTS_CSV}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_exports.cmake
    DEPENDS ${_opt} ${CMAKE_CURRENT_SOURCE_DIR}/verify_exports.cmake
    COMMENT "Disassembling and verifying exports for ${base_arch}..."
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${_final}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBDIR}
    COMMAND ${LLVM_AS_EXECUTABLE} ${_ll} -o ${_final}
    DEPENDS ${_ll}
    COMMENT "Assembling final bitcode -> librccl_device_${base_arch}.bc..."
    VERBATIM
  )

  set(RCCL_IR_ARCH_BC_FILES ${RCCL_IR_ARCH_BC_FILES} ${_final} PARENT_SCOPE)
endfunction()

list(LENGTH RCCL_IR_ARCHS _ir_n_archs)
math(EXPR _ir_last "${_ir_n_archs} - 1")
foreach(_i RANGE ${_ir_last})
  list(GET RCCL_IR_ARCHS ${_i} _full)
  list(GET RCCL_IR_ARCH_BASES ${_i} _base)
  rccl_ir_add_arch(${_full} ${_base})
endforeach()

##### Unsuffixed librccl_device.bc (first arch) #####
add_custom_command(
  OUTPUT ${FINAL_BC}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${LIBDIR}/librccl_device_${RCCL_IR_PRIMARY_ARCH}.bc ${FINAL_BC}
  DEPENDS ${LIBDIR}/librccl_device_${RCCL_IR_PRIMARY_ARCH}.bc
  COMMENT "Staging librccl_device.bc from ${RCCL_IR_PRIMARY_ARCH}..."
  VERBATIM
)

# Visible to the parent for install rules.
set(RCCL_IR_ARCH_BC_FILES ${RCCL_IR_ARCH_BC_FILES} PARENT_SCOPE)

##### Public target #####
add_custom_target(llvm_ir
  DEPENDS ${FINAL_BC} ${RCCL_IR_ARCH_BC_FILES} ${WRAPPER_HEADER}
  COMMENT "RCCL LLVM IR / bitcode generation"
)

##### Post-build summary #####
add_custom_command(TARGET llvm_ir POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "RCCL LLVM IR and bitcode generated successfully:"
  COMMAND ${CMAKE_COMMAND} -E echo "  C++ Standard:      ${BITCODE_CXX_STD}"
  COMMAND ${CMAKE_COMMAND} -E echo "  GPU Architectures: ${RCCL_IR_ARCH_BASES}"
  COMMAND ${CMAKE_COMMAND} -E echo "  Per-arch BC:       ${RCCL_IR_ARCH_BC_FILES}"
  COMMAND ${CMAKE_COMMAND} -E echo "  Final BC:          ${FINAL_BC}"
  COMMAND ${CMAKE_COMMAND} -E echo "  Wrapper Header:    ${WRAPPER_HEADER}"
  VERBATIM
)
