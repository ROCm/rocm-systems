# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# RCCL LLVM bitcode toolchain discovery (EMIT_LLVM_IR).
#==================================================================================================
# Locates the ROCm LLVM tools used by bindings/ir/CMakeLists.txt to build the
# librccl_device.bc artifact. This is the RCCL counterpart to upstream NCCL's
#   find_program(CLANG_EXECUTABLE clang REQUIRED) ...
# block; it is isolated in a module so that section of bindings/ir stays a small,
# easy-to-merge delta against NCCL.
#
# Included (not add_subdirectory'd) from bindings/ir/CMakeLists.txt, so the tool
# variables it sets land in that subdirectory scope. bindings/ir is only added
# when EMIT_LLVM_IR=ON, so the FATAL_ERROR below never fires for ordinary library
# builds.
#
# Expects BITCODE_LIB_ARCH / BITCODE_CXX_STD to be set by the caller (used only
# for the status message).
#==================================================================================================

# ROCM_PATH is normally set by the top-level CMakeLists.txt; provide a
# safe default so this file is also usable standalone (cmake -P style).
if(NOT DEFINED ROCM_PATH OR ROCM_PATH STREQUAL "")
  set(ROCM_PATH "/opt/rocm")
endif()
set(ROCM_LLVM_BIN "${ROCM_PATH}/llvm/bin")

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
