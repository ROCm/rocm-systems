# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21.3)
add_custom_target(build_cookbook)
add_subdirectory(0_MatrixTranspose)
add_subdirectory(1_hipEvent)
add_subdirectory(3_shared_memory)
add_subdirectory(4_shfl)
add_subdirectory(5_2dshfl)
add_subdirectory(6_dynamic_shared)
add_subdirectory(7_streams)
add_subdirectory(8_peer2peer)
add_subdirectory(9_unroll)
add_subdirectory(10_inline_asm)
add_subdirectory(11_texture_driver)
add_subdirectory(12_cmake_hip_add_executable)
add_subdirectory(13_occupancy)
add_subdirectory(14_gpu_arch)
add_subdirectory(15_static_library/device_functions)
add_subdirectory(15_static_library/host_functions)
# Disable samples that call hipcc directly for static builds.
# Also check for required LLVM tools on Windows
if(BUILD_SHARED_LIBS)
  add_subdirectory(16_assembly_to_executable)
  
  # 17_llvm_ir_to_executable requires llvm-dis which may not be available
  find_program(LLVM_DIS_EXE 
    NAMES llvm-dis llvm-dis.exe
    HINTS "${ROCM_PATH}/lib/llvm/bin" "${HIP_PATH}/lib/llvm/bin"
  )
  if(LLVM_DIS_EXE)
    add_subdirectory(17_llvm_ir_to_executable)
  else()
    message(STATUS "Skipping 17_llvm_ir_to_executable - llvm-dis not found")
  endif()
endif()
add_subdirectory(18_cmake_hip_device)
add_subdirectory(19_cmake_lang)
add_subdirectory(21_cmake_hip_cxx_clang)
# Disable hip-lang test for static builds.
# Once the issues are fixed, will enable the test
if(BUILD_SHARED_LIBS)
  add_subdirectory(22_cmake_hip_lang)
endif()
add_subdirectory(23_cmake_hiprtc)
