# Copyright (c) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
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
if(BUILD_SHARED_LIBS)
  add_subdirectory(16_assembly_to_executable)
  add_subdirectory(17_llvm_ir_to_executable)
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
