# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

cmake_minimum_required(VERSION 3.21)

if(UNIX)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCM Path")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    endif()
  endif()
  # Search for rocm in common locations
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

project(generic_target LANGUAGES CXX HIP)

# Find hiprtc
find_package(hiprtc REQUIRED)

# Find hip (for HIP_PLATFORM check)
find_package(hip REQUIRED)

if(NOT HIP_PLATFORM MATCHES "amd")
  message("Generic target is only supported on AMD GPU")
  return()
endif()

# Create the excutable
if(TARGET build_intro)
set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
set(EXCLUDE_OPTION)
endif()

# Test generic target without feature
set(OFFLOAD_ARCH_GENERIC_STR "--offload-arch=gfx9-generic --offload-arch=gfx9-4-generic --offload-arch=gfx10-1-generic --offload-arch=gfx10-3-generic --offload-arch=gfx11-generic --offload-arch=gfx12-generic")
separate_arguments(OFFLOAD_ARCH_GENERIC_LIST UNIX_COMMAND "${OFFLOAD_ARCH_GENERIC_STR}")

# Mark source files as HIP code (contains __global__ kernels and uses HIP APIs)
set_source_files_properties(square.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(../../../catch/hipTestMain/hip_test_features.cc PROPERTIES LANGUAGE HIP)

add_executable(squareGenericTarget ${EXCLUDE_OPTION} square.cpp ../../../catch/hipTestMain/hip_test_features.cc)
target_include_directories(squareGenericTarget PRIVATE ../../common ${CMAKE_CURRENT_SOURCE_DIR}/../../../catch/include/)
target_compile_options(squareGenericTarget PRIVATE 
  $<$<COMPILE_LANGUAGE:HIP>:-mcode-object-version=6>
  $<$<COMPILE_LANGUAGE:HIP>:-w>
  $<$<COMPILE_LANGUAGE:HIP>:${OFFLOAD_ARCH_GENERIC_LIST}>
)

# Test generic target with features
set(OFFLOAD_ARCH_GENERIC_FEATURE_STR "--offload-arch=gfx9-generic --offload-arch=gfx9-4-generic:sramecc+:xnack- --offload-arch=gfx9-4-generic:sramecc-:xnack- --offload-arch=gfx9-4-generic:xnack+ --offload-arch=gfx10-1-generic --offload-arch=gfx10-3-generic --offload-arch=gfx11-generic --offload-arch=gfx12-generic")
separate_arguments(OFFLOAD_ARCH_GENERIC_FEATURE_LIST UNIX_COMMAND "${OFFLOAD_ARCH_GENERIC_FEATURE_STR}")

add_executable(squareGenericTargetWithFeatures ${EXCLUDE_OPTION} square.cpp ../../../catch/hipTestMain/hip_test_features.cc)
target_include_directories(squareGenericTargetWithFeatures PRIVATE ../../common ${CMAKE_CURRENT_SOURCE_DIR}/../../../catch/include/)
target_compile_options(squareGenericTargetWithFeatures PRIVATE 
  $<$<COMPILE_LANGUAGE:HIP>:-mcode-object-version=6>
  $<$<COMPILE_LANGUAGE:HIP>:-w>
  $<$<COMPILE_LANGUAGE:HIP>:${OFFLOAD_ARCH_GENERIC_FEATURE_LIST}>
)

# Create the excutable for HIPRTC test
# Mark source files as HIP code (contains __global__ kernels and uses HIP APIs)
set_source_files_properties(saxpy.cpp PROPERTIES LANGUAGE HIP)
# hip_test_features.cc already marked above

add_executable(saxpyGenericTarget ${EXCLUDE_OPTION} saxpy.cpp ../../../catch/hipTestMain/hip_test_features.cc)
target_include_directories(saxpyGenericTarget PRIVATE ../../common ${CMAKE_CURRENT_SOURCE_DIR}/../../../catch/include/)

# Link with HIPRTC
target_link_libraries(saxpyGenericTarget hiprtc)
# Link with HIP

if(NOT BUILD_SHARED_LIBS)
  target_link_libraries(saxpyGenericTarget hiprtc-builtins)
endif()

# Set RPATH so executables can find HIP libraries at runtime
if(UNIX)
  set_target_properties(squareGenericTarget squareGenericTargetWithFeatures saxpyGenericTarget PROPERTIES
    BUILD_RPATH "${ROCM_PATH}/lib"
    INSTALL_RPATH "${ROCM_PATH}/lib"
  )
endif()

if(TARGET build_intro)
add_dependencies(build_intro saxpyGenericTarget)
add_dependencies(build_intro squareGenericTarget)
add_dependencies(build_intro squareGenericTargetWithFeatures)
endif()
