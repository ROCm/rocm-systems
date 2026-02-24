# Copyright (c) 2020 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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

# Windows ROCM_PATH setup
if(WIN32)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCM Path")
    elseif(DEFINED ENV{HIP_PATH})
      set(ROCM_PATH $ENV{HIP_PATH} CACHE STRING "ROCM Path from HIP_PATH")
    endif()
  endif()
  if(DEFINED ROCM_PATH)
    set(HIP_PATH ${ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
  endif()
elseif(UNIX)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCM Path")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    endif()
  endif()
  if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
      set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
    else()
      set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

project(llvm_ir_to_exe LANGUAGES CXX HIP)

# Find hip (for HIP_PLATFORM check)
find_package(hip REQUIRED)

# Set compiler and toolchain paths - using amdclang++ directly
set(HIP_CLANG ${CMAKE_HIP_COMPILER})
if(WIN32)
  set(CLANG "${ROCM_PATH}/lib/llvm/bin/clang.exe")
  set(LLVM_MC "${ROCM_PATH}/lib/llvm/bin/llvm-mc.exe")
  set(CLANG_OFFLOAD_BUNDLER "${ROCM_PATH}/lib/llvm/bin/clang-offload-bundler.exe")
  set(LLVM_AS "${ROCM_PATH}/lib/llvm/bin/llvm-as.exe")
  set(LLVM_DIS "${ROCM_PATH}/lib/llvm/bin/llvm-dis.exe")
else()
  set(CLANG ${HIP_PATH}/llvm/bin/clang)
  set(LLVM_MC ${HIP_PATH}/llvm/bin/llvm-mc)
  set(CLANG_OFFLOAD_BUNDLER ${HIP_PATH}/llvm/bin/clang-offload-bundler)
  set(LLVM_AS ${HIP_PATH}/llvm/bin/llvm-as)
  set(LLVM_DIS ${HIP_PATH}/llvm/bin/llvm-dis)
endif()

set(INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../../common)

set(SRCS ${CMAKE_CURRENT_SOURCE_DIR}/square.cpp)
set(SQ_HOST_BC ${CMAKE_CURRENT_BINARY_DIR}/square_host.bc)
set(SQ_HOST_LL ${CMAKE_CURRENT_BINARY_DIR}/square_host.ll)
if(WIN32)
  set(SQ_HOST_OBJ ${CMAKE_CURRENT_BINARY_DIR}/square_host.obj)
  set(SQ_DEVICE_OBJ ${CMAKE_CURRENT_BINARY_DIR}/square_device.obj)
  set(SQ_IR_EXE ${CMAKE_CURRENT_BINARY_DIR}/square_ir.exe)
  set(OBJ_EXT ".obj")
  set(HOST_TARGET "x86_64-pc-windows-msvc")
  set(NULL_DEVICE "NUL")
else()
  set(SQ_HOST_OBJ ${CMAKE_CURRENT_BINARY_DIR}/square_host.o)
  set(SQ_DEVICE_OBJ ${CMAKE_CURRENT_BINARY_DIR}/square_device.o)
  set(SQ_IR_EXE ${CMAKE_CURRENT_BINARY_DIR}/square_ir.out)
  set(OBJ_EXT ".o")
  set(HOST_TARGET "x86_64-linux-gnu")
  set(NULL_DEVICE "/dev/null")
endif()
set(SQ_DEVICE_HIPFB ${CMAKE_CURRENT_BINARY_DIR}/offload_bundle.hipfb)

# Use platform-specific mcin file (ELF vs COFF assembly syntax)
if(WIN32)
  set(MCIN_OBJ_GEN ${CMAKE_CURRENT_SOURCE_DIR}/hip_obj_gen_win.mcin)
else()
  set(MCIN_OBJ_GEN ${CMAKE_CURRENT_SOURCE_DIR}/hip_obj_gen.mcin)
endif()

# Append Current device arch from amdgpu-arch
# if amdgpu-arch is not found, support --offload-arch
# to pass arch use format like -DOFFLOAD_ARCH_STR="--offload-arch=gfx1032 --offload-arch=gfx1031"

if(UNIX)
  set(ARCH_PATH "${ROCM_PATH}/llvm/bin/amdgpu-arch")
else()
  set(ARCH_PATH "${ROCM_PATH}/bin/amdgpu-arch")
endif()

# On Windows, prefer CMAKE_HIP_ARCHITECTURES if set
if(WIN32 AND DEFINED CMAKE_HIP_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "")
  foreach(_hip_gpu_arch ${CMAKE_HIP_ARCHITECTURES})
    list(APPEND GPU_ARCH ${_hip_gpu_arch})
  endforeach()
elseif(NOT DEFINED OFFLOAD_ARCH_STR
   AND EXISTS "${ARCH_PATH}"
   AND HIP_PLATFORM STREQUAL "amd")
  execute_process(COMMAND "${ARCH_PATH}"
    OUTPUT_VARIABLE HIP_GPU_ARCH
    RESULT_VARIABLE ROCM_AGENT_ENUM_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT HIP_GPU_ARCH STREQUAL "")
    string(REGEX REPLACE "\n" ";" HIP_GPU_ARCH_LIST "${HIP_GPU_ARCH}")
    list(REMOVE_DUPLICATES HIP_GPU_ARCH_LIST)
    foreach(_hip_gpu_arch ${HIP_GPU_ARCH_LIST})
      list(APPEND GPU_ARCH ${_hip_gpu_arch})
    endforeach()
  else()
    message(STATUS "amdgpu-arch found no valid architectures")
  endif()
elseif(DEFINED OFFLOAD_ARCH_STR)
  string(REPLACE "--offload-arch=" "" HIP_GPU_ARCH_LIST ${OFFLOAD_ARCH_STR})
  string(REGEX REPLACE " " ";" HIP_GPU_ARCH_LIST "${HIP_GPU_ARCH_LIST}")
  foreach(_hip_gpu_arch ${HIP_GPU_ARCH_LIST})
    list(APPEND GPU_ARCH ${_hip_gpu_arch})
  endforeach()
endif()

if(TARGET build_cookbook)
  set(ALL_OPTION)
else()
  set(ALL_OPTION ALL)
endif()

list(JOIN GPU_ARCH "," OFFLOAD_ARCH)
add_custom_target(src_to_ir ${ALL_OPTION}
  COMMAND ${HIP_CLANG} -x hip -c -emit-llvm -I${INCLUDES} --cuda-host-only -fuse-cuid=none -target ${HOST_TARGET} -o ${SQ_HOST_BC} ${SRCS}
  COMMAND ${HIP_CLANG} -x hip -c -emit-llvm -I${INCLUDES} --cuda-device-only --offload-arch=${OFFLOAD_ARCH} ${SRCS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Target for BC to LL
add_custom_command(OUTPUT ll_obj 
  COMMAND ${LLVM_DIS} ${SQ_HOST_BC} -o ${SQ_HOST_LL}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

foreach(ARCH ${GPU_ARCH})
  set(arch_obj ${ARCH}_ll_obj)
  add_custom_command(OUTPUT ${arch_obj} 
    COMMAND ${LLVM_DIS} square-hip-amdgcn-amd-amdhsa-${ARCH}.bc -o square-hip-amdgcn-amd-amdhsa-${ARCH}.ll
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  list(APPEND arch_ll_obj ${arch_obj})
endforeach()

add_custom_target(bc_to_ll ${ALL_OPTION} DEPENDS src_to_ir ll_obj ${arch_ll_obj})

# Target for LL to BC
add_custom_command(OUTPUT bc_obj 
  COMMAND ${LLVM_AS} ${SQ_HOST_LL} -o ${SQ_HOST_BC}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
foreach(ARCH ${GPU_ARCH})
  set(arch_obj ${ARCH}_bc_obj)
  add_custom_command(OUTPUT ${arch_obj} 
    COMMAND ${LLVM_AS} square-hip-amdgcn-amd-amdhsa-${ARCH}.ll -o square-hip-amdgcn-amd-amdhsa-${ARCH}.bc
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  list(APPEND arch_bc_obj ${arch_obj})
endforeach()

add_custom_target(ll_to_bc ${ALL_OPTION} DEPENDS bc_to_ll bc_obj ${arch_bc_obj})

# Target for IR to EXEC
add_custom_command(OUTPUT host_obj 
  COMMAND ${HIP_CLANG} -c ${SQ_HOST_BC} -o ${SQ_HOST_OBJ}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

foreach(ARCH ${GPU_ARCH})
  set(arch_obj ${ARCH}_obj)
  add_custom_command(OUTPUT ${arch_obj} 
    COMMAND ${CLANG} -target amdgcn-amd-amdhsa -mcpu=${ARCH} square-hip-amdgcn-amd-amdhsa-${ARCH}.bc -o square-hip-amdgcn-amd-amdhsa-${ARCH}${OBJ_EXT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  list(APPEND arch_obj_list ${arch_obj})
  list(APPEND TARGETS hip-amdgcn-amd-amdhsa--${ARCH})
  list(APPEND INPUTS square-hip-amdgcn-amd-amdhsa-${ARCH}${OBJ_EXT})
endforeach()

list(JOIN TARGETS "," TARGET_STR)
list(TRANSFORM INPUTS PREPEND "-input=")

if(WIN32)
  set(BUNDLE_HOST_TARGET "host-x86_64-pc-windows-msvc")
  add_custom_target(ir_to_exec ${ALL_OPTION} DEPENDS ll_to_bc host_obj ${arch_obj_list}
    COMMAND ${CLANG_OFFLOAD_BUNDLER} -type=o -bundle-align=4096 -targets=${BUNDLE_HOST_TARGET},${TARGET_STR} -input=${NULL_DEVICE} ${INPUTS} -output=${SQ_DEVICE_HIPFB}
    COMMAND ${LLVM_MC} ${MCIN_OBJ_GEN} -o ${SQ_DEVICE_OBJ} --filetype=obj -triple=x86_64-pc-windows-msvc
    COMMAND ${HIP_CLANG} ${SQ_HOST_OBJ} ${SQ_DEVICE_OBJ} -o ${SQ_IR_EXE} -L${ROCM_PATH}/lib ${ROCM_PATH}/lib/amdhip64.lib
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
else()
  add_custom_target(ir_to_exec ${ALL_OPTION} DEPENDS ll_to_bc host_obj ${arch_obj_list}
    COMMAND ${CLANG_OFFLOAD_BUNDLER} -type=o -bundle-align=4096 -targets=host-x86_64-unknown--linux,${TARGET_STR} -input=/dev/null ${INPUTS} -output=${SQ_DEVICE_HIPFB}
    COMMAND ${LLVM_MC} ${MCIN_OBJ_GEN} -o ${SQ_DEVICE_OBJ} --filetype=obj
    COMMAND ${HIP_CLANG} ${SQ_HOST_OBJ} ${SQ_DEVICE_OBJ} -o ${SQ_IR_EXE} -L${ROCM_PATH}/lib -lamdhip64 -Wl,-rpath,${ROCM_PATH}/lib
  )
endif()

if(TARGET build_cookbook)
  add_dependencies(build_cookbook src_to_ir bc_to_ll ll_to_bc ir_to_exec)
endif()
