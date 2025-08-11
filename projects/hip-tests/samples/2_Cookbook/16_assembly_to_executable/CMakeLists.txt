project(asm_to_exe)

cmake_minimum_required(VERSION 3.10)

if(UNIX)
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


# Find hip
find_package(hip REQUIRED)

# Set compiler and linker
set(HIPCC ${HIP_HIPCC_EXECUTABLE})
set(CLANG   ${HIP_PATH}/llvm/bin/clang)
set(LLVM_MC   ${HIP_PATH}/llvm/bin/llvm-mc)
set(CLANG_OFFLOAD_BUNDLER   ${HIP_PATH}/llvm/bin/clang-offload-bundler)

set(INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../../common)

set(SRCS ${CMAKE_CURRENT_SOURCE_DIR}/square.cpp)
set(SQ_HOST_ASM ${CMAKE_CURRENT_BINARY_DIR}/square_host.s)
set(SQ_HOST_OBJ ${CMAKE_CURRENT_BINARY_DIR}/square_host.o)
set(SQ_DEVICE_HIPFB ${CMAKE_CURRENT_BINARY_DIR}/offload_bundle.hipfb)
set(SQ_DEVICE_OBJ   ${CMAKE_CURRENT_BINARY_DIR}/square_device.o)
set(SQ_ASM_EXE ${CMAKE_CURRENT_BINARY_DIR}/square_asm.out)

set(MCIN_OBJ_GEN ${CMAKE_CURRENT_SOURCE_DIR}/hip_obj_gen.mcin)

# Append Current device arch from rocm_agent_enumerator
# if rocm_agent_enumerator is not found, support --offload-arch
# to pass arch use format like -DOFFLOAD_ARCH_STR="--offload-arch=gfx1032  --offload-arch=gfx1031"
if(UNIX)
set(ARCH_PATH "${ROCM_PATH}/llvm/bin/amdgpu-arch")
else()
set(ARCH_PATH "${ROCM_PATH}/bin/amdgpu-arch")
endif()

if(NOT DEFINED OFFLOAD_ARCH_STR
   AND EXISTS "${ARCH_PATH}"
   AND HIP_PLATFORM STREQUAL "amd")
    execute_process(COMMAND "${ARCH_PATH}"
         OUTPUT_VARIABLE HIP_GPU_ARCH
         RESULT_VARIABLE ROCM_AGENT_ENUM_RESULT
         OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT HIP_GPU_ARCH STREQUAL "")
          string(REGEX REPLACE "\n" ";" HIP_GPU_ARCH_LIST "${HIP_GPU_ARCH}")
          list(REMOVE_DUPLICATES  HIP_GPU_ARCH_LIST)
          foreach(_hip_gpu_arch ${HIP_GPU_ARCH_LIST})
              list(APPEND GPU_ARCH ${_hip_gpu_arch})
          endforeach()
    else()
          message(STATUS "ROCm Agent Enumerator found no valid architectures")
    endif()
elseif(DEFINED OFFLOAD_ARCH_STR)
    string(REPLACE "--offload-arch=" ""  HIP_GPU_ARCH_LIST ${OFFLOAD_ARCH_STR})
    string(REGEX REPLACE " " ";" HIP_GPU_ARCH_LIST "${HIP_GPU_ARCH_LIST}")
    foreach(_hip_gpu_arch ${HIP_GPU_ARCH_LIST})
          list(APPEND GPU_ARCH ${_hip_gpu_arch})
    endforeach()
endif()

if(TARGET build_cookbook)
set(ALL_OPTION )
else()
set(ALL_OPTION ALL )
endif()

list(JOIN GPU_ARCH "," OFFLOAD_ARCH)
add_custom_target(src_to_asm ${ALL_OPTION}  COMMAND ${HIPCC} -c -S -I${INCLUDES} --cuda-host-only -fuse-cuid=none -target x86_64-linux-gnu -o ${SQ_HOST_ASM} ${SRCS}
                               COMMAND ${HIPCC} -c -S -I${INCLUDES} --cuda-device-only --offload-arch=${OFFLOAD_ARCH} ${SRCS})


add_custom_command(OUTPUT host_obj COMMAND ${HIPCC} -c ${SQ_HOST_ASM} -o ${SQ_HOST_OBJ})

foreach(ARCH ${GPU_ARCH})
  list(APPEND TARGETS hip-amdgcn-amd-amdhsa--${ARCH})
  list(APPEND INPUTS square-hip-amdgcn-amd-amdhsa-${ARCH}.o)
  set(arch_obj ${ARCH}_obj)
  add_custom_command(OUTPUT ${arch_obj} COMMAND ${CLANG} -target amdgcn-amd-amdhsa -mcpu=${ARCH} square-hip-amdgcn-amd-amdhsa-${ARCH}.s -o square-hip-amdgcn-amd-amdhsa-${ARCH}.o)
  list(APPEND arch_obj_targets ${arch_obj})
endforeach()

list(JOIN TARGETS "," TARGET_STR)
list(TRANSFORM INPUTS PREPEND "-input=")
add_custom_target(asm_to_exec ${ALL_OPTION} DEPENDS src_to_asm host_obj ${arch_obj_targets}
                               COMMAND ${CLANG_OFFLOAD_BUNDLER} -type=o -bundle-align=4096 -targets=host-x86_64-unknown--linux,${TARGET_STR} -input=/dev/null ${INPUTS} -output=${SQ_DEVICE_HIPFB}
                               COMMAND ${LLVM_MC} ${MCIN_OBJ_GEN} -o ${SQ_DEVICE_OBJ} --filetype=obj
                               COMMAND ${HIPCC} ${SQ_HOST_OBJ} ${SQ_DEVICE_OBJ} -o ${SQ_ASM_EXE})

if(TARGET build_cookbook)
add_dependencies(build_cookbook src_to_asm asm_to_exec)
endif()
