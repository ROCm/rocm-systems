project(llvm_ir_to_exe)

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


# Get the actual LLVM bin directory from hipcc location
get_filename_component(HIPCC_DIR ${HIP_HIPCC_EXECUTABLE} DIRECTORY)
get_filename_component(HIP_ROOT_DIR ${HIPCC_DIR} DIRECTORY)

find_program(CLANG_OFFLOAD_BUNDLER clang-offload-bundler
  HINTS
    "${HIP_ROOT_DIR}/llvm/bin"
    "${ROCM_PATH}/llvm/bin"
    "${ROCM_PATH}/bin"
  REQUIRED)

get_filename_component(LLVM_BIN_DIR ${CLANG_OFFLOAD_BUNDLER} DIRECTORY)

set(CLANG   ${LLVM_BIN_DIR}/clang)
set(LLVM_MC   ${LLVM_BIN_DIR}/llvm-mc)
set(LLVM_AS ${LLVM_BIN_DIR}/llvm-as)
set(LLVM_DIS ${LLVM_BIN_DIR}/llvm-dis)

message(STATUS "Using LLVM tools from: ${LLVM_BIN_DIR}")

set(SRCS ${CMAKE_CURRENT_SOURCE_DIR}/saxpy.cpp)
set(DEVICE_HIPFB ${CMAKE_CURRENT_BINARY_DIR}/../offload_bundle.hipfb)


# Append Current device arch from rocm_agent_enumerator
# if rocm_agent_enumerator is not found, support --offload-arch
# to pass arch use format like -DOFFLOAD_ARCH_STR="--offload-arch=gfx1032  --offload-arch=gfx1031"

find_program(ARCH_PATH amdgpu-arch
  HINTS
    "${LLVM_BIN_DIR}"
    "${ROCM_PATH}/llvm/bin"
    "${ROCM_PATH}/bin")

if(NOT DEFINED OFFLOAD_ARCH_STR
   AND ARCH_PATH
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


list(JOIN GPU_ARCH "," OFFLOAD_ARCH)

# Step 1: Generate bitcode for all GPU architectures
add_custom_target(src_to_ir 
  COMMAND ${HIP_HIPCC_EXECUTABLE} -c -emit-llvm  --cuda-device-only --offload-arch=${OFFLOAD_ARCH} ${SRCS}
  COMMENT "Generating GPU bitcode for architectures: ${OFFLOAD_ARCH}")

# Step 2: Build target and input lists for bundling
foreach(ARCH ${GPU_ARCH})
  list(APPEND TARGETS hip-amdgcn-amd-amdhsa--${ARCH})
  list(APPEND INPUTS -input=${CMAKE_CURRENT_BINARY_DIR}/saxpy-hip-amdgcn-amd-amdhsa-${ARCH}.bc)
endforeach()

# Convert list to comma-separated string for bundler
list(JOIN TARGETS "," TARGET_STR)

# Step 3: Bundle all bitcode files into single .hipfb
add_custom_target(bundle_bitcode ALL
  DEPENDS src_to_ir
  COMMAND ${CLANG_OFFLOAD_BUNDLER} 
    -type=bc 
    -targets=${TARGET_STR}
    ${INPUTS}
    -output=${DEVICE_HIPFB}
  COMMENT "Bundling bitcode for targets: ${TARGET_STR}"
  BYPRODUCTS ${DEVICE_HIPFB})

message(STATUS "Will generate bundled bitcode for GPU architectures: ${GPU_ARCH}")