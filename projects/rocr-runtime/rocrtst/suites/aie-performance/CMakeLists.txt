#  Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

cmake_minimum_required(VERSION 3.20)

project(aie_tests
  VERSION 1.0
  DESCRIPTION "A set of benchmarks to compare backends"
  LANGUAGES C CXX
)

# Set a default build type
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS
      "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

include(FetchContent)

# Import GoogleBenchmark
FetchContent_Declare(googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG main)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(googlebenchmark)

# HSA
find_package(hsa-runtime64 CONFIG REQUIRED NAMES hsa_runtime64 hsa-runtime64)

# XRT
find_package(XRT REQUIRED)
if(NOT XRT_FOUND)
  set(XRT_INC_DIR "/opt/xilinx/xrt/include" CACHE PATH "XRT include dir")
  set(XRT_LIB_DIR "/opt/xilinx/xrt/lib" CACHE PATH "XRT library dir")
else()
  if(XRT_INCLUDE_DIRS)
    set(XRT_INC_DIR "${XRT_INCLUDE_DIRS}" CACHE PATH "XRT include dir")
  elseif(XRT_INCLUDE_DIR)
    set(XRT_INC_DIR "${XRT_INCLUDE_DIR}" CACHE PATH "XRT include dir")
  endif()
  if(XRT_LINK_DIRS)
    set(XRT_LIB_DIR "${XRT_LINK_DIRS}" CACHE PATH "XRT library dir")
  endif()
endif()

# xclbinutil
find_program(XCLBINUTIL xclbinutil
  HINTS "${XRT_LIB_DIR}/../bin/unwrapped" "${XRT_LIB_DIR}/../bin"
)
if(NOT XCLBINUTIL)
  message(FATAL_ERROR "xclbinutil not found")
endif()

# Python (for kernel build pipeline)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ---------------------------------------------------------------------------
# Kernel artifact build
# ---------------------------------------------------------------------------
set(KERNEL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/kernel_build")
set(VECTOR_SCALAR_ADD_PY "${CMAKE_CURRENT_SOURCE_DIR}/vector_scalar_add.py")

# Step 1: Python -> MLIR
set(AIE_MLIR "${KERNEL_BUILD_DIR}/aie.mlir")
add_custom_command(
  OUTPUT "${AIE_MLIR}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${KERNEL_BUILD_DIR}"
  COMMAND "${Python3_EXECUTABLE}" "${VECTOR_SCALAR_ADD_PY}" npu > "${AIE_MLIR}"
  DEPENDS "${VECTOR_SCALAR_ADD_PY}"
  COMMENT "Generating aie.mlir from vector_scalar_add.py"
  VERBATIM
)

# Step 2: MLIR -> xclbin + insts.bin + design.pdi
# aiecc needs xclbinutil on PATH to generate the xclbin container.
get_filename_component(XCLBINUTIL_DIR "${XCLBINUTIL}" DIRECTORY)
set(FINAL_XCLBIN "${KERNEL_BUILD_DIR}/final.xclbin")
set(INSTS_BIN    "${KERNEL_BUILD_DIR}/insts.bin")
set(DESIGN_PDI   "${KERNEL_BUILD_DIR}/design.pdi")
add_custom_command(
  OUTPUT "${FINAL_XCLBIN}" "${INSTS_BIN}" "${DESIGN_PDI}"
  COMMAND "${CMAKE_COMMAND}" -E env "PATH=${XCLBINUTIL_DIR}:$ENV{PATH}"
          aiecc
          --aie-generate-xclbin
          --aie-generate-npu-insts
          --aie-generate-pdi
          --no-compile-host
          --no-xchesscc
          --no-xbridge
          --xclbin-name=final.xclbin
          --npu-insts-name=insts.bin
          --pdi-name=design.pdi
          "${AIE_MLIR}"
  WORKING_DIRECTORY "${KERNEL_BUILD_DIR}"
  DEPENDS "${AIE_MLIR}"
  COMMENT "Compiling AIE kernel with aiecc"
  VERBATIM
)

add_custom_target(aie_kernel_artifacts
  DEPENDS "${FINAL_XCLBIN}" "${INSTS_BIN}" "${DESIGN_PDI}"
)

# Step 3: MLIR -> full ELF (standalone, contains PDI + instructions)
set(FULL_ELF "${KERNEL_BUILD_DIR}/aie.elf")
add_custom_command(
  OUTPUT "${FULL_ELF}"
  COMMAND "${CMAKE_COMMAND}" -E env "PATH=${XCLBINUTIL_DIR}:$ENV{PATH}"
          aiecc
          --generate-full-elf
          --full-elf-name=aie.elf
          --no-compile-host
          --no-xchesscc
          --no-xbridge
          "${AIE_MLIR}"
  WORKING_DIRECTORY "${KERNEL_BUILD_DIR}"
  DEPENDS "${AIE_MLIR}"
  COMMENT "Generating full ELF from AIE kernel"
  VERBATIM
)

add_custom_target(aie_full_elf DEPENDS "${FULL_ELF}")

# ---------------------------------------------------------------------------
# XRT PDI/insts benchmark
# ---------------------------------------------------------------------------
add_executable(vector-scalar-add-xrt vector_scalar_add_xrt.cpp)
target_compile_features(vector-scalar-add-xrt PRIVATE cxx_std_20)
target_include_directories(vector-scalar-add-xrt PRIVATE "${XRT_INC_DIR}")
target_compile_definitions(vector-scalar-add-xrt PRIVATE
  DEFAULT_XCLBIN_PATH=${FINAL_XCLBIN}
  DEFAULT_INSTS_PATH=${INSTS_BIN}
)
target_link_directories(vector-scalar-add-xrt PRIVATE "${XRT_LIB_DIR}")
target_link_libraries(vector-scalar-add-xrt PRIVATE
  benchmark::benchmark_main
  xrt_coreutil)
add_dependencies(vector-scalar-add-xrt aie_kernel_artifacts)

# ---------------------------------------------------------------------------
# XRT full-ELF benchmark
# ---------------------------------------------------------------------------

add_executable(vector-scalar-add-xrt-elf vector_scalar_add_xrt_elf.cpp)
target_compile_features(vector-scalar-add-xrt-elf PRIVATE cxx_std_20)
target_include_directories(vector-scalar-add-xrt-elf PRIVATE "${XRT_INC_DIR}")
target_compile_definitions(vector-scalar-add-xrt-elf PRIVATE
  DEFAULT_ELF_PATH=${FULL_ELF}
  DEFAULT_KERNEL_NAME="vector_scalar_add:sequence"
)
target_link_directories(vector-scalar-add-xrt-elf PRIVATE "${XRT_LIB_DIR}")
target_link_libraries(vector-scalar-add-xrt-elf PRIVATE
  benchmark::benchmark_main
  xrt_coreutil)
add_dependencies(vector-scalar-add-xrt-elf aie_full_elf)

# ---------------------------------------------------------------------------
# HSA PDI/insts benchmark
# ---------------------------------------------------------------------------

add_executable(vector-scalar-add-hsa vector_scalar_add_hsa.cpp)
target_compile_features(vector-scalar-add-hsa PRIVATE cxx_std_20)
target_compile_definitions(vector-scalar-add-hsa PRIVATE
  DEFAULT_PDI_PATH=${DESIGN_PDI}
  DEFAULT_INSTS_PATH=${INSTS_BIN}
)
target_link_libraries(vector-scalar-add-hsa PRIVATE
  benchmark::benchmark_main
  hsa-runtime64::hsa-runtime64
)
add_dependencies(vector-scalar-add-hsa aie_kernel_artifacts)
