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

# aiebu-asm (AIE binary utilities, shipped alongside XRT) assembles the
# control code embedded in the full ELF.
find_program(AIEBU_ASM aiebu-asm
  HINTS "${XRT_LIB_DIR}/../bin" "/opt/xilinx/aiebu/bin"
)
if(NOT AIEBU_ASM)
  message(FATAL_ERROR "aiebu-asm not found")
endif()

# aiecc drives aiebu-asm with the 'aie2_config' target. Older aiebu releases
# (e.g. the one shipped with XRT 2.20.0) only know that target as 'config', and
# aiebu-asm exits 0 on an unrecognised target -- so aiecc reports success and
# never writes an ELF. Detect that here rather than letting the build silently
# produce nothing.
execute_process(
  COMMAND "${AIEBU_ASM}" --help
  OUTPUT_VARIABLE AIEBU_ASM_HELP
  ERROR_VARIABLE AIEBU_ASM_HELP
)
if(AIEBU_ASM_HELP MATCHES "aie2_config")
  set(AIE_FULL_ELF_SUPPORTED TRUE)
else()
  set(AIE_FULL_ELF_SUPPORTED FALSE)
  message(WARNING
    "aiebu-asm at ${AIEBU_ASM} does not support the 'aie2_config' target, so "
    "the full-ELF kernel cannot be built; vector-scalar-add-xrt-elf will be "
    "skipped.")
endif()

# Python (for kernel build pipeline)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ---------------------------------------------------------------------------
# NPU detection
# ---------------------------------------------------------------------------
find_program(XRT_SMI xrt-smi)
if(XRT_SMI)
  execute_process(
    COMMAND "${XRT_SMI}" examine
    OUTPUT_VARIABLE XRT_SMI_OUTPUT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(XRT_SMI_OUTPUT MATCHES "NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[456]")
    set(NPU_TARGET "npu2")
  elseif(XRT_SMI_OUTPUT MATCHES "NPU Phoenix|RyzenAI-npu1")
    set(NPU_TARGET "npu")
  endif()
endif()
if(NOT NPU_TARGET)
  message(FATAL_ERROR "Could not auto-detect NPU device. Please set NPU_TARGET to either 'npu' or 'npu2'.")
else()
  message(STATUS "NPU target: ${NPU_TARGET}")
endif()

# ---------------------------------------------------------------------------
# Peano (llvm-aie) detection
#
# aiecc compiles the AIE cores with Peano, which ships as the llvm-aie wheel
# installed alongside mlir-aie in the IRON environment.
# ---------------------------------------------------------------------------
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import os, sysconfig; print(os.path.join(sysconfig.get_paths()['purelib'], 'llvm-aie'))"
  OUTPUT_VARIABLE PEANO_INSTALL_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT EXISTS "${PEANO_INSTALL_DIR}")
  message(FATAL_ERROR "Could not find Peano (llvm-aie) at '${PEANO_INSTALL_DIR}'. "
                      "Set up the IRON environment with iron_env_setup.sh first.")
endif()
message(STATUS "Peano install dir: ${PEANO_INSTALL_DIR}")

# ---------------------------------------------------------------------------
# Kernel artifact build
# ---------------------------------------------------------------------------
set(KERNEL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/kernel_build")
set(VECTOR_SCALAR_ADD_PY "${CMAKE_CURRENT_SOURCE_DIR}/vector_scalar_add.py")

# Step 1: Python -> MLIR (xclbin/PDI variant)
set(AIE_MLIR "${KERNEL_BUILD_DIR}/aie.mlir")
add_custom_command(
  OUTPUT "${AIE_MLIR}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${KERNEL_BUILD_DIR}"
  COMMAND "${Python3_EXECUTABLE}" "${VECTOR_SCALAR_ADD_PY}" "${NPU_TARGET}" > "${AIE_MLIR}"
  DEPENDS "${VECTOR_SCALAR_ADD_PY}"
  COMMENT "Generating aie.mlir from vector_scalar_add.py (target: ${NPU_TARGET})"
  VERBATIM
)

# Step 1b: Python -> MLIR (full-ELF variant)
#
# The full-ELF design differs from the xclbin one: its runtime sequence starts
# with aiex.npu.load_pdi, because no xclbin configures the array for it. It also
# gets its own directory -- aiecc writes <input>.prj next to its input, so
# sharing a directory would make the two aiecc runs clobber each other's
# intermediates under 'make -j'.
set(FULL_ELF_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/kernel_build_full_elf")
set(AIE_ELF_MLIR "${FULL_ELF_BUILD_DIR}/aie.mlir")
add_custom_command(
  OUTPUT "${AIE_ELF_MLIR}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${FULL_ELF_BUILD_DIR}"
  COMMAND "${Python3_EXECUTABLE}" "${VECTOR_SCALAR_ADD_PY}" "${NPU_TARGET}" --full-elf
          > "${AIE_ELF_MLIR}"
  DEPENDS "${VECTOR_SCALAR_ADD_PY}"
  COMMENT "Generating full-ELF aie.mlir from vector_scalar_add.py (target: ${NPU_TARGET})"
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
          --peano=${PEANO_INSTALL_DIR}
          --get-xclbin
          --get-npu-insts
          --get-pdi
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
# aiecc needs aiebu-asm on PATH to assemble the embedded control code.
get_filename_component(AIEBU_ASM_DIR "${AIEBU_ASM}" DIRECTORY)
set(FULL_ELF "${FULL_ELF_BUILD_DIR}/aie.elf")
if(AIE_FULL_ELF_SUPPORTED)
  add_custom_command(
    OUTPUT "${FULL_ELF}"
    # aiebu-asm exits 0 even when it fails, and aiecc does not check that the
    # ELF appeared. Delete first so that a silent failure leaves nothing behind
    # and the build stops, instead of resurrecting the previous ELF.
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${FULL_ELF}"
    COMMAND "${CMAKE_COMMAND}" -E env "PATH=${XCLBINUTIL_DIR}:${AIEBU_ASM_DIR}:$ENV{PATH}"
            aiecc
            --peano=${PEANO_INSTALL_DIR}
            --get-full-elf
            --full-elf-name=aie.elf
            "${AIE_ELF_MLIR}"
    # Errors out (non-zero) if aiecc produced no ELF.
    COMMAND "${CMAKE_COMMAND}" -E md5sum "${FULL_ELF}"
    WORKING_DIRECTORY "${FULL_ELF_BUILD_DIR}"
    DEPENDS "${AIE_ELF_MLIR}"
    COMMENT "Generating full ELF from AIE kernel"
    VERBATIM
  )

  add_custom_target(aie_full_elf DEPENDS "${FULL_ELF}")
endif()

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

# The kernel is addressed as "<device>:<sequence>", the names aiecc writes into
# aie.mlir.prj/full_elf_config.json. vector_scalar_add.py leaves the aie.device
# and the runtime sequence unnamed, so aiecc defaults them to "main" and
# "sequence".
if(AIE_FULL_ELF_SUPPORTED)
  add_executable(vector-scalar-add-xrt-elf vector_scalar_add_xrt_elf.cpp)
  target_compile_features(vector-scalar-add-xrt-elf PRIVATE cxx_std_20)
  target_include_directories(vector-scalar-add-xrt-elf PRIVATE "${XRT_INC_DIR}")
  target_compile_definitions(vector-scalar-add-xrt-elf PRIVATE
    DEFAULT_ELF_PATH=${FULL_ELF}
    DEFAULT_KERNEL_NAME="main:sequence"
  )
  target_link_directories(vector-scalar-add-xrt-elf PRIVATE "${XRT_LIB_DIR}")
  target_link_libraries(vector-scalar-add-xrt-elf PRIVATE
    benchmark::benchmark_main
    xrt_coreutil)
  add_dependencies(vector-scalar-add-xrt-elf aie_full_elf)
endif()

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

# ---------------------------------------------------------------------------
# HSA vmem alloc/release microbenchmark
# ---------------------------------------------------------------------------

add_executable(vmem-alloc-hsa vmem_alloc_hsa.cpp)
target_compile_features(vmem-alloc-hsa PRIVATE cxx_std_20)
target_link_libraries(vmem-alloc-hsa PRIVATE
  benchmark::benchmark_main
  hsa-runtime64::hsa-runtime64
)
