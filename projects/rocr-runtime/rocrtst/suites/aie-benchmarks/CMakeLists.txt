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
  # Bounded, so that RyzenAI-npu10/npu11 (npu3, below) do not match as npu1.
  elseif(XRT_SMI_OUTPUT MATCHES "NPU Phoenix|RyzenAI-npu1([^0-9]|$)")
    set(NPU_TARGET "npu")
  elseif(XRT_SMI_OUTPUT MATCHES "NPU3|RyzenAI-npu9|RyzenAI-npu10|RyzenAI-npu11|Medusa")
    set(NPU_TARGET "npu3")
  endif()
endif()
if(NOT NPU_TARGET)
  message(FATAL_ERROR
    "Could not auto-detect NPU device. Please set NPU_TARGET to 'npu', 'npu2' or 'npu3'.")
else()
  message(STATUS "NPU target: ${NPU_TARGET}")
endif()

# ---------------------------------------------------------------------------
# xclbinutil / aiebu-asm
#
# Only the PDI+insts (xclbin) flow needs these, and npu3 (AIE4) has no such flow -- aiecc
# assembles its full ELF's control code itself, without aiebu-asm. Required everywhere else: npu
# and npu2 always build the PDI+insts benchmarks below.
# ---------------------------------------------------------------------------
if(NOT NPU_TARGET STREQUAL "npu3")
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

  # Full-ELF dispatch is aie2p only -- the design's npu_load_pdi has no aie2 equivalent -- so there
  # is nothing to build for an npu1 target even with a complete toolchain.
  if(AIE_FULL_ELF_SUPPORTED AND NOT NPU_TARGET STREQUAL "npu2")
    set(AIE_FULL_ELF_SUPPORTED FALSE)
    message(STATUS "Full-ELF kernel will not be built: it needs an npu2 target.")
  endif()
else()
  # npu3 (AIE4) has only the full-ELF flow; it needs neither tool.
  set(AIE_FULL_ELF_SUPPORTED TRUE)
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
# The standard llvm-aie wheel installs Peano at a top-level 'llvm-aie' dir. npu3 (AIE4) instead
# uses the mlir-aie-xrlabs venv, which nests it under 'lnx64.o/tools/peano' -- try that layout
# only for npu3, so npu/npu2 keep using the standard one.
if(NPU_TARGET STREQUAL "npu3" AND NOT EXISTS "${PEANO_INSTALL_DIR}")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import os, sysconfig; print(os.path.join(sysconfig.get_paths()['purelib'], 'lnx64.o', 'tools', 'peano'))"
    OUTPUT_VARIABLE PEANO_INSTALL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()
if(NOT EXISTS "${PEANO_INSTALL_DIR}")
  message(FATAL_ERROR "Could not find Peano (llvm-aie) at '${PEANO_INSTALL_DIR}'. "
                      "Set up the IRON environment with iron_env_setup.sh first.")
endif()
message(STATUS "Peano install dir: ${PEANO_INSTALL_DIR}")

# aie-hsaco packages the kernel artifacts into an hsaco. It ships with mlir-aie, next to aiecc.
#
# AIE_HSACO_PYTHON_DIR runs it from an mlir-aie checkout's python/compiler directory with this
# build's Python, instead of finding an aie-hsaco executable on PATH: npu3 needs the
# mlir-aie-xrlabs venv for its compiler, and that venv ships no aie-hsaco of its own.
set(AIE_HSACO_PYTHON_DIR "" CACHE PATH
    "mlir-aie checkout's python/compiler directory to run aie-hsaco from, instead of AIE_HSACO")
if(AIE_HSACO_PYTHON_DIR)
  set(AIE_HSACO_CMD "${CMAKE_COMMAND}" -E env "PYTHONPATH=${AIE_HSACO_PYTHON_DIR}"
                    "${Python3_EXECUTABLE}" -m hsaco.pack)
else()
  find_program(AIE_HSACO aie-hsaco)
  if(AIE_HSACO)
    set(AIE_HSACO_CMD "${AIE_HSACO}")
  endif()
endif()
if(NOT AIE_HSACO_CMD)
  message(FATAL_ERROR "Could not find aie-hsaco, which packages the kernels into an hsaco. It "
                      "ships with mlir-aie; set up the IRON environment, or set AIE_HSACO_PYTHON_DIR, "
                      "first.")
endif()
list(JOIN AIE_HSACO_CMD " " AIE_HSACO_CMD_STR)
message(STATUS "aie-hsaco: ${AIE_HSACO_CMD_STR}")

# ---------------------------------------------------------------------------
# Kernel artifact build
# ---------------------------------------------------------------------------
set(KERNEL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/kernel_build")
set(VECTOR_SCALAR_ADD_PY "${CMAKE_CURRENT_SOURCE_DIR}/vector_scalar_add.py")

# Steps 1/2 (PDI+insts via aiecc's xclbin flow) don't apply to npu3 (AIE4): it has only the
# full-ELF flow below. aie_kernel_artifacts is not defined for npu3; nothing below may depend on
# it in that case.
if(NOT NPU_TARGET STREQUAL "npu3")
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
endif()

# Step 1b: Python -> MLIR (full-ELF variant)
#
# The full-ELF design differs from the xclbin one: its runtime sequence starts
# with aiex.npu.load_pdi, because no xclbin configures the array for it (except on npu3, whose own
# control code already configures the array -- see vector_scalar_add.py). It also gets its own
# directory -- aiecc writes <input>.prj next to its input, so sharing a directory would make the
# two aiecc runs clobber each other's intermediates under 'make -j'.
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

# Step 3: MLIR -> full ELF (standalone, contains PDI + instructions)
# npu3 (AIE4) needs neither xclbinutil nor aiebu-asm on PATH (see AIE_FULL_ELF_SUPPORTED above).
set(FULL_ELF "${FULL_ELF_BUILD_DIR}/aie.elf")
if(AIE_FULL_ELF_SUPPORTED)
  if(NOT NPU_TARGET STREQUAL "npu3")
    get_filename_component(XCLBINUTIL_DIR "${XCLBINUTIL}" DIRECTORY)
    get_filename_component(AIEBU_ASM_DIR "${AIEBU_ASM}" DIRECTORY)
    set(FULL_ELF_PATH_ENV "PATH=${XCLBINUTIL_DIR}:${AIEBU_ASM_DIR}:$ENV{PATH}")
  else()
    set(FULL_ELF_PATH_ENV "PATH=$ENV{PATH}")
  endif()

  add_custom_command(
    OUTPUT "${FULL_ELF}"
    # aiebu-asm exits 0 even when it fails, and aiecc does not check that the
    # ELF appeared. Delete first so that a silent failure leaves nothing behind
    # and the build stops, instead of resurrecting the previous ELF.
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${FULL_ELF}"
    COMMAND "${CMAKE_COMMAND}" -E env "${FULL_ELF_PATH_ENV}"
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
#
# No npu3 (AIE4) equivalent -- it has only the full-ELF flow below.
# ---------------------------------------------------------------------------
if(NOT NPU_TARGET STREQUAL "npu3")
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
endif()

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
# hsaco packaging
#
# Both HSA benchmarks dispatch by kernel object, so each needs an hsaco rather than the raw
# artifacts. aie-hsaco names sections by architecture, which is what the loader matches against
# the agent; AIE_HSACO_ARCH is the aie-hsaco spelling of NPU_TARGET. PDI+insts works on either
# aie2/aie2p architecture, so this mapping is not gated on full-ELF support.
# ---------------------------------------------------------------------------
if(NPU_TARGET STREQUAL "npu2")
  set(AIE_HSACO_ARCH "aie2p")
elseif(NPU_TARGET STREQUAL "npu3")
  set(AIE_HSACO_ARCH "aie4")
else()
  set(AIE_HSACO_ARCH "aie2")
endif()

# ---------------------------------------------------------------------------
# HSA PDI/insts benchmark
#
# No npu3 (AIE4) equivalent -- it has only the full-ELF flow below.
# ---------------------------------------------------------------------------
if(NOT NPU_TARGET STREQUAL "npu3")
  # vsadd's kernarg buffer is 2 pointers + 2 sizes; the design uses a single NPU column.
  set(PDI_HSACO_KERNEL_NAME "vsadd")
  set(PDI_HSACO "${CMAKE_CURRENT_BINARY_DIR}/vsadd.hsaco")
  add_custom_command(
    OUTPUT "${PDI_HSACO}"
    COMMAND ${AIE_HSACO_CMD} --hsaco "${PDI_HSACO}" --arch "${AIE_HSACO_ARCH}"
            --kernel "${PDI_HSACO_KERNEL_NAME}:${INSTS_BIN}:${DESIGN_PDI}:32:1"
    # Depend on the target, not its outputs: two custom commands racing aiecc in one build
    # directory produce concurrent runs that clobber each other's intermediates.
    DEPENDS aie_kernel_artifacts
    COMMENT "Packaging the vsadd PDI+insts artifacts into an hsaco for the benchmark"
    VERBATIM)
  add_custom_target(vsadd_pdi_hsaco_bench DEPENDS "${PDI_HSACO}")

  add_executable(vector-scalar-add-hsa vector_scalar_add_hsa.cpp)
  target_compile_features(vector-scalar-add-hsa PRIVATE cxx_std_20)
  target_compile_definitions(vector-scalar-add-hsa PRIVATE
    DEFAULT_HSACO_PATH=${PDI_HSACO}
    DEFAULT_HSACO_KERNEL_NAME="${PDI_HSACO_KERNEL_NAME}"
  )
  target_link_libraries(vector-scalar-add-hsa PRIVATE
    benchmark::benchmark_main
    hsa-runtime64::hsa-runtime64
  )
  add_dependencies(vector-scalar-add-hsa vsadd_pdi_hsaco_bench)
endif()

# ---------------------------------------------------------------------------
# HSA full-ELF benchmark
# ---------------------------------------------------------------------------

if(AIE_FULL_ELF_SUPPORTED)
  # The application dispatches by kernel object, so it needs an hsaco rather than a bare ELF.
  set(ELF_HSACO "${CMAKE_CURRENT_BINARY_DIR}/vsadd_elf.hsaco")
  add_custom_command(
    OUTPUT "${ELF_HSACO}"
    COMMAND ${AIE_HSACO_CMD} --hsaco "${ELF_HSACO}" --arch "${AIE_HSACO_ARCH}"
            --kernel "elf:${FULL_ELF}:0:1"
    DEPENDS aie_full_elf
    COMMENT "Packaging the full ELF into an hsaco for the benchmark"
    VERBATIM)
  add_custom_target(vsadd_elf_hsaco_bench DEPENDS ${ELF_HSACO})

  add_executable(vector-scalar-add-hsa-elf vector_scalar_add_hsa_elf.cpp)
  target_compile_features(vector-scalar-add-hsa-elf PRIVATE cxx_std_20)
  target_compile_definitions(vector-scalar-add-hsa-elf PRIVATE
    DEFAULT_ELF_HSACO_PATH=${ELF_HSACO}
    DEFAULT_ELF_KERNEL_NAME="main:sequence"
    DEFAULT_REQUIRED_ARCH="${AIE_HSACO_ARCH}"
  )
  target_link_libraries(vector-scalar-add-hsa-elf PRIVATE
    benchmark::benchmark_main
    hsa-runtime64::hsa-runtime64
  )
  add_dependencies(vector-scalar-add-hsa-elf vsadd_elf_hsaco_bench)
endif()

# ---------------------------------------------------------------------------
# HSA vmem alloc/release microbenchmark
# ---------------------------------------------------------------------------

add_executable(vmem-alloc-hsa vmem_alloc_hsa.cpp)
target_compile_features(vmem-alloc-hsa PRIVATE cxx_std_20)
target_link_libraries(vmem-alloc-hsa PRIVATE
  benchmark::benchmark_main
  hsa-runtime64::hsa-runtime64
)
