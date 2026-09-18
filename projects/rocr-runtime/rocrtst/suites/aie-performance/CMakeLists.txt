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

# Full-ELF dispatch is aie2p only -- the design's npu_load_pdi has no aie2 equivalent -- so there
# is nothing to build for an npu1 target even with a complete toolchain.
if(AIE_FULL_ELF_SUPPORTED AND NOT NPU_TARGET STREQUAL "npu2")
  set(AIE_FULL_ELF_SUPPORTED FALSE)
  message(STATUS "Full-ELF kernel will not be built: it needs an npu2 target.")
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

# aie_hsaco.py shells out to llvm-objcopy to attach the AIE section. It ships in the mlir_aie
# wheel's bin directory, which iron_env_setup.sh does not put on PATH -- so probe for it here
# rather than letting the packaging step die with a Python traceback halfway through a build.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import os, sysconfig; print(os.path.join(sysconfig.get_paths()['purelib'], 'mlir_aie'))"
  OUTPUT_VARIABLE MLIR_AIE_INSTALL_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_program(LLVM_OBJCOPY llvm-objcopy
  HINTS "${MLIR_AIE_INSTALL_DIR}/bin" "${PEANO_INSTALL_DIR}/bin"
)
if(NOT LLVM_OBJCOPY)
  message(FATAL_ERROR "Could not find llvm-objcopy, which aie_hsaco.py needs to package an hsaco. "
                      "It ships with mlir_aie; add its bin directory to PATH.")
endif()
get_filename_component(LLVM_OBJCOPY_DIR "${LLVM_OBJCOPY}" DIRECTORY)
message(STATUS "llvm-objcopy: ${LLVM_OBJCOPY}")

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
# hsaco packaging
#
# Both HSA benchmarks dispatch by kernel object, so each needs an hsaco rather than the raw
# artifacts. aie_hsaco.py names sections by architecture, which is what the loader matches against
# the agent; NPU_TARGET is the aiecc spelling of the same thing. PDI+insts works on either
# architecture, so this mapping is not gated on full-ELF support.
# ---------------------------------------------------------------------------
if(NPU_TARGET STREQUAL "npu2")
  set(AIE_ARCH "aie2p")
else()
  set(AIE_ARCH "aie2")
endif()
set(AIE_HSACO_PY "${CMAKE_CURRENT_SOURCE_DIR}/../aie/aie_hsaco.py")

# ---------------------------------------------------------------------------
# HSA PDI/insts benchmark
# ---------------------------------------------------------------------------

# vsadd's kernarg buffer is 2 pointers + 2 sizes; the design uses a single NPU column.
set(PDI_HSACO_KERNEL_NAME "vsadd")
set(PDI_HSACO "${CMAKE_CURRENT_BINARY_DIR}/vsadd.hsaco")
add_custom_command(
  OUTPUT "${PDI_HSACO}"
  # aie_hsaco.py creates the container before it can fail, so a failed run would otherwise leave
  # a newer-than-dependencies hsaco behind that the next build treats as up to date.
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${PDI_HSACO}"
  COMMAND "${CMAKE_COMMAND}" -E env "PATH=${LLVM_OBJCOPY_DIR}:$ENV{PATH}"
          "${Python3_EXECUTABLE}" "${AIE_HSACO_PY}"
          --hsaco "${PDI_HSACO}" --arch "${AIE_ARCH}"
          --kernel "${PDI_HSACO_KERNEL_NAME}:${INSTS_BIN}:${DESIGN_PDI}:32:1"
  # Depend on the target, not its outputs: two custom commands racing aiecc in one build
  # directory produce concurrent runs that clobber each other's intermediates.
  DEPENDS aie_kernel_artifacts "${AIE_HSACO_PY}"
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

# ---------------------------------------------------------------------------
# HSA full-ELF benchmark
# ---------------------------------------------------------------------------

if(AIE_FULL_ELF_SUPPORTED)
  # The application dispatches by kernel object, so it needs an hsaco rather than a bare ELF.
  set(ELF_HSACO "${CMAKE_CURRENT_BINARY_DIR}/vsadd_elf.hsaco")
  add_custom_command(
    OUTPUT "${ELF_HSACO}"
    # Same as the PDI+insts rule above: clear the output so a failed run cannot be mistaken for
    # an up-to-date one.
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${ELF_HSACO}"
    COMMAND "${CMAKE_COMMAND}" -E env "PATH=${LLVM_OBJCOPY_DIR}:$ENV{PATH}"
            "${Python3_EXECUTABLE}" "${AIE_HSACO_PY}"
            --hsaco "${ELF_HSACO}" --arch "${AIE_ARCH}"
            --kernel "elf:${FULL_ELF}:0:1"
    DEPENDS aie_full_elf "${AIE_HSACO_PY}"
    COMMENT "Packaging the full ELF into an hsaco for the benchmark"
    VERBATIM)
  add_custom_target(vsadd_elf_hsaco_bench DEPENDS ${ELF_HSACO})

  add_executable(vector-scalar-add-hsa-elf vector_scalar_add_hsa_elf.cpp)
  target_compile_features(vector-scalar-add-hsa-elf PRIVATE cxx_std_20)
  target_compile_definitions(vector-scalar-add-hsa-elf PRIVATE
    DEFAULT_ELF_HSACO_PATH=${ELF_HSACO}
    DEFAULT_ELF_KERNEL_NAME="main:sequence"
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

# ---------------------------------------------------------------------------
# HSA vmem size-sweep microbenchmark
# ---------------------------------------------------------------------------

add_executable(vmem-size-hsa vmem_size_hsa.cpp)
target_compile_features(vmem-size-hsa PRIVATE cxx_std_20)
target_link_libraries(vmem-size-hsa PRIVATE
  benchmark::benchmark_main
  hsa-runtime64::hsa-runtime64
)

# ---------------------------------------------------------------------------
# XRT bo size-sweep microbenchmark
# ---------------------------------------------------------------------------

add_executable(vmem-size-xrt vmem_size_xrt.cpp)
target_compile_features(vmem-size-xrt PRIVATE cxx_std_20)
target_include_directories(vmem-size-xrt PRIVATE "${XRT_INC_DIR}")
target_compile_definitions(vmem-size-xrt PRIVATE
  DEFAULT_XCLBIN_PATH=${FINAL_XCLBIN}
)
target_link_directories(vmem-size-xrt PRIVATE "${XRT_LIB_DIR}")
target_link_libraries(vmem-size-xrt PRIVATE
  benchmark::benchmark_main
  xrt_coreutil)
add_dependencies(vmem-size-xrt aie_kernel_artifacts)
