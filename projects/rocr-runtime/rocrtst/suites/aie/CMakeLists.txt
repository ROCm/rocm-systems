cmake_minimum_required(VERSION 3.20)

project(aie_tests)

enable_testing()

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(hsa-runtime64 CONFIG REQUIRED NAMES hsa_runtime64 hsa-runtime64)

# memory test
add_executable(memory memory.cc)
set_target_properties(memory
  PROPERTIES
    CXX_STANDARD                17
    CXX_EXTENSIONS              OFF
    CXX_VISIBILITY_PRESET       hidden
    VISIBILITY_INLINES_HIDDEN   ON
    POSITION_INDEPENDENT_CODE   ON
)
target_compile_options(memory
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
target_link_libraries(memory
  PRIVATE
    hsa-runtime64::hsa-runtime64
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(memory)

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
# Kernel artifact build pipeline
# ---------------------------------------------------------------------------
set(KERNEL ${CMAKE_CURRENT_SOURCE_DIR}/vector_scalar_add.py)
set(KERNEL_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernel)
set(AIE_MLIR ${KERNEL_BUILD_DIR}/aie.mlir)
set(INSTS_BIN  ${KERNEL_BUILD_DIR}/insts.bin)
set(DESIGN_PDI ${KERNEL_BUILD_DIR}/design.pdi)

# Step 1: Python -> MLIR
add_custom_command(
  OUTPUT "${AIE_MLIR}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${KERNEL_BUILD_DIR}"
  COMMAND "${Python3_EXECUTABLE}" "${KERNEL}" "${NPU_TARGET}" > "${AIE_MLIR}"
  DEPENDS "${KERNEL}"
  COMMENT "Generating aie.mlir from vector_scalar_add.py (target: ${NPU_TARGET})"
  VERBATIM
)

# Step 2: MLIR -> insts.bin + design.pdi (via aiecc)
add_custom_command(
  OUTPUT "${INSTS_BIN}" "${DESIGN_PDI}"
  COMMAND aiecc
          --peano=${PEANO_INSTALL_DIR}
          --get-npu-insts
          --get-pdi
          --npu-insts-name=insts.bin
          --pdi-name=design.pdi
          "${AIE_MLIR}"
  WORKING_DIRECTORY "${KERNEL_BUILD_DIR}"
  DEPENDS "${AIE_MLIR}"
  COMMENT "Compiling AIE kernel with aiecc"
  VERBATIM
)

add_custom_target(dispatch_kernel_artifacts
  DEPENDS "${INSTS_BIN}" "${DESIGN_PDI}"
)

# ---------------------------------------------------------------------------
# Full-ELF kernel artifact
#
# The full-ELF design differs from the PDI one: its runtime sequence starts with
# aiex.npu.load_pdi, because nothing configures the array for it out of band. It
# also gets its own directory -- aiecc writes <input>.prj next to its input, so
# sharing a directory would make the two aiecc runs clobber each other's
# intermediates under 'make -j'.
# ---------------------------------------------------------------------------

# xclbinutil and aiebu-asm ship alongside XRT; aiecc needs both on PATH to
# assemble the control code embedded in the full ELF.
find_program(XCLBINUTIL xclbinutil
  HINTS "/opt/xilinx/xrt/bin/unwrapped" "/opt/xilinx/xrt/bin"
)
find_program(AIEBU_ASM aiebu-asm
  HINTS "/opt/xilinx/xrt/bin" "/opt/xilinx/aiebu/bin"
)

# aiecc drives aiebu-asm with the 'aie2_config' target. Older aiebu releases
# only know that target as 'config', and aiebu-asm exits 0 on an unrecognised
# target -- so aiecc reports success and never writes an ELF. Detect that here
# rather than letting the build silently produce nothing.
# Full-ELF dispatch is aie2p only -- the design's npu_load_pdi has no aie2 equivalent -- so there
# is nothing to build for an npu1 target even with a complete toolchain.
set(AIE_FULL_ELF_SUPPORTED FALSE)
if(NPU_TARGET STREQUAL "npu2" AND XCLBINUTIL AND AIEBU_ASM)
  execute_process(
    COMMAND "${AIEBU_ASM}" --help
    OUTPUT_VARIABLE AIEBU_ASM_HELP
    ERROR_VARIABLE AIEBU_ASM_HELP
  )
  if(AIEBU_ASM_HELP MATCHES "aie2_config")
    set(AIE_FULL_ELF_SUPPORTED TRUE)
  endif()
endif()
if(NOT AIE_FULL_ELF_SUPPORTED)
  message(STATUS
    "Full-ELF kernel will not be built (needs an npu2 target plus xclbinutil and an aiebu-asm "
    "supporting the 'aie2_config' target); the full-ELF dispatch tests will skip themselves.")
endif()

set(FULL_ELF_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/kernel_full_elf")
set(AIE_ELF_MLIR "${FULL_ELF_BUILD_DIR}/aie.mlir")
set(FULL_ELF "${FULL_ELF_BUILD_DIR}/aie.elf")

if(AIE_FULL_ELF_SUPPORTED)
  get_filename_component(XCLBINUTIL_DIR "${XCLBINUTIL}" DIRECTORY)
  get_filename_component(AIEBU_ASM_DIR "${AIEBU_ASM}" DIRECTORY)

  add_custom_command(
    OUTPUT "${AIE_ELF_MLIR}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${FULL_ELF_BUILD_DIR}"
    COMMAND "${Python3_EXECUTABLE}" "${KERNEL}" "${NPU_TARGET}" --full-elf > "${AIE_ELF_MLIR}"
    DEPENDS "${KERNEL}"
    COMMENT "Generating full-ELF aie.mlir from vector_scalar_add.py (target: ${NPU_TARGET})"
    VERBATIM
  )

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

  add_custom_target(dispatch_full_elf_artifacts DEPENDS "${FULL_ELF}")
endif()

# ---------------------------------------------------------------------------
# dispatch test executable
# ---------------------------------------------------------------------------
add_executable(dispatch dispatch.cc)
set_target_properties(dispatch
  PROPERTIES
    CXX_STANDARD                17
    CXX_EXTENSIONS              OFF
    CXX_VISIBILITY_PRESET       hidden
    VISIBILITY_INLINES_HIDDEN   ON
    POSITION_INDEPENDENT_CODE   ON
)
target_compile_options(dispatch
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
# Some tests are compiled out because the behaviour they check is not implemented
# yet; they abort the test binary rather than fail. Flip this on once the runtime
# reports asynchronous dispatch failures instead of aborting. See dispatch.cc.
option(AIE_TEST_ASYNC_ERRORS
  "Build the dispatch tests that require rejected dispatches to be observable" OFF)
if(AIE_TEST_ASYNC_ERRORS)
  target_compile_definitions(dispatch PRIVATE ROCRTST_AIE_ASYNC_ERROR_REPORTING)
endif()

target_compile_definitions(dispatch
  PRIVATE
    DEFAULT_PDI_PATH=${DESIGN_PDI}
    DEFAULT_INSTS_PATH=${INSTS_BIN}
    # The kernel is addressed as "<device>:<sequence>", the names aiecc writes
    # into aie.mlir.prj/full_elf_config.json. vector_scalar_add.py leaves the
    # aie.device and the runtime sequence unnamed, so aiecc defaults them to
    # "main" and "sequence".
    DEFAULT_ELF_PATH=${FULL_ELF}
    DEFAULT_ELF_KERNEL_NAME="main:sequence"
)
target_link_libraries(dispatch
  PRIVATE
    hsa-runtime64::hsa-runtime64
    GTest::gtest_main
)
add_dependencies(dispatch dispatch_kernel_artifacts)
if(AIE_FULL_ELF_SUPPORTED)
  add_dependencies(dispatch dispatch_full_elf_artifacts)
endif()

gtest_discover_tests(dispatch)
