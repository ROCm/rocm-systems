cmake_minimum_required(VERSION 3.20)

project(aie_tests)

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

# Map the NPU target to the hsaco arch section name.
if(NPU_TARGET STREQUAL "npu2")
  set(NPU_TARGET_ARCH "aie2p")
else()
  set(NPU_TARGET_ARCH "aie2")
endif()

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
          --aie-generate-npu-insts
          --aie-generate-pdi
          --no-compile-host
          --no-xchesscc
          --no-xbridge
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
# hsaco packaging: inject an arch section into a base ELF template.
# ---------------------------------------------------------------------------
# The base template only needs to be a valid ELF64 the loader's reader accepts;
# the arch section carries all AIE payloads. Generate a minimal object with the
# project compiler so nothing binary is checked in.
set(HSACO_TEMPLATE ${KERNEL_BUILD_DIR}/empty_hsaco_template.o)
add_custom_command(
  OUTPUT "${HSACO_TEMPLATE}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${KERNEL_BUILD_DIR}"
  COMMAND "${CMAKE_C_COMPILER}" -c -o "${HSACO_TEMPLATE}" -xc /dev/null
  COMMENT "Generating empty hsaco ELF template"
  VERBATIM
)

set(AIE_HSACO_PY ${CMAKE_CURRENT_SOURCE_DIR}/aie_hsaco.py)
set(TEST_HSACO ${KERNEL_BUILD_DIR}/vector_scalar_add.hsaco)
set(TEST_HSACO2 ${KERNEL_BUILD_DIR}/vector_scalar_add2.hsaco)
set(TEST_HSACO_NOPDI ${KERNEL_BUILD_DIR}/vector_scalar_add_nopdi.hsaco)

# KERNARG_SIZE=32 (2 ptr + 2 size = 4*8), NUM_COLS=1.
add_custom_command(
  OUTPUT "${TEST_HSACO}" "${TEST_HSACO2}" "${TEST_HSACO_NOPDI}"
  COMMAND "${CMAKE_COMMAND}" -E copy "${HSACO_TEMPLATE}" "${TEST_HSACO}"
  COMMAND "${Python3_EXECUTABLE}" "${AIE_HSACO_PY}"
          --hsaco "${TEST_HSACO}" --arch "${NPU_TARGET_ARCH}"
          --kernel "vector_scalar_add:${INSTS_BIN}:${DESIGN_PDI}:32:1"
  COMMAND "${CMAKE_COMMAND}" -E copy "${TEST_HSACO}" "${TEST_HSACO2}"
  COMMAND "${CMAKE_COMMAND}" -E copy "${HSACO_TEMPLATE}" "${TEST_HSACO_NOPDI}"
  COMMAND "${Python3_EXECUTABLE}" "${AIE_HSACO_PY}"
          --hsaco "${TEST_HSACO_NOPDI}" --arch "${NPU_TARGET_ARCH}"
          --kernel "vector_scalar_add:${INSTS_BIN}:32:1"
  DEPENDS "${INSTS_BIN}" "${DESIGN_PDI}" "${AIE_HSACO_PY}" "${HSACO_TEMPLATE}"
  COMMENT "Packaging AIE kernels into hsaco files"
  VERBATIM
)

add_custom_target(dispatch_hsaco
  DEPENDS "${TEST_HSACO}" "${TEST_HSACO2}" "${TEST_HSACO_NOPDI}"
)

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
target_compile_definitions(dispatch
  PRIVATE
    DEFAULT_PDI_PATH=${DESIGN_PDI}
    DEFAULT_INSTS_PATH=${INSTS_BIN}
    DEFAULT_HSACO_PATH=${TEST_HSACO}
    DEFAULT_HSACO2_PATH=${TEST_HSACO2}
    DEFAULT_HSACO_NOPDI_PATH=${TEST_HSACO_NOPDI}
)
target_link_libraries(dispatch
  PRIVATE
    hsa-runtime64::hsa-runtime64
    GTest::gtest_main
)
add_dependencies(dispatch dispatch_kernel_artifacts dispatch_hsaco)

gtest_discover_tests(dispatch)
