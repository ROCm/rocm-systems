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
)
target_link_libraries(dispatch
  PRIVATE
    hsa-runtime64::hsa-runtime64
    GTest::gtest_main
)
add_dependencies(dispatch dispatch_kernel_artifacts)

gtest_discover_tests(dispatch)
