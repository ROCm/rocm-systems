cmake_minimum_required(VERSION 3.20)
project(hotswap-transpiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Paths to LLVM ------------------------------------------------------------
# We require an LLVM *build tree* (not an install tree) because the raiser uses
# AMDGPU target-private headers (SIDefines.h, SIInstrInfo.h, AMDGPUBaseInfo.h)
# and the TableGen-generated AMDGPUGen*.inc files.  Standard `ninja install`
# does not copy those into the install prefix.
#
# Point LLVM_DIR at <llvm-build>/lib/cmake/llvm.  This is what every other
# LLVM out-of-tree project does (clang-tools-extra, bolt, etc.).
#
# Example:
#   cmake -DLLVM_DIR=$HOME/llvm-build/lib/cmake/llvm ..
find_package(LLVM REQUIRED CONFIG)

if(NOT DEFINED LLVM_BUILD_MAIN_SRC_DIR OR NOT LLVM_BUILD_MAIN_SRC_DIR OR
   NOT DEFINED LLVM_BINARY_DIR OR NOT LLVM_BINARY_DIR)
  message(FATAL_ERROR
    "hotswap-transpiler requires LLVM_DIR to point at a build tree, not an "
    "install tree. The install tree does not expose AMDGPU target-private "
    "headers (SIDefines.h, AMDGPUBaseInfo.h, AMDGPUGen*.inc). "
    "Pass -DLLVM_DIR=<llvm-build>/lib/cmake/llvm.")
endif()

message(STATUS "Using LLVM from: ${LLVM_DIR}")
message(STATUS "LLVM source tree: ${LLVM_BUILD_MAIN_SRC_DIR}")
message(STATUS "LLVM build tree:  ${LLVM_BINARY_DIR}")
message(STATUS "LLVM version:     ${LLVM_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

# --- Include paths ------------------------------------------------------------
include_directories(${LLVM_INCLUDE_DIRS})

# --- Link paths ---------------------------------------------------------------
link_directories(${LLVM_LIBRARY_DIRS})

# --- LLVM definitions ---------------------------------------------------------
add_definitions(${LLVM_DEFINITIONS})

# --- Raiser + Pipeline library ------------------------------------------------
add_library(hotswap-transpiler STATIC
  raiser.cpp
  raise_context.cpp
  mc_state.cpp
  opcode_map.cpp
  handle_sopp.cpp
  handle_smem.cpp
  handle_sopc.cpp
  handle_sop1.cpp
  handle_sopk.cpp
  handle_sop2.cpp
  handle_valu.cpp
  handle_flat.cpp
  handle_ds.cpp
  handle_mubuf.cpp
  handle_mfma.cpp
  handle_vopd.cpp
  wmma_lowering.cpp
  pipeline.cpp
  code_object_utils.cpp
)

# AMDGPU target-private headers (not installed by `ninja install`).
# SIDefines.h, SIInstrInfo.h → source tree Target/AMDGPU/{,Utils}
# AMDGPUGenInstrInfo.inc etc.  → build tree Target/AMDGPU
target_include_directories(hotswap-transpiler PRIVATE
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU"
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU/Utils"
  "${LLVM_BINARY_DIR}/lib/Target/AMDGPU"
)

# `llvm-mc`, `llvm-objdump` etc. live in the build tree's bin/.
target_compile_definitions(hotswap-transpiler PRIVATE
  LLVM_TOOLS_DIR="${LLVM_TOOLS_BINARY_DIR}"
)

target_link_libraries(hotswap-transpiler
  PUBLIC
    # LLVM AMDGPU
    LLVMAMDGPUDisassembler
    LLVMAMDGPUCodeGen
    LLVMAMDGPUAsmParser
    LLVMAMDGPUDesc
    LLVMAMDGPUInfo
    LLVMAMDGPUUtils
    # LLVM CodeGen
    LLVMCodeGen
    LLVMCodeGenTypes
    LLVMTarget
    LLVMAsmPrinter
    LLVMSelectionDAG
    LLVMGlobalISel
    LLVMMIRParser
    # LLVM MC
    LLVMMCDisassembler
    LLVMMC
    LLVMMCParser
    # LLVM Core + Support
    LLVMCore
    LLVMSupport
    LLVMObject
    LLVMBinaryFormat
    LLVMRemarks
    LLVMBitReader
    LLVMBitWriter
    LLVMBitstreamReader
    LLVMScalarOpts
    LLVMTransformUtils
    LLVMAnalysis
    LLVMipo
    LLVMIRReader
    LLVMAsmParser
    LLVMInstCombine
    LLVMAggressiveInstCombine
    LLVMInstrumentation
    LLVMVectorize
    LLVMPasses
    LLVMProfileData
    LLVMDebugInfoDWARF
    LLVMDebugInfoCodeView
    LLVMCFGuard
    LLVMSymbolize
    LLVMDebugInfoMSF
    LLVMDebugInfoPDB
    LLVMDemangle
    LLVMTextAPI
    LLVMFrontendOffloading
    LLVMFrontendOpenMP
    LLVMLinker
    LLVMObjCARCOpts
)

# --- GoogleTest-based unified test binary -------------------------------------
find_package(GTest REQUIRED)
enable_testing()
include(GoogleTest)

set(TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../test_data")
set(HOTSWAP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(AITER_CORPUS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/aiter_gfx950")
set(GFX1250_TEST_DATA "${CMAKE_CURRENT_SOURCE_DIR}/../test_data/gfx1250")

set(TEST_SOURCES
  tests/test_main.cpp
  tests/batch_raise_test.cpp
  tests/corpus_test.cpp
)

find_package(hip QUIET)
if(hip_FOUND)
  set(MVE_CO "${HOTSWAP_DIR}/build/mve_vecadd_gfx942.co")
  set(MFMA_CO "${CMAKE_CURRENT_SOURCE_DIR}/build/mfma_gemm_gfx942_unbundled.co")
  set(NATIVE_HSACO "/opt/rocm-7.2.1/lib/rocblas/library/Kernels.so-000-gfx942-xnack-.hsaco")
  set(SOURCE_HSACO "/opt/rocm-7.2.1/lib/rocblas/library/Kernels.so-000-gfx1200.hsaco")
  set(VECADD_CO "${HOTSWAP_DIR}/build/mve_vecadd_gfx1250.co")

  list(APPEND TEST_SOURCES
    tests/ir_gpu_test.cpp
    tests/mfma_gpu_test.cpp
    tests/gfx1250_gpu_test.cpp
    tests/integration_test.cpp
    tests/cross_arch_gpu_test.cpp
  )
  message(STATUS "GPU tests enabled (HIP found)")
else()
  message(STATUS "GPU tests DISABLED (HIP not found)")
endif()

add_executable(transpiler_tests ${TEST_SOURCES})

target_compile_definitions(transpiler_tests PRIVATE
  TRANSPILER_SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  TEST_DATA_DIR="${TEST_DATA_DIR}"
  AITER_CORPUS_DIR="${AITER_CORPUS_DIR}"
  LLVM_TOOLS_DIR="${LLVM_TOOLS_BINARY_DIR}"
)

if(hip_FOUND)
  target_compile_definitions(transpiler_tests PRIVATE
    MVE_CO_PATH="${MVE_CO}"
    MFMA_CO_PATH="${MFMA_CO}"
    NATIVE_HSACO_PATH="${NATIVE_HSACO}"
    SOURCE_HSACO_PATH="${SOURCE_HSACO}"
    VECADD_CO_PATH="${VECADD_CO}"
    GFX1250_TEST_DATA_DIR="${GFX1250_TEST_DATA}"
  )
endif()

target_link_libraries(transpiler_tests
  PRIVATE
    -Wl,--exclude-libs,ALL
    hotswap-transpiler
    GTest::GTest
)

if(hip_FOUND)
  target_link_libraries(transpiler_tests PRIVATE hip::host)
endif()

gtest_discover_tests(transpiler_tests
  PROPERTIES TIMEOUT 120 LABELS "transpiler"
)

# Include expected-failure (XFAIL) properties after test discovery.
# The xfail.cmake file uses set_tests_properties(... WILL_FAIL TRUE)
# so that known-failing tests are run and expected to fail.  If one
# unexpectedly passes, CTest flags it.
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/xfail.cmake"
)
