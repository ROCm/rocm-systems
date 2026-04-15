cmake_minimum_required(VERSION 3.20)
project(llvm-ir-proto LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Paths to LLVM ------------------------------------------------------------
set(LLVM_INSTALL_DIR "$ENV{HOME}/shared-llvm" CACHE PATH "LLVM install prefix")

# --- Find LLVM ----------------------------------------------------------------
list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}")
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using LLVM from: ${LLVM_DIR}")
message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

# --- Include paths ------------------------------------------------------------
include_directories(${LLVM_INCLUDE_DIRS})

# --- Link paths ---------------------------------------------------------------
link_directories(${LLVM_INSTALL_DIR}/lib)

# --- LLVM definitions ---------------------------------------------------------
add_definitions(${LLVM_DEFINITIONS})

# --- Raiser + Pipeline library ------------------------------------------------
add_library(llvm-ir-raiser STATIC
  raiser.cpp
  pipeline.cpp
  code_object_utils.cpp
)

target_compile_definitions(llvm-ir-raiser PRIVATE
  LLVM_TOOLS_DIR="${LLVM_INSTALL_DIR}/bin"
)

target_link_libraries(llvm-ir-raiser
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

# --- Batch raiser test (no HIP needed) ----------------------------------------
add_executable(batch_raise_test
  tests/batch_raise_test.cpp
)
target_link_libraries(batch_raise_test
  PRIVATE
    llvm-ir-raiser
)

# --- GPU tests ----------------------------------------------------------------
find_package(hip QUIET)
if(hip_FOUND)
  set(HOTSWAP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
  set(MVE_CO "${HOTSWAP_DIR}/build/mve_vecadd_gfx942.co")
  set(MFMA_CO "${CMAKE_CURRENT_SOURCE_DIR}/build/mfma_gemm_gfx942_unbundled.co")

  # vecadd GPU test
  add_executable(ir_gpu_test
    tests/ir_gpu_test.cpp
  )
  target_compile_definitions(ir_gpu_test PRIVATE
    MVE_CO_PATH="${MVE_CO}"
    LLVM_TOOLS_DIR="${LLVM_INSTALL_DIR}/bin"
  )
  target_link_libraries(ir_gpu_test
    PRIVATE
      -Wl,--exclude-libs,ALL
      llvm-ir-raiser
      hip::host
  )

  # MFMA GPU test
  add_executable(mfma_gpu_test
    tests/mfma_gpu_test.cpp
  )
  target_compile_definitions(mfma_gpu_test PRIVATE
    MFMA_CO_PATH="${MFMA_CO}"
    LLVM_TOOLS_DIR="${LLVM_INSTALL_DIR}/bin"
  )
  target_link_libraries(mfma_gpu_test
    PRIVATE
      -Wl,--exclude-libs,ALL
      llvm-ir-raiser
      hip::host
  )

  message(STATUS "GPU tests enabled (HIP found)")
else()
  message(STATUS "GPU tests DISABLED (HIP not found)")
endif()
