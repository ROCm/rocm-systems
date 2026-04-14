cmake_minimum_required(VERSION 3.20)
project(llvm-mir-proto LANGUAGES CXX)

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

# --- Lifter + Pipeline library ------------------------------------------------
add_library(llvm-mir-lifter STATIC
  lifter.cpp
  pipeline.cpp
  code_object_utils.cpp
)

target_compile_definitions(llvm-mir-lifter PRIVATE
  LLVM_TOOLS_DIR="${LLVM_INSTALL_DIR}/bin"
)

target_link_libraries(llvm-mir-lifter
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

# --- GPU test -----------------------------------------------------------------
find_package(hip QUIET)
if(hip_FOUND)
  set(HOTSWAP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
  set(MVE_CO "${HOTSWAP_DIR}/build/mve_vecadd_gfx942.co")

  add_executable(mir_gpu_test
    tests/mir_gpu_test.cpp
  )
  target_compile_definitions(mir_gpu_test PRIVATE
    MVE_CO_PATH="${MVE_CO}"
    LLVM_TOOLS_DIR="${LLVM_INSTALL_DIR}/bin"
  )
  target_link_libraries(mir_gpu_test
    PRIVATE
      -Wl,--exclude-libs,ALL
      llvm-mir-lifter
      hip::host
  )
  message(STATUS "GPU test enabled (HIP found)")
else()
  message(STATUS "GPU test DISABLED (HIP not found)")
endif()
