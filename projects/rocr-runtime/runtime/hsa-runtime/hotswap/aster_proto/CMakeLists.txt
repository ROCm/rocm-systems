cmake_minimum_required(VERSION 3.20)
project(aster-proto LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Paths to Aster and LLVM --------------------------------------------------
set(ASTER_SRC_DIR "$ENV{HOME}/aster" CACHE PATH "Aster source directory")
set(ASTER_BUILD_DIR "${ASTER_SRC_DIR}/build" CACHE PATH "Aster build directory")
set(LLVM_INSTALL_DIR "$ENV{HOME}/shared-llvm" CACHE PATH "LLVM install prefix")
set(LLVM_BUILD_DIR "$ENV{HOME}/llvm-build" CACHE PATH "LLVM build directory")

# --- Find MLIR/LLVM -----------------------------------------------------------
list(APPEND CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${LLVM_BUILD_DIR}")
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIR from: ${MLIR_DIR}")
message(STATUS "Using LLVM from: ${LLVM_DIR}")
message(STATUS "Using Aster source: ${ASTER_SRC_DIR}")
message(STATUS "Using Aster build: ${ASTER_BUILD_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

# --- Include paths ------------------------------------------------------------
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
# Aster source headers
include_directories(${ASTER_SRC_DIR}/include)
# Aster generated headers (TableGen output)
include_directories(${ASTER_BUILD_DIR}/include)

# --- Aster and LLVM libraries (from build tree + install) ---------------------
link_directories(${ASTER_BUILD_DIR}/lib)
link_directories(${LLVM_INSTALL_DIR}/lib)

# --- LLVM definitions ---------------------------------------------------------
add_definitions(${LLVM_DEFINITIONS})

# --- Lifter + Pipeline library ------------------------------------------------
add_library(aster-proto-lifter STATIC
  lifter.cpp
  pipeline.cpp
  code_object_utils.cpp
)

target_compile_definitions(aster-proto-lifter PRIVATE
  LLVM_TOOLS_DIR="${LLVM_INSTALL_DIR}/bin"
)

target_link_libraries(aster-proto-lifter
  PUBLIC
    # Aster libraries (from build tree)
    AMDGCNDialect
    AMDGCNTransforms
    AMDGCNAnalysis
    AsterExportASM
    AsterInterfaces
    ASTERIR
    ASTERSupport
    ASTERInit
    AsterLegalizer
    AsterCodeGen
    AsterCodeGenTransforms
    AsterTransforms
    AsterAMDGCNCodeGen
    AsterLSIRCodeGen
    LSIRDialect
    LayoutDialect
    LayoutTransforms
    AsterUtilsDialect
    AsterUtilsTransforms
    ASTERAnalysis
    MLIRNormalFormDialect
    MLIRNormalFormTransforms
    # MLIR core
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRArithDialect
    MLIRTransformUtils
    MLIRTransforms
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRMathDialect
    MLIRAffineDialect
    MLIRBufferizationDialect
    MLIRComplexDialect
    MLIRVectorDialect
    MLIRPtrDialect
    MLIRPtrMemorySpaceInterfaces
    # LLVM AMDGPU
    LLVMAMDGPUDisassembler
    LLVMAMDGPUCodeGen
    LLVMAMDGPUAsmParser
    LLVMAMDGPUDesc
    LLVMAMDGPUInfo
    LLVMAMDGPUUtils
    LLVMMCDisassembler
    LLVMMC
    LLVMMCParser
    LLVMSupport
    LLVMObject
    LLVMCore
    LLVMTarget
    LLVMCodeGen
    LLVMAsmPrinter
    LLVMCodeGenTypes
    LLVMScalarOpts
    LLVMTransformUtils
    LLVMAnalysis
    LLVMBinaryFormat
    LLVMRemarks
    LLVMBitReader
    LLVMBitWriter
    LLVMBitstreamReader
    LLVMSelectionDAG
    LLVMMIRParser
    LLVMGlobalISel
    LLVMipo
    LLVMFrontendOffloading
    LLVMFrontendOpenMP
    LLVMLinker
    LLVMIRReader
    LLVMAsmParser
    LLVMInstCombine
    LLVMAggressiveInstCombine
    LLVMInstrumentation
    LLVMVectorize
    LLVMPasses
    LLVMProfileData
    LLVMObjCARCOpts
    LLVMDebugInfoDWARF
    LLVMDebugInfoCodeView
    LLVMCFGuard
    LLVMSymbolize
    LLVMDebugInfoMSF
    LLVMDebugInfoPDB
    LLVMDemangle
    LLVMTextAPI
    LLVMObject
)

# --- GPU test -----------------------------------------------------------------
find_package(hip QUIET)
if(hip_FOUND)
  # Build the MVE artifacts (reuse from parent build or build our own)
  set(HOTSWAP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
  set(MVE_CO "${HOTSWAP_DIR}/build/mve_vecadd_gfx942.co")
  set(MVE_DEVICE_ASM "${HOTSWAP_DIR}/build/mve_vecadd_device.s")

  add_executable(aster_gpu_test
    tests/aster_gpu_test.cpp
  )
  target_compile_definitions(aster_gpu_test PRIVATE
    MVE_CO_PATH="${MVE_CO}"
    MVE_DEVICE_ASM_PATH="${MVE_DEVICE_ASM}"
    LLVM_TOOLS_DIR="${LLVM_INSTALL_DIR}/bin"
  )
  target_link_libraries(aster_gpu_test
    PRIVATE
      -Wl,--exclude-libs,ALL
      aster-proto-lifter
      hip::host
  )
  message(STATUS "GPU test enabled (HIP found)")
else()
  message(STATUS "GPU test DISABLED (HIP not found)")
endif()
