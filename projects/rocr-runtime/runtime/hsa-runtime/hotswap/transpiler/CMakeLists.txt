cmake_minimum_required(VERSION 3.20)
project(hotswap-transpiler LANGUAGES CXX)

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
add_library(hotswap-transpiler STATIC
  raiser.cpp
  raise_context.cpp
  mc_state.cpp
  opcode_map.cpp
  canonicalize.cpp
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

target_compile_definitions(hotswap-transpiler PRIVATE
  LLVM_TOOLS_DIR="${LLVM_INSTALL_DIR}/bin"
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

# --- Batch raiser test (no HIP needed) ----------------------------------------
add_executable(batch_raise_test
  tests/batch_raise_test.cpp
)
target_link_libraries(batch_raise_test
  PRIVATE
    hotswap-transpiler
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
      hotswap-transpiler
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
      hotswap-transpiler
      hip::host
  )

  # Cross-architecture GPU test (gfx1200 -> gfx942)
  set(NATIVE_HSACO "/opt/rocm-7.2.1/lib/rocblas/library/Kernels.so-000-gfx942-xnack-.hsaco")
  set(SOURCE_HSACO "/opt/rocm-7.2.1/lib/rocblas/library/Kernels.so-000-gfx1200.hsaco")

  add_executable(cross_arch_gpu_test
    tests/cross_arch_gpu_test.cpp
  )
  set(VECADD_CO "${HOTSWAP_DIR}/build/mve_vecadd_gfx1250.co")

  target_compile_definitions(cross_arch_gpu_test PRIVATE
    NATIVE_HSACO_PATH="${NATIVE_HSACO}"
    SOURCE_HSACO_PATH="${SOURCE_HSACO}"
    VECADD_CO_PATH="${VECADD_CO}"
    LLVM_TOOLS_DIR="${LLVM_INSTALL_DIR}/bin"
  )
  target_link_libraries(cross_arch_gpu_test
    PRIVATE
      -Wl,--exclude-libs,ALL
      hotswap-transpiler
      hip::host
  )

  # gfx1250 → gfx942 end-to-end GPU test
  set(GFX1250_TEST_DATA "${CMAKE_CURRENT_SOURCE_DIR}/../test_data/gfx1250")
  add_executable(gfx1250_gpu_test
    tests/gfx1250_gpu_test.cpp
  )
  target_compile_definitions(gfx1250_gpu_test PRIVATE
    GFX1250_TEST_DATA_DIR="${GFX1250_TEST_DATA}"
    LLVM_TOOLS_DIR="${LLVM_INSTALL_DIR}/bin"
  )
  target_link_libraries(gfx1250_gpu_test
    PRIVATE
      -Wl,--exclude-libs,ALL
      hotswap-transpiler
      hip::host
  )

  # Integration test: runPipelineAllKernels + GPU execution
  add_executable(integration_test
    tests/integration_test.cpp
  )
  target_compile_definitions(integration_test PRIVATE
    GFX1250_TEST_DATA_DIR="${GFX1250_TEST_DATA}"
    LLVM_TOOLS_DIR="${LLVM_INSTALL_DIR}/bin"
  )
  target_link_libraries(integration_test
    PRIVATE
      -Wl,--exclude-libs,ALL
      hotswap-transpiler
      hip::host
  )

  message(STATUS "GPU tests enabled (HIP found)")
else()
  message(STATUS "GPU tests DISABLED (HIP not found)")
endif()
