# Optional standalone build for hotswap library (for testing outside ROCR)
cmake_minimum_required(VERSION 3.20)
project(rocr-hotswap LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

llvm_map_components_to_libnames(LLVM_LIBS
  AMDGPUDisassembler
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUAsmParser
  MC
  MCParser
  MCDisassembler
  Support
)

# ── waveasm MLIR dialect ─────────────────────────────────────────────────────
add_subdirectory(waveasm)

# ── lifter library (MCInst -> waveasm MLIR) ──────────────────────────────────
add_library(rocr-lifter STATIC
  lifter.cpp
  ssa_construction.cpp
  cross_target.cpp
  wave_width.cpp
  emit_assembly.cpp
  pipeline.cpp
)
target_include_directories(rocr-lifter PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/waveasm/include
  ${CMAKE_CURRENT_BINARY_DIR}/waveasm/include
)
target_link_libraries(rocr-lifter PUBLIC
  MLIRWaveASMDialect
  MLIRWaveASMTransforms
  MLIRWaveASMTarget
  MLIRIR
  MLIRSupport
  MLIRPass
  ${LLVM_LIBS}
)

# ── hotswap library (existing) ───────────────────────────────────────────────
add_library(rocr-hotswap STATIC
  hotswap.cpp
  hotswap_rules.cpp
  trampoline.cpp
  transpiler.cpp
)

target_include_directories(rocr-hotswap PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../inc
)

target_link_libraries(rocr-hotswap PUBLIC ${LLVM_LIBS})
target_compile_definitions(rocr-hotswap PUBLIC ROCR_HOTSWAP_ENABLED=1)

# ── code-object builder library ───────────────────────────────────────────────
add_library(rocr-co-builder STATIC
  code_object_builder.cpp
)
target_include_directories(rocr-co-builder PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(rocr-co-builder PUBLIC rocr-lifter)

# ── tests ────────────────────────────────────────────────────────────────────
add_executable(hotswap_test tests/hotswap_test.cpp)
target_link_libraries(hotswap_test PRIVATE rocr-hotswap)

add_executable(waveasm_roundtrip_test tests/waveasm_roundtrip_test.cpp)
target_include_directories(waveasm_roundtrip_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/waveasm/include
  ${CMAKE_CURRENT_BINARY_DIR}/waveasm/include
)
target_link_libraries(waveasm_roundtrip_test PRIVATE
  MLIRWaveASMDialect
  MLIRWaveASMTransforms
  MLIRWaveASMTarget
  MLIRIR
  MLIRSupport
  MLIRPass
  ${LLVM_LIBS}
)

add_executable(lifter_test tests/lifter_test.cpp)
target_link_libraries(lifter_test PRIVATE rocr-lifter)

# ── MVE GPU round-trip test ──────────────────────────────────────────────────
set(MVE_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tests/mve_vecadd.hip)
set(MVE_HSACO      ${CMAKE_CURRENT_BINARY_DIR}/mve_vecadd.hsaco)
set(MVE_CO         ${CMAKE_CURRENT_BINARY_DIR}/mve_vecadd_gfx942.co)
set(MVE_DEVICE_ASM ${CMAKE_CURRENT_BINARY_DIR}/mve_vecadd_device.s)
set(CLANG_BUNDLER  /opt/rocm/llvm/bin/clang-offload-bundler)

add_custom_command(
  OUTPUT ${MVE_HSACO}
  COMMAND hipcc --genco --offload-arch=gfx942 -O2 -o ${MVE_HSACO} ${MVE_KERNEL_SRC}
  DEPENDS ${MVE_KERNEL_SRC}
  COMMENT "Compiling MVE kernel to code object bundle"
)

add_custom_command(
  OUTPUT ${MVE_CO}
  COMMAND ${CLANG_BUNDLER} --type=o
    --input=${MVE_HSACO}
    --targets=hip-amdgcn-amd-amdhsa--gfx942
    --output=${MVE_CO}
    --unbundle
  DEPENDS ${MVE_HSACO}
  COMMENT "Unbundling device code object"
)

add_custom_command(
  OUTPUT ${MVE_DEVICE_ASM}
  COMMAND hipcc -S --offload-arch=gfx942 -O2 -o ${MVE_DEVICE_ASM}.tmp ${MVE_KERNEL_SRC}
  COMMAND ${CMAKE_COMMAND} -DINPUT=${MVE_DEVICE_ASM}.tmp -DOUTPUT=${MVE_DEVICE_ASM}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/extract_device_asm.cmake
  DEPENDS ${MVE_KERNEL_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/extract_device_asm.cmake
  COMMENT "Extracting device assembly for MVE template"
)

# ── gfx1250 (RDNA4) build for cross-ISA test ────────────────────────────────
set(MVE_HSACO_1250    ${CMAKE_CURRENT_BINARY_DIR}/mve_vecadd_gfx1250.hsaco)
set(MVE_CO_1250       ${CMAKE_CURRENT_BINARY_DIR}/mve_vecadd_gfx1250.co)
set(MVE_DEVICE_ASM_1250 ${CMAKE_CURRENT_BINARY_DIR}/mve_vecadd_gfx1250_device.s)

add_custom_command(
  OUTPUT ${MVE_HSACO_1250}
  COMMAND hipcc --genco --offload-arch=gfx1250 -O2 -o ${MVE_HSACO_1250} ${MVE_KERNEL_SRC}
  DEPENDS ${MVE_KERNEL_SRC}
  COMMENT "Compiling MVE kernel to gfx1250 code object bundle"
)

add_custom_command(
  OUTPUT ${MVE_CO_1250}
  COMMAND ${CLANG_BUNDLER} --type=o
    --input=${MVE_HSACO_1250}
    --targets=hip-amdgcn-amd-amdhsa--gfx1250
    --output=${MVE_CO_1250}
    --unbundle
  DEPENDS ${MVE_HSACO_1250}
  COMMENT "Unbundling gfx1250 device code object"
)

add_custom_command(
  OUTPUT ${MVE_DEVICE_ASM_1250}
  COMMAND hipcc -S --offload-arch=gfx1250 -O2 -o ${MVE_DEVICE_ASM_1250}.tmp ${MVE_KERNEL_SRC}
  COMMAND ${CMAKE_COMMAND} -DINPUT=${MVE_DEVICE_ASM_1250}.tmp -DOUTPUT=${MVE_DEVICE_ASM_1250}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/extract_device_asm.cmake
  DEPENDS ${MVE_KERNEL_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/extract_device_asm.cmake
  COMMENT "Extracting gfx1250 device assembly for cross-ISA test"
)

add_custom_target(mve_kernel_artifacts
  DEPENDS ${MVE_CO} ${MVE_DEVICE_ASM} ${MVE_CO_1250} ${MVE_DEVICE_ASM_1250}
)

find_package(hip QUIET CONFIG HINTS /opt/rocm/lib/cmake/hip)

if(hip_FOUND)
  add_executable(mve_gpu_test tests/mve_gpu_test.cpp)
  target_link_libraries(mve_gpu_test PRIVATE rocr-co-builder hip::host)
  target_compile_definitions(mve_gpu_test PRIVATE
    MVE_CO_PATH="${MVE_CO}"
    MVE_DEVICE_ASM_PATH="${MVE_DEVICE_ASM}"
  )
  add_dependencies(mve_gpu_test mve_kernel_artifacts)

  add_executable(mve_cross_isa_test tests/mve_cross_isa_test.cpp)
  target_link_libraries(mve_cross_isa_test PRIVATE rocr-co-builder hip::host)
  target_compile_definitions(mve_cross_isa_test PRIVATE
    MVE_CO_1250_PATH="${MVE_CO_1250}"
    MVE_GFX942_TEMPLATE_PATH="${MVE_DEVICE_ASM}"
  )
  add_dependencies(mve_cross_isa_test mve_kernel_artifacts)
else()
  message(STATUS "HIP not found -- skipping mve_gpu_test")
endif()
