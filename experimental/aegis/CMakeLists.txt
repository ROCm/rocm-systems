cmake_minimum_required(VERSION 3.20)
project(AegisBit VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for editors
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Match LLVM's RTTI setting (LLVM is typically built without RTTI)
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()

# Find LLVM
# Prefer amd-staging build, then upstream build, then system LLVM 23
find_package(LLVM REQUIRED CONFIG
  PATHS
    "${CMAKE_SOURCE_DIR}/../llvm-project-amd-staging/build/lib/cmake/llvm"
    "${CMAKE_SOURCE_DIR}/../llvm-project/build/lib/cmake/llvm"
    "/usr/lib/llvm-23/lib/cmake/llvm"
  NO_DEFAULT_PATH
)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# LLVM components we need
set(LLVM_LINK_COMPONENTS
  Core
  Support
  MC
  MCDisassembler
  MCParser
  Object
  ObjectYAML
  BinaryFormat
  CodeGen              # MachineFunction, MachineInstr, RegScavenger
  Target               # TargetMachine, TargetSubtargetInfo
  DebugInfoDWARF       # DWARF .debug_line parsing for source mapping
  AMDGPUInfo
  AMDGPUDesc
  AMDGPUUtils
  AMDGPUDisassembler
  AMDGPUAsmParser
  AMDGPUCodeGen
)

# ROCm paths
set(ROCM_PATH "${CMAKE_SOURCE_DIR}/../rocm-systems" CACHE PATH "Path to ROCm systems")
set(ROCPROFILER_PATH "${ROCM_PATH}/projects/rocprofiler" CACHE PATH "Path to rocprofiler")

# Include directories
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${ROCPROFILER_PATH}/include
)

# Enable testing
enable_testing()

# Fuzz testing option (requires Clang)
option(ENABLE_FUZZ_TESTS "Build libFuzzer-based fuzz tests" OFF)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)
