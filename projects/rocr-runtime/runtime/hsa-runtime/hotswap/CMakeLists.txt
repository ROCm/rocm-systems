# Optional standalone build for hotswap library (for testing outside ROCR)
cmake_minimum_required(VERSION 3.16)
project(rocr-hotswap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Map LLVM components to libraries
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

add_library(rocr-hotswap STATIC
  hotswap.cpp
  hotswap_rules.cpp
  trampoline.cpp
  transpiler.cpp
)

target_include_directories(rocr-hotswap PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(rocr-hotswap PUBLIC ${LLVM_LIBS})
target_compile_definitions(rocr-hotswap PUBLIC ROCR_HOTSWAP_ENABLED=1)
