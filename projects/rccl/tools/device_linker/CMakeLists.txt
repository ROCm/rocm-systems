# Device linker executable only.
# Build by hand from this directory. Binary is placed here (outside main build tree).
#
#   cmake -B build -S . -DLLVM_DIR=/path/to/llvm
#   cmake --build build
#
# Output: device_linker (in this directory, tools/device_linker/)
#
# The main RCCL build (install.sh -l --device-linker) does NOT build this binary;
# it expects tools/device_linker/device_linker to exist. Build this first when using --device-linker.
#
# Full pipeline (specialized kernels + dispatcher + rccl + device_link) is in the repo root
# CMakeLists.txt when DEVICE_LINKER=ON; that pipeline runs this binary.
# Backup of the previous standalone full-pipeline CMake: CMakeLists_full_pipeline.txt
#
cmake_minimum_required(VERSION 3.21)
project(rccl_device_linker LANGUAGES CXX C)

set(DEVICE_LINKER_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# LLVM (required for DWARF linking)
message(STATUS "Using LLVM_DIR ${LLVM_DIR}")


find_package(LLVM REQUIRED CONFIG PATHS "${LLVM_DIR}/lib/cmake" NO_DEFAULT_PATH)

message(STATUS "Found LLVM ${LLVM_VERSION}")

set(LLVM_CONFIG_PATH "${LLVM_TOOLS_BINARY_DIR}/llvm-config")
if(EXISTS "${LLVM_CONFIG_PATH}")
  execute_process(
    COMMAND ${LLVM_CONFIG_PATH} --ldflags --libs
    OUTPUT_VARIABLE LLVM_LINK_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE LLVM_CONFIG_RESULT
  )
  if(LLVM_CONFIG_RESULT EQUAL 0 AND LLVM_LINK_FLAGS)
    string(REPLACE "\n" " " LLVM_LINK_FLAGS "${LLVM_LINK_FLAGS}")
    string(STRIP "${LLVM_LINK_FLAGS}" LLVM_LINK_FLAGS)
    string(REGEX MATCH "-L([^ ]+)" LLVM_LIBDIR_MATCH "${LLVM_LINK_FLAGS}")
    if(LLVM_LIBDIR_MATCH)
      string(REGEX REPLACE "-L" "" LLVM_LIBDIR "${LLVM_LIBDIR_MATCH}")
      string(STRIP "${LLVM_LIBDIR}" LLVM_LIBDIR)
    endif()
    string(REGEX MATCH "-l([^ ]+)" LLVM_LIBNAME_MATCH "${LLVM_LINK_FLAGS}")
    if(LLVM_LIBNAME_MATCH)
      string(REGEX REPLACE "-l" "" LLVM_LIBNAME "${LLVM_LIBNAME_MATCH}")
      string(STRIP "${LLVM_LIBNAME}" LLVM_LIBNAME)
    endif()
    message(STATUS "Using llvm-config: libdir='${LLVM_LIBDIR}', lib='${LLVM_LIBNAME}'")
  else()
    message(STATUS "llvm-config failed (result=${LLVM_CONFIG_RESULT}), using known paths")
  endif()
else()
  message(STATUS "llvm-config not found, using known LLVM paths")
endif()

add_executable(device_linker
  ${DEVICE_LINKER_DIR}/device_linker.cpp
  ${DEVICE_LINKER_DIR}/CustomStreamer.cpp
)
target_compile_features(device_linker PRIVATE cxx_std_17)
# LLVM headers; Classic subdir so CustomStreamer.h can #include "DWARFLinker.h"
target_include_directories(device_linker PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}/llvm/DWARFLinker/Classic
  ${DEVICE_LINKER_DIR}
)
target_compile_options(device_linker PRIVATE -fno-rtti -fno-exceptions)
target_compile_definitions(device_linker PRIVATE
  _GNU_SOURCE
  __STDC_CONSTANT_MACROS
  __STDC_FORMAT_MACROS
  __STDC_LIMIT_MACROS
)
target_link_directories(device_linker PRIVATE ${LLVM_LIBDIR})
target_link_libraries(device_linker PRIVATE ${LLVM_LIBNAME})

# Binary lives in tools/device_linker/ (outside main build tree)
set_target_properties(device_linker PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${DEVICE_LINKER_DIR}
)

message(STATUS "Device linker executable will be built to: ${DEVICE_LINKER_DIR}/device_linker")
