cmake_minimum_required(VERSION 3.20)
project(rccl-build-server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------------------------------------------------
# Find LLVM/Clang/LLD from AOMP (or any LLVM installation with dev files).
#
# Usage:
#   cmake -B build -DLLVM_DIR=/work/lmeadows/rocm/aomp_23.0-1/lib/llvm/lib/cmake/llvm \
#                  -DClang_DIR=/work/lmeadows/rocm/aomp_23.0-1/lib/llvm/lib/cmake/clang \
#                  -DLLD_DIR=/work/lmeadows/rocm/aomp_23.0-1/lib/llvm/lib/cmake/lld
# -------------------------------------------------------------------

# Satisfy transitive dependencies that LLVM's CMake config may require.
# We link against the shared libLLVM.so which already bundles these, but
# CMake's imported targets still need them resolved.
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "")
set(zstd_FOUND TRUE CACHE BOOL "")
if(NOT TARGET zstd::libzstd_shared)
  add_library(zstd::libzstd_shared SHARED IMPORTED)
  set_target_properties(zstd::libzstd_shared PROPERTIES
    IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/libzstd.so.1"
    INTERFACE_INCLUDE_DIRECTORIES "")
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
find_package(LLD REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "  LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "  LLVM_LIBRARY_DIRS: ${LLVM_LIBRARY_DIRS}")
message(STATUS "Found Clang")
message(STATUS "Found LLD")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()
if(NOT LLVM_ENABLE_EH)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
endif()

add_executable(rccl-build-server
  src/main.cpp
)

target_include_directories(rccl-build-server PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
)

# Link against the shared LLVM dylib and clang-cpp (all-in-one shared libs).
# This avoids listing hundreds of individual static libraries.
target_link_libraries(rccl-build-server PRIVATE
  LLVM
  clang-cpp
  lldELF
  lldCommon
)
