################################################################################
# RcclSmokeTest — standalone build
#
# Can be built independently (no full RCCL source tree required).
# By default links against the installed RCCL in /opt/rocm.
#
# Quick build:
#   mkdir build && cd build
#   cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
#   make -j$(nproc)
#
# Custom RCCL install:
#   cmake .. -DRCCL_PATH=/path/to/rccl/install
################################################################################

cmake_minimum_required(VERSION 3.16)
project(RcclSmokeTest LANGUAGES CXX HIP)

# ── version ───────────────────────────────────────────────────────────────────

set(SMOKETEST_VERSION_MAJOR 1)
set(SMOKETEST_VERSION_MINOR 0)
set(SMOKETEST_VERSION_PATCH 0)

find_package(Git QUIET)
if(Git_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE SMOKETEST_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()
if(NOT SMOKETEST_GIT_COMMIT)
  set(SMOKETEST_GIT_COMMIT "unknown")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# ── ROCm / HIP ────────────────────────────────────────────────────────────────

# Respect ROCM_PATH env var or default to /opt/rocm.
if(NOT DEFINED ROCM_PATH)
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "$ENV{ROCM_PATH}")
  else()
    set(ROCM_PATH "/opt/rocm")
  endif()
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

find_package(hip REQUIRED CONFIG
  PATHS "${ROCM_PATH}/lib/cmake/hip" "${ROCM_PATH}"
)

# ── RCCL ──────────────────────────────────────────────────────────────────────

# Allow caller to point at a non-standard RCCL installation.
if(NOT DEFINED RCCL_PATH)
  set(RCCL_PATH "${ROCM_PATH}")
endif()

find_library(RCCL_LIB
  NAMES rccl
  HINTS "${RCCL_PATH}/lib" "${RCCL_PATH}/lib64"
  REQUIRED
)

find_path(RCCL_INCLUDE_DIR
  NAMES rccl/rccl.h rccl.h
  HINTS "${RCCL_PATH}/include"
  REQUIRED
)

message(STATUS "RCCL library : ${RCCL_LIB}")
message(STATUS "RCCL include : ${RCCL_INCLUDE_DIR}")

# ── target ────────────────────────────────────────────────────────────────────

add_executable(RcclSmokeTest RcclSmokeTest.cpp)
set_source_files_properties(RcclSmokeTest.cpp PROPERTIES LANGUAGE HIP)

target_include_directories(RcclSmokeTest PRIVATE "${RCCL_INCLUDE_DIR}")

target_link_libraries(RcclSmokeTest PRIVATE
  hip::device     # HIP runtime + device code
  "${RCCL_LIB}"  # RCCL / librccl.so
)

target_compile_options(RcclSmokeTest PRIVATE
  -Wall
  -Wextra
  -Wno-unused-parameter
)

target_compile_definitions(RcclSmokeTest PRIVATE
  SMOKETEST_VERSION_MAJOR=${SMOKETEST_VERSION_MAJOR}
  SMOKETEST_VERSION_MINOR=${SMOKETEST_VERSION_MINOR}
  SMOKETEST_VERSION_PATCH=${SMOKETEST_VERSION_PATCH}
  SMOKETEST_GIT_COMMIT="${SMOKETEST_GIT_COMMIT}"
)

install(TARGETS RcclSmokeTest DESTINATION bin)
