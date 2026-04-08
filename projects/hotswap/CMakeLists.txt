cmake_minimum_required(VERSION 3.16)
project(hotswap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# HSA runtime headers (needed for hsa.h, hsa_api_trace.h)
set(HSA_RUNTIME_INC "${CMAKE_CURRENT_SOURCE_DIR}/../rocr-runtime/runtime/hsa-runtime/inc"
    CACHE PATH "Path to HSA runtime include directory")

# TODO(COMGR-upstream): Remove mock-comgr once amd_comgr_hotswap_rewrite
# is available in a released libamd_comgr.so package. At that point, the
# real amd_comgr CMake package should be required for non-test builds and
# the mock should only be used under BUILD_TESTING.
add_library(mock-comgr SHARED tests/mock_comgr.cpp)
set_target_properties(mock-comgr PROPERTIES
  OUTPUT_NAME "amd_comgr"
  POSITION_INDEPENDENT_CODE ON)

# Find real COMGR via its exported CMake package, or fall back to the mock.
# TODO(COMGR-upstream): Once COMGR ships the hotswap API in a released package,
# make find_package(amd_comgr CONFIG REQUIRED) for normal builds.
find_package(amd_comgr CONFIG QUIET)
if(TARGET amd_comgr)
  set(COMGR_LINK_LIB amd_comgr)
  message(STATUS "Found COMGR package: amd_comgr")
else()
  set(COMGR_LINK_LIB mock-comgr)
  message(STATUS "COMGR not found, using mock for build/test")
endif()

# Shared library for HSA_TOOLS_LIB usage — links COMGR directly
set(HOTSWAP_PLATFORM_IO_SRC hotswap_platform_io_posix.cpp)
if(WIN32)
  set(HOTSWAP_PLATFORM_IO_SRC hotswap_platform_io_win.cpp)
endif()

add_library(hsa-hotswap SHARED
  hotswap_tool.cpp
  hotswap.cpp
  ${HOTSWAP_PLATFORM_IO_SRC}
)

target_include_directories(hsa-hotswap PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${HSA_RUNTIME_INC}
  ${HSA_RUNTIME_INC}/..
)

target_link_libraries(hsa-hotswap PRIVATE ${COMGR_LINK_LIB})
set_target_properties(hsa-hotswap PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# Test executable — links COMGR directly (mock or real)
add_executable(hotswap_test tests/hotswap_test.cpp
  hotswap.cpp
)
target_include_directories(hotswap_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hotswap_test PRIVATE ${COMGR_LINK_LIB})

enable_testing()
add_test(NAME hotswap_test COMMAND hotswap_test)
