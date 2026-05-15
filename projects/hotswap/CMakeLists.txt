cmake_minimum_required(VERSION 3.16)
project(hotswap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# HSA runtime headers (needed for hsa.h, hsa_api_trace.h)
set(HSA_RUNTIME_INC "${CMAKE_CURRENT_SOURCE_DIR}/../rocr-runtime/runtime/hsa-runtime/inc"
    CACHE PATH "Path to HSA runtime include directory")

# COMGR is required — provides amd_comgr_hotswap_rewrite.
find_package(amd_comgr CONFIG REQUIRED)

find_path(HSA_INCLUDE_DIR hsa.h PATHS ${HSA_RUNTIME_INC} NO_DEFAULT_PATH REQUIRED)

# Shared library for HSA_TOOLS_LIB usage
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

target_link_libraries(hsa-hotswap PRIVATE amd_comgr)
set_target_properties(hsa-hotswap PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# Test executable
add_executable(hotswap_test tests/hotswap_test.cpp hotswap.cpp)
target_include_directories(hotswap_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hotswap_test PRIVATE amd_comgr)

enable_testing()
add_test(NAME hotswap_test COMMAND hotswap_test)
