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

# libdrm / libdrm_amdgpu are used on Linux to read the GPU's internal chip
# revision (chip_rev) so HotSwap can be gated to gfx1250 A0 silicon only.
if(NOT WIN32)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(DRM REQUIRED IMPORTED_TARGET libdrm)
  pkg_check_modules(DRM_AMDGPU REQUIRED IMPORTED_TARGET libdrm_amdgpu)
endif()

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
if(NOT WIN32)
  target_link_libraries(hsa-hotswap PRIVATE PkgConfig::DRM PkgConfig::DRM_AMDGPU)
endif()
set_target_properties(hsa-hotswap PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# Test executable
add_executable(hotswap_test tests/hotswap_test.cpp hotswap.cpp)
target_include_directories(hotswap_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hotswap_test PRIVATE amd_comgr)

# Tool-internal unit tests for the gfx1250-A0 agent gating logic. The test TU
# #includes hotswap_tool.cpp directly and supplies its own stubs for the HSA,
# libdrm and COMGR entry points, so this target needs the relevant headers but
# must NOT link the real HSA / libdrm / COMGR libraries (doing so would clash
# with the in-file stubs).
if(NOT WIN32)
  add_executable(hotswap_tool_test
    tests/hotswap_tool_test.cpp
    hotswap_platform_io_posix.cpp
  )
  target_include_directories(hotswap_tool_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HSA_RUNTIME_INC}
    ${HSA_RUNTIME_INC}/..
    ${DRM_INCLUDE_DIRS}
    ${DRM_AMDGPU_INCLUDE_DIRS}
  )
endif()

enable_testing()
add_test(NAME hotswap_test COMMAND hotswap_test)
if(NOT WIN32)
  add_test(NAME hotswap_tool_test COMMAND hotswap_tool_test)
endif()