cmake_minimum_required(VERSION 3.16)
project(hotswap LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# HSA runtime headers (needed for hsa.h, hsa_api_trace.h)
set(HSA_RUNTIME_INC "${CMAKE_CURRENT_SOURCE_DIR}/../rocr-runtime/runtime/hsa-runtime/inc"
    CACHE PATH "Path to HSA runtime include directory")

# COMGR is required — provides amd_comgr_hotswap_rewrite.
find_package(amd_comgr CONFIG REQUIRED)
find_package(hsa-runtime64 CONFIG REQUIRED)

find_path(HSA_INCLUDE_DIR hsa.h PATHS ${HSA_RUNTIME_INC} NO_DEFAULT_PATH REQUIRED)

# The gfx target and ASIC revision used to gate HotSwap are read through the HSA
# runtime (HSA_AMD_AGENT_INFO_ASIC_REVISION), so no libdrm / KFD-sysfs access is
# needed and the gating logic builds identically on Linux and Windows.

# Shared library for HSA_TOOLS_LIB usage
set(HOTSWAP_PLATFORM_IO_SRC hotswap_platform_io_posix.cpp)
if(WIN32)
  set(HOTSWAP_PLATFORM_IO_SRC hotswap_platform_io_win.cpp)
endif()

add_library(hsa-hotswap SHARED
  hotswap_tool.cpp
  hotswap.cpp
  hotswap_gfx_query.cpp
  ${HOTSWAP_PLATFORM_IO_SRC}
)

target_include_directories(hsa-hotswap PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${HSA_RUNTIME_INC}
  ${HSA_RUNTIME_INC}/..
)

target_link_libraries(hsa-hotswap PRIVATE
  amd_comgr
  hsa-runtime64::hsa-runtime64
)
set_target_properties(hsa-hotswap PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

install(TARGETS hsa-hotswap
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Test executable
add_executable(hotswap_test tests/hotswap_test.cpp hotswap.cpp)
target_include_directories(hotswap_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hotswap_test PRIVATE amd_comgr)

# Embed the gfx1250 fixture code object as a byte array so the test exercises the
# real parse + rewrite path with no GPU and no runtime file dependency (works in
# any CI). Regenerate the .hsaco per tests/fixtures/README.md.
set(_hotswap_co "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/gfx1250_min.hsaco")
set(_hotswap_co_hdr "${CMAKE_CURRENT_BINARY_DIR}/gfx1250_min_hsaco.h")
file(READ "${_hotswap_co}" _hotswap_co_hex HEX)
string(REGEX REPLACE "(..)" "0x\\1," _hotswap_co_arr "${_hotswap_co_hex}")
file(WRITE "${_hotswap_co_hdr}"
  "// Generated from tests/fixtures/gfx1250_min.hsaco. Do not edit.\n"
  "static const unsigned char kGfx1250MinCo[] = {${_hotswap_co_arr}};\n")
target_include_directories(hotswap_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Unit tests for the gfx-target / ASIC-revision query logic. The test compiles
# the portable hotswap_gfx_query.cpp unit alongside the test translation unit 
# and supplies its own stubs for the HSA entry points, so this target needs 
# the relevant headers but must NOT link the real HSA library (doing so would 
# clash with the in-file stubs). 
add_executable(hotswap_tool_test
  tests/hotswap_tool_test.cpp
  hotswap_gfx_query.cpp
)
target_include_directories(hotswap_tool_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${HSA_RUNTIME_INC}
  ${HSA_RUNTIME_INC}/..
)

enable_testing()
add_test(NAME hotswap_test COMMAND hotswap_test)
add_test(NAME hotswap_tool_test COMMAND hotswap_tool_test)
