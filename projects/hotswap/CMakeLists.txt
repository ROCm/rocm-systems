cmake_minimum_required(VERSION 3.16)
project(hotswap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# HSA runtime headers (needed for hsa.h, hsa_api_trace.h)
set(HSA_RUNTIME_INC "${CMAKE_CURRENT_SOURCE_DIR}/../rocr-runtime/runtime/hsa-runtime/inc"
    CACHE PATH "Path to HSA runtime include directory")

# Static library for standalone testing
add_library(hotswap-static STATIC
  hotswap.cpp
  hotswap_comgr_client.cpp
)

target_include_directories(hotswap-static PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(UNIX)
  target_link_libraries(hotswap-static PUBLIC ${CMAKE_DL_LIBS})
endif()

# Shared library for HSA_TOOLS_LIB usage
add_library(hsa-hotswap SHARED
  hotswap_tool.cpp
  hotswap.cpp
  hotswap_comgr_client.cpp
)

target_include_directories(hsa-hotswap PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${HSA_RUNTIME_INC}
  ${HSA_RUNTIME_INC}/..
)

if(UNIX)
  target_link_libraries(hsa-hotswap PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(hsa-hotswap PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
endif()

# Mock COMGR library for testing the dlopen path
add_library(mock-comgr SHARED tests/mock_comgr.cpp)
set_target_properties(mock-comgr PROPERTIES
  OUTPUT_NAME "amd_comgr"
  POSITION_INDEPENDENT_CODE ON)

# Test executable (without mock — tests the "COMGR unavailable" paths)
add_executable(hotswap_test tests/hotswap_test.cpp)
target_link_libraries(hotswap_test PRIVATE hotswap-static)

# Test with mock COMGR (tests the dlopen/rewrite paths)
add_executable(hotswap_test_with_comgr tests/hotswap_test.cpp)
target_link_libraries(hotswap_test_with_comgr PRIVATE hotswap-static)
add_dependencies(hotswap_test_with_comgr mock-comgr)

enable_testing()
add_test(NAME hotswap_test COMMAND hotswap_test)
add_test(NAME hotswap_test_with_comgr COMMAND hotswap_test_with_comgr)
set_tests_properties(hotswap_test_with_comgr PROPERTIES
  ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:mock-comgr>")
