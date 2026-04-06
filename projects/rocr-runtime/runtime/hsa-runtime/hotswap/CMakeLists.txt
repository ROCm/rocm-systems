if(NOT TARGET hsa-runtime64)
  cmake_minimum_required(VERSION 3.16)
  project(rocr-hotswap LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# HSA runtime headers (needed for hsa.h, hsa_api_trace.h)
set(HSA_RUNTIME_INC "${CMAKE_CURRENT_SOURCE_DIR}/../inc"
    CACHE PATH "Path to HSA runtime include directory")

# Static library for standalone testing
add_library(rocr-hotswap STATIC
  hotswap.cpp
  hotswap_comgr_client.cpp
)

target_include_directories(rocr-hotswap PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(UNIX)
  target_link_libraries(rocr-hotswap PUBLIC ${CMAKE_DL_LIBS})
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

# Test executable
add_executable(hotswap_test tests/hotswap_test.cpp)
target_link_libraries(hotswap_test PRIVATE rocr-hotswap)
