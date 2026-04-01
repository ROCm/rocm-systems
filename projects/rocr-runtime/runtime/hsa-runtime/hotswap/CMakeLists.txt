if(NOT TARGET hsa-runtime64)
  cmake_minimum_required(VERSION 3.16)
  project(rocr-hotswap LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_executable(hotswap_test tests/hotswap_test.cpp)
target_link_libraries(hotswap_test PRIVATE rocr-hotswap)
