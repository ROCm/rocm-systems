cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(omnitrace-user-api LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Debug")
find_package(Threads REQUIRED)
add_executable(user-api user-api.cpp)
target_link_libraries(user-api PRIVATE Threads::Threads omnitrace::omnitrace-user-library)

if(NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set_target_properties(user-api PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                              ${CMAKE_BINARY_DIR})
endif()
