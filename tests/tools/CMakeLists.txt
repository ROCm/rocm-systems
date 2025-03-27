#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-tools
    LANGUAGES C CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# tool library supporting JSON and perfetto output
add_library(rocprofiler-sdk-json-tool SHARED)
target_sources(rocprofiler-sdk-json-tool PRIVATE json-tool.cpp)
target_link_libraries(
    rocprofiler-sdk-json-tool
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::rocprofiler-sdk-cereal
            rocprofiler-sdk::tests-build-flags rocprofiler-sdk::tests-common-library
            rocprofiler-sdk::tests-perfetto)
set_target_properties(
    rocprofiler-sdk-json-tool
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/rocprofiler-sdk"
               SOVERSION ${PROJECT_VERSION_MINOR}
               VERSION
               ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")

# tool library which just checks that tools can be compiled with C language
add_library(rocprofiler-sdk-c-tool SHARED)
target_sources(rocprofiler-sdk-c-tool PRIVATE c-tool.c)
target_link_libraries(rocprofiler-sdk-c-tool PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                                     rocprofiler-sdk::tests-build-flags)
set_target_properties(
    rocprofiler-sdk-c-tool
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/rocprofiler-sdk"
               SOVERSION ${PROJECT_VERSION_MINOR}
               VERSION
               ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")

if(NOT CMAKE_STRIP)
    find_program(CMAKE_STRIP NAMES strip REQUIRED)
endif()

add_custom_command(
    TARGET rocprofiler-sdk-c-tool
    POST_BUILD
    COMMAND ${CMAKE_STRIP} $<TARGET_FILE:rocprofiler-sdk-c-tool>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Stripping rocprofiler-sdk-c-tool...")
