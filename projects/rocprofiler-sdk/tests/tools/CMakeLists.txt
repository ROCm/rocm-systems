#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-json-tool
    LANGUAGES C CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# tool library supporting JSON and perfetto output
add_library(rocprofiler-sdk-json-tool SHARED)
target_sources(rocprofiler-sdk-json-tool PRIVATE json-tool.cpp)
target_link_libraries(
    rocprofiler-sdk-json-tool
    PRIVATE rocprofiler::rocprofiler rocprofiler::cereal rocprofiler::tests-build-flags
            rocprofiler::tests-common-library rocprofiler::tests-perfetto)
set_target_properties(
    rocprofiler-sdk-json-tool
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/rocprofiler-sdk"
               SOVERSION ${PROJECT_VERSION_MINOR}
               VERSION
               ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")

install(
    TARGETS rocprofiler-sdk-json-tool
    DESTINATION lib/rocprofiler-sdk
    COMPONENT tests)

# tool library which just checks that tools can be compiled with C language
add_library(rocprofiler-sdk-c-tool SHARED)
target_sources(rocprofiler-sdk-c-tool PRIVATE c-tool.c)
target_link_libraries(rocprofiler-sdk-c-tool PRIVATE rocprofiler::rocprofiler
                                                     rocprofiler::tests-build-flags)
set_target_properties(
    rocprofiler-sdk-c-tool
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/rocprofiler-sdk"
               SOVERSION ${PROJECT_VERSION_MINOR}
               VERSION
               ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")
