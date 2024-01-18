#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-json-tool
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

add_library(rocprofiler-sdk-json-tool SHARED)
target_sources(rocprofiler-sdk-json-tool PRIVATE json-tool.cpp)
target_link_libraries(
    rocprofiler-sdk-json-tool
    PRIVATE rocprofiler::rocprofiler rocprofiler::cereal rocprofiler::tests-build-flags
            rocprofiler::tests-common-library)
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
