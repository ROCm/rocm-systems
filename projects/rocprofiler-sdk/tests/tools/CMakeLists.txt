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
    PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/.."
               INSTALL_RPATH_USE_LINK_PATH ON
               SOVERSION ${PROJECT_VERSION_MINOR}
               VERSION
               ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})

install(
    TARGETS rocprofiler-sdk-json-tool
    DESTINATION lib/rocprofiler-sdk
    COMPONENT tests)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:rocprofiler-sdk-json-tool>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:rocprofiler-sdk-json-tool>")
endif()
