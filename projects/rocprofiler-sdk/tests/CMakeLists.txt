#
# Integration tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-tests LANGUAGES C CXX)

if(COMMAND rocprofiler_deactivate_clang_tidy)
    rocprofiler_deactivate_clang_tidy()
endif()

if(NOT TARGET rocprofiler::cereal)
    get_filename_component(ROCPROFILER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.."
                           REALPATH)

    rocprofiler_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/cereal
        WORKING_DIRECTORY ${ROCPROFILER_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/cereal.git
        REPO_BRANCH "rocprofiler")

    add_library(rocprofiler-cereal INTERFACE)
    add_library(rocprofiler::cereal ALIAS rocprofiler-cereal)
    target_compile_definitions(rocprofiler-cereal
                               INTERFACE $<BUILD_INTERFACE:CEREAL_THREAD_SAFE=1>)
    target_include_directories(
        rocprofiler-cereal
        INTERFACE $<BUILD_INTERFACE:${ROCPROFILER_SOURCE_DIR}/external/cereal/include>)
endif()

add_library(rocprofiler-tests-build-flags INTERFACE)
add_library(rocprofiler::tests-build-flags ALIAS rocprofiler-tests-build-flags)
target_compile_options(rocprofiler-tests-build-flags INTERFACE -W -Wall -Wextra -Wshadow)

if(ROCPROFILER_BUILD_CI OR ROCPROFILER_BUILD_WERROR)
    target_compile_options(rocprofiler-tests-build-flags INTERFACE -Werror)
endif()

# needed for validation
find_package(Python3 REQUIRED)

# applications used by integration tests
add_subdirectory(apps)

# tool libraries
add_subdirectory(kernel-tracing)
