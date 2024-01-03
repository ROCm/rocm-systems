cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND CMAKE_CURRENT_SOURCE_DIR STREQUAL
                                                  CMAKE_SOURCE_DIR)
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintended, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(AUTHOR_WARNING "In-source build")
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" FULL_VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1.\\2.\\3" ROCPROFILER_VERSION
                     "${FULL_VERSION_STRING}")

foreach(_LANG C CXX)
    set(CMAKE_${_LANG}_FLAGS_COVERAGE_INIT
        "-Og -g3 -fno-omit-frame-pointer -fprofile-abs-path -fprofile-arcs -ftest-coverage --coverage -DCODECOV=1"
        CACHE STRING "${_LANG} flags for code coverage builds")
    set(CMAKE_${_LANG}_FLAGS_COVERAGE
        "${CMAKE_${_LANG}_FLAGS_COVERAGE_INIT}"
        CACHE STRING "${_LANG} flags for code coverage builds")
endforeach()

project(
    rocprofiler
    LANGUAGES C CXX
    VERSION ${ROCPROFILER_VERSION}
    DESCRIPTION "ROCm GPU performance analysis SDK"
    HOMEPAGE_URL "https://github.com/ROCm/rocprofiler-sdk-internal")

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "core")

find_package(Git)

if(Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags
        OUTPUT_VARIABLE ROCPROFILER_GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _GIT_DESCRIBE_RESULT
        ERROR_QUIET)

    if(NOT _GIT_DESCRIBE_RESULT EQUAL 0)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe
            OUTPUT_VARIABLE ROCPROFILER_GIT_DESCRIBE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _GIT_DESCRIBE_RESULT
            ERROR_QUIET)
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        OUTPUT_VARIABLE ROCPROFILER_GIT_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
else()
    set(ROCPROFILER_GIT_DESCRIBE "v${ROCPROFILER_VERSION}")
    set(ROCPROFILER_GIT_REVISION "")
endif()

message(
    STATUS
        "[${PROJECT_NAME}] version ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH} (${FULL_VERSION_STRING})"
    )
message(STATUS "[${PROJECT_NAME}] git revision: ${ROCPROFILER_GIT_REVISION}")
message(STATUS "[${PROJECT_NAME}] git describe: ${ROCPROFILER_GIT_DESCRIBE}")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/cmake/Modules
                      ${CMAKE_MODULE_PATH})

include(GNUInstallDirs) # install directories
set(CMAKE_INSTALL_LIBDIR "lib") # rocm doesn't use lib64

set(ROCPROFILER_INTERNAL_BUILD_DOCS
    OFF
    CACHE INTERNAL "Generates rocprofiler/version.h and exits (no build targets)")
if(ROCPROFILER_INTERNAL_BUILD_DOCS)
    add_subdirectory(source/include)
    return()
endif()

include(rocprofiler_utilities) # various functions/macros
include(rocprofiler_interfaces) # interface libraries
include(rocprofiler_compilers) # compiler identification
include(rocprofiler_options) # project options
include(rocprofiler_build_settings) # build flags
include(rocprofiler_formatting) # formatting
include(rocprofiler_linting) # clang-tidy
include(rocprofiler_config_interfaces) # configure interface libraries

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

enable_testing()
include(CTest)

add_subdirectory(external)
add_subdirectory(source)

include(rocprofiler_config_install)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(ROCPROFILER_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

include(rocprofiler_config_packaging)

rocprofiler_print_features()
