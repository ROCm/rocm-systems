#
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND CMAKE_CURRENT_SOURCE_DIR STREQUAL
                                                  CMAKE_SOURCE_DIR)
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintended, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(AUTHOR_WARNING "In-source build")
endif()

foreach(_LANG C CXX)
    set(CMAKE_${_LANG}_FLAGS_COVERAGE_INIT
        "-Og -g3 -fno-omit-frame-pointer -fprofile-abs-path -fprofile-arcs -ftest-coverage --coverage -DCODECOV=1"
        CACHE STRING "${_LANG} flags for code coverage builds")
    set(CMAKE_${_LANG}_FLAGS_COVERAGE
        "${CMAKE_${_LANG}_FLAGS_COVERAGE_INIT}"
        CACHE STRING "${_LANG} flags for code coverage builds")
endforeach()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../cmake ${CMAKE_MODULE_PATH})
include(rocm_versioning)

rocm_versioning_read_version_file(
    rocprofiler-sdk ROCPROFILER_SDK ${CMAKE_CURRENT_LIST_DIR}/VERSION
    ${CMAKE_CURRENT_LIST_DIR}/versioning.yml)

project(
    rocprofiler-sdk
    LANGUAGES C CXX
    VERSION ${ROCPROFILER_SDK_VERSION}
    DESCRIPTION "ROCm GPU performance analysis SDK"
    HOMEPAGE_URL "https://github.com/ROCm/rocprofiler-sdk")

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "core")
set(PACKAGE_NAME ${PROJECT_NAME})

if(NOT PROJECT_IS_TOP_LEVEL)
    set(CMAKE_MESSAGE_INDENT "[${PROJECT_NAME}] ")
endif()

find_package(Git)

if(Git_FOUND AND (EXISTS "${PROJECT_SOURCE_DIR}/../../.git"
                  OR EXISTS "${PROJECT_SOURCE_DIR}/.git"))
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags
        OUTPUT_VARIABLE ROCPROFILER_SDK_GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _GIT_DESCRIBE_RESULT
        ERROR_QUIET)

    if(NOT _GIT_DESCRIBE_RESULT EQUAL 0)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe
            OUTPUT_VARIABLE ROCPROFILER_SDK_GIT_DESCRIBE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _GIT_DESCRIBE_RESULT
            ERROR_QUIET)
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE ROCPROFILER_SDK_GIT_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
else()
    set(ROCPROFILER_SDK_GIT_DESCRIBE "v${ROCPROFILER_SDK_VERSION}")
    set(ROCPROFILER_SDK_GIT_REVISION "")
endif()

# for backward compatibility
set(FULL_VERSION_STRING "${ROCPROFILER_SDK_FULL_VERSION_STRING}")
# for SOVERSION
set(ROCPROFILER_SDK_SOVERSION "${ROCPROFILER_SDK_VERSION_MAJOR}")

# make sure that cmake re-runs when version file changes
rocm_versioning_configure_version_files(FILES ${PROJECT_SOURCE_DIR}/VERSION
                                        ${PROJECT_SOURCE_DIR}/versioning.yml)

message(
    STATUS
        "[${PROJECT_NAME}] version ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH} (${FULL_VERSION_STRING})"
    )
message(STATUS "[${PROJECT_NAME}] git revision:  ${ROCPROFILER_SDK_GIT_REVISION}")
message(STATUS "[${PROJECT_NAME}] git describe:  ${ROCPROFILER_SDK_GIT_DESCRIBE}")
message(STATUS "[${PROJECT_NAME}] build version: ${ROCPROFILER_SDK_VERSION_BUILD}")
message(STATUS "[${PROJECT_NAME}] version hash:  ${ROCPROFILER_SDK_VERSION_HASH}")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/cmake/Modules
                      ${CMAKE_MODULE_PATH})

include(GNUInstallDirs) # install directories
set(CMAKE_INSTALL_LIBDIR "lib") # rocm doesn't use lib64
set(CMAKE_BUILD_RPATH "${PROJECT_BINARY_DIR}/lib:\$ORIGIN:\$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH
    OFF
    CACHE BOOL "")
set(CMAKE_DEBUG_POSTFIX
    ""
    CACHE STRING "Suffix for non-executable build targets")
mark_as_advanced(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_DEBUG_POSTFIX)

set(ROCPROFILER_INTERNAL_BUILD_DOCS
    OFF
    CACHE INTERNAL "Generates rocprofiler-sdk/version.h and exits (no build targets)")
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
include(rocprofiler_config_install_roctx)
include(rocprofiler_config_install_rocpd)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(ROCPROFILER_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

if(ROCPROFILER_BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()

include(rocprofiler_config_packaging)

rocprofiler_print_features()
