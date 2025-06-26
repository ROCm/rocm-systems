cmake_minimum_required(VERSION 3.22.0 FATAL_ERROR)

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
string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1.\\2.\\3"
                     ROCPROFILER_REGISTER_VERSION "${FULL_VERSION_STRING}")

foreach(_LANG C CXX)
    set(CMAKE_${_LANG}_FLAGS_COVERAGE_INIT
        "-Og -g3 -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline-functions -fprofile-abs-path -fprofile-arcs -ftest-coverage --coverage"
        CACHE STRING "${_LANG} flags for code coverage builds")

    set(CMAKE_${_LANG}_FLAGS_COVERAGE
        "${CMAKE_${_LANG}_FLAGS_COVERAGE_INIT}"
        CACHE STRING "${_LANG} flags for code coverage builds")
endforeach()

project(
    rocprofiler-register
    LANGUAGES C CXX
    VERSION ${ROCPROFILER_REGISTER_VERSION}
    DESCRIPTION "Registration library for rocprofiler"
    HOMEPAGE_URL "https://github.com/ROCm/rocprofiler-register-internal")

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME core)
# configure the version file so that if it changes, cmake automatically re-runs
configure_file(${PROJECT_SOURCE_DIR}/VERSION ${PROJECT_BINARY_DIR}/VERSION COPYONLY)

# needed for modulefile and setup-env.sh
string(REPLACE "-" "_" PROJECT_NAME_UNDERSCORED "${PROJECT_NAME}")

find_package(Git)

if(Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags
        OUTPUT_VARIABLE ROCPROFILER_REGISTER_GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _GIT_DESCRIBE_RESULT
        ERROR_QUIET)

    if(NOT _GIT_DESCRIBE_RESULT EQUAL 0)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe
            OUTPUT_VARIABLE ROCPROFILER_REGISTER_GIT_DESCRIBE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _GIT_DESCRIBE_RESULT
            ERROR_QUIET)
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        OUTPUT_VARIABLE ROCPROFILER_REGISTER_GIT_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
else()
    set(ROCPROFILER_REGISTER_GIT_DESCRIBE "v${ROCPROFILER_REGISTER_VERSION}")
    set(ROCPROFILER_REGISTER_GIT_REVISION "")
endif()

message(
    STATUS
        "[${PROJECT_NAME}] version ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH} (${FULL_VERSION_STRING})"
    )
message(STATUS "[${PROJECT_NAME}] git revision: ${ROCPROFILER_REGISTER_GIT_REVISION}")
message(STATUS "[${PROJECT_NAME}] git describe: ${ROCPROFILER_REGISTER_GIT_DESCRIBE}")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/cmake/Modules
                      ${CMAKE_MODULE_PATH})

include(GNUInstallDirs) # install directories

# ROCm does not use lib64
set(CMAKE_INSTALL_LIBDIR "lib")

include(rocprofiler_register_utilities) # various functions/macros
include(rocprofiler_register_interfaces) # interface libraries
include(rocprofiler_register_compilers) # compiler identification
include(rocprofiler_register_options) # project options
include(rocprofiler_register_build_settings) # build flags
include(rocprofiler_register_formatting) # formatting
include(rocprofiler_register_linting) # clang-tidy
include(rocprofiler_register_config_interfaces) # configure interface libraries

set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

enable_testing()
include(CTest)

add_subdirectory(external)
add_subdirectory(source)

include(rocprofiler_register_config_install)

if(ROCPROFILER_REGISTER_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(ROCPROFILER_REGISTER_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

include(rocprofiler_register_config_packaging)

rocprofiler_register_print_features()
