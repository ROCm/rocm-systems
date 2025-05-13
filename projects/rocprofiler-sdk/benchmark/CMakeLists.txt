#
# Integration tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(COMMAND rocprofiler_deactivate_clang_tidy)
    rocprofiler_deactivate_clang_tidy()
endif()

project(rocprofiler-sdk-benchmark LANGUAGES C CXX)

#
# project options
#
option(ROCPROFILER_BENCHMARK_INSTALL_TIMEM "Install timem" ON)

#
# cmake overrides
#
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "benchmark")
set(ROCPROFILER_SDK_BENCHMARK_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(ROCPROFILER_SDK_BENCHMARK_BINARY_DIR "${PROJECT_BINARY_DIR}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        "Release"
        CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

enable_testing()
include(CTest)

include(GNUInstallDirs)
# always use lib instead of lib64
set(CMAKE_INSTALL_LIBDIR "lib")
set(CMAKE_INSTALL_LIB64DIR "lib64")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_DATAROOT_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}")

# define the library output directory
if(PROJECT_IS_TOP_LEVEL)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" FULL_VERSION_STRING LIMIT_COUNT 1)
    string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
    string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1.\\2.\\3"
                         ROCPROFILER_SDK_VERSION "${FULL_VERSION_STRING}")

    find_package(Git)

    if(Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/../.git")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/..
            OUTPUT_VARIABLE ROCPROFILER_SDK_GIT_REVISION
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    else()
        set(ROCPROFILER_SDK_GIT_REVISION "")
    endif()
else()
    set(ROCPROFILER_BENCHMARK_INSTALL_PREFIX
        ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-sdk/benchmark
        CACHE PATH "relative install prefix for rocprofiler-sdk benchmark")
    set(_GNUInstallDirs_VARIABLES
        BINDIR
        SBINDIR
        LIBEXECDIR
        SYSCONFDIR
        SHAREDSTATEDIR
        LOCALSTATEDIR
        RUNSTATEDIR
        LIBDIR
        LIB64DIR # locally defined
        INCLUDEDIR
        OLDINCLUDEDIR
        DATAROOTDIR
        DATADIR
        INFODIR
        LOCALEDIR
        MANDIR
        DOCDIR)

    foreach(_INSTALL_DIR ${_GNUInstallDirs_VARIABLES})
        set(CMAKE_INSTALL_${_INSTALL_DIR}
            ${ROCPROFILER_BENCHMARK_INSTALL_PREFIX}/${CMAKE_INSTALL_${_INSTALL_DIR}})
    endforeach()
endif()

# include the timem executable
include(timem)

add_subdirectory(source)
