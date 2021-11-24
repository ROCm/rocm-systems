cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND CMAKE_CURRENT_SOURCE_DIR STREQUAL
                                                  CMAKE_SOURCE_DIR)
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(AUTHOR_WARNING "In-source build")
endif()

if(NOT UNIX OR APPLE)
    message(
        AUTHOR_WARNING
            "omnitrace only supports Linux. Configure and/or build is likely to fail")
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" FULL_VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1.\\2.\\3" OMNITRACE_VERSION
                     "${FULL_VERSION_STRING}")

project(
    omnitrace
    LANGUAGES C CXX
    VERSION ${OMNITRACE_VERSION}
    DESCRIPTION "CPU/GPU Application tracing with static/dynamic binary instrumentation"
    HOMEPAGE_URL "https://github.com/AARInternal/omnitrace")

message(
    STATUS
        "[${PROJECT_NAME}] version ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    )
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/cmake/Modules
                      ${CMAKE_MODULE_PATH})
set(BUILD_SHARED_LIBS
    ON
    CACHE BOOL "Build shared libraries")
set(BUILD_STATIC_LIBS
    OFF
    CACHE BOOL "Build static libraries")
set(CMAKE_POSITION_INDEPENDENT_CODE
    ON
    CACHE BOOL "Build position independent code")

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE
        RelWithDebInfo
        CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs) # install directories
include(MacroUtilities) # various functions and macros
include(Compilers) # compiler identification
include(BuildSettings) # compiler flags

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "CXX language standard")
omnitrace_add_feature(CMAKE_CXX_STANDARD "CXX language standard")
omnitrace_add_option(CMAKE_CXX_STANDARD_REQUIRED "Require C++ language standard" ON)
omnitrace_add_option(CMAKE_CXX_EXTENSIONS "Compiler specific language extensions" OFF)
omnitrace_add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Enable rpath to linked libraries"
                     ON)
omnitrace_add_option(OMNITRACE_USE_CLANG_TIDY "Enable clang-tidy" OFF)
omnitrace_add_option(OMNITRACE_USE_MPI "Enable MPI support" OFF)
omnitrace_add_option(OMNITRACE_CUSTOM_DATA_SOURCE "Enable custom data source" OFF)
omnitrace_add_option(OMNITRACE_USE_ROCTRACER "Enable roctracer support" ON)
omnitrace_add_option(OMNITRACE_BUILD_DYNINST "Build dyninst from submodule" OFF)
omnitrace_add_option(OMNITRACE_USE_MPI_HEADERS
                     "Enable wrapping MPI functions w/o enabling MPI dependency" OFF)

include(ProcessorCount)
processorcount(OMNITRACE_PROCESSOR_COUNT)
math(EXPR OMNITRACE_THREAD_COUNT "8 * ${OMNITRACE_PROCESSOR_COUNT}")
set(OMNITRACE_MAX_THREADS
    "${OMNITRACE_THREAD_COUNT}"
    CACHE
        STRING
        "Maximum number of threads in the host application. Likely only needs to be increased if host app does not use thread-pool but creates many threads"
    )
omnitrace_add_feature(
    OMNITRACE_MAX_THREADS
    "Maximum number of total threads supported in the host application (default: 8 * nproc)"
    )

# ensure synced
set(TIMEMORY_USE_MPI
    ${OMNITRACE_USE_MPI}
    CACHE BOOL "Enable MPI support" FORCE)

# default visibility settings
set(CMAKE_C_VISIBILITY_PRESET "default")
set(CMAKE_CXX_VISIBILITY_PRESET "default")
set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(Formatting) # format target
include(Packages) # finds third-party libraries

if(OMNITRACE_USE_ROCTRACER)
    find_package(HIP QUIET)
    if(HIP_VERSION_MAJOR GREATER_EQUAL 4 AND HIP_VERSION_MINOR GREATER 3)
        set(roctracer_kfdwrapper_LIBRARY)
    endif()
else()
    set(HIP_VERSION "0.0.0")
    set(HIP_VERSION_MAJOR 0)
    set(HIP_VERSION_MINOR 0)
    set(HIP_VERSION_PATCH 0)
endif()

configure_file(${PROJECT_SOURCE_DIR}/include/library/defines.hpp.in
               ${PROJECT_BINARY_DIR}/include/library/defines.hpp @ONLY)

omnitrace_activate_clang_tidy()

# custom visibility settings
set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(OMNITRACE_BUILD_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# ------------------------------------------------------------------------------#
#
# omnitrace-library target
#
# ------------------------------------------------------------------------------#

set(library_sources
    ${CMAKE_CURRENT_LIST_DIR}/src/library.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/critical_trace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/fork_gotcha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/omnitrace_component.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/mpi_gotcha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/perfetto.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/ptl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/thread_data.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/timemory.cpp
    ${perfetto_DIR}/sdk/perfetto.cc)

set(library_headers
    ${CMAKE_CURRENT_LIST_DIR}/include/library.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/api.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/config.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/common.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/critical_trace.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/debug.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/fork_gotcha.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/omnitrace_component.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/mpi_gotcha.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/perfetto.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/ptl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/state.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/thread_data.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/timemory.hpp
    ${perfetto_DIR}/sdk/perfetto.h)

if(NOT TIMEMORY_USE_PERFETTO)

endif()

add_library(omnitrace-library SHARED ${library_sources} ${library_headers})

if(OMNITRACE_USE_ROCTRACER)
    target_sources(
        omnitrace-library
        PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include/library/roctracer.hpp
                ${CMAKE_CURRENT_LIST_DIR}/src/library/roctracer.cpp
                ${CMAKE_CURRENT_LIST_DIR}/include/library/roctracer_callbacks.hpp
                ${CMAKE_CURRENT_LIST_DIR}/src/library/roctracer_callbacks.cpp)
endif()

target_include_directories(omnitrace-library SYSTEM PRIVATE ${perfetto_DIR}/sdk)

target_compile_definitions(
    omnitrace-library
    PRIVATE $<IF:$<BOOL:${OMNITRACE_CUSTOM_DATA_SOURCE}>,CUSTOM_DATA_SOURCE,>)

target_link_libraries(
    omnitrace-library
    PRIVATE omnitrace::omnitrace-headers
            omnitrace::omnitrace-threading
            omnitrace::omnitrace-compile-options
            omnitrace::omnitrace-roctracer
            omnitrace::omnitrace-mpi
            omnitrace::omnitrace-ptl
            $<BUILD_INTERFACE:timemory::timemory-headers>
            $<BUILD_INTERFACE:timemory::timemory-gotcha>
            $<BUILD_INTERFACE:timemory::timemory-cxx-shared>
            $<IF:$<BOOL:${OMNITRACE_USE_SANITIZER}>,omnitrace::omnitrace-sanitizer,>)

if(OMNITRACE_DYNINST_API_RT)
    get_filename_component(OMNITRACE_DYNINST_API_RT_DIR "${OMNITRACE_DYNINST_API_RT}"
                           DIRECTORY)
endif()

set_target_properties(
    omnitrace-library PROPERTIES OUTPUT_NAME omnitrace
                                 INSTALL_RPATH "\$ORIGIN:\$ORIGIN/dyninst-tpls/libs")

install(
    TARGETS omnitrace-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    OPTIONAL)

# ------------------------------------------------------------------------------#
#
# omnitrace-exe target
#
# ------------------------------------------------------------------------------#

add_executable(
    omnitrace-exe
    ${_EXCLUDE} ${CMAKE_CURRENT_LIST_DIR}/src/omnitrace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/include/omnitrace.hpp
    ${CMAKE_CURRENT_LIST_DIR}/src/omnitrace/details.cpp)

target_link_libraries(
    omnitrace-exe
    PRIVATE omnitrace::omnitrace-headers
            omnitrace::omnitrace-dyninst
            omnitrace::omnitrace-compile-options
            $<BUILD_INTERFACE:timemory::timemory-headers>
            $<IF:$<BOOL:${OMNITRACE_USE_SANITIZER}>,omnitrace::omnitrace-sanitizer,>)

set_target_properties(
    omnitrace-exe
    PROPERTIES
        OUTPUT_NAME omnitrace
        INSTALL_RPATH_USE_LINK_PATH ON
        INSTALL_RPATH
        "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/dyninst-tpls/lib"
    )

install(
    TARGETS omnitrace-exe
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)

# build the timemory-avail exe
add_dependencies(omnitrace-exe timemory-avail)

# ------------------------------------------------------------------------------#
#
# miscellaneous installs
#
# ------------------------------------------------------------------------------#

configure_file(${PROJECT_SOURCE_DIR}/scripts/setup-env.sh.in
               ${PROJECT_BINARY_DIR}/scripts/setup-env.sh @ONLY)

install(
    PROGRAMS ${PROJECT_SOURCE_DIR}/scripts/omnitrace-merge.jl
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)

install(
    FILES ${PROJECT_SOURCE_DIR}/roctrace.cfg
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}
    OPTIONAL)

install(
    FILES ${PROJECT_BINARY_DIR}/scripts/setup-env.sh
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}
    OPTIONAL)

# ------------------------------------------------------------------------------#
#
# examples
#
# ------------------------------------------------------------------------------#

add_subdirectory(examples)

# ------------------------------------------------------------------------------#
#
# tests
#
# ------------------------------------------------------------------------------#

include(CTest)
enable_testing()

add_subdirectory(tests)

# ------------------------------------------------------------------------------#
#
# packaging
#
# ------------------------------------------------------------------------------#

include(ConfigCPack)

# ------------------------------------------------------------------------------#
#
# config info
#
# ------------------------------------------------------------------------------#

omnitrace_print_features()
