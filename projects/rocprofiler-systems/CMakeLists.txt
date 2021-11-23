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
            "hosttrace only supports Linux. Configure and/or build is likely to fail")
endif()

project(
    hosttrace
    LANGUAGES C CXX
    VERSION 0.0.3)

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
hosttrace_add_feature(CMAKE_CXX_STANDARD "CXX language standard")
hosttrace_add_option(CMAKE_CXX_STANDARD_REQUIRED "Require C++ language standard" ON)
hosttrace_add_option(CMAKE_CXX_EXTENSIONS "Compiler specific language extensions" OFF)
hosttrace_add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Enable rpath to linked libraries"
                     ON)
hosttrace_add_option(HOSTTRACE_USE_CLANG_TIDY "Enable clang-tidy" OFF)
hosttrace_add_option(HOSTTRACE_USE_MPI "Enable MPI support" OFF)
hosttrace_add_option(HOSTTRACE_CUSTOM_DATA_SOURCE "Enable custom data source" OFF)
hosttrace_add_option(HOSTTRACE_USE_ROCTRACER "Enable roctracer support" ON)
hosttrace_add_option(HOSTTRACE_BUILD_DYNINST "Build dyninst from submodule" OFF)
hosttrace_add_option(HOSTTRACE_USE_MPI_HEADERS
                     "Enable wrapping MPI functions w/o enabling MPI dependency" OFF)

include(ProcessorCount)
processorcount(HOSTTRACE_PROCESSOR_COUNT)
math(EXPR HOSTTRACE_THREAD_COUNT "8 * ${HOSTTRACE_PROCESSOR_COUNT}")
set(HOSTTRACE_MAX_THREADS
    "${HOSTTRACE_THREAD_COUNT}"
    CACHE
        STRING
        "Maximum number of threads in the host application. Likely only needs to be increased if host app does not use thread-pool but creates many threads"
    )
hosttrace_add_feature(
    HOSTTRACE_MAX_THREADS
    "Maximum number of total threads supported in the host application (default: 8 * nproc)"
    )

# ensure synced
set(TIMEMORY_USE_MPI
    ${HOSTTRACE_USE_MPI}
    CACHE BOOL "Enable MPI support" FORCE)

# default visibility settings
set(CMAKE_C_VISIBILITY_PRESET "default")
set(CMAKE_CXX_VISIBILITY_PRESET "default")
set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(Formatting) # format target
include(Packages) # finds third-party libraries

if(HOSTTRACE_USE_ROCTRACER)
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

hosttrace_activate_clang_tidy()

# custom visibility settings
set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(HOSTTRACE_BUILD_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# ------------------------------------------------------------------------------#
#
# hosttrace-library target
#
# ------------------------------------------------------------------------------#

set(library_sources
    ${CMAKE_CURRENT_LIST_DIR}/src/library.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/critical_trace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/fork_gotcha.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/library/hosttrace_component.cpp
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
    ${CMAKE_CURRENT_LIST_DIR}/include/library/hosttrace_component.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/mpi_gotcha.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/perfetto.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/ptl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/state.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/thread_data.hpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library/timemory.hpp
    ${perfetto_DIR}/sdk/perfetto.h)

if(NOT TIMEMORY_USE_PERFETTO)

endif()

add_library(hosttrace-library SHARED ${library_sources} ${library_headers})

if(HOSTTRACE_USE_ROCTRACER)
    target_sources(
        hosttrace-library
        PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include/library/roctracer.hpp
                ${CMAKE_CURRENT_LIST_DIR}/src/library/roctracer.cpp
                ${CMAKE_CURRENT_LIST_DIR}/include/library/roctracer_callbacks.hpp
                ${CMAKE_CURRENT_LIST_DIR}/src/library/roctracer_callbacks.cpp)
endif()

target_include_directories(hosttrace-library SYSTEM PRIVATE ${perfetto_DIR}/sdk)

target_compile_definitions(
    hosttrace-library
    PRIVATE $<IF:$<BOOL:${HOSTTRACE_CUSTOM_DATA_SOURCE}>,CUSTOM_DATA_SOURCE,>)

target_link_libraries(
    hosttrace-library
    PRIVATE hosttrace::hosttrace-headers
            hosttrace::hosttrace-threading
            hosttrace::hosttrace-compile-options
            hosttrace::hosttrace-roctracer
            hosttrace::hosttrace-mpi
            hosttrace::hosttrace-ptl
            $<BUILD_INTERFACE:timemory::timemory-headers>
            $<BUILD_INTERFACE:timemory::timemory-gotcha>
            $<BUILD_INTERFACE:timemory::timemory-cxx-shared>
            $<IF:$<BOOL:${HOSTTRACE_USE_SANITIZER}>,hosttrace::hosttrace-sanitizer,>)

if(HOSTTRACE_DYNINST_API_RT)
    get_filename_component(HOSTTRACE_DYNINST_API_RT_DIR "${HOSTTRACE_DYNINST_API_RT}"
                           DIRECTORY)
endif()

set_target_properties(
    hosttrace-library PROPERTIES OUTPUT_NAME hosttrace
                                 INSTALL_RPATH "\$ORIGIN:\$ORIGIN/dyninst-tpls/libs")

install(
    TARGETS hosttrace-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    OPTIONAL)

# ------------------------------------------------------------------------------#
#
# hosttrace-exe target
#
# ------------------------------------------------------------------------------#

add_executable(
    hosttrace-exe
    ${_EXCLUDE} ${CMAKE_CURRENT_LIST_DIR}/src/hosttrace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/include/hosttrace.hpp
    ${CMAKE_CURRENT_LIST_DIR}/src/hosttrace/details.cpp)

target_link_libraries(
    hosttrace-exe
    PRIVATE hosttrace::hosttrace-headers
            hosttrace::hosttrace-dyninst
            hosttrace::hosttrace-compile-options
            $<BUILD_INTERFACE:timemory::timemory-headers>
            $<IF:$<BOOL:${HOSTTRACE_USE_SANITIZER}>,hosttrace::hosttrace-sanitizer,>)

set_target_properties(
    hosttrace-exe
    PROPERTIES
        OUTPUT_NAME hosttrace
        INSTALL_RPATH_USE_LINK_PATH ON
        INSTALL_RPATH
        "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/dyninst-tpls/lib"
    )

install(
    TARGETS hosttrace-exe
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)

# build the timemory-avail exe
add_dependencies(hosttrace-exe timemory-avail)

# ------------------------------------------------------------------------------#
#
# miscellaneous installs
#
# ------------------------------------------------------------------------------#

configure_file(${PROJECT_SOURCE_DIR}/scripts/setup-env.sh.in
               ${PROJECT_BINARY_DIR}/scripts/setup-env.sh @ONLY)

install(
    PROGRAMS ${PROJECT_SOURCE_DIR}/scripts/hosttrace-merge.jl
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

hosttrace_print_features()
