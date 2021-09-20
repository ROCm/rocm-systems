cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND CMAKE_CURRENT_SOURCE_DIR STREQUAL
                                                  CMAKE_SOURCE_DIR)
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(AUTHOR_WARNING "In-source build")
endif()

project(
    hosttrace
    LANGUAGES C CXX
    VERSION 0.0.1)

message(
    STATUS
        "[${PROJECT_NAME}] version ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    )
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
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
add_option(CMAKE_CXX_STANDARD_REQUIRED "Require C++ language standard" ON)
add_option(CMAKE_CXX_EXTENSIONS "Compiler specific language extensions" OFF)
add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Enable rpath to linked libraries" ON)
add_option(HOSTTRACE_USE_CLANG_TIDY "Enable clang-tidy" OFF)
add_option(HOSTTRACE_USE_MPI "Enable MPI support" OFF)
add_option(HOSTTRACE_CUSTOM_DATA_SOURCE "Enable custom data source" OFF)
add_option(HOSTTRACE_USE_ROCTRACER "Enable roctracer support" ON)
add_option(HOSTTRACE_BUILD_DYNINST "Build dyninst from submodule" OFF)

# ensure synced
set(TIMEMORY_USE_MPI
    ${HOSTTRACE_USE_MPI}
    CACHE BOOL "Enable MPI support" FORCE)
set(TRANSPOSE_USE_MPI
    ${HOSTTRACE_USE_MPI}
    CACHE BOOL "Enable MPI support" FORCE)

include(Packages) # finds third-party libraries

hosttrace_activate_clang_tidy()

set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# ------------------------------------------------------------------------------#
#
# hosttrace-library target
#
# ------------------------------------------------------------------------------#

add_library(
    hosttrace-library SHARED
    ${CMAKE_CURRENT_LIST_DIR}/src/library.cpp ${CMAKE_CURRENT_LIST_DIR}/src/libmisc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/include/library.hpp ${perfetto_DIR}/sdk/perfetto.cc)

if(HOSTTRACE_USE_ROCTRACER)
    target_sources(
        hosttrace-library PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include/roctracer.hpp
                                  ${CMAKE_CURRENT_LIST_DIR}/src/roctracer.cpp)
endif()

target_include_directories(hosttrace-library PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)

target_include_directories(hosttrace-library SYSTEM PRIVATE ${perfetto_DIR}/sdk)

target_compile_definitions(
    hosttrace-library
    PRIVATE $<IF:$<BOOL:${HOSTTRACE_CUSTOM_DATA_SOURCE}>,CUSTOM_DATA_SOURCE,>)

target_link_libraries(
    hosttrace-library
    PRIVATE hosttrace::hosttrace-threading
            hosttrace::hosttrace-compile-options
            hosttrace::hosttrace-roctracer
            $<BUILD_INTERFACE:timemory::timemory-headers>
            $<BUILD_INTERFACE:timemory::timemory-gotcha>
            $<BUILD_INTERFACE:timemory::timemory-cxx-shared>
            $<IF:$<BOOL:${HOSTTRACE_USE_SANITIZER}>,hosttrace::hosttrace-sanitizer,>)

if(DYNINST_API_RT)
    get_filename_component(DYNINST_API_RT_DIR "${DYNINST_API_RT}" DIRECTORY)
endif()

set_target_properties(
    hosttrace-library PROPERTIES OUTPUT_NAME hosttrace INSTALL_RPATH
                                                       "\$ORIGIN:${DYNINST_API_RT_DIR}")

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
    ${CMAKE_CURRENT_LIST_DIR}/src/hosttrace-details.cpp)

target_include_directories(hosttrace-exe PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)

target_link_libraries(
    hosttrace-exe
    PRIVATE hosttrace::hosttrace-dyninst hosttrace::hosttrace-compile-options
            $<BUILD_INTERFACE:timemory::timemory-headers>
            $<IF:$<BOOL:${HOSTTRACE_USE_SANITIZER}>,hosttrace::hosttrace-sanitizer,>)

set_target_properties(
    hosttrace-exe
    PROPERTIES OUTPUT_NAME hosttrace
               INSTALL_RPATH_USE_LINK_PATH ON
               INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}:${DYNINST_API_RT_DIR}")

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

install(
    PROGRAMS ${PROJECT_SOURCE_DIR}/scripts/hosttrace-merge.jl
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)

install(
    FILES ${PROJECT_SOURCE_DIR}/roctrace.cfg
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}
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
