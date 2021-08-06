cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND
   CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(AUTHOR_WARNING "In-source build")
endif()

project(
    hosttrace
    LANGUAGES CXX
    VERSION   0.0.1)

message(STATUS "[${PROJECT_NAME}] version ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)     # install directories
include(MacroUtilities)     # various functions and macros
include(Compilers)          # compiler identification
include(BuildSettings)      # compiler flags

set(CMAKE_CXX_STANDARD 17 CACHE STRING "CXX language standard")
add_option(CMAKE_CXX_STANDARD_REQUIRED "Require C++ language standard" ON)
add_option(CMAKE_CXX_EXTENSIONS "Compiler specific language extensions" OFF)
add_option(HOSTTRACE_USE_CLANG_TIDY "Enable clang-tidy" OFF)

include(Packages)           # finds third-party libraries

hosttrace_activate_clang_tidy()

#------------------------------------------------------------------------------#
#
#                   hosttrace-library target
#
#------------------------------------------------------------------------------#

add_library(hosttrace-library SHARED
    ${CMAKE_CURRENT_LIST_DIR}/src/library.cpp
    ${perfetto_DIR}/sdk/perfetto.cc)

target_include_directories(hosttrace-library PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include)

target_include_directories(hosttrace-library SYSTEM PRIVATE
    ${perfetto_DIR}/sdk)

target_link_libraries(hosttrace-library PRIVATE
    hosttrace::hosttrace-threading
    $<IF:$<BOOL:${hosttrace_USE_SANITIZER}>,hosttrace::hosttrace-sanitizer,>)

set_target_properties(hosttrace-library PROPERTIES
    OUTPUT_NAME hosttrace)

install(
    TARGETS     hosttrace-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    OPTIONAL)

#------------------------------------------------------------------------------#
#
#                   hosttrace-exe target
#
#------------------------------------------------------------------------------#

add_executable(hosttrace-exe ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/src/hosttrace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/include/hosttrace.hpp
    ${CMAKE_CURRENT_LIST_DIR}/src/hosttrace-details.cpp)

target_include_directories(hosttrace-exe PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include)

target_link_libraries(hosttrace-exe PRIVATE
    hosttrace::hosttrace-dyninst
    hosttrace::hosttrace-compile-options)

set_target_properties(hosttrace-exe PROPERTIES
    OUTPUT_NAME                 hosttrace
    INSTALL_RPATH_USE_LINK_PATH ON)

install(
    TARGETS     hosttrace-exe
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)

#------------------------------------------------------------------------------#
#
#                   clang-format target
#
#------------------------------------------------------------------------------#

find_program(CLANG_FORMAT_EXE
    NAMES
        clang-format-12
        clang-format-11
        clang-format-10
        clang-format-9
        clang-format)

if(CLANG_FORMAT_EXE)
    file(GLOB sources
        ${PROJECT_SOURCE_DIR}/src/*.cpp)
    file(GLOB headers
        ${PROJECT_SOURCE_DIR}/include/*.hpp)
    file(GLOB_RECURSE examples
        ${PROJECT_SOURCE_DIR}/examples/*.cpp
        ${PROJECT_SOURCE_DIR}/examples/*.hpp)
    add_custom_target(format
        ${CLANG_FORMAT_EXE} -i ${sources} ${headers} ${examples}
        COMMENT "Running ${CLANG_FORMAT_EXE}...")
else()
    message(AUTHOR_WARNING "clang-format could not be found. format build target not available.")
endif()

#------------------------------------------------------------------------------#
#
#                   examples
#
#------------------------------------------------------------------------------#

add_subdirectory(examples)

#------------------------------------------------------------------------------#
#
#                   tests
#
#------------------------------------------------------------------------------#

include(CTest)
enable_testing()

add_subdirectory(tests)

#------------------------------------------------------------------------------#
#
#                   packaging
#
#------------------------------------------------------------------------------#

include(ConfigCPack)
