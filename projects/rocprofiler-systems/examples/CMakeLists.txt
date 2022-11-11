cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(omnitrace-examples LANGUAGES C CXX)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE
        "RelWithDebInfo"
        CACHE STRING "Build type" FORCE)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)

set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
set(CMAKE_CXX_VISIBILITY_PRESET "default")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_CLANG_TIDY)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME examples)

if(OMNITRACE_BUILD_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g  -fno-omit-frame-pointer")
endif()

option(BUILD_SHARED_LIBS "Build dynamic libraries" ON)

if(CMAKE_PROJECT_NAME STREQUAL "omnitrace")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
    omnitrace_add_option(OMNITRACE_INSTALL_EXAMPLES "Install omnitrace examples" OFF)
else()
    option(OMNITRACE_INSTALL_EXAMPLES "Install omnitrace examples" ON)
endif()

if(OMNITRACE_INSTALL_EXAMPLES)
    include(GNUInstallDirs)
endif()

add_subdirectory(transpose)
add_subdirectory(parallel-overhead)
add_subdirectory(code-coverage)
add_subdirectory(user-api)
add_subdirectory(openmp)
add_subdirectory(mpi)
add_subdirectory(python)
add_subdirectory(lulesh)
add_subdirectory(rccl)
add_subdirectory(rewrite-caller)
