cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(omnitrace-dyninst-examples LANGUAGES CXX)

set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
set(CMAKE_CXX_VISIBILITY_PRESET "default")

add_subdirectory(transpose)
add_subdirectory(parallel-overhead)
add_subdirectory(code-coverage)
add_subdirectory(user-api)
add_subdirectory(openmp)

if(CMAKE_VERSION VERSION_LESS 3.16 AND (NOT DEFINED LULESH_BUILD_KOKKOS
                                        OR LULESH_BUILD_KOKKOS))
    message(
        AUTHOR_WARNING
            "lulesh example cannot build Kokkos (requires CMake 3.16+, current version is ${CMAKE_VERSION})"
        )
else()
    option(BUILD_SHARED_LIBS "Build dynamic libraries" ON)

    add_subdirectory(lulesh)
endif()
