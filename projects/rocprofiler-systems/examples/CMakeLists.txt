cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(omnitrace-dyninst-examples LANGUAGES CXX)

set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
set(CMAKE_CXX_VISIBILITY_PRESET "default")
set(CMAKE_CXX_CLANG_TIDY)
option(BUILD_SHARED_LIBS "Build dynamic libraries" ON)

add_subdirectory(transpose)
add_subdirectory(parallel-overhead)
add_subdirectory(code-coverage)
add_subdirectory(user-api)
add_subdirectory(openmp)
add_subdirectory(mpi)
add_subdirectory(lulesh)
