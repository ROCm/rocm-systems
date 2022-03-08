cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(omnitrace-dyninst-examples LANGUAGES CXX)

set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
set(CMAKE_CXX_VISIBILITY_PRESET "default")

add_subdirectory(transpose)
add_subdirectory(parallel-overhead)
add_subdirectory(user-api)
add_subdirectory(openmp)

option(BUILD_SHARED_LIBS "Build dynamic libraries" ON)

add_subdirectory(lulesh)
