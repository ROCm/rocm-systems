cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(hosttrace-dyninst-examples
    LANGUAGES CXX)

add_subdirectory(transpose)
add_subdirectory(parallel-overhead)
