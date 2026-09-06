cmake_minimum_required(VERSION 3.20)
project(abce LANGUAGES CXX)

add_library(abce INTERFACE)
add_library(abce::abce ALIAS abce)
target_include_directories(
  abce
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
target_compile_features(abce INTERFACE cxx_std_17)
