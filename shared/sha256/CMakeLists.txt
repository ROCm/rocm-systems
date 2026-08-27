# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# SHA-256 / HMAC-SHA-256. Consumed with add_subdirectory() rather than
# find_package(): it depends on nothing but the standard library, so it can be
# pulled in from either direction without creating a cycle.

cmake_minimum_required(VERSION 3.16)
project(rocm-sha256 VERSION 1.0.0 LANGUAGES CXX)

add_library(rocm-sha256 STATIC src/sha256.cc)
add_library(rocm::sha256 ALIAS rocm-sha256)

target_include_directories(
    rocm-sha256
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(rocm-sha256 PUBLIC cxx_std_17)

# Linked into shared libraries.
set_target_properties(rocm-sha256 PROPERTIES POSITION_INDEPENDENT_CODE ON)
