# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

# SHA-256 / HMAC-SHA-256. Consumed with add_subdirectory() rather than
# find_package(): it depends on nothing but the standard library, so it can be
# pulled in from either direction without creating a cycle.

cmake_minimum_required(VERSION 3.16)
project(rocm-sha256 VERSION 1.0.0 LANGUAGES CXX)

# Create as a STATIC and OBJECT library. rocprofiler-sdk links statically since
# rocm-sha256 can be propagated through its build-interface dependency targets.
# CUID uses the object target so the SHA-256 implementation is embedded directly
# in libamdcuid_static.a, preserving its standalone installed archive contract.
# Other in-tree consumers can choose the form that matches their packaging
# model.
add_library(rocm-sha256-objects OBJECT src/sha256.cc)

target_include_directories(
    rocm-sha256-objects
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(rocm-sha256-objects PUBLIC cxx_std_17)

# Linked into shared libraries.
set_target_properties(
    rocm-sha256-objects
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)

add_library(rocm-sha256 STATIC)
add_library(rocm::sha256 ALIAS rocm-sha256)

target_link_libraries(rocm-sha256 PUBLIC rocm-sha256-objects)
