# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

# ELFIO, the header-only ELF reader/writer, vendored from upstream with the
# patches in patches/ applied (see README.md). Consumed with add_subdirectory()
# by rocm-systems projects; nothing here is installed, since ELFIO is a private
# build-time dependency of every consumer.

cmake_minimum_required(VERSION 3.16)
project(rocm-elfio LANGUAGES CXX)

add_library(rocm-elfio INTERFACE)
add_library(rocm::elfio ALIAS rocm-elfio)

# SYSTEM so consumers building with -Werror are not failing on upstream code.
target_include_directories(
    rocm-elfio
    SYSTEM
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_features(rocm-elfio INTERFACE cxx_std_17)
