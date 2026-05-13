# *************************************************************************
#  Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  See LICENSE.txt for license information
# *************************************************************************

# Standalone CMake consumer that exercises the CMake install + find_package
# integration added in NCCL 2.29.7:
#   "Added cmake install along with target export. Now `find_package(NCCL)`
#    is supported using the configuration file mode."
#
# This project is intentionally NOT a subdirectory of the RCCL CMake
# project; it is a downstream consumer that links against an *installed*
# RCCL via `find_package(rccl)` (RCCL's CMake export uses lowercase
# package name; the equivalent NCCL config-mode upstream uses NCCL).
#
# Usage from a CI shell:
#   ./install.sh -l                           # builds + installs RCCL
#   cmake -S test/cmake_consumer -B /tmp/cc \
#       -DCMAKE_PREFIX_PATH=$RCCL_INSTALL_DIR
#   cmake --build /tmp/cc
#   /tmp/cc/cmake_consumer

cmake_minimum_required(VERSION 3.16)
project(cmake_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to find a packaged copy under either the rccl-specific config or the
# upstream NCCL config-mode name. The RCCL package is named "rccl"; the
# upstream NCCL config-mode package is named "NCCL".
find_package(rccl QUIET)
if(NOT rccl_FOUND)
    find_package(NCCL QUIET)
endif()

if(NOT rccl_FOUND AND NOT NCCL_FOUND)
    message(FATAL_ERROR
        "Neither find_package(rccl) nor find_package(NCCL) succeeded. "
        "Set CMAKE_PREFIX_PATH to the RCCL install prefix.")
endif()

add_executable(cmake_consumer main.cpp)

# Prefer the imported target, fall back to the variables provided by the
# package config file.
if(TARGET rccl)
    target_link_libraries(cmake_consumer PRIVATE rccl)
elseif(TARGET NCCL::NCCL)
    target_link_libraries(cmake_consumer PRIVATE NCCL::NCCL)
elseif(DEFINED NCCL_LIBRARIES)
    target_include_directories(cmake_consumer PRIVATE ${NCCL_INCLUDE_DIRS})
    target_link_libraries(cmake_consumer PRIVATE ${NCCL_LIBRARIES})
else()
    message(FATAL_ERROR
        "RCCL/NCCL package was found but does not expose an importable "
        "target or NCCL_LIBRARIES variable.")
endif()

# Add a CTest invocation so the consumer can be exercised via the normal
# build_and_test pipeline.
enable_testing()
add_test(NAME cmake_consumer_smoke COMMAND cmake_consumer)
