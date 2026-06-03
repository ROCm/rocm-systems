###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

include(${CMAKE_SOURCE_DIR}/cmake/setup_project.cmake)

find_package(MPI)

add_executable(gin_qp_factory_test gin_qp_factory_test.cpp)

target_include_directories(gin_qp_factory_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(gin_qp_factory_test
  PRIVATE
    roc::rocshmem
    $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
)
