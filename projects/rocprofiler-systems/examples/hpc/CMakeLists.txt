# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(rocprofiler-systems-hpc-examples LANGUAGES C CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

if(CMAKE_PROJECT_NAME STREQUAL "rocprofiler-systems")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

include(${CMAKE_CURRENT_LIST_DIR}/hpc-helper.cmake)

# HPCTrainingExamples/Pragma_examples/OpenMP/Fortran/8_jacobi/3_jacobi_targetdata_markers
add_subdirectory(jacobi-fortran-targetdata-markers)

# HPCTrainingExamples/Pragma_examples/OpenMP/Fortran/8_jacobi/1_jacobi_usm
# Only runs on GPUs that support xnack (instinct cards)
add_subdirectory(jacobi-fortran-usm)

# HPCTrainingExamples/HIP/jacobi
add_subdirectory(jacobi-hip)

# External
add_subdirectory(julia)

# HPCtrainingExamples/Libraries/matrix_exponential/streams_sync/hip
add_subdirectory(matrix-exponential-streams-sync-hip)

# HPCtrainingExamples/HIP/Stream_Overlap/1-split-copy-compute-hq-queues
add_subdirectory(split-copy-compute-hw-queues)
