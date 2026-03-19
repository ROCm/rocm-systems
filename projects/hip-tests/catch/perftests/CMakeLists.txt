# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

add_custom_target(perf_test COMMAND "${CMAKE_CTEST_COMMAND}" -R "Perf_"
                            COMMENT "Build complete, now executing the performnce test ...")

add_subdirectory(memory)
add_subdirectory(stream)
add_subdirectory(dispatch)
add_subdirectory(compute)
add_subdirectory(graph)
add_subdirectory(event)
add_subdirectory(vmm)