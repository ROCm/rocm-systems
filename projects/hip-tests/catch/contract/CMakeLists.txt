# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

add_custom_target(contract_tests)

add_subdirectory(memory)
add_subdirectory(transfer)
add_subdirectory(runtime)
add_subdirectory(device)
add_subdirectory(stream_event)
add_subdirectory(async_transfer)
add_subdirectory(memset)
add_subdirectory(error_api)
add_subdirectory(kernel)
add_subdirectory(graph)
add_subdirectory(occupancy)
add_subdirectory(graph_capture)
add_subdirectory(graph_kernel)
add_subdirectory(graph_event)
add_subdirectory(graph_topology)
add_subdirectory(graph_clone)
add_subdirectory(graph_update)
add_subdirectory(host_memory)
add_subdirectory(pitched_memory)
