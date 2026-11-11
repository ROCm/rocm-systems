# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# PMC pollers: the sampled sources of the collection framework
set(pmc_pollers_sources
    ${CMAKE_CURRENT_LIST_DIR}/catalog/cpu.hpp
    ${CMAKE_CURRENT_LIST_DIR}/catalog/gpu_smi.hpp
    ${CMAKE_CURRENT_LIST_DIR}/catalog/nic_smi.hpp
    ${CMAKE_CURRENT_LIST_DIR}/cpu.hpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_perf_counter.hpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_smi.hpp
    ${CMAKE_CURRENT_LIST_DIR}/hipfile.hpp
    ${CMAKE_CURRENT_LIST_DIR}/nic_smi.hpp
    ${CMAKE_CURRENT_LIST_DIR}/row_stream.hpp
)

set(pmc_pollers_sources ${pmc_pollers_sources} PARENT_SCOPE)
