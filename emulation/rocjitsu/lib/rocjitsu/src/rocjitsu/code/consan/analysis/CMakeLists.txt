# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    rocjitsu_consan_analysis
    STATIC
    analyze.cpp
    render.cpp
    lds_race.cpp
    lds_race_render.cpp
)
target_link_libraries(rocjitsu_consan_analysis PUBLIC rocjitsu_consan_contract)
rj_configure_target(rocjitsu_consan_analysis WARNINGS HIDDEN)
