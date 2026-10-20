# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocjitsu_consan_compiler STATIC input.cpp prepare.cpp lds_race.cpp)
target_link_libraries(
    rocjitsu_consan_compiler
    PUBLIC rocjitsu_consan_image rocjitsu_dbt_internal simdojo_headers
)
rj_configure_target(rocjitsu_consan_compiler PRIVATE_INCLUDES WARNINGS HIDDEN)
