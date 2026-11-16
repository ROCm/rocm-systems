# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Waitcheck's target policies can be tested independently of the checker driver.
rj_add_object_library(rocjitsu_waitcheck_target waitcheck/target.cpp)
target_link_libraries(
    rocjitsu_waitcheck_target
    INTERFACE $<TARGET_OBJECTS:rocjitsu_wait_counter_policy>
)

# State transfer depends on target policy, independently of the checker driver.
rj_add_object_library(rocjitsu_waitcheck_state waitcheck/state.cpp)
target_link_libraries(rocjitsu_waitcheck_state PUBLIC rocjitsu_waitcheck_target)

# The stream checker consumes target policies and pending-event transfer.
rj_add_object_library(rocjitsu_waitcheck_stream waitcheck/stream.cpp)
target_link_libraries(rocjitsu_waitcheck_stream PUBLIC rocjitsu_waitcheck_state)
