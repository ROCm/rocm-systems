# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The guest-visible clock, alone in its own object library because its readers
# are not the timing plane's readers: the KFD ioctl handlers, the command
# processor and the generated ISA execute bodies all read it, and those end up
# in images -- the DBT translation tool, the HSA hook library -- that have no
# business linking the rest of the simulator. It depends on nothing but the
# standard library, so keeping it separate costs one target and keeps that
# dependency edge from existing.
rj_add_object_library(rocjitsu_timing_clock simulated_clock.cpp)

# The timed machine: the components, the event loop they run on, and the
# parser for the `timing` block of the architecture config.
rj_add_object_library(
    rocjitsu_timing
    cache_des.cpp
    classify.cpp
    collector.cpp
    cu_des.cpp
    dispatch_des.cpp
    dram_des.cpp
    engine.cpp
    timed_component.cpp
    timing_plane.cpp
    tuning.cpp
)
# tuning.cpp reads the config block with FlexBuffers, through the same
# JSON-to-FlexBuffers helper the plugin block is read with.
target_link_libraries(rocjitsu_timing PRIVATE flatbuffers)
add_dependencies(rocjitsu_timing flatbuffers_schemas)
