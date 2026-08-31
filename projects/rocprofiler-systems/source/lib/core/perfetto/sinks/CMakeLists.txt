# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(perfetto_sink_sources
    ${CMAKE_CURRENT_LIST_DIR}/per_pid_file_sink.cpp
    ${CMAKE_CURRENT_LIST_DIR}/recording_sink.cpp
    ${CMAKE_CURRENT_LIST_DIR}/single_file_sink.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tee_sink.cpp
)

set(perfetto_sink_headers
    ${CMAKE_CURRENT_LIST_DIR}/append_mode.hpp
    ${CMAKE_CURRENT_LIST_DIR}/io_helpers.hpp
    ${CMAKE_CURRENT_LIST_DIR}/per_pid_file_sink.hpp
    ${CMAKE_CURRENT_LIST_DIR}/polymorphic_sink_view.hpp
    ${CMAKE_CURRENT_LIST_DIR}/recording_sink.hpp
    ${CMAKE_CURRENT_LIST_DIR}/single_file_sink.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tee_sink.hpp
    ${CMAKE_CURRENT_LIST_DIR}/trace_sink.hpp
)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${perfetto_sink_sources} ${perfetto_sink_headers}
)
