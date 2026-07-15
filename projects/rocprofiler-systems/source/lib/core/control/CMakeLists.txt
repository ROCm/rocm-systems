# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(control_sources
    ${CMAKE_CURRENT_LIST_DIR}/clocks/steady.cpp
    ${CMAKE_CURRENT_LIST_DIR}/session.cpp
    ${CMAKE_CURRENT_LIST_DIR}/triggers/roctx.cpp
)

set(control_headers
    ${CMAKE_CURRENT_LIST_DIR}/clock.hpp
    ${CMAKE_CURRENT_LIST_DIR}/clocks/steady.hpp
    ${CMAKE_CURRENT_LIST_DIR}/session.hpp
    ${CMAKE_CURRENT_LIST_DIR}/subscriber.hpp
    ${CMAKE_CURRENT_LIST_DIR}/trigger.hpp
    ${CMAKE_CURRENT_LIST_DIR}/triggers/roctx.hpp
    ${CMAKE_CURRENT_LIST_DIR}/triggers/time_window.hpp
)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${control_sources} ${control_headers}
)
