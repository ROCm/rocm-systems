# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(perfetto_sources
    ${CMAKE_CURRENT_LIST_DIR}/cached_perfetto_session.cpp
    ${CMAKE_CURRENT_LIST_DIR}/emitter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/engine.cpp
    ${CMAKE_CURRENT_LIST_DIR}/locked_file_append.cpp
)

set(perfetto_headers
    ${CMAKE_CURRENT_LIST_DIR}/cached_perfetto_session.hpp
    ${CMAKE_CURRENT_LIST_DIR}/category_registry.hpp
    ${CMAKE_CURRENT_LIST_DIR}/counter_track.hpp
    ${CMAKE_CURRENT_LIST_DIR}/emitter.hpp
    ${CMAKE_CURRENT_LIST_DIR}/engine.hpp
    ${CMAKE_CURRENT_LIST_DIR}/engine_impl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/fwd.hpp
    ${CMAKE_CURRENT_LIST_DIR}/locked_file_append.hpp
    ${CMAKE_CURRENT_LIST_DIR}/packet_framing.hpp
    ${CMAKE_CURRENT_LIST_DIR}/session_backend.hpp
)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${perfetto_sources} ${perfetto_headers}
)

add_subdirectory(sinks)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
