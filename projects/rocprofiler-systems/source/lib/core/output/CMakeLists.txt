# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

target_sources(
    rocprofiler-systems-core-library
    PRIVATE
        output_summary.cpp
        perfetto_log_filter.cpp
        process_tree_builder.cpp
        run_metadata.cpp
        text_layout.cpp
        output_summary.hpp
        perfetto_log_filter.hpp
        perfetto_log_filter_detail.hpp
        process_metadata.hpp
        process_tree_builder.hpp
        run_metadata.hpp
        text_layout.hpp
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
