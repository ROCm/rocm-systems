# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    output-summary-tests
    OBJECT
    format_gpu_ids_tests.cpp
    format_size_human_tests.cpp
    small_process_collapse_tests.cpp
    output_summary_tests.cpp
    perfetto_log_filter_tests.cpp
    process_tree_builder_tests.cpp
    summarize_command_tests.cpp
    output_summary_pipeline_tests.cpp
    output_summary_print_tests.cpp
)

target_link_libraries(
    output-summary-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems::rocprofiler-systems-perfetto
        rocprofiler-systems::rocprofiler-systems-spdlog
)

target_include_directories(
    output-summary-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${CMAKE_CURRENT_SOURCE_DIR}
)
