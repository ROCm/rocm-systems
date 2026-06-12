# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(demangler-tests OBJECT demangler_test.cpp)

target_link_libraries(
    demangler-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(demangler-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)

add_library(
    output-summary-tests
    OBJECT
    format_gpu_ids_tests.cpp
    format_size_human_tests.cpp
    helper_collapser_tests.cpp
    output_file_registry_tests.cpp
    perfetto_log_filter_tests.cpp
    process_tree_builder_tests.cpp
    role_classifier_tests.cpp
    summarize_command_tests.cpp
    summary_renderer_pipeline_tests.cpp
    summary_renderer_tests.cpp
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
