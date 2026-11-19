#
# rocpd command-line tests
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocpd
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Test Install Build configures this directory as its own project, so CMAKE_BINARY_DIR
# does not contain the schema. The common helper records the directory that does.
if(ROCPROFILER_SDK_TESTS_ROCPD_SCHEMA_PATH)
    set(rocpd-schema-path "${ROCPROFILER_SDK_TESTS_ROCPD_SCHEMA_PATH}")
else()
    set(rocpd-schema-path "${CMAKE_BINARY_DIR}/share/rocprofiler-sdk-rocpd")
endif()
set(rocpd-site-packages
    "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )
set(rocpd-env "ROCPD_SCHEMA_PATH=${rocpd-schema-path}"
              "PYTHONPATH=${rocpd-site-packages}")

#########################################################################################
#
# Test the --help flag works
#
#########################################################################################

foreach(_SUBPARSER "" "-convert" "-query" "-summary")
    string(REPLACE "-" "" _CMD "${_SUBPARSER}")

    rocprofiler_add_integration_execute_test(
        rocpd${_SUBPARSER}-help
        COMMAND ${Python3_EXECUTABLE} -m rocpd ${_CMD} --help
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 120
        LABELS "integration-tests;rocpd"
        PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
        ENVIRONMENT "${rocpd-env}")
endforeach()

foreach(_MODULE "csv" "pftrace" "otf2" "query" "summary")
    rocprofiler_add_integration_execute_test(
        rocpd-module-${_MODULE}-help
        COMMAND ${Python3_EXECUTABLE} -m rocpd.${_MODULE} --help
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 120
        LABELS "integration-tests;rocpd"
        PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
        ENVIRONMENT "${rocpd-env}")
endforeach()

# validate.py holds more than one group of tests, so each validate test names the
# functions it runs. Parametrized tests are listed by function name and run every case.
rocprofiler_add_integration_validate_test(
    rocpd-time-window
    CONFIG pytest.ini validate.py
    BUNDLE_TESTS
    ARGS validate.py::test_loader_finds_installed_module_without_source
         validate.py::test_loader_finds_build_source_without_package
         validate.py::test_bounds_include_timestamp_only_tables
         validate.py::test_pc_sample_only_bounds_and_filtering
         validate.py::test_native_sqlite_uses_event_bounds_not_process_envelope
         validate.py::test_bounds_query_aggregates_each_table_before_union
         validate.py::test_timestamp_only_events_are_not_out_of_range
         validate.py::test_no_timed_events_warning_accounts_for_timestamp_tables
         validate.py::test_empty_window_reports_no_timed_events
         validate.py::test_zero_length_trace_does_not_divide_by_zero
         validate.py::test_valid_percentage_is_converted
         validate.py::test_numeric_endpoints_are_accepted
         validate.py::test_large_timestamp_boundaries_are_preserved
         validate.py::test_percentage_offsets_preserve_integer_nanoseconds
         validate.py::test_malformed_percentages_are_rejected
         validate.py::test_overhanging_percentages_are_clamped
         validate.py::test_non_finite_timestamps_are_rejected
         validate.py::test_invalid_timestamp_identifies_the_argument
         validate.py::test_overhanging_bounds_are_clamped
         validate.py::test_bounds_inside_the_trace_are_not_clamped
         validate.py::test_disjoint_window_is_rejected
         validate.py::test_missing_timing_data_is_reported
         validate.py::test_missing_marker_view_has_specific_error
         validate.py::test_marker_schema_errors_are_not_reclassified
         validate.py::test_unrelated_database_errors_are_not_reclassified
         validate.py::test_unknown_marker_is_reported
         validate.py::test_ambiguous_marker_is_reported
         validate.py::test_marker_endpoints_read_their_own_column
         validate.py::test_end_marker_uses_the_marker_end_timestamp
         validate.py::test_invalid_marker_type_is_rejected
         validate.py::test_reversed_markers_report_marker_arguments
         validate.py::test_reversed_timestamps_report_time_arguments
         validate.py::test_mixed_marker_and_timestamp_endpoints_are_supported
         validate.py::test_same_endpoint_cannot_use_time_and_marker
         validate.py::test_no_window_arguments_is_a_no_op
         validate.py::test_process_args_or_exit_reports_time_window_errors
         validate.py::test_process_args_or_exit_does_not_swallow_other_errors
         validate.py::test_standalone_main_reports_invalid_window_without_traceback
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "ROCPD_SCHEMA_PATH=${rocpd-schema-path}"
    PYTHON_PATH "${rocpd-site-packages}" LIBRARY_PATH "${rocprofiler-sdk_LIB_DIR}"
    DEPENDS rocprofiler-sdk::rocprofv3)

# Pure Python test of the rank variable selection in rocpd.summary. summary.py is loaded
# by path with stubbed importer/query modules, so this needs no profiled database, GPU,
# libpyrocpd, or generation fixture.
rocprofiler_add_integration_validate_test(
    rocpd-rank-env
    CONFIG pytest.ini validate.py
    BUNDLE_TESTS
    ARGS validate.py::test_rank_env_var_rejects_malicious_name
         validate.py::test_rank_env_cache_is_per_connection
    TIMEOUT 60
    LABELS "integration-tests;rocpd"
    PYTHON_PATH "${rocpd-site-packages}")

#########################################################################################
#
# PC-sampling post-processing on synthetic databases (no GPU required)
#
#########################################################################################

set(_rocpd_python_bindings_dependency
    rocprofiler-sdk-rocpd-python-bindings-${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}
    )
if(NOT TARGET ${_rocpd_python_bindings_dependency})
    set(_rocpd_python_bindings_dependency "")
endif()

rocprofiler_add_integration_validate_test(
    rocpd-pc-sampling-postprocess
    TEST_PATHS pc_sampling_postprocess.py
    CONFIG pytest.ini
    TIMEOUT 300
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    DEPENDS ${_rocpd_python_bindings_dependency})
