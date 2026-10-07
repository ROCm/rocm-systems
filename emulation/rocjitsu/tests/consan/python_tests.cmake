# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# These are host unit tests using temporary fixtures and mocked workloads.
# Discover new validation suites automatically so CTest cannot silently omit
# target admission, fault-runner, coverage, or provenance regressions.
file(
    GLOB _consan_python_suites
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/../dbi/consan/test_consan_*.py"
)
foreach(_suite IN LISTS _consan_python_suites)
    get_filename_component(_module "${_suite}" NAME_WE)
    # Preserve the names of suites already exposed to CTest callers.
    if(_module STREQUAL "test_consan_benchmark")
        set(_name ConSanBenchmarkUnitTest)
    elseif(_module STREQUAL "test_consan_sweep")
        set(_name ConSanSweepUnitTest)
    elseif(_module STREQUAL "test_consan_hipblaslt_benchmark_workload")
        set(_name ConSanHipblasltBenchmarkWorkloadUnitTest)
    else()
        set(_name "ConSanPython.${_module}")
    endif()
    add_test(NAME "${_name}" COMMAND "${Python3_EXECUTABLE}" "${_suite}" -v)
    set_tests_properties("${_name}" PROPERTIES LABELS "consan;consan-python")
endforeach()
unset(_consan_python_suites)
unset(_suite)
unset(_module)
unset(_name)
