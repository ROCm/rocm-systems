# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Existing builds may retain an interpreter without the host-test dependencies.
# Check availability without importing PyTorch or changing test registration.
execute_process(
    COMMAND
        "${Python3_EXECUTABLE}" -c
        "import importlib.util; print(', '.join(m for m in ('torch', 'yaml') if importlib.util.find_spec(m) is None))"
    OUTPUT_VARIABLE _consan_missing_python_modules
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _consan_python_probe_result
)
if(NOT _consan_python_probe_result STREQUAL "0")
    message(
        WARNING
        "Could not check ConSan Python test dependencies with ${Python3_EXECUTABLE}."
    )
elseif(_consan_missing_python_modules)
    message(
        WARNING
        "Python3_EXECUTABLE=${Python3_EXECUTABLE} cannot find these ConSan Python "
        "test dependencies: ${_consan_missing_python_modules}. To run these "
        "suites, install ${CMAKE_CURRENT_LIST_DIR}/requirements.txt in a separate "
        "virtual environment and reconfigure with "
        "-DPython3_EXECUTABLE=<venv>/bin/python. All suites remain registered."
    )
endif()
unset(_consan_missing_python_modules)
unset(_consan_python_probe_result)

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
