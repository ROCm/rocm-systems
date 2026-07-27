#
# Broad, data-driven rocprofv3 counter collection option coverage
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-extended-options
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(_args --rocprofv3 $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --vector-ops
          $<TARGET_FILE:vector-ops>)
set(_depends vector-ops)

foreach(_target IN ITEMS reproducible-dispatch-count openmp-target)
    if(TARGET ${_target})
        string(REPLACE "_" "-" _option "${_target}")
        list(APPEND _args --${_option} $<TARGET_FILE:${_target}>)
        list(APPEND _depends ${_target})
    endif()
endforeach()

add_custom_target(rocprofv3-counter-collection-extended-options-dependencies)
add_dependencies(rocprofv3-counter-collection-extended-options-dependencies ${_depends})

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-extended-options
    TEST_PATHS test_extended_options.py
    COPY conftest.py
    CONFIG pytest.ini
    BUNDLE_TESTS
    ARGS ${_args}
    DEPENDS rocprofv3-counter-collection-extended-options-dependencies
    TIMEOUT 600
    LABELS "integration-tests;counter-collection")
