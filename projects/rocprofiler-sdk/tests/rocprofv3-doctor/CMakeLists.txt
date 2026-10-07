#
# rocprofv3-doctor tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-doctor
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

set(PRELOAD_ENV "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}")

# NOTE: none of these tests require a GPU. rocprofv3-doctor is expected to report skip
# (not fail) for GPU-dependent checks on a machine without one, and the assertions below
# only cover the schema/exit-code contract, so they hold either way.
#
# The two execute tests go through `rocprofv3 --doctor`, as users do; validate.py also
# drives the libexec script directly (rocprofiler-sdk::rocprofv3-doctor) to test it alone.

rocprofiler_add_integration_execute_test(
    rocprofv3-doctor-text
    COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --doctor
            --no-color
    DEPENDS rocprofiler-sdk::rocprofv3 rocprofiler-sdk::rocprofv3-doctor
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION "Summary:"
    DISABLED OFF)

rocprofiler_add_integration_execute_test(
    rocprofv3-doctor-list-checks
    COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --doctor
            --list-checks
    DEPENDS rocprofiler-sdk::rocprofv3 rocprofiler-sdk::rocprofv3-doctor
    TIMEOUT 60
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION "install\\.rocm-root"
    DISABLED OFF)

foreach(
    _FILE IN
    ITEMS conftest.py pytest.ini test_doctor_framework.py test_doctor_checks.py
          test_doctor_layout.py test_doctor_diagnose.py test_doctor_system.py validate.py)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${_FILE}
                   ${CMAKE_CURRENT_BINARY_DIR}/${_FILE} COPYONLY)
endforeach()

# Locations of the configured launcher and doctor script (plain `rocprofv3`, no .py),
# resolved at configure time from the imported targets.
get_target_property(_ROCPROFV3_EXE rocprofiler-sdk::rocprofv3 IMPORTED_LOCATION)
get_target_property(_DOCTOR_EXE rocprofiler-sdk::rocprofv3-doctor IMPORTED_LOCATION)

# GPU-free unit tests for the check framework: registry ordering, dependency validation,
# skip-propagation, exception isolation, filtering, and the individual checks driven
# through a synthetic accessor. Registered through rocprofiler_sdk_pytest_discover_tests
# rather than the validate-test macro so they carry the unittests label: they need no GPU
# and no installed tool.
rocprofiler_sdk_pytest_discover_tests(
    rocprofv3-doctor-unit-tests
    TEST_PATHS test_doctor_framework.py test_doctor_checks.py test_doctor_layout.py
               test_doctor_diagnose.py test_doctor_system.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PYTHON_EXECUTABLE ${Python3_EXECUTABLE}
    # NOTE: rocprofiler-sdk_LIB_DIR, not PROJECT_BINARY_DIR: the project() call above
    # rebinds PROJECT_BINARY_DIR to this test directory, which would point the tests at a
    # path that never exists. This mirrors tests/rocpd/CMakeLists.txt.
    EXTRA_ARGS --rocprofv3-package-dir
               ${rocprofiler-sdk_LIB_DIR}/python3/site-packages
               # the tests run from this binary directory, where the source-tree fallbacks
               # for these two paths do not exist
               --rocprofv3-path
               ${_ROCPROFV3_EXE}
               --doctor-path
               ${_DOCTOR_EXE}
    PROPERTIES
    LABELS "unittests"
    TIMEOUT 120)

rocprofiler_add_integration_validate_test(
    rocprofv3-doctor
    TEST_PATHS validate.py
    COPY conftest.py test_doctor_framework.py test_doctor_checks.py test_doctor_layout.py
         test_doctor_diagnose.py test_doctor_system.py
    CONFIG pytest.ini
    ARGS --doctor-path $<TARGET_FILE:rocprofiler-sdk::rocprofv3-doctor> --rocprofv3-path
         $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
    TIMEOUT 180
    PRELOAD "${PRELOAD_ENV}"
    LABELS "integration-tests"
    DISABLED OFF)
