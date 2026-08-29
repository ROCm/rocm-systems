#
# rocpd kernel replay schema tests: drive the shipped SQL against SQLite, no GPU required
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocpd-kernel-replay-schema
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(rocpd-kernel-replay-schema-path "${CMAKE_BINARY_DIR}/share/rocprofiler-sdk-rocpd")

# The tests build their databases from the schema SQL directly, so there is nothing to profile
# and no application to run. The execute step only exists to provide the fixture that
# rocprofiler_add_integration_validate_test requires.
rocprofiler_add_integration_execute_test(
    rocpd-kernel-replay-schema-test
    COMMAND ${Python3_EXECUTABLE} -c "pass"
    TIMEOUT 10
    LABELS "integration-tests;rocpd;kernel-replay"
    FIXTURES_SETUP rocpd-kernel-replay-schema)

rocprofiler_add_integration_validate_test(
    rocpd-kernel-replay-schema-test
    TEST_PATHS validate.py
    CONFIG pytest.ini
    TIMEOUT 60
    LABELS "integration-tests;rocpd;kernel-replay"
    ENVIRONMENT "ROCPD_SCHEMA_PATH=${rocpd-kernel-replay-schema-path}"
    FIXTURES_REQUIRED rocpd-kernel-replay-schema)
