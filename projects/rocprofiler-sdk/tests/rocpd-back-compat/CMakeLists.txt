#
# Backward-compatibility tests for rocpd schema versions.
#
# For each supported schema version:
#   1. An execute test builds a minimal SQLite DB via make_db.py.
#   2. Execute tests convert it to CSV, Perfetto (.pftrace), and OTF2.
#   3. A validate test runs pytest validate.py to assert conversion outputs.
#
# Schema version sets and output directories are defined here in CMake so that
# adding a new version only requires updating ROCPD_COMPAT_TESTED_SCHEMAS.
#
#

# ---------------------------------------------------------------------------
# README: How to add backwards compatibility test
# ---------------------------------------------------------------------------

# * Add your schema version to ROCPD_COMPAT_TESTED_SCHEMAS in this CMakeLists.txt
#
# This list is used to discover the schemas to iterate through to test. The
# ROCPD_OLD_SCHEMAS and ROCPD_LATEST_SCHEMA are derived automatically from
# ROCPD_COMPAT_TESTED_SCHEMAS. No need to add schema version anywhere else.
#
# * Add any new fields or tables to check for in make_db.py
#
# When adding a new schema version, ROCPD must maintain backwards compatibility. However,
# rocprofiler-sdk does not guarantee it can generate a DB based on the older schemas, so
# use make_db.py to generate a small fake DB to convert. If the schema introduced new
# fields/tables, add them to make_db.py script so the new schema changes are exercised.
# Use AI to help generate dummy data for the new rows.
#
# * In validate.py, add a test that can check that a change is PRESENT and ABSENT.
#
# Example: Schema 3.0.2 added graph_launch changes to CSV only. Add a CSV test to check
# the graph_launch fields are ABSENT in the CSV. Use the OLD_SCHEMAS list to run that test
# to ensure the graph_launch fields are not output. Important: The old_schema fixture
# grows over time. When schema 3.0.3 is added, OLD_SCHEMAS becomes [3.0.0, 3.0.1, 3.0.2].
# An ABSENT test for a 3.0.2 feature will then run against schema 3.0.2 itself — and fail,
# because the feature IS present there. Guard against this by version-checking old_schema
# inside the test body: if old_schema >= "3.0.2" (the introducing version), then call the
# PRESENT test instead.
#
# Of course, the PRESENT test should be run by the LATEST_SCHEMA, to ensure latest
# continues to support the graph_launch fields (for example).
#
# NOTE: you may need to delete
# build/tests/rocpd-back-compat/tests.integration.validate.rocpd-back-compat_tests.cmake
# and rerun cmake --build to pickup new validate.py tests.
#
# * Add additional tests for other output formats (as required).
#
# If the schema change also affects PFTRACE or OTF2 output, add tests for those formats as
# well.
#

cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocpd-back-compat
    LANGUAGES NONE
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ---------------------------------------------------------------------------
# Schema version definitions
# ---------------------------------------------------------------------------

# All schema versions with explicit backward-compat test coverage. When a new version
# ships, append it here and add the corresponding validate assertions in validate.py. Add
# appropriate changes to make_db.py to insert dummy data into a fake DB to simulate the
# new schema changes.
set(ROCPD_COMPAT_TESTED_SCHEMAS "3.0.0" "3.0.1" "3.0.2" "3.0.3")

# Derive old schemas (all except the latest) and the latest schema.
list(LENGTH ROCPD_COMPAT_TESTED_SCHEMAS _num_schemas)
math(EXPR _old_count "${_num_schemas} - 1")
list(SUBLIST ROCPD_COMPAT_TESTED_SCHEMAS 0 ${_old_count} ROCPD_OLD_SCHEMAS)
list(GET ROCPD_COMPAT_TESTED_SCHEMAS -1 ROCPD_LATEST_SCHEMA)

# ---------------------------------------------------------------------------
# OTF2 availability check
# ---------------------------------------------------------------------------

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import otf2"
    RESULT_VARIABLE _otf2_import_result
    OUTPUT_QUIET ERROR_QUIET)
if(_otf2_import_result EQUAL 0)
    set(ROCPD_BACK_COMPAT_OTF2_DISABLED OFF)
else()
    set(ROCPD_BACK_COMPAT_OTF2_DISABLED ON)
    message(
        STATUS
            "rocpd-back-compat: otf2 Python package not found — OTF2 generation tests disabled"
        )
endif()

# ---------------------------------------------------------------------------
# Shared environment and output root
# ---------------------------------------------------------------------------

set(rocpd-back-compat-schema-path "${CMAKE_BINARY_DIR}/share/rocprofiler-sdk-rocpd")
set(rocpd-back-compat-site-packages
    "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )
set(rocpd-back-compat-env
    "ROCPD_SCHEMA_PATH=${rocpd-back-compat-schema-path}"
    "PYTHONPATH=${ROCPROFILER_SDK_TESTS_BINARY_DIR}/pytest-packages:${rocpd-back-compat-site-packages}"
    )

# Root directory under which per-version output subdirectories are created:
# <output-root>/<version>/db/schema_<version>.db
# <output-root>/<version>/csv/out_kernel_trace.csv  ...
# <output-root>/<version>/pftrace/out_results.pftrace
# <output-root>/<version>/otf2/out_results.otf2
set(ROCPD_BACK_COMPAT_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocpd-back-compat-output")

# ---------------------------------------------------------------------------
# Per-version execute tests: DB creation + format conversions
# ---------------------------------------------------------------------------

set(_gen_fixtures "")

foreach(VERSION IN LISTS ROCPD_COMPAT_TESTED_SCHEMAS)
    string(REPLACE "." "_" VT "${VERSION}")

    set(_db_file "${ROCPD_BACK_COMPAT_OUTPUT_DIR}/${VERSION}/db/schema_${VT}.db")
    set(_csv_dir "${ROCPD_BACK_COMPAT_OUTPUT_DIR}/${VERSION}/csv")
    set(_pftrace_dir "${ROCPD_BACK_COMPAT_OUTPUT_DIR}/${VERSION}/pftrace")
    set(_otf2_dir "${ROCPD_BACK_COMPAT_OUTPUT_DIR}/${VERSION}/otf2")

    # ------------------------------------------------------------------
    # Create minimal SQLite database for this schema version
    # ------------------------------------------------------------------
    rocprofiler_add_integration_execute_test(
        rocpd-back-compat-${VT}-db
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_db.py --version
                ${VERSION} --output ${_db_file}
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 30
        LABELS "integration-tests;rocpd"
        PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
        ENVIRONMENT "${rocpd-back-compat-env}"
        FIXTURES_SETUP rocpd-back-compat-${VT}-db)

    # ------------------------------------------------------------------
    # CSV conversion
    # ------------------------------------------------------------------
    rocprofiler_add_integration_execute_test(
        rocpd-back-compat-${VT}-csv
        COMMAND ${Python3_EXECUTABLE} -m rocpd convert -f csv -d ${_csv_dir} -o out -i
                ${_db_file}
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 60
        LABELS "integration-tests;rocpd"
        PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
        ENVIRONMENT "${rocpd-back-compat-env}"
        FIXTURES_REQUIRED rocpd-back-compat-${VT}-db
        FIXTURES_SETUP rocpd-back-compat-${VT}-gen)

    # ------------------------------------------------------------------
    # Perfetto conversion
    # ------------------------------------------------------------------
    rocprofiler_add_integration_execute_test(
        rocpd-back-compat-${VT}-pftrace
        COMMAND ${Python3_EXECUTABLE} -m rocpd convert -f pftrace -d ${_pftrace_dir} -o
                out -i ${_db_file}
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 60
        LABELS "integration-tests;rocpd"
        PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
        ENVIRONMENT "${rocpd-back-compat-env}"
        FIXTURES_REQUIRED rocpd-back-compat-${VT}-db
        FIXTURES_SETUP rocpd-back-compat-${VT}-gen)

    # ------------------------------------------------------------------
    # OTF2 conversion (disabled if otf2 Python package is absent)
    # ------------------------------------------------------------------
    rocprofiler_add_integration_execute_test(
        rocpd-back-compat-${VT}-otf2
        COMMAND ${Python3_EXECUTABLE} -m rocpd convert -f otf2 -d ${_otf2_dir} -o out -i
                ${_db_file}
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 60
        LABELS "integration-tests;rocpd"
        PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
        ENVIRONMENT "${rocpd-back-compat-env}"
        FIXTURES_REQUIRED rocpd-back-compat-${VT}-db
        FIXTURES_SETUP rocpd-back-compat-${VT}-gen
        DISABLED "${ROCPD_BACK_COMPAT_OTF2_DISABLED}")

    list(APPEND _gen_fixtures "rocpd-back-compat-${VT}-gen")
endforeach()

# ---------------------------------------------------------------------------
# Validate: pytest suite checks pre-generated outputs
# ---------------------------------------------------------------------------
#
# ARGS are appended to every individual test command (discovered at build time).
# DISCOVERY_ARGS are passed to `pytest --collect-only` so pytest_generate_tests in
# conftest.py can parametrize correctly during cmake's test discovery phase.

rocprofiler_add_integration_validate_test(
    rocpd-back-compat
    TEST_PATHS
    COPY conftest.py validate.py
    CONFIG pytest.ini
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "ROCPD_SCHEMA_PATH=${rocpd-back-compat-schema-path}"
    PYTHON_PATH "${rocpd-back-compat-site-packages}"
    FIXTURES_REQUIRED ${_gen_fixtures}
    DEPENDS rocprofiler-sdk::rocprofv3
    ARGS --output-root ${ROCPD_BACK_COMPAT_OUTPUT_DIR} --tested-schemas
         ${ROCPD_COMPAT_TESTED_SCHEMAS} --old-schemas ${ROCPD_OLD_SCHEMAS} --latest-schema
         ${ROCPD_LATEST_SCHEMA}
    DISCOVERY_ARGS --tested-schemas ${ROCPD_COMPAT_TESTED_SCHEMAS} --old-schemas
                   ${ROCPD_OLD_SCHEMAS} --latest-schema ${ROCPD_LATEST_SCHEMA})
