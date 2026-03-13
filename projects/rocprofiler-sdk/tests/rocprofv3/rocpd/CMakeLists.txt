#
# rocprofv3 rocpd tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-rocpd
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(rocprofv3-rocpd-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    "OMPI_ALLOW_RUN_AS_ROOT=1"
    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

find_package(MPI)
find_package(Python3 REQUIRED)

if(MPI_FOUND)
    set(MULTIPROC_IS_DISABLED OFF)
    set(MULTIPROC_LAUNCHER ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                           ${MPIEXEC_PREFLAGS})
else()
    set(MULTIPROC_IS_DISABLED ON)
    set(MULTIPROC_LAUNCHER)
endif()

if(ROCPROFILER_DISABLE_UNSTABLE_CTESTS)
    set(MULTIPROC_IS_DISABLED ON) # fails on mi300 SLES for some unknown reason
endif()

#########################################################################################
#
# generate rocpd database and the old-way outputs csv, otf2, perfetto to compare
#
#########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data -o out --output-format rocpd json
        --runtime-trace --kernel-rename --output-config --pmc SQ_WAVES --
        $<TARGET_FILE:reproducible-dispatch-count> 500 2
    DEPENDS reproducible-dispatch-count
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-rocpd)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-multiproc
    COMMAND
        ${MULTIPROC_LAUNCHER} ${Python3_EXECUTABLE}
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc -o out_mp_%rank%
        --output-format rocpd json --runtime-trace --kernel-rename --output-config --pmc
        SQ_WAVES -- $<TARGET_FILE:reproducible-dispatch-count> 200 1
    DEPENDS reproducible-dispatch-count
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-rocpd-multiproc
    DISABLED "${MULTIPROC_IS_DISABLED}")

# Can remove this test if MULTIPROC is re-enabled and rely on the multiproc databases for
# packaging
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-execute2
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data -o out2 --output-format rocpd
        --runtime-trace --kernel-rename --pmc SQ_WAVES --
        $<TARGET_FILE:reproducible-dispatch-count> 500 2
    DEPENDS reproducible-dispatch-count
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP rocprofv3-test-rocpd)

#########################################################################################
#
# OTF2 generation
#
#########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-otf2-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f otf2 --kernel-rename -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/out_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-generation
    FIXTURES_REQUIRED rocprofv3-test-rocpd)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-otf2-generation-multiproc
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f otf2 --kernel-rename -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data-multiproc -o rocpd-mp-output-test -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc/out_mp_0_results.db
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc/out_mp_1_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-generation
    FIXTURES_REQUIRED rocprofv3-test-rocpd
    DISABLED "${MULTIPROC_IS_DISABLED}")

#########################################################################################
#
# Perfetto generation
#
#########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-perfetto-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f pftrace --kernel-rename -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/out_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-generation
    FIXTURES_REQUIRED rocprofv3-test-rocpd)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-perfetto-generation-multiproc
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f perfetto --kernel-rename
        --group-by-queue -d ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data-multiproc -o
        out_mp -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc/out_mp_0_results.db
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc/out_mp_1_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-generation
    FIXTURES_REQUIRED rocprofv3-test-rocpd
    DISABLED "${MULTIPROC_IS_DISABLED}")

#########################################################################################
#
# CSV generation
#
#########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-csv-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f csv --kernel-rename -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/out_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-generation
    FIXTURES_REQUIRED rocprofv3-test-rocpd)

#########################################################################################
#
# Summary generation
#
#########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-summary-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd summary --domain-summary --summary-by-rank -f csv
        -d ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data/summary -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/out_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-generation
    FIXTURES_REQUIRED rocprofv3-test-rocpd)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-summary-generation-multiproc
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd summary --domain-summary --summary-by-rank -f csv
        -d ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-test/summary -o out_mp -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc/out_mp_0_results.db
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc/out_mp_1_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FIXTURES_SETUP rocprofv3-test-rocpd-generation
    FIXTURES_REQUIRED rocprofv3-test-rocpd
    DISABLED "${MULTIPROC_IS_DISABLED}")

#########################################################################################
#
# Validation
#
#########################################################################################

rocprofiler_add_integration_validate_test(
    rocprofv3-test-rocpd
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/out_results.json
         --otf2-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data/out_results.otf2
         --otf2-sys-tree-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data/out_results.otf2
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data/out_results.pftrace
         --csv-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data/out_agent_info.csv
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data/out_counter_collection_trace.csv
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data/out_kernel_trace.csv
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data/out_memory_allocation_trace.csv
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-data/out_regions_trace.csv
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    FIXTURES_REQUIRED rocprofv3-test-rocpd-generation)

#########################################################################################
#
# Package generation
#
#########################################################################################

# make sure --copy is used with package generation. Package generation will move the
# original databases by default, not copy them. The original databases may be needed for
# tests that have not run yet at this point in the test graph...
#
# Example: when ctest is run with --parallel option, these tests may run before the
# rocprofv3-test-rocpd-generation fixture and, without using --copy, those tests would
# fail since the input database would no longer exist.

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-package-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd package --consolidate --copy -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/test_package -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/out*.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP rocprofv3-test-rocpd-package-generation
    FIXTURES_REQUIRED rocprofv3-test-rocpd)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-package-generation-multiproc
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd package --consolidate --copy -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc/test_package_multi -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc/out*.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED "${MULTIPROC_IS_DISABLED}"
    FIXTURES_SETUP rocprofv3-test-rocpd-package-generation-multiproc
    FIXTURES_REQUIRED rocprofv3-test-rocpd-multiproc)

#########################################################################################
#
# Query SQL string using package
#
#########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-query-using-package
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd query -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/test_package.rpdb --query
        "SELECT id, type FROM rocpd_info_agent LIMIT 10;"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED rocprofv3-test-rocpd-package-generation)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-query-using-package-multiproc
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd query -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc/test_package_multi.rpdb
        --query "SELECT id, type FROM rocpd_info_agent LIMIT 10;"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED "${MULTIPROC_IS_DISABLED}"
    FIXTURES_REQUIRED rocprofv3-test-rocpd-package-generation-multiproc)

#########################################################################################
#
# Merge database generation with package
#
#########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-merge-generation-using-package
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd merge -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data -o merged_db -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/test_package.rpdb
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP rocprofv3-test-rocpd-merge-generation-using-package
    FIXTURES_REQUIRED rocprofv3-test-rocpd-package-generation)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-merge-generation-using-package-multiproc
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd merge -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc -o merged_db_multi.db -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc/test_package_multi.rpdb
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED "${MULTIPROC_IS_DISABLED}"
    FIXTURES_SETUP rocprofv3-test-rocpd-merge-generation-using-package-multiproc
    FIXTURES_REQUIRED rocprofv3-test-rocpd-package-generation-multiproc)

#########################################################################################
#
# Query SQL string using merged database
#
#########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-query-using-mergedDB
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd query -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/merged_db.db --query
        "SELECT id, type FROM rocpd_info_agent LIMIT 10;"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED rocprofv3-test-rocpd-merge-generation-using-package)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-query-using-mergedDB-multiproc
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd query -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc/merged_db_multi.db --query
        "SELECT id, type FROM rocpd_info_agent LIMIT 10;"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED "${MULTIPROC_IS_DISABLED}"
    FIXTURES_REQUIRED rocprofv3-test-rocpd-merge-generation-using-package-multiproc)

#########################################################################################
#
# AI analysis module tests
#
#########################################################################################

# Test the analyze --help flag works
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-analyze-help
    COMMAND ${Python3_EXECUTABLE} -m rocpd analyze --help
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Test standalone module help
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-module-analyze-help
    COMMAND ${Python3_EXECUTABLE} -m rocpd.analyze --help
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Test analyze on existing database
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-analyze
    COMMAND ${Python3_EXECUTABLE} -m rocpd analyze -i
            ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/out_results.db --top-kernels 5
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED rocprofv3-test-rocpd)

# Test analyze with JSON output format (also a fixture for schema validation test)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-analyze-json
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd analyze -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/out_results.db --format json -o
        analysis_results -d ${CMAKE_CURRENT_BINARY_DIR}/rocpd-analyze-output
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP rocprofv3-test-rocpd-analyze-json
    FIXTURES_REQUIRED rocprofv3-test-rocpd)

# Test analyze with custom top-kernels
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-analyze-top-kernels
    COMMAND ${Python3_EXECUTABLE} -m rocpd analyze -i
            ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/out_results.db --top-kernels 20
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED rocprofv3-test-rocpd)

# Test analyze with custom prompt (no LLM, just metadata)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-analyze-custom-prompt
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd analyze -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data/out_results.db --prompt
        "Why is this kernel slow?"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED rocprofv3-test-rocpd)

# Test analyze on multiproc database
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-analyze-multiproc
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd analyze -i
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc/out_mp_0_results.db
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-input-data-multiproc/out_mp_1_results.db
        --top-kernels 5
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    DISABLED "${MULTIPROC_IS_DISABLED}"
    FIXTURES_REQUIRED rocprofv3-test-rocpd-multiproc)

# Test AI analysis Python API import
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-ai-analysis-api
    COMMAND
        ${Python3_EXECUTABLE} -c
        "from rocpd.ai_analysis import analyze_database; print('AI analysis API imported successfully')"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Copy test_analyze.py to build directory to avoid conftest.py issues
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_analyze.py
               ${CMAKE_CURRENT_BINARY_DIR}/test_analyze_standalone.py COPYONLY)

# Test analyze unit tests (run from build dir to avoid conftest.py issues)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-analyze-unit-tests
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_analyze_standalone.py
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd;pytest"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

#########################################################################################
#
# JSON schema tests
#
#########################################################################################

# Copy test_analyze_schema.py to build directory to avoid conftest.py issues
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_analyze_schema.py
               ${CMAKE_CURRENT_BINARY_DIR}/test_analyze_schema_standalone.py COPYONLY)

# Unit schema tests: validate the schema file structure and synthetic JSON output. No
# database fixture needed - uses synthetic data generated in-process.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-analyze-schema-unit-tests
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_BINARY_DIR}/test_analyze_schema_standalone.py
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd;pytest"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Integration schema validation: parse the real JSON output produced by the analyze-json
# test and assert schema_version, required fields, and commands structure.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-analyze-schema-validate
    COMMAND
        ${Python3_EXECUTABLE} -c
        "import json, sys, importlib.resources as pkg; \
         path = '${CMAKE_CURRENT_BINARY_DIR}/rocpd-analyze-output/analysis_results.json'; \
         d = json.load(open(path)); \
         schema = json.loads(pkg.files('rocpd.ai_analysis').joinpath('docs/analysis-output.schema.json').read_text()); \
         allowed = schema['properties']['schema_version']['enum']; \
         assert d.get('schema_version') in allowed, f'Bad schema_version {d.get(\"schema_version\")!r}, expected one of {allowed}'; \
         assert all(k in d for k in ('metadata','recommendations','hardware_counters','hotspots')); \
         cmds = [c for r in d['recommendations'] for c in r.get('commands', [])]; \
         assert len(cmds) > 0, 'No commands found in recommendations'; \
         assert all(c['tool'] in ('rocprofv3','rocprof-sys','rocprof-compute') for c in cmds); \
         print(f'Schema OK: version={d[\"schema_version\"]}, recs={len(d[\"recommendations\"])}, commands={len(cmds)}')"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED rocprofv3-test-rocpd-analyze-json)

#########################################################################################
#
# AI analysis API unit tests
#
#########################################################################################

# Copy test_ai_analysis_standalone.py to build directory to avoid conftest.py issues
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_ai_analysis_standalone.py
               ${CMAKE_CURRENT_BINARY_DIR}/test_ai_analysis_standalone.py COPYONLY)

# AI analysis API unit tests (run from build dir to avoid conftest.py issues) These tests
# do not require a GPU trace fixture — they test the Python API in isolation.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-ai-analysis-unit-tests
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_BINARY_DIR}/test_ai_analysis_standalone.py
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd;pytest"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

#########################################################################################
#
# LLM guide context-aware filtering unit tests
#
#########################################################################################

# Copy test file to build directory to avoid conftest.py discovery issues
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_guide_filter_standalone.py
               ${CMAKE_CURRENT_BINARY_DIR}/test_guide_filter_standalone.py COPYONLY)

# Guide filtering unit tests — no GPU trace fixture required; no LLM API calls made.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-guide-filter-unit-tests
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_BINARY_DIR}/test_guide_filter_standalone.py
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd;pytest"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

#########################################################################################
#
# AI analysis module sub-package unit tests
# Source: source/lib/python/rocpd/ai_analysis/tests/
# Tests are mock-based; no GPU trace fixture or LLM API key required.
# Optional-provider tests (anthropic/openai) are skipped when packages are absent.
#
#########################################################################################

set(_AI_ANALYSIS_TESTS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../source/lib/python/rocpd/ai_analysis/tests")

configure_file("${_AI_ANALYSIS_TESTS_DIR}/test_api_standalone.py"
               "${CMAKE_CURRENT_BINARY_DIR}/test_api_standalone.py" COPYONLY)
configure_file("${_AI_ANALYSIS_TESTS_DIR}/test_llm_conversation.py"
               "${CMAKE_CURRENT_BINARY_DIR}/test_llm_conversation_standalone.py" COPYONLY)
configure_file("${_AI_ANALYSIS_TESTS_DIR}/test_local_llm.py"
               "${CMAKE_CURRENT_BINARY_DIR}/test_local_llm_standalone.py" COPYONLY)
configure_file("${_AI_ANALYSIS_TESTS_DIR}/test_interactive.py"
               "${CMAKE_CURRENT_BINARY_DIR}/test_interactive_standalone.py" COPYONLY)
configure_file("${_AI_ANALYSIS_TESTS_DIR}/test_workflow.py"
               "${CMAKE_CURRENT_BINARY_DIR}/test_workflow_standalone.py" COPYONLY)
configure_file("${_AI_ANALYSIS_TESTS_DIR}/test_tracelens_port.py"
               "${CMAKE_CURRENT_BINARY_DIR}/test_tracelens_port_standalone.py" COPYONLY)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-api-unit-tests
    COMMAND ${Python3_EXECUTABLE} -m pytest --noconftest -x
            ${CMAKE_CURRENT_BINARY_DIR}/test_api_standalone.py
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd;pytest"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-llm-conversation-unit-tests
    COMMAND ${Python3_EXECUTABLE} -m pytest --noconftest -x
            ${CMAKE_CURRENT_BINARY_DIR}/test_llm_conversation_standalone.py
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd;pytest"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-local-llm-unit-tests
    COMMAND ${Python3_EXECUTABLE} -m pytest --noconftest -x
            ${CMAKE_CURRENT_BINARY_DIR}/test_local_llm_standalone.py
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd;pytest"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-interactive-unit-tests
    COMMAND ${Python3_EXECUTABLE} -m pytest --noconftest -x
            ${CMAKE_CURRENT_BINARY_DIR}/test_interactive_standalone.py
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd;pytest"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-workflow-unit-tests
    COMMAND ${Python3_EXECUTABLE} -m pytest --noconftest -x
            ${CMAKE_CURRENT_BINARY_DIR}/test_workflow_standalone.py
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd;pytest"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-tracelens-port-unit-tests
    COMMAND ${Python3_EXECUTABLE} -m pytest --noconftest -x
            ${CMAKE_CURRENT_BINARY_DIR}/test_tracelens_port_standalone.py
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;rocpd;pytest"
    ENVIRONMENT "${rocprofv3-rocpd-env}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
