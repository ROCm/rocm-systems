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

rocprofiler_add_integration_execute_test(
    rocprofv3-test-rocpd-package-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd package --consolidate -d
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
        ${Python3_EXECUTABLE} -m rocpd package --consolidate -d
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
