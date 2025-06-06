#
# rocprofv3 rocpd tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-rocpd
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(tracing-env
    "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    "OMPI_ALLOW_RUN_AS_ROOT=1"
    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY conftest.py validate.py)

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

#########################################################################################
#
# generate rocpd database and the old-way outputs csv, otf2, perfetto to compare
#
#########################################################################################

add_test(
    NAME rocprofv3-test-rocpd-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-test -o out --output-format rocpd json
        --runtime-trace --kernel-rename --output-config --pmc SQ_WAVES --
        $<TARGET_FILE:transpose> 2 500 50)

set_tests_properties(
    rocprofv3-test-rocpd-execute
    PROPERTIES TIMEOUT 120 LABELS "integration-tests;rocpd" ENVIRONMENT "${tracing-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-rocpd-execute-multiproc
    COMMAND
        ${MULTIPROC_LAUNCHER} ${Python3_EXECUTABLE}
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-mp-test -o out_transpose_%rank% --output-format
        rocpd json --runtime-trace --kernel-rename --output-config --pmc SQ_WAVES --
        $<TARGET_FILE:transpose> 1 250 50)

set_tests_properties(
    rocprofv3-test-rocpd-execute-multiproc
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests;rocpd"
               ENVIRONMENT
               "${tracing-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${MULTIPROC_IS_DISABLED}")

#########################################################################################
#
# OTF2 generation
#
#########################################################################################

add_test(
    NAME rocprofv3-test-rocpd-otf2-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f otf2 --kernel-rename -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-test -i
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-test/out_results.db)

set_tests_properties(
    rocprofv3-test-rocpd-otf2-generation
    PROPERTIES TIMEOUT 120 LABELS "integration-tests;rocpd" ENVIRONMENT "${tracing-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-rocpd-otf2-generation-multiproc
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f otf2 --kernel-rename -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-test -o rocpd-mp-output-test -i
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-mp-test/out_transpose_0_results.db
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-mp-test/out_transpose_1_results.db)

set_tests_properties(
    rocprofv3-test-rocpd-otf2-generation-multiproc
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests;rocpd"
               ENVIRONMENT
               "${tracing-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${MULTIPROC_IS_DISABLED}")

#########################################################################################
#
# perfetto generate
#
#########################################################################################

add_test(
    NAME rocprofv3-test-rocpd-perfetto-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f pftrace --kernel-rename -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-test -i
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-test/out_results.db)

set_tests_properties(
    rocprofv3-test-rocpd-perfetto-generation
    PROPERTIES TIMEOUT 120 LABELS "integration-tests;rocpd" ENVIRONMENT "${tracing-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-rocpd-perfetto-generation-multiproc
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f perfetto --kernel-rename
        --group-by-queue -d ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-test -o out_mp -i
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-mp-test/out_transpose_0_results.db
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-mp-test/out_transpose_1_results.db)

set_tests_properties(
    rocprofv3-test-rocpd-perfetto-generation-multiproc
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests;rocpd"
               ENVIRONMENT
               "${tracing-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${MULTIPROC_IS_DISABLED}")

#########################################################################################
#
# CSV generate
#
#########################################################################################

add_test(
    NAME rocprofv3-test-rocpd-csv-generation
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f csv --kernel-rename -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-test -i
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-test/out_results.db)

set_tests_properties(
    rocprofv3-test-rocpd-csv-generation
    PROPERTIES TIMEOUT 45 LABELS "integration-tests;rocpd" ENVIRONMENT "${tracing-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

#########################################################################################
#
# Validation
#
#########################################################################################

add_test(
    NAME rocprofv3-test-rocpd-validation
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/transpose-test/out_results.json --otf2-input
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-test/out_results.otf2 --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/rocpd-output-test/out_results.pftrace)

set_tests_properties(
    rocprofv3-test-rocpd-validation
    PROPERTIES TIMEOUT 45 LABELS "integration-tests;rocpd" ENVIRONMENT "${tracing-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
