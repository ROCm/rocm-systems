#
# rocprofv3 tool test
#
# Regression test for the hsa-runtime InterceptQueue retry/overflow hang that
# manifested during counter collection. Counter collection registers a
# dispatch_counter_collection context, which forces the non-inline (hsa-runtime
# InterceptQueue) intercept path, so every dispatch is rewritten with counter
# start/stop packets. Under a high dispatch count this stresses the queue
# overflow + retry-barrier logic that previously stranded packets (hang) or
# overcommitted the ring copy (OOB). multistream launches 100 streams x 100
# iterations = 10000 dispatches; a regression hangs (TIMEOUT) or drops
# dispatches (validate fails on the expected count).
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-queue-overflow
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-queue-overflow
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES -d
        ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_queue_overflow -o
        queue_overflow --output-format csv json -- $<TARGET_FILE:multistream>
    DEPENDS multistream
    TIMEOUT 300
    LABELS "integration-tests;counter-collection"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-queue-overflow)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-queue-overflow
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --input
         ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_queue_overflow/queue_overflow_counter_collection.csv
         --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/out_counter_collection_queue_overflow/queue_overflow_results.json
    TIMEOUT 120
    LABELS "integration-tests;counter-collection"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-queue-overflow)
