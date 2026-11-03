rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

set(ROCPROFILER_LIB_PC_SAMPLING_TEST_SOURCES
    configure_service.cpp
    cid_manager.cpp
    # samples_processing.cpp
    pc_sampling_vs_counter_collection.cpp
    query_configuration.cpp
    multiple_contexts_one_per_agent.cpp
    queue_hooks_test.cpp
    local_context.cpp)
set(ROCPROFILER_LIB_PC_SAMPLING_TEST_HEADERS pc_sampling_internals.hpp)

add_executable(pcs-test)

target_sources(pcs-test PRIVATE ${ROCPROFILER_LIB_PC_SAMPLING_TEST_SOURCES}
                                ${ROCPROFILER_LIB_PC_SAMPLING_TEST_HEADERS})

target_link_libraries(
    pcs-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            rocprofiler-sdk::rocprofiler-sdk-static-library
            GTest::gtest
            GTest::gtest_main)

# pcs-test links the static library, so the counter definitions (config.yaml) are looked
# up relative to the test executable. That works in the build tree (<build>/bin) but not
# for the installed copy of the test (share/rocprofiler-sdk/tests/unit-tests/bin), so
# point the lookup at the share directory explicitly like the other counter-related unit
# tests.
set(_ROCPROFILER_SHARE_DIR
    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}")

rocprofiler_add_unit_test(
    TARGET pcs-test
    SOURCES ${ROCPROFILER_LIB_PC_SAMPLING_TEST_SOURCES}
    TIMEOUT 45
    LABELS "unittests;pc-sampling"
    SKIP_REGULAR_EXPRESSION "PC sampling unavailable"
    ENVIRONMENT
        "ROCPROFILER_METRICS_PATH=${_ROCPROFILER_SHARE_DIR}"
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:$ENV{LD_LIBRARY_PATH}"
        "ROCPROFILER_PC_SAMPLING_BETA_ENABLED=on"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
