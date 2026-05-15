rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

set(ROCPROFILER_LIB_PC_SAMPLING_TEST_SOURCES
    configure_service.cpp cid_manager.cpp
    # samples_processing.cpp
    pc_sampling_vs_counter_collection.cpp query_configuration.cpp)
set(ROCPROFILER_LIB_PC_SAMPLING_TEST_HEADERS pc_sampling_internals.hpp)

add_executable(pcs-test)

target_sources(pcs-test PRIVATE ${ROCPROFILER_LIB_PC_SAMPLING_TEST_SOURCES}
                                ${ROCPROFILER_LIB_PC_SAMPLING_TEST_HEADERS})

target_link_libraries(
    pcs-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET pcs-test
    SOURCES ${ROCPROFILER_LIB_PC_SAMPLING_TEST_SOURCES}
    TIMEOUT 45
    LABELS "unittests;pc-sampling"
    SKIP_REGULAR_EXPRESSION "PC sampling unavailable"
    ENVIRONMENT
        "LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:/opt/rocm/lib:/opt/rocm/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
