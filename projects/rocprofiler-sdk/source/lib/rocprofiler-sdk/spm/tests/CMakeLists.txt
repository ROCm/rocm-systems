rocprofiler_deactivate_clang_tidy()

project(rocprofiler-sdk-unit-tests-spm-counters LANGUAGES C CXX)

include(GoogleTest)

find_program(
    amdclangpp_EXECUTABLE REQUIRED
    NAMES amdclang++
    HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES bin llvm/bin NO_CACHE)

set(_ROCPROFILER_SHARE_DIR
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}")

set(ROCPROFILER_LIB_SPM_COUNTER_TEST_SOURCES core.cpp spm_memory_pool_test.cpp)

add_executable(spm-counter-test)

target_sources(spm-counter-test PRIVATE ${ROCPROFILER_LIB_SPM_COUNTER_TEST_SOURCES})

add_dependencies(spm-counter-test agent_hsaco_targets)

target_link_libraries(
    spm-counter-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-hip
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library
            GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET spm-counter-test
    SOURCES ${ROCPROFILER_LIB_SPM_COUNTER_TEST_SOURCES}
    TIMEOUT 60
    ENVIRONMENT
        "ROCPROFILER_METRICS_PATH=${_ROCPROFILER_SHARE_DIR}"
        "ROCPROFILER_SPM_BETA_ENABLED=True"
        "LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:/opt/rocm/lib:/opt/rocm/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    SKIP_REGULAR_EXPRESSION "SPM unavailable"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

set_tests_properties(unit.spm_core.check_packet_generation unit.spm_core.check_callbacks
                     PROPERTIES TIMEOUT 180)
