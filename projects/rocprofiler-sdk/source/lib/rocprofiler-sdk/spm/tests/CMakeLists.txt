rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

find_program(
    amdclangpp_EXECUTABLE REQUIRED
    NAMES amdclang++
    HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES bin llvm/bin NO_CACHE)

function(generate_hsaco TARGET_ID INPUT_FILE OUTPUT_FILE)
    separate_arguments(
        CLANG_ARG_LIST
        UNIX_COMMAND
        "-O2 -x cl -Xclang -finclude-default-header -cl-denorms-are-zero -cl-std=CL2.0 -Wl,--build-id=sha1
    -target amdgcn-amd-amdhsa -mcpu=${TARGET_ID} -o ${OUTPUT_FILE} ${INPUT_FILE}")
    add_custom_command(
        OUTPUT ${PROJECT_BINARY_DIR}/${OUTPUT_FILE}
        COMMAND ${amdclangpp_EXECUTABLE} ${CLANG_ARG_LIST}
        COMMAND
            ${CMAKE_COMMAND} -E copy
            ${PROJECT_BINARY_DIR}/source/lib/rocprofiler-sdk/spm/tests/${OUTPUT_FILE}
            ${CMAKE_BINARY_DIR}/bin/${OUTPUT_FILE}
        OUTPUT ${CMAKE_BINARY_DIR}/bin/${OUTPUT_FILE}
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/bin/${OUTPUT_FILE}
                ${CMAKE_BINARY_DIR}/bin/${OUTPUT_FILE}
        COMMENT "Building ${OUTPUT_FILE}...")
    set(HSACO_TARGET_LIST
        ${HSACO_TARGET_LIST} ${PROJECT_BINARY_DIR}/${OUTPUT_FILE}
        PARENT_SCOPE)
endfunction(generate_hsaco)

foreach(target_id ${GPU_TARGETS})
    # generate kernel bitcodes
    generate_hsaco(${target_id} ${CMAKE_CURRENT_SOURCE_DIR}/agent_kernels.cl
                   ${target_id}_agent_kernels.hsaco)
endforeach()

add_custom_target(spm_agent_hasco_targets DEPENDS ${HSACO_TARGET_LIST})

set(ROCPROFILER_LIB_COUNTER_TEST_SOURCES spm_core.cpp)

add_executable(spm-counter-test)

target_sources(spm-counter-test PRIVATE ${ROCPROFILER_LIB_COUNTER_TEST_SOURCES})

add_dependencies(spm-counter-test agent_hasco_targets)

target_link_libraries(
    spm-counter-test
    PRIVATE rocprofiler-sdk::counter-test-constants
            rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-hip
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library
            GTest::gtest
            GTest::gtest_main)

gtest_add_tests(
    TARGET spm-counter-test
    SOURCES ${ROCPROFILER_LIB_COUNTER_TEST_SOURCES}
    TEST_LIST spm-counter-tests-TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${spm-counter-test_TESTS}
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "unittests"
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION
        "Running non-intercept test(.*)could not be locked for profiling due to lack of permissions.*"
    )
