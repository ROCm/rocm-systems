# Attachment coexistence tests

if(TARGET rocprofiler-register::headers)
    set(ATTACHMENT_TEST_HEADERS_TARGET rocprofiler-register::headers)
elseif(TARGET rocprofiler-register::rocprofiler-register-headers)
    set(ATTACHMENT_TEST_HEADERS_TARGET rocprofiler-register::rocprofiler-register-headers)
else()
    message(FATAL_ERROR "rocprofiler-register headers target not found")
endif()

add_library(rocprofiler-register-tests-attach-fixture SHARED)
target_sources(rocprofiler-register-tests-attach-fixture
               PRIVATE attachment-selection-fixture.cpp)
target_link_libraries(
    rocprofiler-register-tests-attach-fixture PRIVATE ${ATTACHMENT_TEST_HEADERS_TARGET}
                                                      rocprofiler-register::tests-common)
set_target_properties(
    rocprofiler-register-tests-attach-fixture
    PROPERTIES OUTPUT_NAME rocprofiler-sdk-attach
               LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mock-attach")

add_executable(test-ambient-configure-attachment)
target_sources(test-ambient-configure-attachment PRIVATE test_attachment_selection.cpp)
target_link_libraries(
    test-ambient-configure-attachment
    PRIVATE ${ATTACHMENT_TEST_HEADERS_TARGET} rocprofiler-register::tests-common
            hsa-runtime::hsa-runtime)
set_target_properties(test-ambient-configure-attachment PROPERTIES ENABLE_EXPORTS ON)
add_dependencies(test-ambient-configure-attachment
                 rocprofiler-register-tests-attach-fixture rocprofiler)

add_executable(test-anytime-runtime-attachment)
target_sources(test-anytime-runtime-attachment PRIVATE test_attachment_selection.cpp)
target_compile_definitions(test-anytime-runtime-attachment
                           PRIVATE ROCPROFILER_REGISTER_TEST_NO_AMBIENT_CONFIGURE=1)
target_link_libraries(
    test-anytime-runtime-attachment
    PRIVATE ${ATTACHMENT_TEST_HEADERS_TARGET} rocprofiler-register::tests-common
            hsa-runtime::hsa-runtime)
add_dependencies(test-anytime-runtime-attachment
                 rocprofiler-register-tests-attach-fixture generic-tool rocprofiler)

function(rocp_register_add_attachment_test NAME)
    cmake_parse_arguments(TEST "" "TARGET" "ARGS;ENVIRONMENT" ${ARGN})
    add_test(
        NAME ${NAME}
        COMMAND $<TARGET_FILE:${TEST_TARGET}> ${TEST_ARGS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(
        ${NAME}
        PROPERTIES
            ENVIRONMENT
            "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/mock-attach:${CMAKE_LIBRARY_OUTPUT_DIRECTORY};ROCP_TOOL_ATTACH=1;ROCPROFILER_REGISTER_LOG_LEVEL=warning;${TEST_ENVIRONMENT}"
            PASS_REGULAR_EXPRESSION
            "Test PASSED: attachment coexists with startup initialization"
            FAIL_REGULAR_EXPRESSION
            "Test FAILED"
            LABELS
            "attachment;insecure")
endfunction()

rocp_register_add_attachment_test(
    test-ambient-configure-attachment TARGET test-ambient-configure-attachment
    ENVIRONMENT "${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}")

rocp_register_add_attachment_test(
    test-anytime-runtime-attachment TARGET test-anytime-runtime-attachment ARGS
    --runtime-attach ENVIRONMENT
    "ROCPROFILER_REGISTER_FORCE_LOAD=0;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}")
