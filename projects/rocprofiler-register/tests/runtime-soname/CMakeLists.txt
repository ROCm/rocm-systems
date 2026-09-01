#
# Runtime library-name loading tests
#

set(RUNTIME_SONAME_SDK_V0_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/fixtures/sdk-v0")
set(RUNTIME_SONAME_SDK_V1_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/fixtures/sdk-v1")
set(RUNTIME_SONAME_ATTACH_V1_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/fixtures/attach-v1")

if(TARGET rocprofiler-register::headers)
    set(RUNTIME_SONAME_HEADERS_TARGET rocprofiler-register::headers)
elseif(TARGET rocprofiler-register::rocprofiler-register-headers)
    set(RUNTIME_SONAME_HEADERS_TARGET rocprofiler-register::rocprofiler-register-headers)
else()
    message(FATAL_ERROR "rocprofiler-register headers target not found")
endif()

add_library(runtime-soname-sdk-fixture SHARED runtime-soname-fixture.cpp)
target_compile_definitions(runtime-soname-sdk-fixture
                           PRIVATE ROCPROFILER_REGISTER_TEST_SDK_FIXTURE=1)
target_link_libraries(runtime-soname-sdk-fixture PRIVATE ${RUNTIME_SONAME_HEADERS_TARGET})
set_target_properties(
    runtime-soname-sdk-fixture
    PROPERTIES OUTPUT_NAME rocprofiler-sdk
               SOVERSION 1
               VERSION 1.0.0
               LIBRARY_OUTPUT_DIRECTORY "${RUNTIME_SONAME_SDK_V1_FIXTURE_DIR}")

add_library(runtime-soname-sdk-v0-fixture SHARED runtime-soname-fixture.cpp)
target_compile_definitions(runtime-soname-sdk-v0-fixture
                           PRIVATE ROCPROFILER_REGISTER_TEST_SDK_FIXTURE=1)
target_link_libraries(runtime-soname-sdk-v0-fixture
                      PRIVATE ${RUNTIME_SONAME_HEADERS_TARGET})
set_target_properties(
    runtime-soname-sdk-v0-fixture
    PROPERTIES OUTPUT_NAME rocprofiler-sdk
               SOVERSION 0
               VERSION 0.7.0
               LIBRARY_OUTPUT_DIRECTORY "${RUNTIME_SONAME_SDK_V0_FIXTURE_DIR}")

add_library(runtime-soname-attach-fixture SHARED runtime-soname-fixture.cpp)
target_compile_definitions(runtime-soname-attach-fixture
                           PRIVATE ROCPROFILER_REGISTER_TEST_ATTACH_FIXTURE=1)
target_link_libraries(runtime-soname-attach-fixture
                      PRIVATE ${RUNTIME_SONAME_HEADERS_TARGET})
set_target_properties(
    runtime-soname-attach-fixture
    PROPERTIES OUTPUT_NAME rocprofiler-sdk-attach
               SOVERSION 1
               VERSION 1.0.0
               LIBRARY_OUTPUT_DIRECTORY "${RUNTIME_SONAME_ATTACH_V1_FIXTURE_DIR}")

if(TARGET rocprofiler-register)
    set(RUNTIME_SONAME_REGISTER_TARGET rocprofiler-register)
else()
    set(RUNTIME_SONAME_REGISTER_TARGET rocprofiler-register::rocprofiler-register)
endif()

add_executable(test-runtime-soname-only test-runtime-soname-only.cpp)
target_link_libraries(test-runtime-soname-only PRIVATE rocprofiler-register::tests-common
                                                       hsa-runtime::hsa-runtime)
add_dependencies(test-runtime-soname-only runtime-soname-sdk-fixture
                 runtime-soname-sdk-v0-fixture runtime-soname-attach-fixture)

function(rocprofiler_register_add_runtime_library_layout_test _TEST_NAME _LAYOUT)
    add_test(
        NAME ${_TEST_NAME}
        COMMAND
            $<TARGET_FILE:test-runtime-soname-only> ${_LAYOUT}
            $<TARGET_FILE:${RUNTIME_SONAME_REGISTER_TARGET}>
            $<TARGET_SONAME_FILE_NAME:${RUNTIME_SONAME_REGISTER_TARGET}>
            $<TARGET_FILE:runtime-soname-sdk-fixture>
            $<TARGET_SONAME_FILE_NAME:runtime-soname-sdk-fixture>
            $<TARGET_LINKER_FILE_NAME:runtime-soname-sdk-fixture>
            $<TARGET_FILE:runtime-soname-sdk-v0-fixture>
            $<TARGET_SONAME_FILE_NAME:runtime-soname-sdk-v0-fixture>
            $<TARGET_FILE:runtime-soname-attach-fixture>
            $<TARGET_SONAME_FILE_NAME:runtime-soname-attach-fixture>
            $<TARGET_LINKER_FILE_NAME:runtime-soname-attach-fixture>
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(
        ${_TEST_NAME}
        PROPERTIES ENVIRONMENT
                   "${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
                   TIMEOUT
                   10
                   PASS_REGULAR_EXPRESSION
                   "Test PASSED: ${_LAYOUT}"
                   FAIL_REGULAR_EXPRESSION
                   "Test FAILED"
                   LABELS
                   "dlopen;insecure;runtime")
endfunction()

rocprofiler_register_add_runtime_library_layout_test(test-runtime-soname-only soname-v1)
rocprofiler_register_add_runtime_library_layout_test(test-runtime-unversioned-only
                                                     unversioned)
rocprofiler_register_add_runtime_library_layout_test(test-runtime-sdk-soversion-zero
                                                     sdk-soversion-zero)
rocprofiler_register_add_runtime_library_layout_test(test-runtime-sdk-soversion-precedence
                                                     sdk-soversion-precedence)
rocprofiler_register_add_runtime_library_layout_test(test-runtime-sdk-entrypoint-fallback
                                                     sdk-entrypoint-fallback)
rocprofiler_register_add_runtime_library_layout_test(test-runtime-sdk-colocated-precedence
                                                     sdk-colocated-precedence)
rocprofiler_register_add_runtime_library_layout_test(
    test-runtime-sdk-explicit-override-no-fallback sdk-explicit-override-no-fallback)
