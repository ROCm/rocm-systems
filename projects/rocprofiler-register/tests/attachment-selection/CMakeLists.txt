# Attachment and startup-profiling selection tests

if(TARGET rocprofiler-register::headers)
    set(ATTACHMENT_SELECTION_HEADERS_TARGET rocprofiler-register::headers)
elseif(TARGET rocprofiler-register::rocprofiler-register-headers)
    set(ATTACHMENT_SELECTION_HEADERS_TARGET
        rocprofiler-register::rocprofiler-register-headers)
else()
    message(FATAL_ERROR "rocprofiler-register headers target not found")
endif()

add_library(rocprofiler-register-tests-implicit-attach SHARED)
target_sources(rocprofiler-register-tests-implicit-attach
               PRIVATE attachment-selection-fixture.cpp)
target_compile_definitions(rocprofiler-register-tests-implicit-attach
                           PRIVATE ROCPROFILER_REGISTER_TEST_ATTACH_FIXTURE=1)
target_link_libraries(
    rocprofiler-register-tests-implicit-attach
    PRIVATE ${ATTACHMENT_SELECTION_HEADERS_TARGET} rocprofiler-register::tests-common)
set_target_properties(
    rocprofiler-register-tests-implicit-attach
    PROPERTIES OUTPUT_NAME rocprofiler-sdk-attach
               LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mock-attach")

# A preload can resolve rocprofiler_configure from one of its dependencies. Give both DSOs
# the same basename to verify discovery compares loaded-object identity instead of
# inferring ownership from the filename.
add_library(rocprofiler-register-tests-preload-collision-dependency SHARED)
target_sources(rocprofiler-register-tests-preload-collision-dependency
               PRIVATE attachment-selection-fixture.cpp)
target_compile_definitions(rocprofiler-register-tests-preload-collision-dependency
                           PRIVATE ROCPROFILER_REGISTER_TEST_PRELOAD_DEPENDENCY=1)
target_link_libraries(
    rocprofiler-register-tests-preload-collision-dependency
    PRIVATE ${ATTACHMENT_SELECTION_HEADERS_TARGET} rocprofiler-register::tests-common)
set_target_properties(
    rocprofiler-register-tests-preload-collision-dependency
    PROPERTIES OUTPUT_NAME rocprofiler-register-tests-preload-collision
               NO_SONAME ON
               LIBRARY_OUTPUT_DIRECTORY
               "${CMAKE_CURRENT_BINARY_DIR}/preload-collision/dependency")

add_library(rocprofiler-register-tests-preload-collision-wrapper SHARED)
target_sources(rocprofiler-register-tests-preload-collision-wrapper
               PRIVATE attachment-selection-fixture.cpp)
target_compile_definitions(rocprofiler-register-tests-preload-collision-wrapper
                           PRIVATE ROCPROFILER_REGISTER_TEST_PRELOAD_WRAPPER=1)
# Pass the dependency path directly to the linker. Normal target linking with NO_SONAME
# can emit a generator-dependent, working-directory-relative DT_NEEDED entry.
target_link_libraries(
    rocprofiler-register-tests-preload-collision-wrapper
    PRIVATE ${ATTACHMENT_SELECTION_HEADERS_TARGET} rocprofiler-register::tests-common
            "-Wl,$<TARGET_FILE:rocprofiler-register-tests-preload-collision-dependency>")
add_dependencies(rocprofiler-register-tests-preload-collision-wrapper
                 rocprofiler-register-tests-preload-collision-dependency)
set_target_properties(
    rocprofiler-register-tests-preload-collision-wrapper
    PROPERTIES OUTPUT_NAME rocprofiler-register-tests-preload-collision
               LIBRARY_OUTPUT_DIRECTORY
               "${CMAKE_CURRENT_BINARY_DIR}/preload-collision/wrapper")

set(_ATTACH_REGISTER_PRELOAD_LIBS "librocprofiler-register.so")
set(_ATTACH_TOOL_PRELOAD_LIBS "libgeneric-tool.so")
set(_ATTACH_REGISTER_TOOL_PRELOAD_LIBS "librocprofiler-register.so" "libgeneric-tool.so")
if(ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY)
    list(PREPEND _ATTACH_REGISTER_PRELOAD_LIBS
         "${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY}")
    list(PREPEND _ATTACH_TOOL_PRELOAD_LIBS
         "${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY}")
    list(PREPEND _ATTACH_REGISTER_TOOL_PRELOAD_LIBS
         "${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY}")
endif()
list(JOIN _ATTACH_REGISTER_PRELOAD_LIBS ":" _ATTACH_REGISTER_PRELOAD)
list(JOIN _ATTACH_TOOL_PRELOAD_LIBS ":" _ATTACH_TOOL_PRELOAD)
list(JOIN _ATTACH_REGISTER_TOOL_PRELOAD_LIBS ":" _ATTACH_REGISTER_TOOL_PRELOAD)

set(_PRELOAD_COLLISION_LIBS
    "$<TARGET_FILE:rocprofiler-register-tests-preload-collision-wrapper>")
if(ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY)
    list(PREPEND _PRELOAD_COLLISION_LIBS
         "${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY}")
endif()
list(JOIN _PRELOAD_COLLISION_LIBS ":" _PRELOAD_COLLISION)

add_executable(test-implicit-configure-attachment)
target_sources(test-implicit-configure-attachment PRIVATE test-attachment-selection.cpp)
target_link_libraries(
    test-implicit-configure-attachment
    PRIVATE ${ATTACHMENT_SELECTION_HEADERS_TARGET} rocprofiler-register::tests-common
            hsa-runtime::hsa-runtime)
set_target_properties(test-implicit-configure-attachment PROPERTIES ENABLE_EXPORTS ON)
add_dependencies(
    test-implicit-configure-attachment rocprofiler-register-tests-implicit-attach
    rocprofiler-register-tests-preload-collision-wrapper generic-tool rocprofiler)

add_executable(test-preload-dependency-attachment)
target_sources(test-preload-dependency-attachment PRIVATE test-attachment-selection.cpp)
target_compile_definitions(test-preload-dependency-attachment
                           PRIVATE ROCPROFILER_REGISTER_TEST_NO_AMBIENT_CONFIGURE=1)
target_link_libraries(
    test-preload-dependency-attachment
    PRIVATE ${ATTACHMENT_SELECTION_HEADERS_TARGET} rocprofiler-register::tests-common
            hsa-runtime::hsa-runtime)
add_dependencies(
    test-preload-dependency-attachment rocprofiler-register-tests-implicit-attach
    rocprofiler-register-tests-preload-collision-wrapper)

function(rocp_register_add_attachment_selection_test _NAME)
    cmake_parse_arguments(TEST "" "TARGET;PASS_REGEX;FAIL_REGEX" "ARGS;ENVIRONMENT"
                          ${ARGN})
    if(NOT TEST_TARGET)
        set(TEST_TARGET test-implicit-configure-attachment)
    endif()

    add_test(
        NAME ${_NAME}
        COMMAND $<TARGET_FILE:${TEST_TARGET}> ${TEST_ARGS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(
        ${_NAME}
        PROPERTIES
            ENVIRONMENT
            "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/mock-attach:${CMAKE_LIBRARY_OUTPUT_DIRECTORY};${TEST_ENVIRONMENT}"
            PASS_REGULAR_EXPRESSION
            "${TEST_PASS_REGEX}"
            FAIL_REGULAR_EXPRESSION
            "${TEST_FAIL_REGEX}"
            LABELS
            "attachment;insecure")
endfunction()

rocp_register_add_attachment_selection_test(
    test-implicit-configure-attachment
    ENVIRONMENT
    "ROCP_TOOL_ATTACH=1;ROCPROFILER_REGISTER_LOG_LEVEL=warning;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
    PASS_REGEX
    "Test PASSED: attachment library selected over implicit configure symbol"
    FAIL_REGEX
    "Test FAILED")

rocp_register_add_attachment_selection_test(
    test-runtime-attach-force-load-zero
    ARGS
    --runtime-attach
    ENVIRONMENT
    "ROCP_TOOL_ATTACH=1;ROCPROFILER_REGISTER_FORCE_LOAD=0;ROCPROFILER_REGISTER_LOG_LEVEL=warning;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
    PASS_REGEX
    "Test PASSED: runtime attachment loaded the SDK with FORCE_LOAD=0"
    FAIL_REGEX
    "Test FAILED")

rocp_register_add_attachment_selection_test(
    test-preloaded-register-with-ambient-symbol
    ENVIRONMENT
    "LD_PRELOAD=${_ATTACH_REGISTER_PRELOAD};ROCP_TOOL_ATTACH=1;ROCPROFILER_REGISTER_LOG_LEVEL=warning"
    PASS_REGEX
    "Test PASSED: attachment library selected over implicit configure symbol"
    FAIL_REGEX
    "Test FAILED")

rocp_register_add_attachment_selection_test(
    test-preloaded-dependency-configure-with-ambient-symbol
    TARGET
    test-preload-dependency-attachment
    ARGS
    $<TARGET_FILE:rocprofiler-register-tests-preload-collision-wrapper>
    $<TARGET_FILE:rocprofiler-register-tests-preload-collision-dependency>
    ENVIRONMENT
    "LD_PRELOAD=${_PRELOAD_COLLISION};ROCP_TOOL_ATTACH=1;ROCPROFILER_REGISTER_LOG_LEVEL=warning"
    PASS_REGEX
    "Test PASSED: attachment library selected over implicit configure symbol"
    FAIL_REGEX
    "Test FAILED")

set(_EXPLICIT_SELECTION_FAIL_REGEX
    "Test FAILED|attachment library selected over implicit configure symbol")

rocp_register_add_attachment_selection_test(
    test-explicit-configure-with-attachment
    ENVIRONMENT
    "ROCP_TOOL_ATTACH=1;ROCPROFILER_REGISTER_FORCE_LOAD=1;ROCPROFILER_REGISTER_LOG_LEVEL=warning;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
    PASS_REGEX
    ".rocprofiler.cpp. hsa ::"
    FAIL_REGEX
    "${_EXPLICIT_SELECTION_FAIL_REGEX}")

rocp_register_add_attachment_selection_test(
    test-explicit-tool-libraries-with-attachment
    ENVIRONMENT
    "ROCP_TOOL_ATTACH=1;ROCP_TOOL_LIBRARIES=libgeneric-tool.so;ROCPROFILER_REGISTER_LOG_LEVEL=warning;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
    PASS_REGEX
    ".rocprofiler.cpp. hsa ::"
    FAIL_REGEX
    "${_EXPLICIT_SELECTION_FAIL_REGEX}")

rocp_register_add_attachment_selection_test(
    test-explicit-register-library-with-attachment
    ENVIRONMENT
    "ROCP_TOOL_ATTACH=1;ROCPROFILER_REGISTER_LIBRARY=$<TARGET_SONAME_FILE:rocprofiler>;ROCPROFILER_REGISTER_LOG_LEVEL=warning;${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV}"
    PASS_REGEX
    ".rocprofiler.cpp. hsa ::"
    FAIL_REGEX
    "${_EXPLICIT_SELECTION_FAIL_REGEX}")

rocp_register_add_attachment_selection_test(
    test-preloaded-configure-with-ambient-symbol
    ENVIRONMENT
    "LD_PRELOAD=${_ATTACH_TOOL_PRELOAD};ROCP_TOOL_ATTACH=1;ROCPROFILER_REGISTER_LOG_LEVEL=warning"
    PASS_REGEX
    ".rocprofiler.cpp. hsa ::"
    FAIL_REGEX
    "${_EXPLICIT_SELECTION_FAIL_REGEX}")

add_dependencies(test-amdhip-hsart-roctx generic-tool rocprofiler)
rocp_register_add_attachment_selection_test(
    test-preloaded-configure-with-attachment
    TARGET
    test-amdhip-hsart-roctx
    ENVIRONMENT
    "LD_PRELOAD=${_ATTACH_REGISTER_TOOL_PRELOAD};ROCP_TOOL_ATTACH=1;ROCPROFILER_REGISTER_LOG_LEVEL=warning"
    PASS_REGEX
    ".rocprofiler.cpp. hsa ::"
    FAIL_REGEX
    "attachment library selected over implicit configure symbol")
