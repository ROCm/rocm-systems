#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocattach
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocprofiler-sdk-rocattach REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TODO: sanitizer tests do not cooperate with attachment code injection
set(ROCPROFILER_MEMCHECK_TYPES "AddressSanitizer" "UndefinedBehaviorSanitizer"
                               "ThreadSanitizer")

if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "trace")
endif()

set(rocattach-test-env
    "ROCP_TOOL_ATTACH=1"
    "ROCPROFILER_REGISTER_LOG_LEVEL=${LOG_LEVEL}"
    "ROCPROFILER_LOG_LEVEL=${LOG_LEVEL}"
    "ROCATTACH_LOG_LEVEL=${LOG_LEVEL}"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

add_executable(rocattach-parallel-attach-test)
target_sources(rocattach-parallel-attach-test PRIVATE main.cpp)
target_link_libraries(
    rocattach-parallel-attach-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            rocprofiler-sdk::tests-common-library
            rocprofiler-sdk::rocprofiler-sdk-rocattach-shared-library)

add_executable(rocattach-tool-path-validation-test)
target_sources(rocattach-tool-path-validation-test PRIVATE tool_path_validation.cpp)
target_link_libraries(
    rocattach-tool-path-validation-test
    PRIVATE rocprofiler-sdk::tests-build-flags
            rocprofiler-sdk::rocprofiler-sdk-rocattach-shared-library)

if(TARGET rocprofiler-sdk::rocprofiler-sdk-build-flags
   AND TARGET rocprofiler-sdk::rocprofiler-sdk-common-library)
    set(ROCATTACH_SYMBOL_LOOKUP_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../source")
    set(ROCATTACH_SYMBOL_LOOKUP_SOURCE
        "${ROCATTACH_SYMBOL_LOOKUP_SOURCE_ROOT}/lib/rocprofiler-sdk-rocattach/symbol_lookup.cpp"
        )

    if(NOT EXISTS "${ROCATTACH_SYMBOL_LOOKUP_SOURCE}")
        message(
            FATAL_ERROR
                "Missing rocattach symbol lookup source: ${ROCATTACH_SYMBOL_LOOKUP_SOURCE}"
            )
    endif()

    function(rocattach_add_symbol_lookup_fixture NAME VALUE HASH_STYLE)
        add_library(${NAME} SHARED)
        target_sources(${NAME} PRIVATE symbol_lookup_fixture.cpp)
        target_compile_definitions(${NAME} PRIVATE ROCATTACH_FIXTURE_VALUE=${VALUE})
        target_link_libraries(${NAME} PRIVATE rocprofiler-sdk::tests-build-flags)
        set_target_properties(${NAME} PROPERTIES OUTPUT_NAME
                                                 "rocprofiler-register.so.${NAME}")
        if(HASH_STYLE)
            target_link_options(${NAME} PRIVATE "LINKER:--hash-style=${HASH_STYLE}")
        endif()
    endfunction()

    rocattach_add_symbol_lookup_fixture(rocattach-symbol-lookup-fixture-normal 1 "")
    rocattach_add_symbol_lookup_fixture(rocattach-symbol-lookup-fixture-gnu 2 "gnu")
    rocattach_add_symbol_lookup_fixture(rocattach-symbol-lookup-fixture-sysv 3 "sysv")
    rocattach_add_symbol_lookup_fixture(rocattach-symbol-lookup-fixture-shifted 6 "")
    target_compile_definitions(rocattach-symbol-lookup-fixture-shifted
                               PRIVATE ROCATTACH_FIXTURE_PADDED=1)
    rocattach_add_symbol_lookup_fixture(rocattach-symbol-lookup-fixture-ambiguous-a 4 "")
    rocattach_add_symbol_lookup_fixture(rocattach-symbol-lookup-fixture-ambiguous-b 5 "")

    add_executable(rocattach-symbol-lookup-test)
    target_sources(rocattach-symbol-lookup-test
                   PRIVATE symbol_lookup_test.cpp "${ROCATTACH_SYMBOL_LOOKUP_SOURCE}")
    target_include_directories(
        rocattach-symbol-lookup-test
        PRIVATE "${ROCATTACH_SYMBOL_LOOKUP_SOURCE_ROOT}"
                "${ROCATTACH_SYMBOL_LOOKUP_SOURCE_ROOT}/lib/rocprofiler-sdk-rocattach")
    target_link_libraries(
        rocattach-symbol-lookup-test
        PRIVATE rocprofiler-sdk::rocprofiler-sdk-shared-library
                rocprofiler-sdk::rocprofiler-sdk-headers
                rocprofiler-sdk::rocprofiler-sdk-build-flags
                rocprofiler-sdk::rocprofiler-sdk-common-library
                ${CMAKE_DL_LIBS})
    add_dependencies(
        rocattach-symbol-lookup-test
        rocattach-symbol-lookup-fixture-normal
        rocattach-symbol-lookup-fixture-gnu
        rocattach-symbol-lookup-fixture-sysv
        rocattach-symbol-lookup-fixture-shifted
        rocattach-symbol-lookup-fixture-ambiguous-a
        rocattach-symbol-lookup-fixture-ambiguous-b)

    rocprofiler_add_integration_execute_test(
        rocattach-symbol-lookup-test-execute
        TARGET rocattach-symbol-lookup-test
        ARGS $<TARGET_FILE:rocattach-symbol-lookup-fixture-normal>
             $<TARGET_FILE:rocattach-symbol-lookup-fixture-gnu>
             $<TARGET_FILE:rocattach-symbol-lookup-fixture-sysv>
             $<TARGET_FILE:rocattach-symbol-lookup-fixture-shifted>
             $<TARGET_FILE:rocattach-symbol-lookup-fixture-ambiguous-a>
             $<TARGET_FILE:rocattach-symbol-lookup-fixture-ambiguous-b>
        TIMEOUT 30
        LABELS "attachment"
        ENVIRONMENT "${rocattach-test-env}"
        PASS_REGULAR_EXPRESSION
            "Test PASSED: target ELF resolver resolved exact mapped libraries and rejected ambiguous and malformed mappings"
        FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
else()
    message(
        STATUS
            "Skipping rocattach-symbol-lookup-test: internal rocprofiler-sdk targets are unavailable"
        )
endif()

# Utilize rocattach to use a generic C tool in a basic app
rocprofiler_add_integration_execute_test(
    rocattach-parallel-attach-test-execute
    COMMAND $<TARGET_FILE:rocattach-parallel-attach-test> $<TARGET_FILE:attachment-test>
            $<TARGET_FILE:rocprofiler-sdk-c-tool>
    DEPENDS rocattach-parallel-attach-test
    TIMEOUT 45
    LABELS "attachment"
    ENVIRONMENT "${rocattach-test-env}"
    DISABLED ${IS_DISABLED}
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    PASS_REGULAR_EXPRESSION "Test C tool \\(priority=0\\) is using rocprofiler-sdk v"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Verify absolute tool library paths fail early when target-side dlopen could not resolve
# them from the target mount namespace.
rocprofiler_add_integration_execute_test(
    rocattach-missing-tool-path-test-execute
    TARGET rocattach-tool-path-validation-test
    ARGS /tmp/rocattach-missing-tool-library-does-not-exist.so
    TIMEOUT 5
    LABELS "attachment"
    ENVIRONMENT "${rocattach-test-env}"
    DISABLED ${IS_DISABLED}
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    PASS_REGULAR_EXPRESSION "is absolute but is not visible"
    FAIL_REGULAR_EXPRESSION "Expected ROCATTACH_STATUS_ERROR_INVALID_ARGUMENT")

rocprofiler_add_integration_execute_test(
    rocattach-nonregular-tool-path-test-execute
    TARGET rocattach-tool-path-validation-test
    ARGS /tmp
    TIMEOUT 5
    LABELS "attachment"
    ENVIRONMENT "${rocattach-test-env}"
    DISABLED ${IS_DISABLED}
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    PASS_REGULAR_EXPRESSION "does not refer to a regular file"
    FAIL_REGULAR_EXPRESSION "Expected ROCATTACH_STATUS_ERROR_INVALID_ARGUMENT")

rocprofiler_add_integration_execute_test(
    rocattach-empty-tool-path-test-execute
    TARGET rocattach-tool-path-validation-test
    ARGS --empty
    TIMEOUT 5
    LABELS "attachment"
    ENVIRONMENT "${rocattach-test-env}"
    DISABLED ${IS_DISABLED}
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    PASS_REGULAR_EXPRESSION "Tool library path must not be empty"
    FAIL_REGULAR_EXPRESSION "Expected ROCATTACH_STATUS_ERROR_INVALID_ARGUMENT")

# Utilize rocattach to verify signals are correctly forwarded to a target application
# during attachment
rocprofiler_add_integration_execute_test(
    rocattach-parallel-attach-test-signal-execute
    COMMAND $<TARGET_FILE:rocattach-parallel-attach-test> $<TARGET_FILE:attachment-test>
            $<TARGET_FILE:rocprofiler-sdk-c-tool> --send-signal
    DEPENDS rocattach-parallel-attach-test
    TIMEOUT 45
    LABELS "attachment"
    ENVIRONMENT "${rocattach-test-env}"
    DISABLED ${IS_DISABLED}
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    PASS_REGULAR_EXPRESSION "Attachment test process [0-9]+ received signal 28"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_executable(rocattach-attach-tree-test)
target_sources(rocattach-attach-tree-test PRIVATE attach_tree.cpp)
target_link_libraries(
    rocattach-attach-tree-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            rocprofiler-sdk::tests-common-library
            rocprofiler-sdk::rocprofiler-sdk-rocattach-shared-library)

rocprofiler_add_integration_execute_test(
    rocattach-attach-tree-test-execute
    TARGET rocattach-attach-tree-test
    ARGS $<TARGET_FILE:attachment-test> $<TARGET_FILE:rocprofiler-sdk-c-tool>
    TIMEOUT 45
    LABELS "attachment"
    ENVIRONMENT "${rocattach-test-env}"
    DISABLED ${IS_DISABLED}
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    PASS_REGULAR_EXPRESSION "verified: grandchild pid [0-9]+ loaded the tool library"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Test that attach fails gracefully when target process lacks ROCP_TOOL_ATTACH=1. Verifies
# that rocattach returns an error status without crashing the target.
add_executable(rocattach-without-env-test)
target_sources(rocattach-without-env-test PRIVATE attach_without_env.cpp)
target_link_libraries(
    rocattach-without-env-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            rocprofiler-sdk::tests-common-library
            rocprofiler-sdk::rocprofiler-sdk-rocattach-shared-library)

# Deliberately omit ROCP_TOOL_ATTACH=1 to test error handling
set(rocattach-without-env-test-env
    "ROCPROFILER_REGISTER_LOG_LEVEL=${LOG_LEVEL}"
    "ROCPROFILER_LOG_LEVEL=${LOG_LEVEL}"
    "ROCATTACH_LOG_LEVEL=${LOG_LEVEL}"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

rocprofiler_add_integration_execute_test(
    rocattach-without-env-test-execute
    TARGET rocattach-without-env-test
    ARGS $<TARGET_FILE:attachment-test> attach
    TIMEOUT 45
    DISABLED ${IS_DISABLED}
    LABELS "attachment"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${rocattach-without-env-test-env}"
    PASS_REGULAR_EXPRESSION
        "Test PASSED: attach failed gracefully without crashing target"
    FAIL_REGULAR_EXPRESSION "Test FAILED|error: attach succeeded|unexpected attach status"
    )

rocprofiler_add_integration_execute_test(
    rocattach-without-env-tree-test-execute
    TARGET rocattach-without-env-test
    ARGS $<TARGET_FILE:attachment-test> attach-tree
    TIMEOUT 45
    DISABLED ${IS_DISABLED}
    LABELS "attachment"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${rocattach-without-env-test-env}"
    PASS_REGULAR_EXPRESSION
        "Test PASSED: attach failed gracefully without crashing target"
    FAIL_REGULAR_EXPRESSION "Test FAILED|error: attach succeeded|unexpected attach status"
    )
